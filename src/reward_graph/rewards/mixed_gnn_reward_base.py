'''
Base class and shared functions for mixed GNN reward.
Provides BaseBatchMixedGNNRewardWithWarmup and factory generators.
Related: mixed_gnn_reward_batch_{qwen3b,qwen1_5b}.py variants.
'''

import os
import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from reward_graph.rewards.utils.gnn_models import UnifiedGNNDotProduct
from reward_graph.utils.cache_utils import load_or_create_unified_cache
from reward_graph.rewards.utils.multi_domain_reward import math_reward, qa_reward, code_reward

logger = logging.getLogger(__name__)


def extract_math_answer(response: str) -> float:
    '''
    Numeric answer extractor for math responses.
    Parses ####, \\boxed{}, or last number from response text.
    Related: math_reward() in multi_domain_reward.py.
    '''
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', response)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass
    numbers = re.findall(r'-?[\d,]+\.?\d*', response)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass
    return float('nan')


def extract_qa_answer(response: str) -> str:
    '''
    Letter choice extractor for QA responses.
    Parses ####, "the answer is", or trailing letter patterns.
    Related: qa_reward() in multi_domain_reward.py.
    '''
    response_upper = response.upper()

    match = re.search(r'####\s*([A-E])\b', response_upper)
    if match:
        return match.group(1)

    match = re.search(r'(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+IS\s*:?\s*([A-E])\b', response_upper)
    if match:
        return match.group(1)

    match = re.search(r'\b([A-D])\b\s*$', response_upper.strip())
    if match:
        return match.group(1)

    return 'X'


def split_think_and_answer(full_answer: str) -> Tuple[str, str]:
    '''
    Think-answer splitter for response text.
    Splits on #### delimiter into (think, answer) tuple.
    Related: _gnn_predict_with_warmup() embedding pipeline.
    '''
    if '####' in full_answer:
        parts = full_answer.split('####')
        think = parts[0].strip()
        answer = parts[1].strip() if len(parts) > 1 else ""
        return think, answer
    else:
        return full_answer, full_answer[:100] if full_answer else ""


class BaseBatchMixedGNNRewardWithWarmup:
    '''
    Base class for batch mixed GT/GNN reward with warmup graph.
    Subclasses set class attributes and override hooks for variant behavior.
    Related: compute_score() VERL entry point.
    '''

    # Subclass must set these class attributes:
    DEFAULT_CHECKPOINT: str = ''
    DEFAULT_WARMUP: str = ''
    DEFAULT_GT: str = ''
    CACHE_DIR_NAME: str = ''
    CACHE_PREFIX: str = ''
    SUPPORTS_MULTI_ARCH: bool = True
    ENV_KEYS: tuple = ()
    DOMAIN_FILTER_QQ_EDGES: bool = False

    def __init__(
        self,
        gnn_checkpoint_path: str = None,
        warmup_embeddings_path: str = None,
        gt_identifiers_path: str = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str = "cuda",
        intra_k: int = 7,
        intra_query_think_k: int = 0,
        think_cross_k: int = 0,
        num_responses_per_query: int = 8,
    ):
        gnn_checkpoint_path, gt_identifiers_path, warmup_embeddings_path = \
            self._resolve_paths(gnn_checkpoint_path, gt_identifiers_path, warmup_embeddings_path)

        self._setup_device()

        self.gt_identifiers = self._load_gt_identifiers(gt_identifiers_path)
        self.intra_k = intra_k
        self.intra_query_think_k = intra_query_think_k
        self.think_cross_k = think_cross_k
        self.num_responses_per_query = num_responses_per_query

        logger.info(f"[WarmupGNN] Loading embedding model: {embedding_model}")
        self.embed_model = SentenceTransformer(embedding_model)
        self.embed_model = self.embed_model.to(self.device)
        self.embed_dim = self.embed_model.get_sentence_embedding_dimension()

        logger.warning(f"[GNN-INIT] Loading GNN from: {gnn_checkpoint_path}")
        print(f"[GNN-INIT] Loading GNN from: {gnn_checkpoint_path}", flush=True)

        self._load_gnn_model(gnn_checkpoint_path)

        logger.info(f"[WarmupGNN] Loading warmup embeddings from: {warmup_embeddings_path}")
        self._load_warmup_embeddings(warmup_embeddings_path)

        self.stats = {'gt_count': 0, 'gnn_count': 0}

    def _resolve_paths(self, gnn_checkpoint_path, gt_identifiers_path, warmup_embeddings_path):
        '''
        Path resolver with env var overrides.
        Reads env vars listed in ENV_KEYS, falls back to class defaults.
        Related: __init__() path setup.
        '''
        if 'GNN_CHECKPOINT_PATH' in self.ENV_KEYS:
            env_val = os.environ.get('GNN_CHECKPOINT_PATH')
            if env_val:
                gnn_checkpoint_path = env_val
                logger.info(f"[GNN-INIT] Using checkpoint from GNN_CHECKPOINT_PATH env: {gnn_checkpoint_path}")
                print(f"[GNN-INIT] Using checkpoint from GNN_CHECKPOINT_PATH env: {gnn_checkpoint_path}", flush=True)

        if gnn_checkpoint_path is None:
            gnn_checkpoint_path = self.DEFAULT_CHECKPOINT

        if 'GT_IDENTIFIERS_PATH' in self.ENV_KEYS:
            env_val = os.environ.get('GT_IDENTIFIERS_PATH')
            if env_val:
                gt_identifiers_path = env_val
                logger.info(f"[GNN-INIT] Using GT identifiers from GT_IDENTIFIERS_PATH env: {gt_identifiers_path}")
                print(f"[GNN-INIT] Using GT identifiers from GT_IDENTIFIERS_PATH env: {gt_identifiers_path}", flush=True)

        if gt_identifiers_path is None:
            gt_identifiers_path = self.DEFAULT_GT

        if 'WARMUP_EMBEDDINGS_PATH' in self.ENV_KEYS:
            env_val = os.environ.get('WARMUP_EMBEDDINGS_PATH')
            if env_val:
                warmup_embeddings_path = env_val
                logger.info(f"[GNN-INIT] Using warmup embeddings from WARMUP_EMBEDDINGS_PATH env: {warmup_embeddings_path}")
                print(f"[GNN-INIT] Using warmup embeddings from WARMUP_EMBEDDINGS_PATH env: {warmup_embeddings_path}", flush=True)

        if warmup_embeddings_path is None:
            warmup_embeddings_path = self.DEFAULT_WARMUP

        return gnn_checkpoint_path, gt_identifiers_path, warmup_embeddings_path

    def _setup_device(self):
        '''
        GPU/CPU device selector.
        Uses GNN_CUDA_DEVICE env var or auto-detects available GPU.
        Related: __init__() device setup.
        '''
        gnn_cuda_device = os.environ.get('GNN_CUDA_DEVICE', None)
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'all')

        if gnn_cuda_device:
            os.environ['CUDA_VISIBLE_DEVICES'] = gnn_cuda_device
            self.device = torch.device("cuda:0")
            logger.warning(f"[GNN-INIT] Using GNN_CUDA_DEVICE={gnn_cuda_device} -> cuda:0")
            print(f"[GNN-INIT] Using GNN_CUDA_DEVICE={gnn_cuda_device} -> cuda:0", flush=True)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logger.warning(f"[GNN-INIT] Using GPU: cuda:0 (CUDA_VISIBLE_DEVICES={cuda_visible})")
            print(f"[GNN-INIT] Using GPU: cuda:0 (CUDA_VISIBLE_DEVICES={cuda_visible})", flush=True)
        else:
            self.device = torch.device("cpu")
            logger.warning(f"[GNN-INIT] WARNING: Using CPU (no GPU available)")
            print(f"[GNN-INIT] WARNING: Using CPU (no GPU available)", flush=True)

    def _load_gnn_model(self, gnn_checkpoint_path):
        '''
        GNN model loader with answer feature dim auto-detection.
        Loads UnifiedGNNDotProduct from checkpoint state_dict.
        Related: gnn_models.py UnifiedGNNDotProduct class.
        '''
        state_dict = torch.load(gnn_checkpoint_path, map_location=self.device)

        self.model_type = 'dotproduct'
        if 'answer_feat_proj.0.weight' in state_dict:
            self.answer_feat_dim = state_dict['answer_feat_proj.0.weight'].shape[1]
        else:
            self.answer_feat_dim = 6
        logger.info(f"[GNN-INIT] Dot Product model (answer_feat_dim={self.answer_feat_dim})")
        print(f"[GNN-INIT] Dot Product model (answer_feat_dim={self.answer_feat_dim})", flush=True)
        self.gnn = UnifiedGNNDotProduct(
            query_dim=self.embed_dim, think_dim=self.embed_dim, answer_dim=self.embed_dim,
            answer_feat_dim=self.answer_feat_dim, hidden_dim=512, num_layers=2, num_heads=4, dropout=0.1,
        ).to(self.device)

        self.gnn.load_state_dict(state_dict, strict=True)
        self.gnn.eval()
        logger.warning(f"[GNN-INIT] {self.model_type} GNN loaded successfully! Device: {self.device}")
        print(f"[GNN-INIT] {self.model_type} GNN loaded successfully! Device: {self.device}", flush=True)

    def _load_warmup_embeddings(self, path: str):
        '''
        Warmup embedding loader from training cache.
        Loads query/think/answer embeddings and domain labels.
        Related: load_or_create_unified_cache() in cache_utils.py.
        '''
        path = Path(path)

        if not path.exists() and path.parent.name == self.CACHE_DIR_NAME:
            cache_dir = path.parent.parent
            logger.info(f"[WarmupGNN] {self.CACHE_DIR_NAME} not found, auto-merging caches from {cache_dir}")
            warmup_data = load_or_create_unified_cache(str(cache_dir), prefix=self.CACHE_PREFIX)
        else:
            warmup_data = torch.load(path, map_location=self.device, weights_only=False)

        self.warmup_query_emb = warmup_data['query_embeddings'].to(self.device)
        self.warmup_think_emb = warmup_data['think_embeddings'].to(self.device)
        self.warmup_answer_emb = warmup_data['answer_embeddings'].to(self.device)

        raw_domains = warmup_data.get('domains', ['math'] * len(self.warmup_query_emb))
        self.warmup_domains = [self._map_warmup_domain(d) for d in raw_domains]

        self.num_warmup_queries = self.warmup_query_emb.shape[0]
        self.num_warmup_responses = self.warmup_think_emb.shape[0]

        if 'answer_features' in warmup_data:
            af = warmup_data['answer_features'].to(self.device)
            if af.shape[1] > self.answer_feat_dim:
                af = af[:, :self.answer_feat_dim]
                logger.info(f"[WarmupGNN] Truncated warmup answer_features to dim={self.answer_feat_dim}")
            elif af.shape[1] < self.answer_feat_dim:
                af = F.pad(af, (0, self.answer_feat_dim - af.shape[1]))
                logger.info(f"[WarmupGNN] Padded warmup answer_features to dim={self.answer_feat_dim}")
            self.warmup_answer_features = af
            logger.info(f"[WarmupGNN] Loaded warmup answer features: {self.warmup_answer_features.shape}")
        else:
            logger.warning(f"[WarmupGNN] No answer_features in warmup, initializing zeros")
            self.warmup_answer_features = torch.zeros(
                self.num_warmup_responses, self.answer_feat_dim, device=self.device
            )

        domain_counts = {}
        for d in self.warmup_domains:
            domain_counts[d] = domain_counts.get(d, 0) + 1
        logger.info(f"[WarmupGNN] Loaded warmup: {self.num_warmup_queries} queries, "
                   f"{self.num_warmup_responses} responses")
        logger.info(f"[WarmupGNN] Warmup domain distribution: {domain_counts}")

    def _map_warmup_domain(self, domain: str) -> str:
        '''
        Dataset-to-domain mapper (gsm8k->math, mmlu->qa, etc.).
        Groups 10 datasets into 3 GNN domains: math, qa, code.
        Related: _map_domain() for data_source routing.
        '''
        ds = domain.lower()
        if ds in ['gsm8k', 'math', 'gsm_symbolic']:
            return 'math'
        elif ds in ['mmlu', 'commonsenseqa', 'obqa', 'arc_c', 'gpqa']:
            return 'qa'
        elif ds in ['humaneval_plus', 'mbpp_plus', 'humaneval', 'mbpp', 'apps']:
            return 'code'
        if 'gsm' in ds or 'math' in ds:
            return 'math'
        elif 'qa' in ds or 'mmlu' in ds:
            return 'qa'
        elif 'code' in ds or 'humaneval' in ds or 'mbpp' in ds:
            return 'code'
        return 'math'

    def _load_gt_identifiers(self, path: str) -> Dict[str, set]:
        '''
        GT identifier loader for data-source routing.
        Loads (dataset, index) sets from gt_identifiers_train{ratio}.json.
        Related: _should_use_gt_data_source() routing decision.
        '''
        import json

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"GT identifiers file not found: {path}\n"
                "Please run the GT identifier generation script first:\n"
                "  python scripts/generate_gt_identifiers.py"
            )

        gt_ids = {'math': set(), 'qa': set(), 'code': set()}

        key_mapping = {
            'gsm8k': 'math', 'math': 'math', 'gsm_symbolic': 'math',
            'numina_math': 'math',
            'mmlu': 'qa', 'gpqa': 'qa', 'commonsenseqa': 'qa',
            'obqa': 'qa', 'arc_c': 'qa', 'qa': 'qa',
            'siqa': 'qa', 'piqa': 'qa',
            'humaneval_plus': 'code', 'mbpp_plus': 'code',
            'humaneval': 'code', 'mbpp': 'code', 'apps': 'code',
        }

        with open(path, 'r') as f:
            data = json.load(f)

        for json_key, value in data.items():
            internal_key = key_mapping.get(json_key.lower())
            if internal_key:
                if isinstance(value, dict) and 'indices' in value:
                    indices_list = value['indices']
                elif isinstance(value, list):
                    indices_list = value
                else:
                    logger.warning(f"[WarmupGNN] Unknown format for {json_key}: {type(value)}")
                    continue

                for x in indices_list:
                    try:
                        idx = int(float(x))
                        gt_ids[internal_key].add((json_key.lower(), idx))
                    except (ValueError, TypeError):
                        gt_ids[internal_key].add((json_key.lower(), x))

        total = sum(len(v) for v in gt_ids.values())
        logger.info(f"[WarmupGNN] Loaded GT identifiers: {total} total")
        logger.info(f"[WarmupGNN]   - math: {len(gt_ids['math'])}")
        logger.info(f"[WarmupGNN]   - qa: {len(gt_ids['qa'])}")
        logger.info(f"[WarmupGNN]   - code: {len(gt_ids['code'])}")

        return gt_ids

    def _get_query_identifier(self, data_source: str, extra_info: Dict[str, Any]) -> Optional[Any]:
        '''
        Query identifier extractor from extra_info.
        Returns (dataset_name, index) tuple for GT routing lookup.
        Related: _should_use_gt_data_source() and _load_gt_identifiers().
        '''
        ds = data_source.lower()

        dataset_name = ds
        if 'gsm_symbolic' in ds:
            dataset_name = 'gsm_symbolic'
        elif 'gsm8k' in ds or 'gsm' in ds:
            dataset_name = 'gsm8k'
        elif 'mmlu' in ds:
            dataset_name = 'mmlu'
        elif 'commonsenseqa' in ds:
            dataset_name = 'commonsenseqa'
        elif 'obqa' in ds:
            dataset_name = 'obqa'
        elif 'arc_c' in ds or 'arc-c' in ds:
            dataset_name = 'arc_c'
        elif 'gpqa' in ds:
            dataset_name = 'gpqa'
        elif 'humaneval' in ds:
            dataset_name = 'humaneval_plus' if 'plus' in ds else 'humaneval'
        elif 'mbpp' in ds:
            dataset_name = 'mbpp_plus' if 'plus' in ds else 'mbpp'
        elif 'numina' in ds:
            dataset_name = 'numina_math'
        elif 'siqa' in ds:
            dataset_name = 'siqa'
        elif 'piqa' in ds:
            dataset_name = 'piqa'
        elif 'math' in ds:
            dataset_name = 'math'
        elif 'apps' in ds:
            dataset_name = 'apps'

        idx = extra_info.get('index')
        if idx is not None:
            try:
                return (dataset_name, int(idx))
            except (ValueError, TypeError):
                return (dataset_name, idx)

        if 'humaneval' in ds or 'mbpp' in ds or 'apps' in ds:
            pid = extra_info.get('problem_id')
            if pid is not None:
                try:
                    return (dataset_name, int(float(pid)))
                except (ValueError, TypeError):
                    return (dataset_name, pid)

        return None

    def _should_use_gt_data_source(self, data_source: str, extra_info: Dict[str, Any]) -> bool:
        '''
        GT routing decision for a query.
        Checks if (dataset, index) is in gt_identifiers set.
        Related: _get_query_identifier() and _load_gt_identifiers().
        '''
        ds = data_source.lower()
        identifier = self._get_query_identifier(data_source, extra_info)

        if identifier is None:
            return False

        if 'math' in ds or 'gsm' in ds:
            return identifier in self.gt_identifiers.get('math', set())
        elif any(kw in ds for kw in ['mmlu', 'gpqa', 'commonsenseqa', 'obqa', 'arc_c']):
            return identifier in self.gt_identifiers.get('qa', set())
        elif 'humaneval' in ds or 'mbpp' in ds or 'apps' in ds:
            return identifier in self.gt_identifiers.get('code', set())

        return False

    def _compute_gt_reward(self, response: str, domain: str, extra_info: Dict[str, Any]) -> float:
        '''
        Ground truth reward calculator.
        Dispatches to math_reward, qa_reward, or code_reward by domain.
        Related: math_reward(), qa_reward(), code_reward() in multi_domain_reward.py.
        '''
        if domain == 'math':
            return math_reward(response, extra_info.get('answer', ''))
        elif domain == 'qa':
            return qa_reward(response, extra_info.get('answer', ''))
        elif domain in ('code', 'coding'):
            test_list = extra_info.get('test_list', [])
            return code_reward(response, test_list, timeout_seconds=5)
        return 0.0

    def get_answer_features(self, responses: list, domain: str) -> torch.Tensor:
        '''
        Answer feature extractor for GNN input.
        Computes 3-dim features (value, validity, consensus) per response.
        Override in subclass for different QA feature encoding.
        Related: UnifiedGNNDotProduct.forward() answer_features parameter.
        '''
        features = []

        if domain == 'math':
            answers = [extract_math_answer(r) for r in responses]
            valid_answers = [a for a in answers if not np.isnan(a)]
            if valid_answers:
                answer_counts = Counter([round(a, 2) for a in valid_answers])
                most_common = answer_counts.most_common(1)[0][0] if answer_counts else None
            else:
                most_common = None

            for ans in answers:
                if np.isnan(ans):
                    features.append([0.0, 0.0, 0.0])
                else:
                    norm_ans = np.sign(ans) * np.log1p(abs(ans)) / 20.0
                    norm_ans = np.clip(norm_ans, -1.0, 1.0)
                    matches = 1.0 if most_common and abs(round(ans, 2) - most_common) < 0.01 else 0.0
                    features.append([norm_ans, 1.0, matches])

        elif domain == 'qa':
            for _ in responses:
                features.append([0.0, 0.0, 0.0])

        else:
            for _ in responses:
                features.append([0.0, 0.0, 0.0])

        return torch.tensor(features, dtype=torch.float32)

    def _build_qq_edges(self, sim_matrix, total_queries, new_query_domains):
        '''
        Query-query kNN edge builder (no domain filtering).
        Override with DOMAIN_FILTER_QQ_EDGES=True for domain-filtered variant.
        Related: _build_extended_graph() edge construction.
        '''
        qq_edges = []

        if not self.DOMAIN_FILTER_QQ_EDGES:
            k_warmup = min(self.intra_k, self.num_warmup_queries - 1) if self.num_warmup_queries > 1 else 0
            for i in range(self.num_warmup_queries):
                if k_warmup > 0:
                    warmup_sims = sim_matrix[i, :self.num_warmup_queries].copy()
                    warmup_sims[i] = -1
                    top_k_idx = np.argsort(warmup_sims)[-k_warmup:]
                    for j in top_k_idx:
                        if warmup_sims[j] > -1:
                            qq_edges.append([i, int(j)])

            for i in range(self.num_warmup_queries, total_queries):
                if self.num_warmup_queries > 0:
                    warmup_sims = sim_matrix[i, :self.num_warmup_queries].copy()
                    k_new = min(self.intra_k, self.num_warmup_queries)
                    top_k_idx = np.argsort(warmup_sims)[-k_new:]
                    for j in top_k_idx:
                        if warmup_sims[j] > -1:
                            qq_edges.append([i, int(j)])
        else:
            k_warmup = min(self.intra_k, self.num_warmup_queries - 1) if self.num_warmup_queries > 1 else 0
            for i in range(self.num_warmup_queries):
                if k_warmup > 0:
                    warmup_sims = sim_matrix[i, :self.num_warmup_queries].copy()
                    warmup_sims[i] = -1
                    for j in range(self.num_warmup_queries):
                        if self.warmup_domains[i] != self.warmup_domains[j]:
                            warmup_sims[j] = -2
                    top_k_idx = np.argsort(warmup_sims)[-k_warmup:]
                    for j in top_k_idx:
                        if warmup_sims[j] > -1:
                            qq_edges.append([i, int(j)])

            for i in range(self.num_warmup_queries, total_queries):
                if self.num_warmup_queries > 0:
                    new_query_idx = i - self.num_warmup_queries
                    warmup_sims = sim_matrix[i, :self.num_warmup_queries].copy()

                    if new_query_domains is not None and new_query_idx < len(new_query_domains):
                        new_domain = new_query_domains[new_query_idx]
                        for j in range(self.num_warmup_queries):
                            if self.warmup_domains[j] != new_domain:
                                warmup_sims[j] = -2

                    k_new = min(self.intra_k, self.num_warmup_queries)
                    top_k_idx = np.argsort(warmup_sims)[-k_new:]
                    for j in top_k_idx:
                        if warmup_sims[j] > -1:
                            qq_edges.append([i, int(j)])

        return qq_edges

    def _build_extended_graph(
        self,
        new_query_emb: torch.Tensor,
        new_think_emb: torch.Tensor,
        new_answer_emb: torch.Tensor,
        num_new_queries: int,
        new_query_domains: List[str] = None,
    ) -> HeteroData:
        '''
        Extended graph builder combining warmup and new queries.
        Constructs HeteroData with query-think-answer nodes and kNN edges.
        Related: build_independent_domain_graph() in graph_builders.py.
        '''
        all_query_emb = torch.cat([self.warmup_query_emb, new_query_emb], dim=0)
        all_think_emb = torch.cat([self.warmup_think_emb, new_think_emb], dim=0)
        all_answer_emb = torch.cat([self.warmup_answer_emb, new_answer_emb], dim=0)

        total_queries = self.num_warmup_queries + num_new_queries
        total_responses = self.num_warmup_responses + new_think_emb.shape[0]

        graph = HeteroData()
        graph['query'].x = all_query_emb
        graph['think'].x = all_think_emb
        graph['answer'].x = all_answer_emb

        # Edge 1: Query -> Answer (structural)
        qa_edges = []
        for q_idx in range(self.num_warmup_queries):
            for r_offset in range(self.num_responses_per_query):
                r_idx = q_idx * self.num_responses_per_query + r_offset
                if r_idx < self.num_warmup_responses:
                    qa_edges.append([q_idx, r_idx])
        for q_idx in range(num_new_queries):
            global_q_idx = self.num_warmup_queries + q_idx
            for r_offset in range(self.num_responses_per_query):
                r_idx = self.num_warmup_responses + q_idx * self.num_responses_per_query + r_offset
                if r_idx < total_responses:
                    qa_edges.append([global_q_idx, r_idx])

        graph[('query', 'generates', 'answer')].edge_index = torch.tensor(
            qa_edges, dtype=torch.long, device=self.device
        ).t().contiguous()
        graph[('answer', 'rev_generates', 'query')].edge_index = graph[
            ('query', 'generates', 'answer')
        ].edge_index.flip(0)

        # Edge 2: Query -> Think (same structure)
        graph[('query', 'generates_reasoning', 'think')].edge_index = graph[
            ('query', 'generates', 'answer')
        ].edge_index.clone()
        graph[('think', 'rev_generates_reasoning', 'query')].edge_index = graph[
            ('query', 'generates_reasoning', 'think')
        ].edge_index.flip(0)

        # Edge 3: Think -> Answer (one-to-one)
        ta_edges = [[i, i] for i in range(total_responses)]
        graph[('think', 'leads_to', 'answer')].edge_index = torch.tensor(
            ta_edges, dtype=torch.long, device=self.device
        ).t().contiguous()
        graph[('answer', 'rev_leads_to', 'think')].edge_index = graph[
            ('think', 'leads_to', 'answer')
        ].edge_index.flip(0)

        # Edge 4: Query <-> Query (kNN, delegates to _build_qq_edges)
        query_emb_np = all_query_emb.cpu().numpy()
        sim_matrix = cosine_similarity(query_emb_np)
        np.fill_diagonal(sim_matrix, -1)

        qq_edges = self._build_qq_edges(sim_matrix, total_queries, new_query_domains)

        graph[('query', 'similar_to', 'query')].edge_index = torch.tensor(
            qq_edges, dtype=torch.long, device=self.device
        ).t().contiguous()

        # Edge 5: Think <-> Think (same-query competition + cross-query kNN)
        tt_edges = []

        if self.intra_query_think_k != 0:
            if self.intra_query_think_k == -1:
                for q_idx in range(self.num_warmup_queries):
                    start = q_idx * self.num_responses_per_query
                    for i in range(self.num_responses_per_query):
                        for j in range(i + 1, self.num_responses_per_query):
                            t_i = start + i
                            t_j = start + j
                            if t_i < self.num_warmup_responses and t_j < self.num_warmup_responses:
                                tt_edges.append([t_i, t_j])
                                tt_edges.append([t_j, t_i])
            else:
                warmup_think_np = all_think_emb[:self.num_warmup_responses].cpu().numpy()
                warmup_sim = cosine_similarity(warmup_think_np)

                for q_idx in range(self.num_warmup_queries):
                    start = q_idx * self.num_responses_per_query
                    end = min(start + self.num_responses_per_query, self.num_warmup_responses)

                    for i in range(start, end):
                        local_sims = warmup_sim[i, start:end].copy()
                        local_sims[i - start] = -1

                        k = min(self.intra_query_think_k, len(local_sims) - 1)
                        if k > 0:
                            top_k_local_idx = np.argsort(local_sims)[-k:]
                            for local_j in top_k_local_idx:
                                j = start + local_j
                                if local_sims[local_j] > -1:
                                    tt_edges.append([i, j])

        if self.intra_query_think_k != 0 and num_new_queries > 0:
            if self.intra_query_think_k == -1:
                for q_idx in range(num_new_queries):
                    start = self.num_warmup_responses + q_idx * self.num_responses_per_query
                    for i in range(self.num_responses_per_query):
                        for j in range(i + 1, self.num_responses_per_query):
                            t_i = start + i
                            t_j = start + j
                            if t_i < total_responses and t_j < total_responses:
                                tt_edges.append([t_i, t_j])
                                tt_edges.append([t_j, t_i])
            else:
                new_think_np = all_think_emb[self.num_warmup_responses:].cpu().numpy()
                new_sim = cosine_similarity(new_think_np)

                for q_idx in range(num_new_queries):
                    start = q_idx * self.num_responses_per_query
                    end = min(start + self.num_responses_per_query, len(new_think_np))

                    for i in range(start, end):
                        local_sims = new_sim[i, start:end].copy()
                        local_sims[i - start] = -1

                        k = min(self.intra_query_think_k, len(local_sims) - 1)
                        if k > 0:
                            top_k_local_idx = np.argsort(local_sims)[-k:]
                            for local_j in top_k_local_idx:
                                j = start + local_j
                                if local_sims[local_j] > -1:
                                    global_i = self.num_warmup_responses + i
                                    global_j = self.num_warmup_responses + j
                                    tt_edges.append([global_i, global_j])

        if self.think_cross_k > 0 and total_responses > self.num_responses_per_query:
            think_emb_np = all_think_emb.cpu().numpy()
            think_sim = cosine_similarity(think_emb_np)

            for i in range(total_responses):
                my_query = i // self.num_responses_per_query
                similarities = think_sim[i].copy()

                start = my_query * self.num_responses_per_query
                end = min(start + self.num_responses_per_query, total_responses)
                similarities[start:end] = -1

                if (similarities > -1).sum() > 0:
                    k = min(self.think_cross_k, (similarities > -1).sum())
                    top_k_idx = np.argsort(similarities)[-k:]
                    for j in top_k_idx:
                        if similarities[j] > -1:
                            tt_edges.append([i, int(j)])

        if tt_edges:
            graph[('think', 'competes_with', 'think')].edge_index = torch.tensor(
                tt_edges, dtype=torch.long, device=self.device
            ).t().contiguous()
        else:
            graph[('think', 'competes_with', 'think')].edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=self.device
            )

        return graph

    def _gnn_predict_with_warmup(
        self,
        query_texts: List[str],
        all_responses: List[List[str]],
        query_domains: List[str] = None,
    ) -> List[List[float]]:
        '''
        GNN reward predictor with warmup graph extension.
        Encodes queries, builds extended graph, runs GNN forward pass.
        Related: _build_extended_graph() and gnn_models.py.
        '''
        num_new_queries = len(query_texts)

        new_query_emb = self.embed_model.encode(
            query_texts, convert_to_tensor=True, device=self.device
        )

        new_think_texts = []
        new_answer_texts = []
        for responses in all_responses:
            for resp in responses:
                think, answer = split_think_and_answer(resp)
                new_think_texts.append(think)
                new_answer_texts.append(answer if answer else think[:100])

        new_think_emb = self.embed_model.encode(
            new_think_texts, convert_to_tensor=True, device=self.device
        )
        new_answer_emb = self.embed_model.encode(
            new_answer_texts, convert_to_tensor=True, device=self.device
        )

        with torch.no_grad():
            graph = self._build_extended_graph(
                new_query_emb, new_think_emb, new_answer_emb, num_new_queries,
                new_query_domains=query_domains
            )

            new_answer_features_list = []
            resp_idx = 0
            for q_idx, responses in enumerate(all_responses):
                domain = query_domains[q_idx] if query_domains else 'math'
                feat = self.get_answer_features(responses, domain)
                if feat.shape[1] < self.answer_feat_dim:
                    feat = F.pad(feat, (0, self.answer_feat_dim - feat.shape[1]))
                new_answer_features_list.append(feat)
                resp_idx += len(responses)

            new_answer_features = torch.cat(new_answer_features_list, dim=0).to(self.device)
            all_answer_features = torch.cat([self.warmup_answer_features, new_answer_features], dim=0)
            all_logits = self.gnn(graph, all_answer_features).squeeze(-1)
            all_probs = torch.sigmoid(all_logits)

            num_warmup_edges = self.num_warmup_queries * self.num_responses_per_query
            new_probs = all_probs[num_warmup_edges:]

            results = []
            idx = 0
            for responses in all_responses:
                n = len(responses)
                query_probs = new_probs[idx:idx+n]
                rewards = (query_probs > 0.5).float().cpu().tolist()
                results.append(rewards)
                idx += n

            return results

    def compute_rewards_batch(
        self,
        data_sources: List[str],
        solution_strs: List[str],
        ground_truths: List[str],
        extra_infos: List[Dict[str, Any]],
    ) -> List[float]:
        '''
        Batch reward computation with GT/GNN routing.
        Groups samples by prompt, routes to GT or GNN, returns scores.
        Related: _should_use_gt_data_source() and _gnn_predict_with_warmup().
        '''
        N = len(solution_strs)
        scores = [0.0] * N

        def parse_extra(e):
            '''
            Extra info parser for batch input.
            Converts string/None extra_info to dict.
            Related: compute_rewards_batch() input processing.
            '''
            if e is None:
                return {}
            if isinstance(e, str):
                try:
                    import json
                    return json.loads(e)
                except:
                    return {}
            return e if isinstance(e, dict) else {}

        parsed_extras = [parse_extra(e) for e in (extra_infos or [{}] * N)]

        # Validation mode: 100% GT reward (skip GNN entirely)
        is_validation = any(not e.get('is_train', True) for e in parsed_extras)
        if is_validation:
            for i in range(N):
                domain = self._map_domain(data_sources[i])
                extra = parsed_extras[i].copy()
                if 'answer' not in extra:
                    extra['answer'] = ground_truths[i]
                scores[i] = self._compute_gt_reward(solution_strs[i], domain, extra)
            return scores

        prompt_groups = defaultdict(list)
        for i in range(N):
            extra = parsed_extras[i]
            prompt = extra.get('question', '') or extra.get('prompt', '')
            if isinstance(prompt, (list, tuple)):
                prompt = str(prompt)
            if not prompt:
                idx = extra.get('index', extra.get('task_id', i))
                prompt = f"{data_sources[i]}_{idx}"

            prompt_hash = hash(prompt)
            prompt_groups[prompt_hash].append({
                'index': i,
                'prompt': prompt,
                'response': solution_strs[i],
                'ground_truth': ground_truths[i],
                'data_source': data_sources[i],
                'extra_info': extra,
            })

        gt_items = []
        gnn_groups = {}

        for prompt_hash, items in prompt_groups.items():
            first_item = items[0]
            use_gt = self._should_use_gt_data_source(
                first_item['data_source'],
                first_item['extra_info']
            )

            if use_gt:
                gt_items.extend(items)
                self.stats['gt_count'] += len(items)
            else:
                gnn_groups[prompt_hash] = items
                self.stats['gnn_count'] += len(items)

        for item in gt_items:
            domain = self._map_domain(item['data_source'])
            extra = item['extra_info'].copy()
            if 'answer' not in extra:
                extra['answer'] = item['ground_truth']
            reward = self._compute_gt_reward(item['response'], domain, extra)
            scores[item['index']] = reward

        if gnn_groups:
            try:
                query_texts = []
                all_responses = []
                query_domains = []
                items_order = []

                for prompt_hash, items in gnn_groups.items():
                    query_texts.append(items[0]['prompt'])
                    all_responses.append([item['response'] for item in items])
                    query_domain = self._map_domain(items[0]['data_source'])
                    query_domains.append(query_domain)
                    items_order.append(items)

                gnn_results = self._gnn_predict_with_warmup(query_texts, all_responses, query_domains)

                for items, rewards in zip(items_order, gnn_results):
                    for item, reward in zip(items, rewards):
                        scores[item['index']] = reward

            except Exception as e:
                logger.warning(f"[WarmupGNN] GNN prediction failed: {e}, using GT fallback")
                for items in gnn_groups.values():
                    for item in items:
                        domain = self._map_domain(item['data_source'])
                        extra = item['extra_info'].copy()
                        if 'answer' not in extra:
                            extra['answer'] = item['ground_truth']
                        scores[item['index']] = self._compute_gt_reward(
                            item['response'], domain, extra
                        )

        return scores

    def _map_domain(self, data_source: str) -> str:
        '''
        Data source to domain mapper for reward routing.
        Maps VERL data_source string to math, qa, or code.
        Related: _map_warmup_domain() for cache domain mapping.
        '''
        ds = data_source.lower()
        if 'math' in ds or 'gsm' in ds:
            return 'math'
        elif 'mmlu' in ds or 'commonsense' in ds or 'obqa' in ds or 'arc' in ds or 'gpqa' in ds or 'qa' in ds or 'musique' in ds:
            return 'qa'
        elif 'humaneval' in ds or 'mbpp' in ds or 'code' in ds or 'apps' in ds:
            return 'code'
        return 'math'

    def get_stats(self) -> Dict[str, Any]:
        '''
        Usage statistics reporter.
        Returns GT/GNN count and percentage breakdown.
        Related: compute_rewards_batch() stats accumulation.
        '''
        total = self.stats['gt_count'] + self.stats['gnn_count']
        if total > 0:
            return {
                'gt_count': self.stats['gt_count'],
                'gnn_count': self.stats['gnn_count'],
                'gt_percentage': self.stats['gt_count'] / total * 100,
                'gnn_percentage': self.stats['gnn_count'] / total * 100,
            }
        return self.stats


def make_get_batch_mixed_reward_function(cls, default_warmup, default_gt):
    '''
    Singleton factory generator for BatchMixedGNNRewardWithWarmup.
    Creates closure with _holder to maintain singleton per variant.
    Related: compute_score() VERL entry point.
    '''
    _holder = [None]

    def get_batch_mixed_reward_function(
        gnn_checkpoint: str = None,
        warmup_embeddings: str = default_warmup,
        gt_identifiers: str = default_gt,
    ):
        if _holder[0] is None:
            _holder[0] = cls(
                gnn_checkpoint_path=gnn_checkpoint,
                warmup_embeddings_path=warmup_embeddings,
                gt_identifiers_path=gt_identifiers,
            )
        return _holder[0]

    return get_batch_mixed_reward_function


def make_compute_score(get_batch_fn):
    '''
    VERL compute_score generator bound to a factory function.
    Supports single and batch modes.
    Related: make_get_batch_mixed_reward_function() singleton.
    '''
    def compute_score(
        data_source=None,
        solution_str=None,
        ground_truth=None,
        extra_info=None,
        data_sources=None,
        solution_strs=None,
        ground_truths=None,
        extra_infos=None,
        **kwargs,
    ):
        import numpy as np

        def to_list(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x.tolist()
            return list(x) if hasattr(x, '__iter__') and not isinstance(x, (str, dict)) else x

        is_batch = data_sources is not None and (
            isinstance(data_sources, (list, np.ndarray)) and len(data_sources) > 0
        )

        reward_fn = get_batch_fn()

        if is_batch:
            data_sources = to_list(data_sources)
            solution_strs = to_list(solution_strs)
            ground_truths = to_list(ground_truths)
            extra_infos = to_list(extra_infos)

            return reward_fn.compute_rewards_batch(
                data_sources=data_sources,
                solution_strs=solution_strs,
                ground_truths=ground_truths,
                extra_infos=extra_infos,
            )
        else:
            return reward_fn.compute_rewards_batch(
                data_sources=[data_source],
                solution_strs=[solution_str],
                ground_truths=[ground_truth],
                extra_infos=[extra_info],
            )[0]

    return compute_score
