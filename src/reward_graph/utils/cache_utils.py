'''
Cache utilities for merging and managing embedding caches.
Merges per-dataset caches into unified cache for GNN training.
Related: mixed_gnn_reward_batch_*.py warmup embedding loading.
'''

import json
import logging
import re
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

DOMAIN_MAP = {
    'gsm8k': 'math', 'math': 'math', 'gsm_symbolic': 'math',
    'mmlu': 'qa', 'commonsenseqa': 'qa', 'obqa': 'qa', 'arc_c': 'qa', 'gpqa': 'qa',
    'humaneval_plus': 'code', 'mbpp_plus': 'code', 'humaneval': 'code', 'mbpp': 'code'
}


def _extract_math_answer(response: str) -> float:
    '''
    Definition: Numeric answer extractor for math responses.
    Purpose: Parses ####, \\boxed{}, or last number from response text.
    Related: _get_answer_features() math feature computation.
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


def _extract_qa_answer(response: str) -> str:
    '''
    Definition: Letter choice extractor for QA responses.
    Purpose: Parses #### marker, "answer is" pattern, or trailing letter.
    Related: _get_answer_features() QA feature computation.
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


def _get_answer_features(responses: list, domain: str, answer_feat_dim: int = 6) -> torch.Tensor:
    '''
    Definition: Answer feature tensor builder per domain.
    Purpose: Extracts numeric or zero features matching GNN training format.
    Related: merge_caches() answer feature computation.
    '''
    features = []

    if domain == 'math':
        answers = [_extract_math_answer(r) for r in responses]
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
                norm_ans = float(np.sign(ans) * np.log1p(abs(ans)) / 20.0)
                norm_ans = float(np.clip(norm_ans, -1.0, 1.0))
                matches = 1.0 if most_common and abs(round(ans, 2) - most_common) < 0.01 else 0.0
                features.append([norm_ans, 1.0, matches])

    elif domain == 'qa':
        # QA answer_features disabled: one-hot choice encoding adds noise
        # Only rely on query/think/answer node embeddings for QA
        for _ in responses:
            features.append([0.0, 0.0, 0.0])

    else:  # code
        for _ in responses:
            features.append([0.0, 0.0, 0.0])

    # Pad to answer_feat_dim
    result = []
    for feat in features:
        if len(feat) < answer_feat_dim:
            feat = feat + [0.0] * (answer_feat_dim - len(feat))
        result.append(feat)

    return torch.tensor(result, dtype=torch.float32)


def merge_caches(
    cache_dir: Path,
    cache_names: List[str],
    output_name: str = "cache_unified",
    force: bool = False,
    answer_feat_dim: int = 6,
    max_per_dataset: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
    '''
    Definition: Multi-cache merger into unified cache directory.
    Purpose: Concatenates embeddings, computes answer features, saves to disk.
    Related: load_or_create_unified_cache() auto-merge logic.
    '''
    cache_dir = Path(cache_dir)
    output_dir = cache_dir / output_name

    if not force and (output_dir / "embeddings.pt").exists():
        logger.info(f"Loading existing unified cache from {output_dir}")
        data = torch.load(output_dir / "embeddings.pt", weights_only=False)
        answer_features = data.get('answer_features', torch.zeros(data['think_embeddings'].shape[0], answer_feat_dim))
        return (
            data['query_embeddings'],
            data['think_embeddings'],
            data['answer_embeddings'],
            data['domains'],
            answer_features
        )

    logger.info(f"Merging caches: {cache_names}")

    all_query_emb = []
    all_think_emb = []
    all_answer_emb = []
    all_answer_features = []
    all_domains = []
    all_responses = []
    cache_response_info = []  # Track (domain, responses) for each cache

    for cache_name in cache_names:
        cache_path = cache_dir / cache_name
        if not cache_path.exists():
            logger.warning(f"Skipping {cache_name} - directory not found")
            continue

        # Try multiple embedding file naming conventions
        emb_path = None
        for emb_name in ["embeddings.pt", "embeddings_train.pt"]:
            candidate = cache_path / emb_name
            if candidate.exists():
                emb_path = candidate
                break

        if emb_path is None:
            logger.warning(f"Skipping {cache_name} - no embeddings file found")
            continue

        # Try multiple response file naming conventions
        resp_path = None
        for resp_name in ["responses.json", "responses_train.json"]:
            candidate = cache_path / resp_name
            if candidate.exists():
                resp_path = candidate
                break

        emb = torch.load(emb_path, weights_only=False)

        domain_name = cache_name
        for prefix in ["qwen3b_cache_", "qwen1.5b_cache_", "cache_"]:
            if cache_name.startswith(prefix):
                domain_name = cache_name[len(prefix):]
                break

        # Apply max_per_dataset truncation if specified
        n_queries_raw = emb['query_embeddings'].shape[0]
        n_queries = min(n_queries_raw, max_per_dataset) if max_per_dataset else n_queries_raw
        n_responses = n_queries * 8  # 8 responses per query

        all_query_emb.append(emb['query_embeddings'][:n_queries])
        all_think_emb.append(emb['think_embeddings'][:n_responses])
        all_answer_emb.append(emb['answer_embeddings'][:n_responses])

        domains = emb.get('domains', [domain_name] * n_queries_raw)
        all_domains.extend(domains[:n_queries])

        responses_data = []
        if resp_path and resp_path.exists():
            with open(resp_path, 'r') as f:
                responses_data = json.load(f)
                all_responses.extend(responses_data[:n_queries])
                responses_data = responses_data[:n_queries]

        grouped_domain = DOMAIN_MAP.get(domain_name, 'math')
        cache_response_info.append((grouped_domain, responses_data, n_queries))

        if max_per_dataset and n_queries_raw > n_queries:
            logger.info(f"  {cache_name}: {n_queries}/{n_queries_raw} queries (truncated, domain: {grouped_domain})")
        else:
            logger.info(f"  {cache_name}: {n_queries} queries (domain: {grouped_domain})")

    if not all_query_emb:
        raise ValueError(f"No valid caches found in {cache_dir}")

    merged_query = torch.cat(all_query_emb, dim=0)
    merged_think = torch.cat(all_think_emb, dim=0)
    merged_answer = torch.cat(all_answer_emb, dim=0)

    logger.info("Computing answer features from responses...")
    for grouped_domain, responses_data, n_queries in cache_response_info:
        if responses_data:
            for item in responses_data:
                if isinstance(item, dict) and 'responses' in item:
                    resps = item['responses']
                else:
                    resps = item if isinstance(item, list) else []
                feat = _get_answer_features(resps, grouped_domain, answer_feat_dim)
                all_answer_features.append(feat)
        else:
            # No responses available, create zeros
            logger.warning(f"No responses for domain {grouped_domain}, using zero features")
            zeros = torch.zeros(n_queries * 8, answer_feat_dim)  # Assume 8 responses per query
            all_answer_features.append(zeros)

    if all_answer_features:
        merged_answer_features = torch.cat(all_answer_features, dim=0)
    else:
        merged_answer_features = torch.zeros(merged_think.shape[0], answer_feat_dim)

    logger.info(f"Merged: {merged_query.shape[0]} queries, {merged_think.shape[0]} responses")
    logger.info(f"Answer features shape: {merged_answer_features.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'query_embeddings': merged_query,
        'think_embeddings': merged_think,
        'answer_embeddings': merged_answer,
        'answer_features': merged_answer_features,
        'domains': all_domains,
    }, output_dir / "embeddings.pt")

    if all_responses:
        with open(output_dir / "responses.json", 'w') as f:
            json.dump(all_responses, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved unified cache to {output_dir}")

    return merged_query, merged_think, merged_answer, all_domains, merged_answer_features


def load_or_create_unified_cache(
    cache_dir: str,
    cache_names: Optional[List[str]] = None,
    unified_name: str = None,
    prefix: str = "cache_",
    answer_feat_dim: int = 6,
    force: bool = False,
    max_per_dataset: Optional[int] = None
) -> Dict[str, torch.Tensor]:
    '''
    Definition: Unified cache loader with auto-creation.
    Purpose: Loads existing unified cache or auto-detects and merges individual caches.
    Related: BatchMixedGNNRewardWithWarmup._load_warmup_embeddings().
    '''
    cache_dir = Path(cache_dir)

    # Determine unified cache name
    if unified_name is None:
        unified_name = f"{prefix}unified"

    unified_path = cache_dir / unified_name / "embeddings.pt"

    # Try to load existing unified cache
    if not force and unified_path.exists():
        logger.info(f"Loading unified cache from {unified_path}")
        data = torch.load(unified_path, weights_only=False)
        if 'answer_features' not in data:
            data['answer_features'] = torch.zeros(data['think_embeddings'].shape[0], answer_feat_dim)
        return data

    # Auto-detect caches if not specified
    # Only include train-split caches (exclude _valid and _backup dirs)
    # Deduplicate: if both old-format (prefix_gsm8k) and new-format (prefix_gsm8k_train)
    # exist, prefer the new-format _train dir and skip the old-format
    if cache_names is None:
        all_dirs = set()
        for item in cache_dir.iterdir():
            if item.is_dir() and item.name.startswith(prefix) and item.name != unified_name:
                if item.name.endswith('_valid') or '_backup' in item.name or 'unified' in item.name:
                    continue
                if (item / "embeddings.pt").exists() or (item / "embeddings_train.pt").exists():
                    all_dirs.add(item.name)
        # Deduplicate: if 'prefix_gsm8k_train' exists, remove 'prefix_gsm8k'
        train_suffixed = {d for d in all_dirs if d.endswith('_train')}
        old_format_with_new = set()
        for t in train_suffixed:
            base = t[:-6]  # Remove '_train' suffix
            if base in all_dirs:
                old_format_with_new.add(base)
        cache_names = sorted(all_dirs - old_format_with_new)
        logger.info(f"Auto-detected caches with prefix '{prefix}': {cache_names}")

    if not cache_names:
        raise ValueError(f"No caches found in {cache_dir} with prefix '{prefix}'")

    query_emb, think_emb, answer_emb, domains, answer_features = merge_caches(
        cache_dir=cache_dir,
        cache_names=cache_names,
        output_name=unified_name,
        force=force,
        answer_feat_dim=answer_feat_dim,
        max_per_dataset=max_per_dataset
    )

    return {
        'query_embeddings': query_emb,
        'think_embeddings': think_emb,
        'answer_embeddings': answer_emb,
        'domains': domains,
        'answer_features': answer_features
    }
