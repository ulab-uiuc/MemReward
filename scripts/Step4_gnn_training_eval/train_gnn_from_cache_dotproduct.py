#!/usr/bin/env python3
'''
GNN training from cached embeddings with scaled dot-product scoring.
Uses UnifiedGNNDotProduct: score = q * r / sqrt(d) + bias.
Related: train_gnn_from_cache.py for MLP predictor variant.
'''

import os
import sys
import json
import random
import re
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from reward_graph.rewards.utils.multi_domain_reward import math_reward, qa_reward, code_reward

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = BASE_DIR / "outputs/gnn_standard_domains"
DATA_DIR = BASE_DIR / "data/sampled_1500"

SEED = 42
NUM_RESPONSES = 8

DATASETS = [
    'gsm8k', 'math', 'gsm_symbolic',
    'mmlu', 'commonsenseqa', 'obqa', 'arc_c', 'gpqa',
    'humaneval_plus', 'mbpp_plus'
]

DOMAIN_MAP = {
    'gsm8k': 'math', 'math': 'math', 'gsm_symbolic': 'math',
    'mmlu': 'qa', 'commonsenseqa': 'qa', 'obqa': 'qa', 'arc_c': 'qa', 'gpqa': 'qa',
    'humaneval_plus': 'code', 'mbpp_plus': 'code'
}


def set_seed(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_math_answer(response: str) -> float:
    """Extract numeric answer from math response."""
    # Look for #### pattern
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', response)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass

    # Look for boxed answer
    match = re.search(r'\\boxed\{([^}]+)\}', response)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass

    # Try to find last number
    numbers = re.findall(r'-?[\d,]+\.?\d*', response)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass

    return float('nan')


def extract_qa_answer(response: str) -> str:
    """Extract letter choice from QA response.

    Uses same patterns as qa_reward in multi_domain_reward.py for consistency.
    """
    response_upper = response.upper()

    # Pattern 1: #### A (letter right after ####, word boundary)
    match = re.search(r'####\s*([A-E])\b', response_upper)
    if match:
        return match.group(1)

    # Pattern 2: "The answer is X" or "correct answer is X"
    match = re.search(r'(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+IS\s*:?\s*([A-E])\b', response_upper)
    if match:
        return match.group(1)

    # Pattern 3: Standalone letter at end (A-D only)
    match = re.search(r'\b([A-D])\b\s*$', response_upper.strip())
    if match:
        return match.group(1)

    return 'X'


def get_answer_features(responses: list, domain: str, gt_answer: str = None) -> torch.Tensor:
    """
    Extract answer features from responses.

    Returns:
        torch.Tensor: Answer features for each response
            - Math: [normalized_answer, answer_valid, answer_matches_common]
            - QA: [one_hot_A, one_hot_B, one_hot_C, one_hot_D, one_hot_E, answer_valid]
            - Code: [0, 0, 0] (use existing embedding)
    """
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
        # QA answer_features disabled: one-hot choice encoding adds noise
        # Only rely on query/think/answer node embeddings for QA
        for _ in responses:
            features.append([0.0, 0.0, 0.0])

    else:  # code
        # For code, we don't have a simple answer to extract
        # Use placeholder features
        for _ in responses:
            features.append([0.0, 0.0, 0.0])

    return torch.tensor(features, dtype=torch.float32)


def load_all_data(model_type='qwen3b', train_ratio=20):
    """Load all datasets with answer features."""
    cache_prefix = f"{model_type}_cache_"

    all_data = {'train': [], 'valid': []}

    import pandas as pd

    for ds in DATASETS:
        cache_dir = OUTPUT_DIR / f"{cache_prefix}{ds}"
        if not cache_dir.exists():
            logger.warning(f"Skipping {ds} - cache not found")
            continue

        domain = DOMAIN_MAP[ds]
        reward_fn = {'math': math_reward, 'qa': qa_reward, 'code': code_reward}[domain]

        for split in ['train', 'valid']:
            emb_file = cache_dir / f"embeddings_{split}.pt"
            resp_file = cache_dir / f"responses_{split}.json"
            if split == 'train':
                data_file = DATA_DIR / f"{ds}_sampled_train_{train_ratio}.parquet"
            else:
                data_file = DATA_DIR / f"{ds}_sampled_valid.parquet"

            if not all(f.exists() for f in [emb_file, resp_file, data_file]):
                continue

            emb = torch.load(emb_file, weights_only=False)
            with open(resp_file) as f:
                responses = json.load(f)

            df = pd.read_parquet(data_file)

            # Subset cache to match parquet size (smaller ratios are prefixes of larger ones)
            n_cache = len(responses)
            n_parquet = len(df)
            num_items = min(n_cache, n_parquet)
            if split == 'train' and n_parquet < n_cache:
                logger.info(f"  {ds}/{split}: Using {num_items}/{n_cache} cache entries for train_ratio={train_ratio}")
            elif split == 'train' and n_parquet > n_cache:
                logger.warning(f"  {ds}/{split}: Cache has {n_cache} entries but train_{train_ratio} has {n_parquet} rows. "
                              f"Using {num_items} (cache limit). Regenerate cache from train_full for full coverage.")

            rewards = []
            answer_features = []

            for idx, item in enumerate(responses[:num_items]):
                extra = df.iloc[idx]['extra_info']
                gt = extra.get('answer', '')

                if domain == 'code':
                    test_list = list(extra.get('test_list', []))
                    for r in item['responses']:
                        rewards.append(reward_fn(r, test_list))
                else:
                    for r in item['responses']:
                        rewards.append(reward_fn(r, gt))

                ans_feat = get_answer_features(item['responses'], domain, gt)
                answer_features.append(ans_feat)

            # Stack answer features
            answer_features = torch.cat(answer_features, dim=0)

            n_resp = num_items * NUM_RESPONSES
            all_data[split].append({
                'dataset': ds,
                'domain': domain,
                'query_emb': emb['query_embeddings'][:num_items],
                'think_emb': emb['think_embeddings'][:n_resp],
                'answer_emb': emb['answer_embeddings'][:n_resp],
                'answer_features': answer_features,
                'rewards': torch.tensor(rewards, dtype=torch.float32),
                'n_queries': num_items
            })

            logger.info(f"  Loaded {ds}/{split}: {num_items} queries, answer_feat_dim={answer_features.shape[1]}")

    return all_data


class UnifiedGNN(nn.Module):
    """GNN with dot product scoring instead of MLP predictor."""

    def __init__(
        self,
        query_dim: int = 384,
        think_dim: int = 384,
        answer_dim: int = 384,
        answer_feat_dim: int = 6,  # Max dimension for answer features
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        dot_dim: int = 256,
    ):
        super().__init__()

        self.dot_dim = dot_dim
        self.query_proj = Linear(query_dim, hidden_dim)
        self.think_proj = Linear(think_dim, hidden_dim)
        self.answer_proj = Linear(answer_dim, hidden_dim)

        # Answer feature processor
        self.answer_feat_proj = nn.Sequential(
            nn.Linear(answer_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ('query', 'generates_reasoning', 'think'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads,
                    dropout=dropout, add_self_loops=False
                ),
                ('think', 'rev_generates_reasoning', 'query'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads,
                    dropout=dropout, add_self_loops=False
                ),
                ('think', 'leads_to', 'answer'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads,
                    dropout=dropout, add_self_loops=False
                ),
                ('answer', 'rev_leads_to', 'think'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads,
                    dropout=dropout, add_self_loops=False
                ),
                ('query', 'similar_to', 'query'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads,
                    dropout=dropout, add_self_loops=True
                ),
                ('think', 'competes_with', 'think'): GATv2Conv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads,
                    dropout=dropout, add_self_loops=False
                ),
            }
            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        # Dot product heads: project query and response into shared dot product space
        self.query_head = nn.Sequential(
            nn.Linear(hidden_dim, dot_dim),
            nn.LayerNorm(dot_dim),
        )
        self.response_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, dot_dim),  # think + answer + answer_feat
            nn.LayerNorm(dot_dim),
        )
        # Learnable bias for dot product score
        self.score_bias = nn.Parameter(torch.zeros(1))

    def forward(self, data: HeteroData, answer_features: torch.Tensor) -> torch.Tensor:
        x_dict = {
            'query': self.query_proj(data['query'].x),
            'think': self.think_proj(data['think'].x),
            'answer': self.answer_proj(data['answer'].x),
        }

        for conv in self.convs:
            filtered_edge_index = {}
            for edge_type in conv.convs.keys():
                if edge_type in data.edge_index_dict:
                    filtered_edge_index[edge_type] = data.edge_index_dict[edge_type]

            if filtered_edge_index:
                x_dict = conv(x_dict, filtered_edge_index)
                x_dict = {k: F.relu(v) for k, v in x_dict.items()}

        # Get node embeddings for prediction
        edge_index = data[('query', 'generates', 'answer')].edge_index
        query_emb = x_dict['query'][edge_index[0]]
        answer_emb = x_dict['answer'][edge_index[1]]
        think_emb = x_dict['think'][edge_index[1]]

        answer_feat_emb = self.answer_feat_proj(answer_features)

        # Project into dot product space
        q = self.query_head(query_emb)
        r = self.response_head(torch.cat([think_emb, answer_emb, answer_feat_emb], dim=-1))

        # Scaled dot product + bias
        score = (q * r).sum(dim=-1) / (self.dot_dim ** 0.5) + self.score_bias

        return score


def build_graph(query, think, answer, num_warmup_queries, device='cuda', use_intra_query=False):
    """Build heterogeneous graph."""
    n_q, n_r = query.shape[0], think.shape[0]

    data = HeteroData()
    data['query'].x = query.to(device)
    data['think'].x = think.to(device)
    data['answer'].x = answer.to(device)

    # Query -> Think/Answer edges
    qa_edges = [[q, q * NUM_RESPONSES + r] for q in range(n_q) for r in range(NUM_RESPONSES) if q * NUM_RESPONSES + r < n_r]
    qa_tensor = torch.tensor(qa_edges, dtype=torch.long, device=device).t().contiguous()
    data[('query', 'generates', 'answer')].edge_index = qa_tensor
    data[('answer', 'rev_generates', 'query')].edge_index = qa_tensor.flip(0)
    data[('query', 'generates_reasoning', 'think')].edge_index = qa_tensor
    data[('think', 'rev_generates_reasoning', 'query')].edge_index = qa_tensor.flip(0)

    # Think -> Answer
    ta_tensor = torch.tensor([[i, i] for i in range(n_r)], dtype=torch.long, device=device).t().contiguous()
    data[('think', 'leads_to', 'answer')].edge_index = ta_tensor
    data[('answer', 'rev_leads_to', 'think')].edge_index = ta_tensor.flip(0)

    # Query-Query (kNN)
    query_np = query.cpu().numpy()
    sim = cosine_similarity(query_np)
    np.fill_diagonal(sim, -1)

    qq_edges = []
    for i in range(num_warmup_queries):
        warmup_sims = sim[i, :num_warmup_queries].copy()
        warmup_sims[i] = -1
        top_k = np.argsort(warmup_sims)[-7:]
        qq_edges.extend([[i, j] for j in top_k if warmup_sims[j] > -1])

    for i in range(num_warmup_queries, n_q):
        warmup_sims = sim[i, :num_warmup_queries]
        top_k = np.argsort(warmup_sims)[-7:]
        qq_edges.extend([[i, j] for j in top_k if warmup_sims[j] > -1])

    if qq_edges:
        data[('query', 'similar_to', 'query')].edge_index = torch.tensor(qq_edges, dtype=torch.long, device=device).t().contiguous()

    # Think-Think competition (optional)
    if use_intra_query:
        think_np = think.cpu().numpy()
        think_sim = cosine_similarity(think_np)
        tt_edges = []
        for q in range(n_q):
            start, end = q * NUM_RESPONSES, min((q + 1) * NUM_RESPONSES, n_r)
            for i in range(start, end):
                local = think_sim[i, start:end].copy()
                local[i - start] = -1
                top_k = np.argsort(local)[-2:]
                tt_edges.extend([[i, start + j] for j in top_k if local[j] > -1])

        if tt_edges:
            data[('think', 'competes_with', 'think')].edge_index = torch.tensor(tt_edges, dtype=torch.long, device=device).t().contiguous()

    return data


def train_model(model, data, answer_features, rewards, train_mask, val_mask,
                num_epochs=150, patience=20, lr=1e-3, device='cuda'):
    """Train with BCE + ranking loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    answer_features = answer_features.to(device)
    rewards = rewards.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)

    best_val_metric = 0
    best_state = None
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(data, answer_features).squeeze(-1)

        # BCE loss
        train_logits = logits[train_mask]
        train_labels = (rewards[train_mask] > 0).float()
        loss = F.binary_cross_entropy_with_logits(train_logits, train_labels)
        loss.backward()
        optimizer.step()

        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits_val = model(data, answer_features).squeeze(-1)
                probs = torch.sigmoid(logits_val)
                val_probs = probs[val_mask].cpu().numpy()
                val_gt = (rewards[val_mask] > 0).float().cpu().numpy()

                # Use F1 for hard label mode (threshold at 0.5)
                val_preds = (val_probs > 0.5).astype(float)
                _, _, metric, _ = precision_recall_fscore_support(val_gt, val_preds, average='binary', zero_division=0)

                if metric > best_val_metric:
                    best_val_metric = metric
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience // 5:
                    logger.info(f"Early stopping at epoch {epoch+1} (best F1: {best_val_metric:.4f})")
                    break

    if best_state:
        model.load_state_dict(best_state)

    return model


def evaluate(model, data, answer_features, rewards, val_mask, all_data, device='cuda'):
    """Evaluate and print metrics."""
    model.eval()
    with torch.no_grad():
        logits = model(data, answer_features.to(device)).squeeze(-1)
        scores = torch.sigmoid(logits).cpu().numpy()

    val_scores = scores[val_mask.cpu().numpy()]
    val_gt = (rewards[val_mask] > 0).float().cpu().numpy()

    # Hard label mode: threshold at 0.5
    val_preds = (val_scores > 0.5).astype(float)

    pos_rate = val_gt.mean()

    # ROC-AUC (always computed as an auxiliary metric)
    try:
        roc_auc = roc_auc_score(val_gt, val_scores)
    except ValueError:
        roc_auc = 0.5

    logger.info(f"\n{'='*100}")
    logger.info(f"OVERALL VALIDATION RESULTS - HARD LABEL (F1)")
    logger.info(f"{'='*100}")
    logger.info(f"Positive Rate: {pos_rate:.4f}")
    logger.info(f"ROC-AUC:       {roc_auc:.4f}")

    # Score distribution
    pos_scores_dist = val_scores[val_gt == 1]
    neg_scores_dist = val_scores[val_gt == 0]
    logger.info(f"Score separation: {pos_scores_dist.mean():.4f} vs {neg_scores_dist.mean():.4f} = {pos_scores_dist.mean() - neg_scores_dist.mean():+.4f}")

    acc = (val_preds == val_gt).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(val_gt, val_preds, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(val_gt, val_preds, labels=[0, 1]).ravel()
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    logger.info(f"Accuracy:      {acc:.4f}")
    logger.info(f"Precision:     {prec:.4f}  (delta from pos_rate: {prec - pos_rate:+.4f})")
    logger.info(f"Recall:        {rec:.4f}")
    logger.info(f"F1:            {f1:.4f}")
    logger.info(f"FP Rate:       {fp_rate:.4f}")

    # Per-dataset metrics
    logger.info(f"\n{'='*100}")
    logger.info("PER-DATASET VALIDATION RESULTS")
    logger.info(f"{'='*100}")

    logger.info(f"{'Dataset':<20} {'Pos Rate':<10} {'ROC-AUC':<10} {'Accuracy':<10} {'Precision':<10} {'P-PosRate':<10} {'Recall':<10} {'F1':<10} {'FP Rate':<10}")
    logger.info("-"*100)

    offset = 0
    results = []
    for item in all_data['valid']:
        n = item['n_queries'] * NUM_RESPONSES
        ds_scores = val_scores[offset:offset+n]
        ds_gt = val_gt[offset:offset+n]
        ds_pos = ds_gt.mean()

        try:
            ds_roc_auc = roc_auc_score(ds_gt, ds_scores)
        except ValueError:
            ds_roc_auc = 0.5

        ds_pos_scores = ds_scores[ds_gt == 1]
        ds_neg_scores = ds_scores[ds_gt == 0]
        ds_score_sep = ds_pos_scores.mean() - ds_neg_scores.mean() if len(ds_pos_scores) > 0 and len(ds_neg_scores) > 0 else 0

        result_item = {
            'dataset': item['dataset'],
            'domain': item['domain'],
            'positive_rate': float(ds_pos),
            'roc_auc': float(ds_roc_auc),
            'score_separation': float(ds_score_sep)
        }

        ds_preds = (ds_scores > 0.5).astype(float)
        ds_acc = (ds_preds == ds_gt).mean()
        ds_prec, ds_rec, ds_f1, _ = precision_recall_fscore_support(ds_gt, ds_preds, average='binary', zero_division=0)
        ds_tn, ds_fp, ds_fn, ds_tp = confusion_matrix(ds_gt, ds_preds, labels=[0, 1]).ravel()
        ds_fp_rate = ds_fp / (ds_fp + ds_tn) if (ds_fp + ds_tn) > 0 else 0

        logger.info(f"{item['dataset']:<20} {ds_pos:<10.4f} {ds_roc_auc:<10.4f} {ds_acc:<10.4f} {ds_prec:<10.4f} {ds_prec-ds_pos:+<10.4f} {ds_rec:<10.4f} {ds_f1:<10.4f} {ds_fp_rate:<10.4f}")

        result_item.update({
            'accuracy': float(ds_acc),
            'precision': float(ds_prec),
            'precision_delta': float(ds_prec - ds_pos),
            'recall': float(ds_rec),
            'f1': float(ds_f1),
            'fp_rate': float(ds_fp_rate)
        })

        results.append(result_item)
        offset += n

    logger.info("-"*100)

    overall_result = {
        'positive_rate': float(pos_rate),
        'roc_auc': float(roc_auc),
        'score_separation': float(pos_scores_dist.mean() - neg_scores_dist.mean())
    }

    overall_result.update({
        'accuracy': float(acc),
        'precision': float(prec),
        'precision_delta': float(prec - pos_rate),
        'recall': float(rec),
        'f1': float(f1),
        'fp_rate': float(fp_rate)
    })

    return {
        'label_mode': 'hard',
        'overall': overall_result,
        'per_dataset': results
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='qwen3b', choices=['qwen3b', 'qwen1.5b'])
    parser.add_argument('--use-intra-query', action='store_true', help='Enable intra-query think-think competition edges (disabled by default)')

    parser.add_argument('--hard-label', action='store_true', default=True, help='Use hard labels (threshold at 0.5, F1 metric)')

    parser.add_argument('--train-ratio', type=int, default=20,
                        choices=[10, 20, 30, 40, 50, 60, 70],
                        help='Percentage of train_full to use for GNN training (default: 20)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Override save directory for model and results only (default: same as data dir)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed (default: 42)')

    args = parser.parse_args()

    # Add model_type subdirectory to OUTPUT_DIR
    global OUTPUT_DIR
    OUTPUT_DIR = OUTPUT_DIR / args.model_type

    # save_dir overrides only where model/results are saved, not where data is read from
    save_dir = Path(args.save_dir) if args.save_dir else OUTPUT_DIR
    if args.save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Save directory overridden: {save_dir}")

    seed = args.seed if args.seed is not None else SEED
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    use_intra_query = args.use_intra_query
    label_mode = "hard"

    logger.info(f"\n{'='*60}")
    logger.info("Training GNN with Answer Features")
    logger.info("  - Extract numeric answer for math")
    logger.info("  - Extract letter choice for QA")
    logger.info(f"  - Intra-query think competition: {use_intra_query}")
    logger.info(f"  - Label mode: {label_mode.upper()}")
    logger.info(f"  - Early stopping metric: F1")
    logger.info(f"{'='*60}")

    all_data = load_all_data(args.model_type, train_ratio=args.train_ratio)

    # Determine max answer feature dimension
    answer_feat_dim = max(
        max(d['answer_features'].shape[1] for d in all_data['train']),
        max(d['answer_features'].shape[1] for d in all_data['valid'])
    )

    # Pad answer features to same dimension before concatenating
    def pad_features(data_list, target_dim):
        padded = []
        for d in data_list:
            feat = d['answer_features']
            if feat.shape[1] < target_dim:
                feat = F.pad(feat, (0, target_dim - feat.shape[1]))
            padded.append(feat)
        return padded

    train_query = torch.cat([d['query_emb'] for d in all_data['train']], dim=0)
    train_think = torch.cat([d['think_emb'] for d in all_data['train']], dim=0)
    train_answer = torch.cat([d['answer_emb'] for d in all_data['train']], dim=0)
    train_answer_feat = torch.cat(pad_features(all_data['train'], answer_feat_dim), dim=0)
    train_rewards = torch.cat([d['rewards'] for d in all_data['train']], dim=0)

    valid_query = torch.cat([d['query_emb'] for d in all_data['valid']], dim=0)
    valid_think = torch.cat([d['think_emb'] for d in all_data['valid']], dim=0)
    valid_answer = torch.cat([d['answer_emb'] for d in all_data['valid']], dim=0)
    valid_answer_feat = torch.cat(pad_features(all_data['valid'], answer_feat_dim), dim=0)
    valid_rewards = torch.cat([d['rewards'] for d in all_data['valid']], dim=0)

    n_train_q = train_query.shape[0]
    n_train_r = train_think.shape[0]
    n_valid_r = valid_think.shape[0]

    logger.info(f"  Train: {n_train_q} queries, {n_train_r} responses")
    logger.info(f"  Valid: {valid_query.shape[0]} queries, {n_valid_r} responses")
    logger.info(f"  Answer feature dim: {answer_feat_dim}")
    logger.info(f"  Positive rate: {(train_rewards > 0).float().mean():.4f}")

    # Combine for transductive learning
    query = torch.cat([train_query, valid_query], dim=0)
    think = torch.cat([train_think, valid_think], dim=0)
    answer = torch.cat([train_answer, valid_answer], dim=0)
    answer_features = torch.cat([train_answer_feat, valid_answer_feat], dim=0)
    rewards = torch.cat([train_rewards, valid_rewards], dim=0)

    logger.info("  Building graph...")
    data = build_graph(query, think, answer, n_train_q, device, use_intra_query=use_intra_query)

    train_mask = torch.zeros(n_train_r + n_valid_r, dtype=torch.bool)
    val_mask = torch.zeros(n_train_r + n_valid_r, dtype=torch.bool)
    train_mask[:n_train_r] = True
    val_mask[n_train_r:] = True

    model = UnifiedGNN(
        query_dim=query.shape[1],
        think_dim=think.shape[1],
        answer_dim=answer.shape[1],
        answer_feat_dim=answer_feat_dim,
        hidden_dim=512,
        num_layers=2,
        num_heads=4,
        dropout=0.1
    )

    logger.info(f"  Model: query_dim={query.shape[1]}, answer_feat_dim={answer_feat_dim}")

    # Train
    model = train_model(model, data, answer_features, rewards, train_mask, val_mask, device=device)

    # Evaluate
    results = evaluate(model, data, answer_features, rewards, val_mask, all_data, device)

    intra_suffix = "_with_intra" if args.use_intra_query else ""
    label_suffix = f"_{label_mode}"
    ratio_suffix = f"_train{args.train_ratio}"

    model_path = save_dir / f"unified_gnn_{args.model_type}{intra_suffix}{label_suffix}{ratio_suffix}.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"\nModel saved: {model_path}")

    results_path = save_dir / f"gnn_results_{args.model_type}{intra_suffix}{label_suffix}{ratio_suffix}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {results_path}")

    # Auto-generate unified cache for VERL inference (skip if using --save-dir)
    if not args.save_dir:
        logger.info(f"\n{'='*60}")
        logger.info("Generating unified cache for VERL inference...")
        logger.info(f"{'='*60}")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
        from reward_graph.utils.cache_utils import load_or_create_unified_cache

        prefix = f"{args.model_type}_cache_"
        unified_name = f"{args.model_type}_cache_unified_train{args.train_ratio}"
        cache_data = load_or_create_unified_cache(
            cache_dir=str(OUTPUT_DIR),
            prefix=prefix,
            unified_name=unified_name,
            force=True
        )
        logger.info(f"Unified cache: {cache_data['query_embeddings'].shape[0]} queries, "
                    f"{cache_data['think_embeddings'].shape[0]} responses, "
                    f"answer_features {cache_data['answer_features'].shape}")
    else:
        logger.info("Skipping unified cache generation (using --save-dir)")


if __name__ == "__main__":
    main()
