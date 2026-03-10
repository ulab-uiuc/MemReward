'''
Generate GT identifier JSON for mixed-reward routing during VERL training.
Maps dataset names to query indices that receive ground-truth rewards.
Related: verify_gt_alignment.py for consistency verification.
'''

import argparse
import json
import pandas as pd
from pathlib import Path

# Project root (relative to this script: scripts/Step3_gnn_verl_data_preparation/generate_and_verify_gt_identifier/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 10 datasets used in GNN training
DATASETS = [
    'gsm8k', 'math', 'gsm_symbolic',
    'mmlu', 'commonsenseqa', 'obqa', 'arc_c', 'gpqa',
    'humaneval_plus', 'mbpp_plus',
]

# Domain mapping
DOMAIN_MAP = {
    'gsm8k': 'math',
    'math': 'math',
    'gsm_symbolic': 'math',
    'mmlu': 'qa',
    'commonsenseqa': 'qa',
    'obqa': 'qa',
    'arc_c': 'qa',
    'gpqa': 'qa',
    'humaneval_plus': 'coding',
    'mbpp_plus': 'coding',
}


def generate_gt_identifiers(train_ratio: int = 20, model_type: str = 'qwen3b'):
    """Generate gt_identifiers JSON for a given train ratio.

    For each dataset, loads the train_{ratio} parquet to determine how many queries
    are in that ratio, then takes the first N unique indices from the cache
    (prefix subsetting, since train_10 ⊂ train_20 ⊂ ... ⊂ train_70).
    """
    CACHE_BASE = PROJECT_ROOT / 'outputs' / 'gnn_standard_domains' / model_type
    DATA_BASE = PROJECT_ROOT / 'data' / 'sampled_1500'

    OUTPUT_PATH = PROJECT_ROOT / 'configs' / f'gt_identifiers_train{train_ratio}.json'

    gt_identifiers = {}

    print("=" * 70)
    print(f"Generating GT Identifiers (train_ratio={train_ratio}%)")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 70)

    total_queries = 0

    for dataset in DATASETS:
        cache_file = CACHE_BASE / f'{model_type}_cache_{dataset}' / 'responses_train.json'
        parquet_file = DATA_BASE / f'{dataset}_sampled_train_{train_ratio}.parquet'

        if not cache_file.exists():
            print(f'{dataset:20} | Cache not found: {cache_file}')
            continue

        if not parquet_file.exists():
            print(f'{dataset:20} | Parquet not found: {parquet_file}')
            continue

        # Load parquet to get query count for this ratio
        df = pd.read_parquet(parquet_file)
        n_parquet = len(df)

        # Load cache and extract unique indices preserving insertion order
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        unique_indices = []
        seen = set()
        for item in cache_data:
            idx = int(item['extra_info']['index'])
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        n_cache = len(unique_indices)

        # Take first n_parquet indices (prefix subsetting)
        n_total = min(n_parquet, n_cache)
        selected_indices = sorted(unique_indices[:n_total])
        total_queries += n_total

        gt_identifiers[dataset] = {
            'indices': selected_indices,
            'n_total': n_total,
            'domain': DOMAIN_MAP[dataset],
            'source': f'First {n_total} queries from cache (train_{train_ratio} prefix, original indices)'
        }

        print(f'{dataset:20} | Queries: {n_total:4} (parquet={n_parquet}, cache={n_cache}) | '
              f'Index range: {min(selected_indices)}-{max(selected_indices)}')

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(gt_identifiers, f, indent=2)

    VERL_TOTAL = 5358
    print("=" * 70)
    print(f"GT identifiers saved to: {OUTPUT_PATH}")
    print(f"Total GT queries: {total_queries}")
    print(f"Coverage: {total_queries}/{VERL_TOTAL} = {100 * total_queries / VERL_TOTAL:.1f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate GT identifiers for mixed training')
    parser.add_argument('--train-ratio', type=int, default=20,
                        help='Train ratio percentage (default: 20). Output file: gt_identifiers_train{ratio}.json')
    parser.add_argument('--model-type', type=str, default='qwen3b',
                        help='Model type for cache prefix (default: qwen3b)')
    args = parser.parse_args()
    generate_gt_identifiers(train_ratio=args.train_ratio, model_type=args.model_type)
