'''
Verify 4-way GT index alignment across GNN cache, gt_identifiers,
verl_train_partial_gt, and mix mode routing.
Related: generate_gt_identifiers.py for GT identifier generation.
'''

import json
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

# Project root (relative to this script: scripts/Step3_gnn_verl_data_preparation/generate_and_verify_gt_identifier/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Dataset configuration
DATASETS = [
    'gsm8k', 'math', 'gsm_symbolic', 'mmlu', 'commonsenseqa',
    'obqa', 'arc_c', 'gpqa', 'humaneval_plus', 'mbpp_plus'
]

BASE_DIR = PROJECT_ROOT
GT_IDS_PATH = BASE_DIR / 'configs' / 'gt_identifiers_train20.json'
PARTIAL_GT_DIR = BASE_DIR / 'data' / 'verl_train_partial_gt'


def load_gnn_cache_indices(model_type='qwen3b'):
    """Load indices from GNN cache (responses_train.json)."""
    gnn_cache_dir = BASE_DIR / 'outputs' / 'gnn_standard_domains' / model_type

    print("="*70)
    print(f"Loading GNN Cache Indices (model_type={model_type})")
    print("="*70)

    gnn_indices = {}

    for dataset in DATASETS:
        cache_file = gnn_cache_dir / f'{model_type}_cache_{dataset}' / 'responses_train.json'

        if not cache_file.exists():
            print(f"  ❌ {dataset:20} cache file not found")
            continue

        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        # Extract unique indices from extra_info
        indices = sorted(list(set([
            int(item['extra_info']['index'])
            for item in cache_data
        ])))

        gnn_indices[dataset] = indices
        print(f"  ✓ {dataset:20} {len(indices):4} indices")

    total = sum(len(v) for v in gnn_indices.values())
    print(f"\nTotal GNN cache indices: {total}")

    return gnn_indices


def load_gt_identifiers():
    """Load gt_identifiers_train20.json."""
    print("\n" + "="*70)
    print("Loading GT Identifiers")
    print("="*70)

    with open(GT_IDS_PATH, 'r') as f:
        gt_ids = json.load(f)

    gt_indices = {}
    for dataset in DATASETS:
        if dataset not in gt_ids:
            print(f"  ❌ {dataset:20} not found in gt_identifiers_train20.json")
            continue

        indices = sorted(gt_ids[dataset]['indices'])
        gt_indices[dataset] = indices
        print(f"  ✓ {dataset:20} {len(indices):4} indices")

    total = sum(len(v) for v in gt_indices.values())
    print(f"\nTotal gt_identifiers indices: {total}")

    return gt_indices


def load_partial_gt_indices():
    """Load indices from Partial GT training data."""
    print("\n" + "="*70)
    print("Loading Partial GT Data Indices")
    print("="*70)

    train_file = PARTIAL_GT_DIR / 'train.parquet'

    if not train_file.exists():
        print("  ❌ Partial GT train.parquet not found")
        return None

    df = pd.read_parquet(train_file)

    # Parse extra_info if needed
    if isinstance(df['extra_info'].iloc[0], str):
        df['extra_info_parsed'] = df['extra_info'].apply(json.loads)
    else:
        df['extra_info_parsed'] = df['extra_info']

    # Group by data_source and extract indices
    partial_gt_indices = {}

    for dataset in DATASETS:
        dataset_df = df[df['data_source'] == dataset]
        if len(dataset_df) == 0:
            # Try alternative names
            if dataset == 'gsm_symbolic':
                dataset_df = df[df['data_source'] == 'gsm_symbolic_main']

        if len(dataset_df) > 0:
            indices = sorted([
                int(row['extra_info_parsed']['index'])
                for _, row in dataset_df.iterrows()
            ])
            partial_gt_indices[dataset] = indices
            print(f"  ✓ {dataset:20} {len(indices):4} indices ({len(set(indices))} unique)")
        else:
            print(f"  ⚠️  {dataset:20} no data found")

    total = sum(len(v) for v in partial_gt_indices.values())
    total_unique = sum(len(set(v)) for v in partial_gt_indices.values())
    print(f"\nTotal Partial GT indices: {total} ({total_unique} unique)")

    return partial_gt_indices


def verify_alignment(gnn_indices, gt_indices, partial_gt_indices):
    """Verify that all three sources have identical indices."""
    print("\n" + "="*70)
    print("Verification: Index Alignment")
    print("="*70)

    all_aligned = True
    alignment_results = []

    for dataset in DATASETS:
        gnn = set(gnn_indices.get(dataset, []))
        gt = set(gt_indices.get(dataset, []))
        partial = set(partial_gt_indices.get(dataset, []))

        # Check if all three match
        if gnn == gt == partial and len(gnn) > 0:
            status = "✓"
            aligned = True
        else:
            status = "❌"
            aligned = False
            all_aligned = False

        print(f"{status} {dataset:20} GNN: {len(gnn):4}  GT: {len(gt):4}  Partial: {len(partial):4}")

        alignment_results.append({
            'dataset': dataset,
            'gnn_count': len(gnn),
            'gt_count': len(gt),
            'partial_count': len(partial),
            'aligned': aligned
        })

        # Show differences if not aligned
        if not aligned:
            if gnn != gt:
                only_gnn = gnn - gt
                only_gt = gt - gnn
                if only_gnn:
                    print(f"     Only in GNN: {len(only_gnn)} indices")
                if only_gt:
                    print(f"     Only in GT:  {len(only_gt)} indices")

            if gt != partial:
                only_gt = gt - partial
                only_partial = partial - gt
                if only_gt:
                    print(f"     Only in GT:      {len(only_gt)} indices")
                if only_partial:
                    print(f"     Only in Partial: {len(only_partial)} indices")

    return all_aligned, alignment_results


def verify_exact_values(gnn_indices, gt_indices, partial_gt_indices):
    """Verify that not only counts but exact index values match."""
    print("\n" + "="*70)
    print("Verification: Exact Index Values")
    print("="*70)

    all_match = True

    for dataset in DATASETS:
        gnn = gnn_indices.get(dataset, [])
        gt = gt_indices.get(dataset, [])
        partial = list(set(partial_gt_indices.get(dataset, [])))  # Remove duplicates

        # Sort for comparison
        gnn_sorted = sorted(gnn)
        gt_sorted = sorted(gt)
        partial_sorted = sorted(partial)

        if gnn_sorted == gt_sorted == partial_sorted:
            # Show first 5 indices as sample
            sample = gnn_sorted[:5]
            print(f"  ✓ {dataset:20} {len(gnn):4} indices - Sample: {sample}")
        else:
            print(f"  ❌ {dataset:20} indices DO NOT match exactly")
            all_match = False

    return all_match


def main(model_type='qwen3b'):
    print("\n" + "="*70)
    print("4-Way GT Index Alignment Verification")
    print("="*70)
    print()
    print(f"Model type: {model_type}")
    print("Checking alignment across:")
    print(f"  1. GNN cache (outputs/gnn_standard_domains/{model_type}/{model_type}_cache_*/responses_train.json)")
    print("  2. gt_identifiers_train20.json (configs/gt_identifiers_train20.json)")
    print("  3. Partial GT data (data/verl_train_partial_gt/train.parquet)")
    print("  4. Mix mode routing (implicitly uses gt_identifiers_train20.json)")
    print()

    # Load all three sources
    gnn_indices = load_gnn_cache_indices(model_type)
    gt_indices = load_gt_identifiers()
    partial_gt_indices = load_partial_gt_indices()

    if not gnn_indices or not gt_indices or not partial_gt_indices:
        print("\n❌ FAILED: Could not load all required data")
        return False

    # Verify alignment
    aligned, results = verify_alignment(gnn_indices, gt_indices, partial_gt_indices)

    # Verify exact values
    exact_match = verify_exact_values(gnn_indices, gt_indices, partial_gt_indices)

    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    total_gnn = sum(len(v) for v in gnn_indices.values())
    total_gt = sum(len(v) for v in gt_indices.values())
    total_partial = sum(len(set(v)) for v in partial_gt_indices.values())

    print(f"Total indices:")
    print(f"  GNN cache:     {total_gnn}")
    print(f"  GT identifiers: {total_gt}")
    print(f"  Partial GT:     {total_partial}")
    print()

    if aligned and exact_match:
        print("✅ SUCCESS: Perfect 4-way alignment!")
        print()
        print("All four components are perfectly synchronized:")
        print(f"  - {total_gnn} queries from {len(DATASETS)} datasets")
        print(f"  - Index counts AND exact values match across all sources")
        print(f"  - GNN training, GT identifiers, and VERL data are aligned")
        print()
        print("Safe to proceed with VERL training!")
        return True
    else:
        print("❌ FAILED: Alignment issues detected!")
        print()
        print("Please check:")
        print("  1. GNN cache is up to date")
        print("  2. gt_identifiers_train20.json was regenerated from current GNN cache")
        print("  3. verl_train_partial_gt was generated from current gt_identifiers")
        print()
        print("To fix:")
        print("  python scripts/Step3_gnn_verl_data_preparation/generate_and_verify_gt_identifier/generate_gt_identifiers.py")
        print("  python scripts/Step3_gnn_verl_data_preparation/generate_verl_data/generate_verl_3modes_data.py")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Verify 4-way GT index alignment')
    parser.add_argument('--model-type', type=str, default='qwen3b',
                        help='Model type for cache prefix (default: qwen3b)')
    args = parser.parse_args()
    success = main(model_type=args.model_type)
    exit(0 if success else 1)
