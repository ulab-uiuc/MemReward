#!/usr/bin/env python3
'''
Sample up to 1500 from each out-of-domain generalization dataset.
Covers NuminaMath, SIQA, PIQA with 50/20/30 train/valid/test split.
Related: sample_1500_datasets.py for in-domain datasets.
'''

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Project root (relative to this script: scripts/Step3_gnn_verl_data_preparation/sample_1500/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Directories
DATA_DIR = str(PROJECT_ROOT / "data" / "generalization")
OUTPUT_DIR = DATA_DIR  # Output to same directory with _sampled suffix

# Sampling parameters
INITIAL_SAMPLE_SIZE = 1500

# Split ratios
TRAIN_RATIO = 0.50   # 50% for training
VALID_RATIO = 0.20   # 20% for validation (GNN testing)
TEST_RATIO = 0.30    # 30% for testing (VERL model testing)

RANDOM_SEED = 42


def sample_dataset(dataset_name: str, data_dir: str):
    """Sample a generalization dataset."""
    print(f"\n{'='*60}")
    print(f"Sampling {dataset_name}...")
    print(f"{'='*60}")

    dataset_path = Path(data_dir) / dataset_name

    # Load original data
    train_file = dataset_path / "train.parquet"
    valid_file = dataset_path / "valid.parquet"
    test_file = dataset_path / "test.parquet"

    dfs = []
    for f, name in [(train_file, 'train'), (valid_file, 'valid'), (test_file, 'test')]:
        if f.exists():
            df = pd.read_parquet(f)
            print(f"  Loaded {name}: {len(df)} samples")
            dfs.append(df)

    if not dfs:
        print(f"  No data files found for {dataset_name}, skipping...")
        return None

    # Combine all splits
    all_data = pd.concat(dfs, ignore_index=True)
    total_samples = len(all_data)
    print(f"  Total samples: {total_samples}")

    # Deduplicate by 'index' column if exists
    if 'index' in all_data.columns:
        unique_data = all_data.drop_duplicates(subset='index', keep='first')
        if len(unique_data) < len(all_data):
            print(f"  Removed {len(all_data) - len(unique_data)} duplicate indices")
            all_data = unique_data
            total_samples = len(all_data)

    # Step 1: Initial sampling (up to 1500)
    if total_samples > INITIAL_SAMPLE_SIZE:
        sampled = all_data.sample(n=INITIAL_SAMPLE_SIZE, replace=False, random_state=RANDOM_SEED)
        print(f"  Sampled {len(sampled)} from {total_samples}")
    else:
        sampled = all_data.copy()
        print(f"  Using all {len(sampled)} samples")

    # Save full sampled
    sampled.to_parquet(f"{data_dir}/{dataset_name}_sampled.parquet", index=False)

    # Step 2: Split into train/valid/test
    n_sampled = len(sampled)
    n_train = int(n_sampled * TRAIN_RATIO)
    n_valid = int(n_sampled * VALID_RATIO)
    n_test = n_sampled - n_train - n_valid

    # Shuffle
    shuffled = sampled.sample(frac=1, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)

    sampled_train = shuffled.iloc[:n_train]
    sampled_valid = shuffled.iloc[n_train:n_train+n_valid]
    sampled_test = shuffled.iloc[n_train+n_valid:]

    print(f"  Split: train={len(sampled_train)}, valid={len(sampled_valid)}, test={len(sampled_test)}")

    # Save splits
    sampled_train.to_parquet(f"{data_dir}/{dataset_name}_sampled_train.parquet", index=False)
    sampled_valid.to_parquet(f"{data_dir}/{dataset_name}_sampled_valid.parquet", index=False)
    sampled_test.to_parquet(f"{data_dir}/{dataset_name}_sampled_test.parquet", index=False)

    print(f"  Saved to {data_dir}/{dataset_name}_sampled_*.parquet")

    return {
        'dataset': dataset_name,
        'original': total_samples,
        'sampled': len(sampled),
        'train': len(sampled_train),
        'valid': len(sampled_valid),
        'test': len(sampled_test)
    }


def main():
    """Sample all generalization datasets."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generalization datasets
    datasets = ['numina_math', 'siqa', 'piqa']

    results = []
    for dataset in datasets:
        try:
            result = sample_dataset(dataset, DATA_DIR)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  Error sampling {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*60}")
    print("GENERALIZATION SAMPLING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Original':>10} {'Sampled':>10} {'Train':>8} {'Valid':>8} {'Test':>8}")
    print('-' * 70)

    for r in results:
        print(f"{r['dataset']:<20} {r['original']:>10} {r['sampled']:>10} {r['train']:>8} {r['valid']:>8} {r['test']:>8}")

    print('-' * 70)
    print(f"{'TOTAL':<20} {sum(r['original'] for r in results):>10} {sum(r['sampled'] for r in results):>10} {sum(r['train'] for r in results):>8} {sum(r['valid'] for r in results):>8} {sum(r['test'] for r in results):>8}")

    # Create README
    readme_content = """# Generalization Test Datasets

These datasets are for testing model generalization ONLY.
DO NOT include in VERL training data or main GNN training.

## Datasets
- numina_math: Competition-level mathematics (AMC, AIME, IMO)
- siqa: Social commonsense reasoning (multiple choice)
- piqa: Physical commonsense reasoning (multiple choice)

## Purpose
- *_sampled_valid.parquet: GNN generalization testing
- *_sampled_test.parquet: VERL model generalization testing

## Warning
These datasets are intentionally separate from the main 10 training datasets
to ensure fair generalization evaluation.
"""
    with open(f"{DATA_DIR}/README.md", 'w') as f:
        f.write(readme_content)

    print(f"\n Created README.md")
    print(f"\n Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
