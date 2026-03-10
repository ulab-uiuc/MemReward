#!/usr/bin/env python3
'''
Sample up to 1500 queries per dataset with 50/20/30 train/valid/test split.
Creates train_20 (20% of train) subsample for GNN warmup training.
Related: generate_multi_ratio_splits.py for other GT ratio splits.
'''

import os
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATA_DIR = str(PROJECT_ROOT / "data")
OUTPUT_DIR = f"{DATA_DIR}/sampled_1500"

INITIAL_SAMPLE_SIZE = 1500

FULL_RATIO = 0.50
VALID_RATIO = 0.20
TEST_RATIO = 0.30

GT_ONLY_RATIO = 0.20

RANDOM_SEED = 42


def sample_dataset(dataset_name: str, data_dir: str, output_dir: str):
    """
    Sample a dataset following the sampled_1500 strategy.

    Args:
        dataset_name: Name of the dataset directory
        data_dir: Base data directory
        output_dir: Output directory for sampled data
    """
    print(f"\n{'='*60}")
    print(f"Sampling {dataset_name}...")
    print(f"{'='*60}")

    dataset_path = Path(data_dir) / dataset_name

    # Load original data
    train_file = dataset_path / "train.parquet"
    valid_file = dataset_path / "valid.parquet"
    test_file = dataset_path / "test.parquet"

    dfs = []
    if train_file.exists():
        train_df = pd.read_parquet(train_file)
        print(f"  Loaded train: {len(train_df)} samples")
        dfs.append(train_df)

    if valid_file.exists():
        valid_df = pd.read_parquet(valid_file)
        print(f"  Loaded valid: {len(valid_df)} samples")
        dfs.append(valid_df)

    if test_file.exists():
        test_df = pd.read_parquet(test_file)
        print(f"  Loaded test: {len(test_df)} samples")
        dfs.append(test_df)

    if not dfs:
        print(f"  ⚠️  No data files found for {dataset_name}, skipping...")
        return

    # Combine all splits
    all_data = pd.concat(dfs, ignore_index=True)
    total_samples = len(all_data)
    print(f"  Total samples: {total_samples}")

    # Deduplicate by 'index' column to ensure no duplicate indices
    if 'index' in all_data.columns:
        unique_data = all_data.drop_duplicates(subset='index', keep='first')
        if len(unique_data) < len(all_data):
            print(f"  ⚠️  Removed {len(all_data) - len(unique_data)} duplicate indices from source data")
            all_data = unique_data
            total_samples = len(all_data)
            print(f"  Total unique samples: {total_samples}")

    # Step 1: Initial sampling (up to 1500) WITHOUT replacement
    if total_samples > INITIAL_SAMPLE_SIZE:
        sampled = all_data.sample(n=INITIAL_SAMPLE_SIZE, replace=False, random_state=RANDOM_SEED)
        print(f"  Step 1: Sampled {len(sampled)} from {total_samples} (WITHOUT replacement)")
    else:
        sampled = all_data.copy()
        print(f"  Step 1: Using all {len(sampled)} samples (< {INITIAL_SAMPLE_SIZE}, no oversampling)")

    # Save sampled
    sampled.to_parquet(f"{output_dir}/{dataset_name}_sampled.parquet", index=False)

    # Step 2: Split sampled into full/valid/test (70%/20%/10%)
    n_sampled = len(sampled)
    n_full = int(n_sampled * FULL_RATIO)
    n_valid = int(n_sampled * VALID_RATIO)
    n_test = n_sampled - n_full - n_valid  # Remainder goes to test

    # Shuffle (WITHOUT replacement, though frac=1 means all data anyway)
    shuffled = sampled.sample(frac=1, replace=False, random_state=RANDOM_SEED).reset_index(drop=True)

    sampled_train_full = shuffled.iloc[:n_full]
    sampled_valid = shuffled.iloc[n_full:n_full+n_valid]
    sampled_test = shuffled.iloc[n_full+n_valid:]

    print(f"  Step 2: Split sampled into train_full/valid/test: {len(sampled_train_full)}/{len(sampled_valid)}/{len(sampled_test)}")
    print(f"          Ratios: {len(sampled_train_full)/n_sampled*100:.1f}% / {len(sampled_valid)/n_sampled*100:.1f}% / {len(sampled_test)/n_sampled*100:.1f}%")

    # Save sampled_train_full, sampled_valid, sampled_test
    sampled_train_full.to_parquet(f"{output_dir}/{dataset_name}_sampled_train_full.parquet", index=False)
    sampled_valid.to_parquet(f"{output_dir}/{dataset_name}_sampled_valid.parquet", index=False)
    sampled_test.to_parquet(f"{output_dir}/{dataset_name}_sampled_test.parquet", index=False)

    # Step 3: Sample 20% from sampled_train_full for GT Only mode (WITHOUT replacement)
    n_gt_only = int(len(sampled_train_full) * GT_ONLY_RATIO)
    sampled_train_20 = sampled_train_full.sample(n=n_gt_only, replace=False, random_state=RANDOM_SEED)

    print(f"  Step 3: Sample {len(sampled_train_20)} from sampled_train_full ({GT_ONLY_RATIO*100:.0f}%) for GT Only mode")

    # Save sampled_train_20
    sampled_train_20.to_parquet(f"{output_dir}/{dataset_name}_sampled_train_20.parquet", index=False)

    print(f"  ✓ {dataset_name} sampling complete!")

    return {
        'dataset': dataset_name,
        'original': total_samples,
        'sampled': len(sampled),
        'train_full': len(sampled_train_full),
        'train_20': len(sampled_train_20),
        'valid': len(sampled_valid),
        'test': len(sampled_test)
    }


def main():
    """Sample all remaining datasets."""

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 10 datasets currently in use
    datasets_to_sample = [
        'arc_c',
        'commonsenseqa',
        'gpqa',
        'gsm_symbolic',
        'gsm8k',
        'humaneval_plus',
        'math',
        'mbpp_plus',
        'mmlu',
        'obqa'
    ]

    results = []

    for dataset in datasets_to_sample:
        try:
            result = sample_dataset(dataset, DATA_DIR, OUTPUT_DIR)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ❌ Error sampling {dataset}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary
    print(f"\n{'='*60}")
    print("SAMPLING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Original':>10} {'Sampled':>10} {'TrainFull':>10} {'Train20':>10} {'Valid':>8} {'Test':>8}")
    print('-' * 96)

    for r in results:
        print(f"{r['dataset']:<20} {r['original']:>10} {r['sampled']:>10} {r['train_full']:>10} {r['train_20']:>10} {r['valid']:>8} {r['test']:>8}")

    print('-' * 96)
    print(f"{'TOTAL':<20} {sum(r['original'] for r in results):>10} {sum(r['sampled'] for r in results):>10} {sum(r['train_full'] for r in results):>10} {sum(r['train_20'] for r in results):>10} {sum(r['valid'] for r in results):>8} {sum(r['test'] for r in results):>8}")

    print(f"\n✓ All datasets sampled successfully!")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
