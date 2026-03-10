#!/usr/bin/env python3
'''
Generate multi-ratio training splits from train_full parquet files.
Samples {10,30,40,50,60,70}% from each dataset for GT ratio ablation.
Related: sample_1500_datasets.py for base train_full generation.
'''

import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sampled_1500"

DATASETS = [
    'gsm8k', 'math', 'gsm_symbolic',
    'mmlu', 'commonsenseqa', 'obqa', 'arc_c', 'gpqa',
    'humaneval_plus', 'mbpp_plus'
]

# New ratios to generate (20 already exists, skip it)
NEW_RATIOS = [10, 30, 40, 50, 60, 70]


def main():
    print("=" * 70)
    print("Generate Multi-Ratio Training Splits")
    print("=" * 70)
    print(f"Data dir: {DATA_DIR}")
    print(f"Ratios to generate: {NEW_RATIOS}")
    print(f"Datasets: {len(DATASETS)}")
    print()

    header = f"{'Dataset':<20} {'Full':>6}"
    for r in NEW_RATIOS:
        header += f" {'t_' + str(r):>6}"
    print(header)
    print("-" * len(header))

    total_created = 0

    for ds in DATASETS:
        full_path = DATA_DIR / f"{ds}_sampled_train_full.parquet"
        if not full_path.exists():
            print(f"  WARNING: {full_path} not found, skipping")
            continue

        df_full = pd.read_parquet(full_path)
        row = f"{ds:<20} {len(df_full):>6}"

        for ratio in NEW_RATIOS:
            n_sample = max(1, int(len(df_full) * ratio / 100))
            df_sample = df_full.sample(n=n_sample, replace=False, random_state=RANDOM_SEED)

            out_path = DATA_DIR / f"{ds}_sampled_train_{ratio}.parquet"
            df_sample.to_parquet(out_path, index=False)
            total_created += 1
            row += f" {len(df_sample):>6}"

        print(row)

    print()
    print(f"Created {total_created} new parquet files in {DATA_DIR}")
    print("Existing _train_20 files are untouched.")


if __name__ == "__main__":
    main()
