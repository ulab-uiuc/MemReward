#!/usr/bin/env python3
'''
Verify is_train field consistency across VERL training/validation datasets.
Checks that train has is_train=True and valid has is_train=False.
Related: fix_validation_is_train.py for fixing incorrect values.
'''

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

CHECKS = [
    {
        'name': 'Qwen2.5 3B Standard Mix (10 domains)',
        'train': PROJECT_ROOT / 'data/qwen2.5-3b/verl_train_mix/train.parquet',
        'valid': PROJECT_ROOT / 'data/qwen2.5-3b/verl_train_mix/valid.parquet',
    },
    {
        'name': 'Qwen2.5 3B Generalization (Mix)',
        'train': PROJECT_ROOT / 'data/generalization/verl_train/train.parquet',
        'valid': PROJECT_ROOT / 'data/generalization/verl_train/valid.parquet',
    },
]

def verify_file(file_path: Path, expected_is_train: bool, dataset_name: str, split: str):
    """Verify a single file"""

    if not file_path.exists():
        print(f"  ⚠️  SKIP: {split} file does not exist")
        return None

    df = pd.read_parquet(file_path)

    if 'is_train' not in df.columns:
        print(f"  ❌ FAIL: {split} missing is_train field")
        return False

    if not (df['is_train'] == expected_is_train).all():
        wrong_count = (df['is_train'] != expected_is_train).sum()
        print(f"  ❌ FAIL: {split} has {wrong_count}/{len(df)} wrong values")
        print(f"      Expected: is_train={expected_is_train}")
        print(f"      Found: {df['is_train'].value_counts().to_dict()}")
        return False

    total = len(df)
    print(f"  ✅ PASS: {split} has correct is_train={expected_is_train} ({total} samples)")
    return True

def main():
    print("=" * 80)
    print("VERIFY is_train FIELDS CONSISTENCY")
    print("=" * 80)
    print()

    all_passed = True

    for check in CHECKS:
        name = check['name']
        train_file = check['train']
        valid_file = check['valid']

        print("=" * 80)
        print(f"Dataset: {name}")
        print("=" * 80)

        print(f"\nTRAIN: {train_file.name}")
        train_result = verify_file(train_file, True, name, 'TRAIN')

        print(f"\nVALID: {valid_file.name}")
        valid_result = verify_file(valid_file, False, name, 'VALID')

        print()
        if train_result and valid_result:
            print(f"✅ {name}: PASS")
        elif train_result is None and valid_result is None:
            print(f"⚠️  {name}: SKIPPED (files not found)")
        else:
            print(f"❌ {name}: FAIL")
            all_passed = False

        print()

    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if all_passed:
        print("✅ ALL CHECKS PASSED")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        return 1

if __name__ == '__main__':
    exit(main())
