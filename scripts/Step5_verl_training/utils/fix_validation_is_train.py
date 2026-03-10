#!/usr/bin/env python3
'''
Fix is_train fields in VERL training and validation parquet files.
Sets is_train=True for train (GNN reward routing) and False for valid (GT-only).
Related: verify_is_train_fields.py for post-fix verification.
'''

import argparse
import pandas as pd
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

DATASETS = {
    'qwen2.5_3b_standard': {
        'name': 'Qwen2.5 3B Standard Mix (10 domains)',
        'train': PROJECT_ROOT / 'data/qwen2.5-3b/verl_train_mix/train.parquet',
        'valid': PROJECT_ROOT / 'data/qwen2.5-3b/verl_train_mix/valid.parquet',
    },
    'qwen2.5_1.5b_standard': {
        'name': 'Qwen2.5 1.5B Standard Mix (10 domains)',
        'train': PROJECT_ROOT / 'data/qwen2.5-1.5b/verl_train_mix/train.parquet',
        'valid': PROJECT_ROOT / 'data/qwen2.5-1.5b/verl_train_mix/valid.parquet',
    },
    'qwen2.5_3b_generalization': {
        'name': 'Qwen2.5 3B Generalization (Mix)',
        'train': PROJECT_ROOT / 'data/generalization/verl_train/train.parquet',
        'valid': PROJECT_ROOT / 'data/generalization/verl_train/valid.parquet',
    },
}

def fix_is_train_field(file_path: Path, target_value: bool, backup: bool = True):
    """Fix is_train field in parquet file"""

    df = pd.read_parquet(file_path)

    if 'is_train' in df.columns:
        if (df['is_train'] == target_value).all():
            return 'skip', None
        else:
            action = 'fix'
    else:
        action = 'add'

    if backup:
        backup_path = file_path.with_suffix('.parquet.bak')
        shutil.copy2(file_path, backup_path)
        backup_info = str(backup_path)
    else:
        backup_info = None

    df['is_train'] = target_value

    def update_extra_info(extra):
        if isinstance(extra, dict):
            extra['is_train'] = target_value
        return extra

    df['extra_info'] = df['extra_info'].apply(update_extra_info)

    df.to_parquet(file_path, index=False)

    return action, backup_info

def main():
    parser = argparse.ArgumentParser(description='Fix is_train fields in training/validation data')
    parser.add_argument('--dataset', choices=['qwen2.5_3b_standard', 'qwen2.5_1.5b_standard', 'qwen2.5_3b_generalization'], required=True,
                        help='Which dataset to fix')
    parser.add_argument('--no-backup', action='store_true',
                        help='Do not create backup files')
    args = parser.parse_args()

    config = DATASETS[args.dataset]
    backup = not args.no_backup

    print("=" * 80)
    print("FIX is_train FIELDS IN TRAINING/VALIDATION DATA")
    print("=" * 80)
    print()
    print(f"Fixing: {config['name']}")
    print("=" * 80)
    print()

    print("=" * 80)
    print(f"DATASET: {config['name']}")
    print("=" * 80)
    print()

    success = True

    print("TRAIN DATA:")
    train_action, train_backup = fix_is_train_field(config['train'], True, backup)

    if train_action == 'skip':
        print("  ✅ SKIP: Already has correct is_train=True")
    elif train_action == 'add':
        print(f"  ✅ ADDED: is_train=True field")
        if train_backup:
            print(f"     Backup: {train_backup}")
    elif train_action == 'fix':
        print(f"  ✅ FIXED: Updated to is_train=True")
        if train_backup:
            print(f"     Backup: {train_backup}")

    print()

    print("VALID DATA:")
    valid_action, valid_backup = fix_is_train_field(config['valid'], False, backup)

    if valid_action == 'skip':
        print("  ✅ SKIP: Already has correct is_train=False")
    elif valid_action == 'add':
        print(f"  ✅ ADDED: is_train=False field")
        if valid_backup:
            print(f"     Backup: {valid_backup}")
    elif valid_action == 'fix':
        print(f"  ✅ FIXED: Updated to is_train=False")
        if valid_backup:
            print(f"     Backup: {valid_backup}")

    print()
    print("-" * 80)

    if success:
        print("✅ PASS: Both train and valid fixed/verified")
    else:
        print("❌ FAIL: Errors occurred")
        return 1

    print()
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print()

    return 0

if __name__ == '__main__':
    exit(main())
