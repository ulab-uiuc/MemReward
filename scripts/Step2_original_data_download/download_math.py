#!/usr/bin/env python3
'''
Download MATH dataset (hendrycks/math) with checksum-based reproducibility guard.
Competition-level math problems from algebra, geometry, number theory, etc.
Related: download_datasets.py for GSM8K/GSM-Symbolic.

Note: Original hendrycks/math was removed from HuggingFace. Uses nlile mirror
with checksum verification to prevent index drift across re-downloads.
Use --force to re-download (indices will change).
'''

import os
import json
import hashlib
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path

# Project root (relative to this script: scripts/Step2_original_data_download/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Output directory
OUTPUT_DIR = str(PROJECT_ROOT / "data")

# Expected checksums for reproducibility verification
CHECKSUMS_FILE = os.path.join(OUTPUT_DIR, "math", "DATA_CHECKSUMS.json")

# System prompt (same as other datasets)
UNIFIED_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful assistant that solves problems step by step.\n"
        "CRITICAL: End your response with #### followed by just the final answer value.\n"
        "Example format: #### 42\n"
        "The value after #### must be ONLY the numerical answer or expression."
    )
}


def extract_boxed_answer(solution: str) -> str:
    """
    Extract answer from \\boxed{...} in MATH solutions, handling nested braces.

    Uses rfind to locate the LAST \\boxed{} (the final answer), then walks
    forward counting brace depth to find the matching closing brace.

    Returns the full solution string if no \\boxed{} is found.
    """
    BOXED_PREFIX = '\\boxed{'
    if BOXED_PREFIX not in solution:
        return solution

    start = solution.rfind(BOXED_PREFIX)
    content_start = start + len(BOXED_PREFIX)

    brace_depth = 0
    i = content_start
    while i < len(solution):
        ch = solution[i]
        if ch == '{':
            brace_depth += 1
        elif ch == '}':
            if brace_depth == 0:
                return solution[content_start:i]
            brace_depth -= 1
        i += 1

    # Fallback: braces never balanced (shouldn't happen in valid LaTeX)
    return solution[content_start:]


def validate_math_answers(data_list: list) -> int:
    """
    Validate extracted answers for truncation (unbalanced braces).
    Returns the number of problematic entries and prints warnings.
    """
    problems = 0
    for item in data_list:
        answer = item['extra_info']['answer']
        if answer.count('{') != answer.count('}'):
            problems += 1
            if problems <= 5:
                idx = item['extra_info']['index']
                print(f"  WARNING: Unbalanced braces in answer [{idx}]: {repr(answer)[:80]}")
    return problems


def create_math_prompt(problem: str) -> list:
    """Create prompt for MATH problems."""
    return [
        UNIFIED_SYSTEM_PROMPT,
        {
            "role": "user",
            "content": (
                f"Solve this math problem:\n\n"
                f"{problem}\n\n"
                "Show your work step by step.\n\n"
                "Write your final answer as:\n"
                "#### [answer]"
            )
        }
    ]


def verify_existing_data(math_dir: str) -> bool:
    """
    Check if existing math parquet files match expected checksums.
    Returns True if all files exist and checksums match.
    """
    if not os.path.exists(CHECKSUMS_FILE):
        return False

    with open(CHECKSUMS_FILE, 'r') as f:
        expected = json.load(f)

    for split in ['train', 'valid', 'test']:
        path = os.path.join(math_dir, f'{split}.parquet')
        if not os.path.exists(path):
            print(f"  {split}.parquet not found")
            return False

        with open(path, 'rb') as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()

        if actual_hash != expected[split]['sha256']:
            print(f"  {split}.parquet checksum mismatch!")
            print(f"    Expected: {expected[split]['sha256'][:16]}...")
            print(f"    Actual:   {actual_hash[:16]}...")
            return False

    return True


def prepare_math_dataset(num_train: int = 6750, num_valid: int = 1125, num_test: int = 625,
                         force: bool = False):
    """
    Download and prepare MATH dataset from hendrycks/math.

    Args:
        num_train: Number of training samples (default: 6750, ~75%)
        num_valid: Number of validation samples (default: 1125, ~12.5%)
        num_test: Number of test samples (default: 625, ~7.5%)
        force: If True, re-download even if existing data passes checksum
    """
    math_dir = f"{OUTPUT_DIR}/math"

    # --- Reproducibility guard: skip download if existing data is valid ---
    if not force and os.path.exists(math_dir):
        print("Checking existing MATH data for reproducibility...")
        if verify_existing_data(math_dir):
            df = pd.read_parquet(os.path.join(math_dir, 'train.parquet'))
            print(f"\n✓ Existing MATH data passes checksum verification. Skipping download.")
            print(f"  Train: {len(pd.read_parquet(os.path.join(math_dir, 'train.parquet')))}")
            print(f"  Valid: {len(pd.read_parquet(os.path.join(math_dir, 'valid.parquet')))}")
            print(f"  Test:  {len(pd.read_parquet(os.path.join(math_dir, 'test.parquet')))}")
            print(f"\n  To re-download, use: python download_math.py --force")
            return (
                pd.read_parquet(os.path.join(math_dir, 'train.parquet')),
                pd.read_parquet(os.path.join(math_dir, 'valid.parquet')),
                pd.read_parquet(os.path.join(math_dir, 'test.parquet')),
            )
        else:
            print("  Existing data missing or checksums don't match. Will re-download.")
            print("  WARNING: Re-downloaded data may have different indices than original!")
            print("           This can break gt_identifiers_train*.json alignment.")

    print("Loading MATH dataset...")

    try:
        # NOTE: Original 'hendrycks/math' was removed from HF Hub.
        # Primary: 'nlile/hendrycks-MATH-benchmark' (full mirror, same format)
        # Fallback: 'EleutherAI/hendrycks_math' (per-topic configs, same data)
        try:
            dataset = load_dataset("nlile/hendrycks-MATH-benchmark", trust_remote_code=True)
            train_data_raw = list(dataset['train'])
            test_data_raw = list(dataset['test'])
            print(f"  Loaded from nlile/hendrycks-MATH-benchmark")
        except Exception:
            print("  Primary source unavailable, trying EleutherAI/hendrycks_math...")
            MATH_TOPICS = [
                'algebra', 'counting_and_probability', 'geometry',
                'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus',
            ]
            train_data_raw = []
            test_data_raw = []
            for topic in MATH_TOPICS:
                ds = load_dataset("EleutherAI/hendrycks_math", topic, trust_remote_code=True)
                train_data_raw.extend(list(ds['train']))
                test_data_raw.extend(list(ds['test']))
                print(f"    Loaded {topic}: {len(ds['train'])} train, {len(ds['test'])} test")

        print(f"  Raw dataset total: {len(train_data_raw)} train, {len(test_data_raw)} test")

        # Combine all data for custom splitting
        all_data = train_data_raw + test_data_raw
        print(f"  Total samples: {len(all_data)}")

    except Exception as e:
        print(f"  Error loading dataset: {e}")
        raise

    def process_item(item, idx, split):
        """Process a single MATH item."""
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        level = item.get('level', '')
        type_category = item.get('type', '')

        answer = extract_boxed_answer(solution)

        prompt = create_math_prompt(problem)

        return {
            'data_source': 'math',
            'prompt': prompt,
            'ability': 'math',
            'reward_model': 'math_reward',  # Uses special MATH reward function
            'extra_info': {
                'answer': answer,
                'solution': solution,
                'level': level,
                'type': type_category,
                'index': idx,
                'split': split
            }
        }

    # Shuffle and split data
    indices = list(range(len(all_data)))
    np.random.seed(42)
    np.random.shuffle(indices)

    total_needed = num_train + num_valid + num_test
    if len(all_data) < total_needed:
        print(f"  Warning: Only {len(all_data)} samples available, adjusting splits...")
        ratio = len(all_data) / total_needed
        num_train = int(num_train * ratio)
        num_valid = int(num_valid * ratio)
        num_test = len(all_data) - num_train - num_valid

    train_data = [process_item(all_data[i], i, 'train') for i in indices[:num_train]]
    valid_data = [process_item(all_data[i], i, 'valid') for i in indices[num_train:num_train+num_valid]]
    test_data = [process_item(all_data[i], i, 'test') for i in indices[num_train+num_valid:num_train+num_valid+num_test]]

    print("\n  Validating extracted answers...")
    for split_name, split_data in [("train", train_data), ("valid", valid_data), ("test", test_data)]:
        n_bad = validate_math_answers(split_data)
        total = len(split_data)
        if n_bad > 0:
            print(f"  ERROR: {split_name} has {n_bad}/{total} answers with unbalanced braces!")
            raise ValueError(
                f"MATH {split_name}: {n_bad}/{total} answers have unbalanced braces "
                f"(likely truncated \\boxed{{}} extraction). Fix extract_boxed_answer()."
            )
        print(f"  {split_name}: {total} answers OK")

    math_dir = f"{OUTPUT_DIR}/math"
    os.makedirs(math_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{math_dir}/train.parquet")
    valid_df.to_parquet(f"{math_dir}/valid.parquet")
    test_df.to_parquet(f"{math_dir}/test.parquet")

    # Post-save integrity check: re-read and verify answers survived serialization
    print("\n  Post-save integrity check...")
    for split_name in ["train", "valid", "test"]:
        df_check = pd.read_parquet(f"{math_dir}/{split_name}.parquet")
        n_bad = 0
        for _, row in df_check.iterrows():
            ei = row['extra_info']
            if isinstance(ei, str):
                import json
                ei = json.loads(ei)
            ans = ei.get('answer', '')
            if ans.count('{') != ans.count('}'):
                n_bad += 1
        if n_bad > 0:
            raise ValueError(
                f"MATH {split_name}: {n_bad} answers corrupted after parquet round-trip! "
                f"Check pyarrow serialization of nested dicts with braces."
            )
        print(f"  {split_name}: round-trip OK")

    print("\n  Generating checksums...")
    checksums = {}
    for split_name in ["train", "valid", "test"]:
        path = f"{math_dir}/{split_name}.parquet"
        with open(path, 'rb') as f:
            checksums[split_name] = {
                'n_rows': len(pd.read_parquet(path)),
                'sha256': hashlib.sha256(open(path, 'rb').read()).hexdigest(),
            }
    checksums_path = f"{math_dir}/DATA_CHECKSUMS.json"
    with open(checksums_path, 'w') as f:
        json.dump(checksums, f, indent=2)
    print(f"  Saved checksums to {checksums_path}")

    print(f"\n✓ MATH dataset saved to {math_dir}")
    print(f"  Train: {len(train_data)}")
    print(f"  Valid: {len(valid_data)}")
    print(f"  Test:  {len(test_data)}")

    # Show sample
    print("\n  Sample problem:")
    sample = train_data[0]
    print(f"    Type: {sample['extra_info']['type']}")
    print(f"    Level: {sample['extra_info']['level']}")
    print(f"    Problem preview: {sample['extra_info'].get('solution', '')[:100]}...")

    return train_df, valid_df, test_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Download and prepare MATH dataset')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if existing data passes checksum. '
                             'WARNING: indices will change and may break gt_identifiers alignment.')
    args = parser.parse_args()
    prepare_math_dataset(force=args.force)
