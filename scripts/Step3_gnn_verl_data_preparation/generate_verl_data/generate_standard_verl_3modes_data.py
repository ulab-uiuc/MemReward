'''
Generate 3-mode VERL training data for in-domain experiments (10 datasets).
Creates Full GT (5358 queries), Partial GT (1104), and Mix mode with GT routing.
Related: generate_generalization_verl_3modes_data.py for OOD datasets.
'''

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add src to path (project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def load_gt_identifiers(path: str) -> dict:
    """Load GT identifiers used for GNN training."""
    with open(path, 'r') as f:
        return json.load(f)


def normalize_data_source(data_source: str) -> str:
    """Normalize data_source name to match gt_identifiers_train{ratio}.json."""
    if data_source == 'gsm_symbolic_main':
        return 'gsm_symbolic'
    return data_source


def is_gt_query(data_source: str, index: int, gt_identifiers: dict) -> bool:
    """Check if a query should use GT reward based on gt_identifiers_train{ratio}.json."""
    data_source = normalize_data_source(data_source)
    if data_source not in gt_identifiers:
        return False
    indices = gt_identifiers[data_source].get('indices', [])
    return int(index) in [int(x) for x in indices]


def normalize_reward_model(reward_model_val, extra_info) -> dict:
    """
    Normalize reward_model to dict format for VERL batch manager.

    VERL batch manager expects reward_model to be a dict with .get() method.
    Some datasets (like math) use 'math_reward' string, which needs conversion.

    Args:
        reward_model_val: Either JSON string or already a dict
        extra_info: Dict containing answer info (for math dataset)

    Returns:
        Dict with 'ground_truth' key
    """
    # Parse JSON string if needed
    if isinstance(reward_model_val, str):
        try:
            parsed = json.loads(reward_model_val)
        except json.JSONDecodeError:
            parsed = reward_model_val
    else:
        parsed = reward_model_val

    # If it's already a proper dict with ground_truth, return as-is
    if isinstance(parsed, dict) and 'ground_truth' in parsed:
        return parsed

    # For math dataset: reward_model is 'math_reward', get ground_truth from extra_info
    if parsed == 'math_reward' or (isinstance(parsed, str) and 'math' in parsed.lower()):
        # Parse extra_info if needed
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        answer = extra_info.get('answer', '')
        return {'ground_truth': str(answer)}

    # Fallback: wrap in dict
    return {'ground_truth': str(parsed) if parsed else ''}


def generate_partial_gt(gt_identifiers: dict, model_name: str = 'qwen2.5'):
    """Generate Partial GT mode from verl_train."""

    print("\n" + "="*70)
    print("Partial GT Mode (1104 GNN training queries)")
    print("="*70)

    INPUT_DIR = Path(f'data/{model_name}/verl_train')
    OUTPUT_DIR = Path(f'data/{model_name}/verl_train_partial_gt')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        input_path = INPUT_DIR / f'{split}.parquet'

        if not input_path.exists():
            print(f"\n{split.upper()}: Input file not found")
            continue

        # Load data
        df = pd.read_parquet(input_path)

        # Parse extra_info to get indices
        if isinstance(df['extra_info'].iloc[0], str):
            df['extra_info_parsed'] = df['extra_info'].apply(json.loads)
        else:
            df['extra_info_parsed'] = df['extra_info']

        # Filter to GT queries only
        gt_mask = df.apply(
            lambda row: is_gt_query(
                row['data_source'],
                row['extra_info_parsed'].get('index'),
                gt_identifiers
            ),
            axis=1
        )

        filtered_df = df[gt_mask].copy()

        # Convert extra_info from parsed version (dict) and drop the temp column
        filtered_df['extra_info'] = filtered_df['extra_info_parsed']
        filtered_df = filtered_df.drop(columns=['extra_info_parsed'])

        # Normalize reward_model to dict format (VERL batch manager expects dict with .get() method)
        filtered_df['reward_model'] = filtered_df.apply(
            lambda row: normalize_reward_model(row['reward_model'], row['extra_info']),
            axis=1
        )
        print(f"  Normalized reward_model to dict format")

        # Validate math ground truths (catch truncated \boxed{} early)
        math_rows = filtered_df[filtered_df['data_source'] == 'math']
        if len(math_rows) > 0:
            n_bad = sum(
                1 for _, r in math_rows.iterrows()
                if str(r['reward_model'].get('ground_truth', '')).count('{')
                != str(r['reward_model'].get('ground_truth', '')).count('}')
            )
            if n_bad > 0:
                print(f"  WARNING: {n_bad}/{len(math_rows)} math ground truths have "
                      f"unbalanced braces! Fix data/math/ upstream first.")

        # Save using pyarrow directly with explicit struct schema for reward_model
        import pyarrow as pa
        import pyarrow.parquet as pq
        output_path = OUTPUT_DIR / f'{split}.parquet'

        # Create reward_model as proper struct array (all have consistent 'ground_truth' key)
        reward_model_struct = pa.StructArray.from_pandas(
            filtered_df['reward_model'].tolist(),
            type=pa.struct([('ground_truth', pa.string())])
        )

        # Build table manually to ensure proper types
        arrays = []
        names = []
        for col in filtered_df.columns:
            if col == 'reward_model':
                arrays.append(reward_model_struct)
            else:
                arrays.append(pa.array(filtered_df[col].tolist()))
            names.append(col)

        table = pa.table(dict(zip(names, arrays)))
        pq.write_table(table, output_path)

        # Statistics
        print(f"\n{split.upper()}:")
        print(f"  Total queries: {len(df)}")
        print(f"  GT queries (filtered): {len(filtered_df)} ({100*len(filtered_df)/len(df):.1f}%)")

        # Dataset distribution
        print(f"  Dataset distribution:")
        for ds in sorted(filtered_df['data_source'].unique()):
            count = len(filtered_df[filtered_df['data_source'] == ds])
            print(f"    {ds:20} {count:4}")

        print(f"  Saved to: {output_path}")

    print(f"\n{'='*70}")
    print("✓ Partial GT data created successfully ")
    print(f"Output directory: {OUTPUT_DIR}")


def generate_mix(model_name: str = 'qwen2.5'):
    """Generate Mix mode from verl_train (simple copy)."""

    print("\n" + "="*70)
    print("Mix Mode (from verl_train)")
    print("="*70)

    INPUT_DIR = Path(f'data/{model_name}/verl_train')
    OUTPUT_DIR = Path(f'data/{model_name}/verl_train_mix')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'valid', 'test']:
        input_path = INPUT_DIR / f'{split}.parquet'

        if not input_path.exists():
            print(f"\n{split.upper()}: Input file not found")
            continue

        df = pd.read_parquet(input_path)

        # Convert extra_info from JSON string to dict if needed
        if isinstance(df['extra_info'].iloc[0], str):
            print(f"  Converting extra_info from JSON string to dict...")
            df['extra_info'] = df['extra_info'].apply(json.loads)

        # Normalize reward_model to dict format (VERL batch manager expects dict with .get() method)
        df['reward_model'] = df.apply(
            lambda row: normalize_reward_model(row['reward_model'], row['extra_info']),
            axis=1
        )
        print(f"  Normalized reward_model to dict format")

        # CRITICAL: Add is_train flag to extra_info for proper validation routing
        # This ensures validation uses 100% GT reward instead of mixed GNN reward
        if split == 'valid':
            def add_is_train_flag(extra_info):
                if isinstance(extra_info, dict):
                    extra_info = extra_info.copy()
                    extra_info['is_train'] = False
                return extra_info
            df['extra_info'] = df['extra_info'].apply(add_is_train_flag)
            print(f"  Added is_train=False to validation extra_info")

        # Save using pyarrow directly with explicit struct schema for reward_model
        import pyarrow as pa
        import pyarrow.parquet as pq
        output_path = OUTPUT_DIR / f'{split}.parquet'

        # Create reward_model as proper struct array (all have consistent 'ground_truth' key)
        reward_model_struct = pa.StructArray.from_pandas(
            df['reward_model'].tolist(),
            type=pa.struct([('ground_truth', pa.string())])
        )

        # Build table manually to ensure proper types
        arrays = []
        names = []
        for col in df.columns:
            if col == 'reward_model':
                arrays.append(reward_model_struct)
            else:
                arrays.append(pa.array(df[col].tolist()))
            names.append(col)

        table = pa.table(dict(zip(names, arrays)))
        pq.write_table(table, output_path)

        print(f"\n{split.upper()}:")
        print(f"  Total queries: {len(df)}")
        print(f"  Saved to: {output_path}")

    print(f"\n{'='*70}")
    print("✓ Mix mode data created successfully ")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nUsage: verl_mixed_reward_qwen3b.py will automatically route:")
    print(f"  - Queries in gt_identifiers_train{{ratio}}.json → GT reward")
    print(f"  - Other queries → GNN prediction")


def verify_alignment_dedup(gt_identifiers: dict, model_name: str = 'qwen2.5'):
    """Verify index alignment with deduplicated data."""

    print("\n" + "="*70)
    print("Verification: Index Alignment (Deduplicated)")
    print("="*70)

    partial_df = pd.read_parquet(f'data/{model_name}/verl_train_partial_gt/train.parquet')

    # Parse extra_info
    if isinstance(partial_df['extra_info'].iloc[0], str):
        partial_df['extra_info_parsed'] = partial_df['extra_info'].apply(json.loads)
    else:
        partial_df['extra_info_parsed'] = partial_df['extra_info']

    print(f"\nPartial GT train data:")
    print(f"  Total queries: {len(partial_df)} (NO duplicates)")

    # Get unique indices
    all_unique_indices = set()
    for row in partial_df['extra_info_parsed']:
        all_unique_indices.add(row.get('index'))
    print(f"  Unique indices: {len(all_unique_indices)}")

    print(f"\nGT identifiers total: {sum(len(v['indices']) for v in gt_identifiers.values())} unique indices")

    # Verify dataset coverage
    print(f"\nDataset coverage (checking unique indices):")
    total_matched = 0

    for dataset in sorted(gt_identifiers.keys()):
        # Normalize dataset name for lookup
        if dataset == 'gsm_symbolic':
            lookup_name = 'gsm_symbolic_main'
        else:
            lookup_name = dataset

        # Get unique indices from partial data
        dataset_df = partial_df[partial_df['data_source'] == lookup_name]
        unique_indices_in_data = set([row['index'] for row in dataset_df['extra_info_parsed']])

        # Get GT indices
        gt_indices_set = set([int(x) for x in gt_identifiers[dataset]['indices']])

        # Check match
        matched = unique_indices_in_data == gt_indices_set
        match_str = "✓" if matched else "✗ MISMATCH"

        print(f"  {dataset:20} Rows: {len(dataset_df):3} | "
              f"Unique: {len(unique_indices_in_data):3} | "
              f"GT IDs: {len(gt_indices_set):3} {match_str}")

        if matched:
            total_matched += 1

    print(f"\n{'='*70}")
    print(f"✓ {total_matched}/{len(gt_identifiers)} datasets perfectly aligned")
    print(f"Note: NO duplicates in deduplicated data!")


def main():
    """Generate all modes from verl_train."""

    import argparse
    parser = argparse.ArgumentParser(description='Generate VERL training modes')
    parser.add_argument('--model', type=str, default='qwen2.5',
                        choices=['qwen2.5'],
                        help='Model name (default: qwen2.5)')
    parser.add_argument('--train-ratio', type=int, default=20,
                        help='Train ratio percentage (default: 20). Uses configs/gt_identifiers_train{ratio}.json')
    args = parser.parse_args()

    model_name = args.model

    # Load GT identifiers
    GT_IDENTIFIERS_PATH = f'configs/gt_identifiers_train{args.train_ratio}.json'
    gt_identifiers = load_gt_identifiers(GT_IDENTIFIERS_PATH)

    print("="*70)
    print(f"VERL Training Data Generation - {model_name.upper()}")
    print("="*70)
    print(f"GT Identifiers: {GT_IDENTIFIERS_PATH}")
    print(f"Total GT queries: {sum(v['n_total'] for v in gt_identifiers.values())}")
    print()

    # Generate Partial GT
    generate_partial_gt(gt_identifiers, model_name)

    # Generate Mix
    generate_mix(model_name)

    # Verification
    verify_alignment_dedup(gt_identifiers, model_name)

    print("\n" + "="*70)
    print(f"Summary: {model_name.upper()} VERL Training Modes")
    print("="*70)
    print(f"1. Full : data/{model_name}/verl_train/")
    print("   - Train: 5358 queries (NO duplicates)")
    print("   - 100% Ground Truth reward")
    print()
    print(f"2. Partial GT : data/{model_name}/verl_train_partial_gt/")
    print("   - Train: 1104 queries (NO duplicates)")
    print("   - 100% Ground Truth reward (GNN training data)")
    print()
    print(f"3. Mix : data/{model_name}/verl_train_mix/")
    print("   - Train: 5358 queries (NO duplicates)")
    print("   - Auto-routing: 1104 GT + 4254 GNN")
    print("="*70)


if __name__ == '__main__':
    main()
