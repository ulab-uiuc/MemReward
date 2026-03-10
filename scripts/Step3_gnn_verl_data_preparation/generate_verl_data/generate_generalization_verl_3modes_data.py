'''
Generate 3-mode VERL training data for out-of-domain generalization experiments.
Combines NuminaMath, SIQA, PIQA into Full GT, Partial GT, and Mix modes.
Related: generate_standard_verl_3modes_data.py for in-domain datasets.
'''

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add src to path (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Constants ---
GENERALIZATION_DATA_DIR = PROJECT_ROOT / "data" / "generalization"
VERL_TRAIN_DIR = GENERALIZATION_DATA_DIR / "verl_train"

GENERALIZATION_DATASETS = ['numina_math', 'siqa', 'piqa']
DOMAIN_MAP = {
    'numina_math': 'math',
    'siqa': 'qa',
    'piqa': 'qa',
}


# ---- Step 1: Prepare verl_train data ----

def _load_and_combine(split: str) -> pd.DataFrame:
    """Load and combine generalization datasets for a given split."""
    dfs = []
    for dataset in GENERALIZATION_DATASETS:
        path = GENERALIZATION_DATA_DIR / f"{dataset}_sampled_{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        df = pd.read_parquet(path)
        df['ability'] = DOMAIN_MAP[dataset]
        df['data_source'] = dataset
        dfs.append(df)
        print(f"  {dataset}: {len(df)} samples")

    combined = pd.concat(dfs, ignore_index=True)
    # Add sequential index for GT routing (used in generalization_gt_identifiers.json)
    combined['extra_info'] = combined.apply(
        lambda row: {**row['extra_info'], 'index': row.name},
        axis=1,
    )
    return combined


def prepare_verl_train():
    """Combine 3 generalization datasets into verl_train/."""
    print("\n" + "="*70)
    print("Step 1: Preparing verl_train data")
    print("="*70)

    VERL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    print("\nTraining data:")
    train_df = _load_and_combine('train')
    print(f"  Total: {len(train_df)} samples")

    print("\nValidation data:")
    valid_df = _load_and_combine('valid')
    print(f"  Total: {len(valid_df)} samples")

    train_df.to_parquet(VERL_TRAIN_DIR / "train.parquet", index=False)
    valid_df.to_parquet(VERL_TRAIN_DIR / "valid.parquet", index=False)

    print(f"\nSaved to: {VERL_TRAIN_DIR}/")

    # Verify schema
    train_loaded = pd.read_parquet(VERL_TRAIN_DIR / "train.parquet")
    print(f"  Columns: {train_loaded.columns.tolist()}")
    for field in ['data_source', 'prompt', 'ability', 'extra_info']:
        present = field in train_loaded.columns
        print(f"    {field:15s}: {'ok' if present else 'MISSING'}")


# ---- Shared helpers ----

def load_gt_identifiers(path: str) -> dict:
    """Load GT identifiers used for GNN training."""
    with open(path, 'r') as f:
        return json.load(f)


def get_relative_index(data_source: str, global_index: int) -> int:
    """
    Convert global index to dataset-relative index.

    Generalization datasets are combined in order: numina_math (750), siqa (749), piqa (750)
    Global indices:
    - numina_math: 0-749 → relative 0-749
    - siqa: 750-1498 → relative 0-748
    - piqa: 1499-2248 → relative 0-749
    """
    if data_source == 'numina_math':
        return global_index  # 0-749
    elif data_source == 'siqa':
        return global_index - 750  # 750-1498 → 0-748
    elif data_source == 'piqa':
        return global_index - 1499  # 1499-2248 → 0-749
    else:
        return global_index


def is_gt_query(data_source: str, global_index: int, gt_identifiers: dict) -> bool:
    """Check if a query should use GT reward based on generalization_gt_identifiers.json."""
    if data_source not in gt_identifiers:
        return False

    # Convert global index to relative index
    relative_index = get_relative_index(data_source, global_index)

    indices = gt_identifiers[data_source].get('indices', [])
    return int(relative_index) in [int(x) for x in indices]


def normalize_reward_model(reward_model_val, extra_info) -> dict:
    """
    Normalize reward_model to dict format for VERL batch manager.

    VERL batch manager expects reward_model to be a dict with .get() method.

    Args:
        reward_model_val: Either JSON string or already a dict
        extra_info: Dict containing answer info

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


# ---- Step 2: Generate partial GT mode ----

def generate_partial_gt(gt_identifiers: dict):
    """Generate Partial GT mode from generalization/verl_train."""

    print("\n" + "="*70)
    print("Step 2: Partial GT Mode (~450 GNN training queries)")
    print("="*70)

    INPUT_DIR = VERL_TRAIN_DIR
    OUTPUT_DIR = GENERALIZATION_DATA_DIR / 'verl_train_partial_gt'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'valid']:
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
    print("Partial GT data created successfully")
    print(f"Output directory: {OUTPUT_DIR}")


# ---- Step 3: Verify alignment ----

def verify_alignment(gt_identifiers: dict):
    """Verify index alignment."""

    print("\n" + "="*70)
    print("Step 3: Verification - Index Alignment")
    print("="*70)

    partial_path = GENERALIZATION_DATA_DIR / 'verl_train_partial_gt' / 'train.parquet'
    partial_df = pd.read_parquet(partial_path)

    # Parse extra_info
    if isinstance(partial_df['extra_info'].iloc[0], str):
        partial_df['extra_info_parsed'] = partial_df['extra_info'].apply(json.loads)
    else:
        partial_df['extra_info_parsed'] = partial_df['extra_info']

    print(f"\nPartial GT train data:")
    print(f"  Total queries: {len(partial_df)}")

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
        # Get unique indices from partial data
        dataset_df = partial_df[partial_df['data_source'] == dataset]
        unique_indices_in_data = set([row['index'] for row in dataset_df['extra_info_parsed']])

        # Get GT indices
        gt_indices_set = set([int(x) for x in gt_identifiers[dataset]['indices']])

        # Check match
        matched = unique_indices_in_data == gt_indices_set
        match_str = "ok" if matched else "MISMATCH"

        print(f"  {dataset:20} Rows: {len(dataset_df):3} | "
              f"Unique: {len(unique_indices_in_data):3} | "
              f"GT IDs: {len(gt_indices_set):3} {match_str}")

        if matched:
            total_matched += 1

    print(f"\n{'='*70}")
    print(f"{total_matched}/{len(gt_identifiers)} datasets perfectly aligned")


def main():
    """Generate all generalization VERL training modes."""

    # Load GT identifiers
    GT_IDENTIFIERS_PATH = str(PROJECT_ROOT / 'configs' / 'generalization_gt_identifiers.json')
    gt_identifiers = load_gt_identifiers(GT_IDENTIFIERS_PATH)

    print("="*70)
    print("GENERALIZATION VERL Training Data Generation")
    print("="*70)
    print(f"GT Identifiers: {GT_IDENTIFIERS_PATH}")
    print(f"Total GT queries: {sum(len(v['indices']) for v in gt_identifiers.values())}")
    print()

    # Step 1: Combine raw datasets into verl_train/
    prepare_verl_train()

    # Step 2: Generate Partial GT
    generate_partial_gt(gt_identifiers)

    # Step 3: Verification
    verify_alignment(gt_identifiers)

    print("\n" + "="*70)
    print("Summary: Generalization VERL Training Modes")
    print("="*70)
    print(f"1. Full GT: {VERL_TRAIN_DIR}/")
    print("   - Train: 2249 queries")
    print("   - 100% Ground Truth reward")
    print()
    print(f"2. Partial GT: {GENERALIZATION_DATA_DIR / 'verl_train_partial_gt'}/")
    print(f"   - Train: ~450 queries (20%)")
    print("   - 100% Ground Truth reward (GNN training data)")
    print()
    print(f"3. Mix: {VERL_TRAIN_DIR}/ (same as Full)")
    print("   - Train: 2249 queries")
    print("   - Auto-routing: ~450 GT + ~1799 GNN")
    print("   - Use with verl_mixed_reward_qwen3b.py")
    print("="*70)


if __name__ == '__main__':
    main()
