#!/usr/bin/env python3
'''
Fix reward_model field format in VERL training parquet files.
Converts JSON string to Python dict that VERL batch manager expects.
Related: generate_standard_verl_3modes_data.py for data generation.
'''

import sys
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def fix_reward_model_format(file_path: Path) -> bool:
    """
    Fix reward_model field in parquet file.

    Converts reward_model from JSON string to Python dict.

    Returns:
        bool: True if fixed or already correct, False on error
    """
    try:
        table = pq.read_table(file_path)

        if 'reward_model' not in table.column_names:
            print(f"⚠️  WARNING: {file_path.name} has no 'reward_model' column")
            return True

        df = table.to_pandas()

        first_rm = df.iloc[0]['reward_model']
        if isinstance(first_rm, dict):
            if 'data_source' in df.columns:
                math_samples = df[df['data_source'] == 'math']
                if len(math_samples) > 0:
                    needs_math_fix = sum(1 for idx, row in math_samples.iterrows()
                                        if isinstance(row['reward_model'], dict) and
                                        row['reward_model'].get('ground_truth') == 'math_reward')
                    if needs_math_fix > 0:
                        print(f"🔧 DETECTED: {needs_math_fix} MATH samples with 'math_reward' marker need conversion")
                    else:
                        print(f"✅ SKIP: {file_path.name} already has dict format")
                        return True
                else:
                    print(f"✅ SKIP: {file_path.name} already has dict format")
                    return True
            else:
                print(f"✅ SKIP: {file_path.name} already has dict format")
                return True

        print(f"🔧 FIXING: {file_path.name} (converting JSON strings to dicts)")

        def parse_reward_model(row):
            """Parse reward_model field with access to extra_info for MATH dataset."""
            rm = row['reward_model']
            extra_info = row.get('extra_info', {})

            if rm is None:
                return {'ground_truth': ''}
            if isinstance(rm, dict):
                gt = rm.get('ground_truth', '')
                if gt == 'math_reward' or (isinstance(gt, str) and 'math' in gt.lower() and 'reward' in gt.lower()):
                    if isinstance(extra_info, dict) and 'answer' in extra_info:
                        answer = str(extra_info['answer'])
                        return {'ground_truth': answer}
                return rm
            if isinstance(rm, str):
                try:
                    parsed = json.loads(rm)
                    if not isinstance(parsed, dict):
                        return {'ground_truth': str(parsed)}
                    if 'ground_truth' not in parsed:
                        parsed['ground_truth'] = ''
                    return parsed
                except json.JSONDecodeError:
                    if rm == 'math_reward' or (isinstance(rm, str) and 'math' in rm.lower() and 'reward' in rm.lower()):
                        if isinstance(extra_info, dict) and 'answer' in extra_info:
                            answer = str(extra_info['answer'])
                            print(f"  🔧 Converting MATH marker '{rm}' → answer: {answer[:50]}...")
                            return {'ground_truth': answer}
                        else:
                            print(f"  ⚠️  MATH marker '{rm}' found but no answer in extra_info")
                            return {'ground_truth': ''}

                    print(f"  ⚠️  Failed to parse: {rm}")
                    return {'ground_truth': ''}
            return {'ground_truth': ''}

        reward_models = df.apply(parse_reward_model, axis=1).tolist()

        all_keys = set()
        for rm in reward_models:
            all_keys.update(rm.keys())

        for rm in reward_models:
            for key in all_keys:
                if key not in rm:
                    rm[key] = ''

        df['reward_model'] = reward_models

        struct_fields = [pa.field(key, pa.string()) for key in sorted(all_keys)]
        reward_model_type = pa.struct(struct_fields)

        original_schema = table.schema

        new_fields = []
        for field in original_schema:
            if field.name == 'reward_model':
                new_fields.append(pa.field('reward_model', reward_model_type))
            else:
                new_fields.append(field)

        new_schema = pa.schema(new_fields)

        new_table = pa.Table.from_pandas(df, schema=new_schema)

        pq.write_table(new_table, file_path)

        print(f"✅ FIXED: {file_path.name} ({len(df)} rows converted)")
        return True

    except Exception as e:
        print(f"❌ ERROR: Failed to fix {file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_reward_model_format.py <parquet_file>")
        sys.exit(1)

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print(f"❌ ERROR: File not found: {file_path}")
        sys.exit(1)

    print("=" * 80)
    print("FIX reward_model FIELD FORMAT")
    print("=" * 80)
    print()

    success = fix_reward_model_format(file_path)

    print()
    if success:
        print("✅ SUCCESS")
        sys.exit(0)
    else:
        print("❌ FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
