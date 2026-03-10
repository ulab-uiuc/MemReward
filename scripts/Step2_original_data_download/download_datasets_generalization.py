#!/usr/bin/env python3
'''
Download out-of-domain generalization test datasets from HuggingFace.
Covers NuminaMath, SIQA, PIQA for evaluating cross-domain transfer.
Related: download_datasets.py for in-domain datasets.
'''

import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import argparse
import re

# =============================================================================
# Prompt Templates (consistent with main training datasets)
# =============================================================================

UNIFIED_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful assistant skilled in mathematics, logical reasoning, and programming. "
        "Solve problems step by step, showing your work clearly.\n\n"
        "CRITICAL FORMAT REQUIREMENT:\n"
        "- You MUST end your response with '#### ' followed by your final answer\n"
        "- NEVER use \\boxed{}, $\\boxed{}$, or any LaTeX boxing format\n"
        "- ALWAYS use #### format, even for mathematical expressions\n"
        "- Example: #### 42 or #### x^2 + 1"
    )
}


def create_math_prompt(question: str) -> list:
    """Create prompt for math competition problems."""
    return [
        UNIFIED_SYSTEM_PROMPT,
        {
            "role": "user",
            "content": (
                f"Solve this math problem:\n\n"
                f"{question}\n\n"
                "Think step by step, showing your work clearly.\n\n"
                "Write your final answer as:\n"
                "#### [answer]"
            )
        }
    ]


def create_code_prompt(task: str, test_cases: list = None) -> list:
    """Create prompt for coding problems."""
    tests_str = ""
    if test_cases and len(test_cases) >= 3:
        tests_str = f"\n\nYour code should pass these tests:\n{test_cases[0]}\n{test_cases[1]}\n{test_cases[2]}\n"
    elif test_cases:
        tests_str = "\n\nYour code should pass these tests:\n" + "\n".join(test_cases[:3]) + "\n"

    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python programmer. Solve coding problems step by step.\n\n"
                "CRITICAL FORMAT REQUIREMENT:\n"
                "- You MUST end your response with '#### ' followed by your complete code\n"
                "- The code after #### should be in a ```python code block\n"
                "- NEVER output code without the #### marker\n"
                "- Example format:\n"
                "  [your reasoning]\n"
                "  #### \n"
                "  ```python\n"
                "  def solution():\n"
                "      pass\n"
                "  ```"
            )
        },
        {
            "role": "user",
            "content": (
                f"Task: {task}{tests_str}\n"
                "Think through your approach, then write the Python function.\n\n"
                "REMEMBER: You MUST end with #### followed by your code in a ```python block."
            )
        }
    ]


def create_mcqa_prompt(context: str, question: str, choices: str) -> list:
    """Create prompt for multiple choice QA (SIQA, PIQA)."""
    return [
        UNIFIED_SYSTEM_PROMPT,
        {
            "role": "user",
            "content": (
                f"Read the following and answer the question:\n\n"
                f"Context: {context}\n\n"
                f"Question: {question}\n\n"
                f"Options:\n{choices}\n\n"
                "Think about which option is most appropriate.\n\n"
                "Write your final answer as:\n"
                "#### [letter]"
            )
        }
    ]


# =============================================================================
# Dataset Preparation Functions
# =============================================================================

def prepare_numina_math_dataset(output_dir: str, num_samples: int = 2000):
    """
    Prepare NuminaMath dataset (competition-level mathematics).

    NuminaMath contains math problems from various competitions:
    - AMC, AIME, IMO
    - Various national olympiads
    - College-level competitions

    Source: AI-MO/NuminaMath-CoT (HuggingFace)
    """
    print("Loading NuminaMath dataset...")

    try:
        # Try loading NuminaMath-CoT which has chain-of-thought solutions
        dataset = load_dataset("AI-MO/NuminaMath-CoT")
        all_data = list(dataset['train'])
        print(f"  Loaded NuminaMath-CoT: {len(all_data)} samples")
    except Exception as e:
        print(f"  NuminaMath-CoT failed ({e}), trying NuminaMath-TIR...")
        try:
            dataset = load_dataset("AI-MO/NuminaMath-TIR")
            all_data = list(dataset['train'])
            print(f"  Loaded NuminaMath-TIR: {len(all_data)} samples")
        except Exception as e2:
            print(f"  Error loading NuminaMath: {e2}")
            return None, None, None

    def extract_answer(solution: str) -> str:
        """Extract final answer from solution."""
        # Look for boxed answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', solution)
        if boxed_match:
            return boxed_match.group(1).strip()

        # Look for #### answer format
        if '####' in solution:
            return solution.split('####')[-1].strip()

        lines = solution.strip().split('\n')
        return lines[-1].strip() if lines else ""

    def process_item(item, idx, split):
        problem = item.get('problem', item.get('question', ''))
        solution = item.get('solution', item.get('answer', ''))
        answer = extract_answer(solution)
        source = item.get('source', 'numina')

        prompt = create_math_prompt(problem)

        return {
            'data_source': 'numina_math',
            'prompt': prompt,
            'ability': 'math',
            'reward_model': {'ground_truth': answer},
            'extra_info': {
                'answer': answer,
                'full_solution': solution,
                'source': source,
                'index': idx,
                'split': split
            }
        }

    # Shuffle and split
    np.random.seed(42)
    indices = list(range(len(all_data)))
    np.random.shuffle(indices)

    # Limit samples
    indices = indices[:num_samples]
    n_train = int(len(indices) * 0.7)
    n_valid = int(len(indices) * 0.15)

    train_data = [process_item(all_data[i], i, 'train') for i in indices[:n_train]]
    valid_data = [process_item(all_data[i], i, 'valid') for i in indices[n_train:n_train+n_valid]]
    test_data = [process_item(all_data[i], i, 'test') for i in indices[n_train+n_valid:]]

    numina_dir = f"{output_dir}/numina_math"
    os.makedirs(numina_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{numina_dir}/train.parquet")
    valid_df.to_parquet(f"{numina_dir}/valid.parquet")
    test_df.to_parquet(f"{numina_dir}/test.parquet")

    print(f"  NuminaMath: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df


def prepare_siqa_dataset(output_dir: str, num_train: int = 5000, num_valid: int = 1000, num_test: int = 1000):
    """
    Prepare SIQA dataset (Social Interaction QA - social commonsense reasoning).

    SIQA tests the ability of models to reason about social situations and
    human interactions. Multiple choice format with 3 options.

    Source: lighteval/social_i_qa (HuggingFace - parquet version)
    """
    print("Loading SIQA dataset...")

    try:
        # Try lighteval version (parquet format, no remote code)
        dataset = load_dataset("lighteval/social_i_qa")
    except Exception as e:
        print(f"  lighteval/social_i_qa failed: {e}")
        try:
            # Try parquet branch of original
            dataset = load_dataset("allenai/social_i_qa", revision="refs/convert/parquet")
        except Exception as e2:
            print(f"  Error loading SIQA: {e2}")
            return None, None, None

    print(f"  Raw data: {len(dataset['train'])} train, {len(dataset['validation'])} validation")

    def format_choices(item) -> str:
        """Format choices with letters."""
        choices = [
            item.get('answerA', ''),
            item.get('answerB', ''),
            item.get('answerC', '')
        ]
        return "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices) if c])

    def process_item(item, idx, split):
        context = item.get('context', '')
        question = item.get('question', '')
        choices = format_choices(item)

        label = int(item.get('label', 1)) - 1
        answer = chr(65 + label)  # 0->A, 1->B, 2->C

        prompt = create_mcqa_prompt(context, question, choices)

        return {
            'data_source': 'siqa',
            'prompt': prompt,
            'ability': 'social_reasoning',
            'reward_model': {'ground_truth': answer},
            'extra_info': {
                'answer': answer,
                'label_idx': label,
                'context': context,
                'question': question,
                'answerA': item.get('answerA', ''),
                'answerB': item.get('answerB', ''),
                'answerC': item.get('answerC', ''),
                'index': idx,
                'split': split
            }
        }

    train_items = list(dataset['train'])
    np.random.seed(42)
    np.random.shuffle(train_items)

    train_data = [process_item(item, i, 'train') for i, item in enumerate(train_items[:num_train])]

    valid_items = list(dataset['validation'])
    np.random.seed(42)
    np.random.shuffle(valid_items)

    valid_data = [process_item(item, i, 'valid') for i, item in enumerate(valid_items[:num_valid])]
    test_data = [process_item(item, i, 'test') for i, item in enumerate(valid_items[num_valid:num_valid+num_test])]

    siqa_dir = f"{output_dir}/siqa"
    os.makedirs(siqa_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{siqa_dir}/train.parquet")
    valid_df.to_parquet(f"{siqa_dir}/valid.parquet")
    test_df.to_parquet(f"{siqa_dir}/test.parquet")

    print(f"  SIQA: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df


def prepare_piqa_dataset(output_dir: str, num_train: int = 5000, num_valid: int = 1000, num_test: int = 1000):
    """
    Prepare PIQA dataset (Physical Interaction QA - physical commonsense reasoning).

    PIQA tests the ability of models to reason about physical interactions
    in the real world. Binary choice format (2 options).

    Source: ybisk/piqa (HuggingFace - original source, parquet format)
    """
    print("Loading PIQA dataset...")

    try:
        # Try ybisk/piqa (parquet format, no remote code needed)
        dataset = load_dataset("ybisk/piqa")
    except Exception as e:
        print(f"  ybisk/piqa failed: {e}")
        try:
            # Try parquet branch
            dataset = load_dataset("piqa", revision="refs/convert/parquet")
        except Exception as e2:
            print(f"  Error loading PIQA: {e2}")
            return None, None, None

    print(f"  Raw data: {len(dataset['train'])} train, {len(dataset['validation'])} validation")

    def format_choices(item) -> str:
        """Format choices with letters."""
        choices = [
            item.get('sol1', ''),
            item.get('sol2', '')
        ]
        return "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices) if c])

    def process_item(item, idx, split):
        goal = item.get('goal', '')
        choices = format_choices(item)

        label = int(item.get('label', 0))
        answer = chr(65 + label)  # 0->A, 1->B

        prompt = create_mcqa_prompt("", goal, choices)  # No separate context for PIQA

        return {
            'data_source': 'piqa',
            'prompt': prompt,
            'ability': 'physical_reasoning',
            'reward_model': {'ground_truth': answer},
            'extra_info': {
                'answer': answer,
                'label_idx': label,
                'goal': goal,
                'sol1': item.get('sol1', ''),
                'sol2': item.get('sol2', ''),
                'index': idx,
                'split': split
            }
        }

    train_items = list(dataset['train'])
    np.random.seed(42)
    np.random.shuffle(train_items)

    train_data = [process_item(item, i, 'train') for i, item in enumerate(train_items[:num_train])]

    valid_items = list(dataset['validation'])
    np.random.seed(42)
    np.random.shuffle(valid_items)

    valid_data = [process_item(item, i, 'valid') for i, item in enumerate(valid_items[:num_valid])]
    test_data = [process_item(item, i, 'test') for i, item in enumerate(valid_items[num_valid:num_valid+num_test])]

    piqa_dir = f"{output_dir}/piqa"
    os.makedirs(piqa_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{piqa_dir}/train.parquet")
    valid_df.to_parquet(f"{piqa_dir}/valid.parquet")
    test_df.to_parquet(f"{piqa_dir}/test.parquet")

    print(f"  PIQA: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Project root (relative to this script: scripts/Step2_original_data_download/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="Download generalization test datasets")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "generalization"),
                        help="Output directory for generalization datasets")
    parser.add_argument("--datasets", type=str, nargs='+',
                        default=['numina_math', 'siqa', 'piqa'],
                        help="Datasets to download")
    parser.add_argument("--num_samples", type=int, default=2000,
                        help="Max samples per dataset (for NuminaMath and APPS)")

    args = parser.parse_args()

    print("=" * 70)
    print("Downloading Generalization Test Datasets")
    print("=" * 70)
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets: {args.datasets}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    for ds_name in args.datasets:
        try:
            print(f"\n{'='*50}")
            print(f"Processing: {ds_name}")
            print('='*50)

            if ds_name == 'numina_math':
                prepare_numina_math_dataset(args.output_dir, num_samples=args.num_samples)
            elif ds_name == 'siqa':
                prepare_siqa_dataset(args.output_dir)
            elif ds_name == 'piqa':
                prepare_piqa_dataset(args.output_dir)
            else:
                print(f"Unknown dataset: {ds_name}")
                print(f"Available: numina_math, siqa, piqa")

        except Exception as e:
            print(f"Error processing {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(" Generalization datasets download complete!")
    print("=" * 70)
