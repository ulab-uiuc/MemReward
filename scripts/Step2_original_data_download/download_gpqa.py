#!/usr/bin/env python3
'''
Download GPQA (Graduate-level QA) dataset via authenticated HuggingFace access.
Shuffles choices per item for unbiased evaluation.
Related: download_datasets.py for other QA datasets.
'''

import os
import random
import pandas as pd
import numpy as np
from datasets import load_dataset
from huggingface_hub import login
from pathlib import Path

# Login with token
HF_TOKEN = os.environ.get("HF_TOKEN", "")
login(token=HF_TOKEN)

# Project root (relative to this script: scripts/Step2_original_data_download/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Output directory
OUTPUT_DIR = str(PROJECT_ROOT / "data")

# System prompt (same as other datasets)
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


def create_qa_prompt(question: str, choices: str) -> list:
    """Create prompt for multiple choice QA."""
    return [
        UNIFIED_SYSTEM_PROMPT,
        {
            "role": "user",
            "content": (
                f"Answer this question:\n\n"
                f"Question: {question}\n\n"
                f"{choices}\n\n"
                "Analyze each option and explain your reasoning.\n\n"
                "Write your final answer as:\n"
                "#### [letter]"
            )
        }
    ]


def prepare_gpqa_dataset():
    """Download and prepare GPQA dataset."""
    print("Loading GPQA dataset with authentication...")

    # Try different configs
    dataset = None
    config_name = None

    for config in ["gpqa_diamond", "gpqa_main", "gpqa_extended"]:
        try:
            print(f"  Trying {config}...")
            dataset = load_dataset("Idavidrein/gpqa", config, token=HF_TOKEN)
            config_name = config
            print(f"  ✓ Successfully loaded {config}")
            break
        except Exception as e:
            print(f"  ✗ {config} failed: {e}")

    if dataset is None:
        raise RuntimeError("Failed to load any GPQA config")

    all_data = list(dataset['train'])
    print(f"  Total samples: {len(all_data)}")

    def process_item(item, idx, split):
        question = item.get('Question', '')
        correct_answer = item.get('Correct Answer', '')
        incorrect_1 = item.get('Incorrect Answer 1', '')
        incorrect_2 = item.get('Incorrect Answer 2', '')
        incorrect_3 = item.get('Incorrect Answer 3', '')

        all_answers = [
            ('correct', correct_answer),
            ('incorrect', incorrect_1),
            ('incorrect', incorrect_2),
            ('incorrect', incorrect_3)
        ]
        all_answers = [(t, a) for t, a in all_answers if a]

        # Shuffle with fixed seed for reproducibility per item
        random.seed(idx)
        random.shuffle(all_answers)

        choices_list = []
        answer_key = None
        for i, (ans_type, ans_text) in enumerate(all_answers):
            letter = chr(65 + i)  # A, B, C, D
            choices_list.append(f"{letter}. {ans_text}")
            if ans_type == 'correct':
                answer_key = letter

        choices_str = "\n".join(choices_list)
        prompt = create_qa_prompt(question, choices_str)

        return {
            'data_source': 'gpqa',
            'prompt': prompt,
            'ability': 'science',
            'reward_model': {'ground_truth': answer_key},
            'extra_info': {
                'answer': answer_key,
                'question': question,
                'correct_answer_text': correct_answer,
                'subdomain': item.get('Subdomain', ''),
                'high_level_domain': item.get('High-level domain', ''),
                'config': config_name,
                'index': idx,
                'split': split
            }
        }

    # Shuffle and split
    indices = list(range(len(all_data)))
    np.random.seed(42)
    np.random.shuffle(indices)

    n_train = int(len(all_data) * 0.7)
    n_valid = int(len(all_data) * 0.15)

    train_data = [process_item(all_data[i], i, 'train') for i in indices[:n_train]]
    valid_data = [process_item(all_data[i], i, 'valid') for i in indices[n_train:n_train+n_valid]]
    test_data = [process_item(all_data[i], i, 'test') for i in indices[n_train+n_valid:]]

    gpqa_dir = f"{OUTPUT_DIR}/gpqa"
    os.makedirs(gpqa_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{gpqa_dir}/train.parquet")
    valid_df.to_parquet(f"{gpqa_dir}/valid.parquet")
    test_df.to_parquet(f"{gpqa_dir}/test.parquet")

    print(f"\n✓ GPQA saved to {gpqa_dir}")
    print(f"  Train: {len(train_data)}")
    print(f"  Valid: {len(valid_data)}")
    print(f"  Test: {len(test_data)}")

    # Show sample
    print("\n  Sample question:")
    sample = train_data[0]
    print(f"    Answer key: {sample['reward_model']['ground_truth']}")
    print(f"    Subdomain: {sample['extra_info']['subdomain']}")

    return train_df, valid_df, test_df


if __name__ == "__main__":
    prepare_gpqa_dataset()
