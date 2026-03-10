#!/usr/bin/env python3
'''
Download 8 standard datasets from HuggingFace in VERL-compatible format.
Covers GSM8K, GSM-Symbolic, HumanEval+, MBPP+, OBQA, MMLU, ARC-C, CommonsenseQA.
Related: download_math.py for MATH, download_gpqa.py for GPQA.
'''

import os
import json
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import argparse

# =============================================================================
# Prompt Templates
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


def create_gsm8k_prompt(question: str) -> list:
    """Create prompt for GSM8K problems (grade school math)."""
    return [
        UNIFIED_SYSTEM_PROMPT,
        {
            "role": "user",
            "content": (
                f"Solve this math word problem:\n\n"
                f"{question}\n\n"
                "Think step by step, showing your calculations.\n\n"
                "Write your final numerical answer as:\n"
                "#### [number]"
            )
        }
    ]


def create_code_prompt(task: str, test_cases: list = None) -> list:
    """Create prompt for coding problems with strong #### format enforcement."""
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




def create_open_qa_prompt(question: str, context: str = None) -> list:
    """Create prompt for open-ended QA (extractive or generative)."""
    if context:
        return [
            UNIFIED_SYSTEM_PROMPT,
            {
                "role": "user",
                "content": (
                    f"Read the following passage and answer the question.\n\n"
                    f"Passage: {context}\n\n"
                    f"Question: {question}\n\n"
                    "Provide a concise answer based on the passage.\n\n"
                    "Write your final answer as:\n"
                    "#### [answer]"
                )
            }
        ]
    else:
        return [
            UNIFIED_SYSTEM_PROMPT,
            {
                "role": "user",
                "content": (
                    f"Answer this question:\n\n"
                    f"{question}\n\n"
                    "Think through your answer carefully.\n\n"
                    "Write your final answer as:\n"
                    "#### [answer]"
                )
            }
        ]




# =============================================================================
# Dataset Preparation Functions
# =============================================================================

def prepare_gsm8k_dataset(output_dir: str, num_train: int = 7000, num_valid: int = 473, num_test: int = 1319):
    '''Prepare GSM8K dataset (grade school math word problems).'''
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")

    print(f"  Raw dataset: {len(dataset['train'])} train, {len(dataset['test'])} test")

    def extract_answer(answer_str: str) -> str:
        """Extract numerical answer from GSM8K format."""
        # GSM8K answers end with #### [number]
        if '####' in answer_str:
            return answer_str.split('####')[-1].strip()
        return answer_str.strip()

    def process_item(item, idx, split):
        question = item['question']
        answer_full = item['answer']
        answer = extract_answer(answer_full)
        prompt = create_gsm8k_prompt(question)

        return {
            'data_source': 'gsm8k',
            'prompt': prompt,
            'ability': 'math',
            'reward_model': {'ground_truth': answer},
            'extra_info': {
                'answer': answer,
                'full_answer': answer_full,
                'index': idx,
                'split': split
            }
        }

    train_data = []
    valid_data = []
    test_data = []

    train_items = list(dataset['train'])
    np.random.seed(42)
    np.random.shuffle(train_items)

    for idx, item in enumerate(train_items[:num_train]):
        train_data.append(process_item(item, idx, 'train'))

    for idx, item in enumerate(train_items[num_train:num_train + num_valid]):
        valid_data.append(process_item(item, idx, 'valid'))

    for idx, item in enumerate(dataset['test']):
        if idx >= num_test:
            break
        test_data.append(process_item(item, idx, 'test'))

    gsm8k_dir = f"{output_dir}/gsm8k"
    os.makedirs(gsm8k_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{gsm8k_dir}/train.parquet")
    valid_df.to_parquet(f"{gsm8k_dir}/valid.parquet")
    test_df.to_parquet(f"{gsm8k_dir}/test.parquet")

    # 10% subsets
    train_10 = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    train_10.to_parquet(f"{gsm8k_dir}/train_10_perc.parquet")

    print(f"✓ GSM8K: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df


def prepare_gsm_symbolic_dataset(output_dir: str, variant: str = "main",
                                  num_train: int = 3500, num_valid: int = 500, num_test: int = 1000):
    """
    Prepare GSM-Symbolic dataset (Apple's symbolic variant of GSM8K).

    GSM-Symbolic tests LLM robustness by:
    - Varying numerical values in problems
    - Adding extra clauses (P1, P2 variants)

    Args:
        output_dir: Output directory
        variant: "main", "p1" (1 extra clause), or "p2" (2 extra clauses)
        num_train: Number of training samples
        num_valid: Number of validation samples
        num_test: Number of test samples

    Dataset source: apple/GSM-Symbolic (HuggingFace)
    Paper: https://arxiv.org/abs/2410.05229
    """
    print(f"Loading GSM-Symbolic dataset (variant: {variant})...")

    try:
        dataset = load_dataset("apple/GSM-Symbolic", name=variant)
    except Exception as e:
        print(f"  Error loading variant '{variant}': {e}")
        print("  Trying default 'main' variant...")
        dataset = load_dataset("apple/GSM-Symbolic", name="main")
        variant = "main"

    # GSM-Symbolic only has 'test' split
    all_data = list(dataset['test'])
    print(f"  Raw dataset: {len(all_data)} samples (variant: {variant})")

    def extract_answer(answer_str: str) -> str:
        """Extract numerical answer from GSM format (#### answer)."""
        if '####' in answer_str:
            return answer_str.split('####')[-1].strip()
        return answer_str.strip()

    def process_item(item, idx, split):
        question = item['question']
        answer_full = item['answer']
        answer = extract_answer(answer_full)
        prompt = create_gsm8k_prompt(question)  # Reuse GSM8K prompt format

        return {
            'data_source': f'gsm_symbolic_{variant}',
            'prompt': prompt,
            'ability': 'math',
            'reward_model': {'ground_truth': answer},
            'extra_info': {
                'answer': answer,
                'full_answer': answer_full,
                'template_id': item.get('id', idx),
                'instance_id': item.get('instance', 0),
                'original_question': item.get('original_question', ''),
                'original_answer': item.get('original_answer', ''),
                'variant': variant,
                'index': idx,
                'split': split
            }
        }

    # Shuffle data
    np.random.seed(42)
    np.random.shuffle(all_data)

    # Split into train/valid/test
    # Note: GSM-Symbolic is primarily a test benchmark, but we create splits for training
    total_needed = num_train + num_valid + num_test
    if len(all_data) < total_needed:
        print(f"  Warning: Only {len(all_data)} samples available, adjusting splits...")
        ratio = len(all_data) / total_needed
        num_train = int(num_train * ratio)
        num_valid = int(num_valid * ratio)
        num_test = len(all_data) - num_train - num_valid

    train_data = [process_item(item, i, 'train') for i, item in enumerate(all_data[:num_train])]
    valid_data = [process_item(item, i, 'valid') for i, item in enumerate(all_data[num_train:num_train+num_valid])]
    test_data = [process_item(item, i, 'test') for i, item in enumerate(all_data[num_train+num_valid:num_train+num_valid+num_test])]

    gsm_sym_dir = f"{output_dir}/gsm_symbolic"
    os.makedirs(gsm_sym_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    # Save with variant suffix if not main
    suffix = f"_{variant}" if variant != "main" else ""
    train_df.to_parquet(f"{gsm_sym_dir}/train{suffix}.parquet")
    valid_df.to_parquet(f"{gsm_sym_dir}/valid{suffix}.parquet")
    test_df.to_parquet(f"{gsm_sym_dir}/test{suffix}.parquet")

    # 10% subset
    if len(train_df) > 0:
        train_10 = train_df.sample(frac=0.1, random_state=42).reset_index(drop=True)
        train_10.to_parquet(f"{gsm_sym_dir}/train_10_perc{suffix}.parquet")

    print(f"✓ GSM-Symbolic ({variant}): {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df


def prepare_humaneval_plus_dataset(output_dir: str):
    """
    Prepare HumanEval dataset with proper test assertions.
    Uses original openai/humaneval which has executable test cases.
    """
    print("Loading original HumanEval dataset (with proper test assertions)...")

    # Force original HumanEval which has proper test code
    dataset = load_dataset("openai/openai_humaneval")
    all_data = list(dataset['test'])

    print(f"  Total samples: {len(all_data)}")

    def process_item(item, idx, split):
        import re
        task_id = item.get('task_id', f'HumanEval/{idx}')
        prompt_code = item['prompt']
        canonical_solution = item.get('canonical_solution', '')
        entry_point = item.get('entry_point', '')

        docstring_match = re.search(r'"""(.*?)"""', prompt_code, re.DOTALL)
        description = docstring_match.group(1).strip() if docstring_match else f"Complete the function {entry_point}"

        test_code = item.get('test', '')
        test_assertions = re.findall(r'assert\s+[^\n]+', test_code)
        # Replace 'candidate' with actual function name
        test_assertions = [t.replace('candidate', entry_point) for t in test_assertions]

        # Debug: show sample test
        if idx == 0:
            print(f"  Sample test assertions: {test_assertions[:2]}")

        prompt = create_code_prompt(
            f"{description}\n\nFunction signature:\n{prompt_code}",
            test_assertions[:3] if test_assertions else None
        )

        return {
            'data_source': 'humaneval_plus',
            'prompt': prompt,
            'ability': 'coding',
            'reward_model': {'ground_truth': prompt_code + canonical_solution},
            'extra_info': {
                'answer': prompt_code + canonical_solution,
                'test_list': test_assertions,
                'task_id': task_id,
                'entry_point': entry_point,
                'index': idx,
                'split': split
            }
        }

    # Split: 130 train, 17 valid, 17 test
    indices = list(range(len(all_data)))
    np.random.seed(42)
    np.random.shuffle(indices)

    train_data = [process_item(all_data[i], i, 'train') for i in indices[:130]]
    valid_data = [process_item(all_data[i], i, 'valid') for i in indices[130:147]]
    test_data = [process_item(all_data[i], i, 'test') for i in indices[147:164]]

    he_dir = f"{output_dir}/humaneval_plus"
    os.makedirs(he_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{he_dir}/train.parquet")
    valid_df.to_parquet(f"{he_dir}/valid.parquet")
    test_df.to_parquet(f"{he_dir}/test.parquet")

    print(f"✓ HumanEval+: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df


def prepare_mbpp_plus_dataset(output_dir: str):
    """
    Prepare MBPP+ from EvalPlus (enhanced test cases).
    """
    print("Loading MBPP+ dataset...")

    try:
        dataset = load_dataset("evalplus/mbppplus")
        all_data = list(dataset['test'])
    except Exception as e:
        print(f"  EvalPlus version failed ({e}), trying original MBPP...")
        dataset = load_dataset("google-research-datasets/mbpp", "full")
        train_data_raw = list(dataset['train'])
        valid_data_raw = list(dataset['validation'])
        test_data_raw = list(dataset['test'])
        all_data = train_data_raw + valid_data_raw + test_data_raw

    print(f"  Total samples: {len(all_data)}")

    def process_item(item, idx, split):
        task = item.get('text', item.get('prompt', ''))
        code = item.get('code', item.get('canonical_solution', ''))
        test_cases = item.get('test_list', [])

        prompt = create_code_prompt(task, test_cases)

        return {
            'data_source': 'mbpp_plus',
            'prompt': prompt,
            'ability': 'coding',
            'reward_model': {'ground_truth': code},
            'extra_info': {
                'answer': code,
                'test_list': test_cases,
                'task_id': str(item.get('task_id', idx)),
                'index': idx,
                'split': split
            }
        }

    # Split data
    indices = list(range(len(all_data)))
    np.random.seed(42)
    np.random.shuffle(indices)

    n_train = int(len(all_data) * 0.7)
    n_valid = int(len(all_data) * 0.15)

    train_data = [process_item(all_data[i], i, 'train') for i in indices[:n_train]]
    valid_data = [process_item(all_data[i], i, 'valid') for i in indices[n_train:n_train+n_valid]]
    test_data = [process_item(all_data[i], i, 'test') for i in indices[n_train+n_valid:]]

    mbpp_dir = f"{output_dir}/mbpp_plus"
    os.makedirs(mbpp_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{mbpp_dir}/train.parquet")
    valid_df.to_parquet(f"{mbpp_dir}/valid.parquet")
    test_df.to_parquet(f"{mbpp_dir}/test.parquet")

    print(f"✓ MBPP+: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df


def prepare_obqa_dataset(output_dir: str):
    """
    Prepare OpenBookQA dataset (science QA with open book).
    """
    print("Loading OpenBookQA dataset...")
    dataset = load_dataset("allenai/openbookqa", "main")

    print(f"  Raw: {len(dataset['train'])} train, {len(dataset['validation'])} valid, {len(dataset['test'])} test")

    def format_choices(item):
        labels = item['choices']['label']
        texts = item['choices']['text']
        return "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])

    def process_item(item, idx, split):
        question = item['question_stem']
        choices = format_choices(item)
        answer = item['answerKey']

        prompt = create_qa_prompt(question, choices)

        return {
            'data_source': 'obqa',
            'prompt': prompt,
            'ability': 'qa',
            'reward_model': {'ground_truth': answer},
            'extra_info': {
                'answer': answer,
                'question': question,
                'choices': item['choices'],
                'index': idx,
                'split': split
            }
        }

    train_data = [process_item(item, i, 'train') for i, item in enumerate(dataset['train'])]
    valid_data = [process_item(item, i, 'valid') for i, item in enumerate(dataset['validation'])]
    test_data = [process_item(item, i, 'test') for i, item in enumerate(dataset['test'])]

    obqa_dir = f"{output_dir}/obqa"
    os.makedirs(obqa_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{obqa_dir}/train.parquet")
    valid_df.to_parquet(f"{obqa_dir}/valid.parquet")
    test_df.to_parquet(f"{obqa_dir}/test.parquet")

    print(f"✓ OBQA: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df




def prepare_mmlu_dataset(output_dir: str, num_train: int = 5000, num_valid: int = 500, num_test: int = 1000):
    """
    Prepare MMLU dataset (multi-task language understanding).
    Uses a subset of subjects for efficiency.
    """
    print("Loading MMLU dataset...")

    # Select diverse subjects
    subjects = [
        'abstract_algebra', 'college_mathematics', 'elementary_mathematics',
        'high_school_physics', 'high_school_chemistry', 'high_school_chemistry',
        'computer_science', 'machine_learning',
        'world_history', 'us_history',
        'logical_fallacies', 'formal_logic'
    ]

    all_train = []
    all_test = []

    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject)
            all_train.extend([(item, subject) for item in ds['test']])  # MMLU 'test' is actually dev
            all_test.extend([(item, subject) for item in ds['validation']])
        except Exception as e:
            print(f"  Warning: Could not load {subject}: {e}")

    print(f"  Loaded {len(all_train)} train, {len(all_test)} test from {len(subjects)} subjects")

    def format_choices(item):
        choices = item['choices']
        return "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])

    def process_item(item_tuple, idx, split):
        item, subject = item_tuple
        question = item['question']
        choices = format_choices(item)
        answer = chr(65 + item['answer'])  # Convert 0->A, etc.

        prompt = create_qa_prompt(question, choices)

        return {
            'data_source': 'mmlu',
            'prompt': prompt,
            'ability': 'knowledge',
            'reward_model': {'ground_truth': answer},
            'extra_info': {
                'answer': answer,
                'question': question,
                'subject': subject,
                'index': idx,
                'split': split
            }
        }

    np.random.seed(42)
    np.random.shuffle(all_train)
    np.random.shuffle(all_test)

    train_data = [process_item(item, i, 'train') for i, item in enumerate(all_train[:num_train])]
    valid_data = [process_item(item, i, 'valid') for i, item in enumerate(all_train[num_train:num_train+num_valid])]
    test_data = [process_item(item, i, 'test') for i, item in enumerate(all_test[:num_test])]

    mmlu_dir = f"{output_dir}/mmlu"
    os.makedirs(mmlu_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{mmlu_dir}/train.parquet")
    valid_df.to_parquet(f"{mmlu_dir}/valid.parquet")
    test_df.to_parquet(f"{mmlu_dir}/test.parquet")

    print(f"✓ MMLU: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df


def prepare_arc_c_dataset(output_dir: str):
    """
    Prepare ARC-Challenge dataset (harder science questions).
    """
    print("Loading ARC-Challenge dataset...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")

    print(f"  Raw: {len(dataset['train'])} train, {len(dataset['validation'])} valid, {len(dataset['test'])} test")

    def format_choices(item):
        labels = item['choices']['label']
        texts = item['choices']['text']
        return "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])

    def process_item(item, idx, split):
        question = item['question']
        choices = format_choices(item)
        answer = item['answerKey']

        prompt = create_qa_prompt(question, choices)

        return {
            'data_source': 'arc_c',
            'prompt': prompt,
            'ability': 'science',
            'reward_model': {'ground_truth': answer},
            'extra_info': {
                'answer': answer,
                'question': question,
                'choices': item['choices'],
                'index': idx,
                'split': split
            }
        }

    train_data = [process_item(item, i, 'train') for i, item in enumerate(dataset['train'])]
    valid_data = [process_item(item, i, 'valid') for i, item in enumerate(dataset['validation'])]
    test_data = [process_item(item, i, 'test') for i, item in enumerate(dataset['test'])]

    arc_dir = f"{output_dir}/arc_c"
    os.makedirs(arc_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{arc_dir}/train.parquet")
    valid_df.to_parquet(f"{arc_dir}/valid.parquet")
    test_df.to_parquet(f"{arc_dir}/test.parquet")

    print(f"✓ ARC-C: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df


def prepare_gpqa_dataset(output_dir: str):
    """
    Prepare GPQA dataset (Graduate-level science QA).
    """
    print("Loading GPQA dataset...")

    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")
        all_data = list(dataset['train'])
        print(f"  Loaded GPQA Diamond: {len(all_data)} samples")
    except Exception as e:
        print(f"  GPQA Diamond failed ({e}), trying main...")
        try:
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")
            all_data = list(dataset['train'])
            print(f"  Loaded GPQA Main: {len(all_data)} samples")
        except Exception as e2:
            print(f"  GPQA Main failed ({e2}), trying extended...")
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_extended")
            all_data = list(dataset['train'])
            print(f"  Loaded GPQA Extended: {len(all_data)} samples")

    def process_item(item, idx, split):
        question = item.get('Question', item.get('question', ''))
        choices_list = []
        answer_key = None
        correct_answer = item.get('Correct Answer', item.get('correct_answer', ''))

        for key in ['A', 'B', 'C', 'D']:
            choice_key = f'Choice {key}' if f'Choice {key}' in item else key.lower()
            if choice_key in item:
                choice_text = item[choice_key]
                choices_list.append(f"{key}. {choice_text}")
                if choice_text == correct_answer:
                    answer_key = key

        if not answer_key:
            answer_key = item.get('Answer', item.get('answer', 'A'))

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
                'index': idx,
                'split': split
            }
        }

    indices = list(range(len(all_data)))
    np.random.seed(42)
    np.random.shuffle(indices)

    n_train = int(len(all_data) * 0.7)
    n_valid = int(len(all_data) * 0.15)

    train_data = [process_item(all_data[i], i, 'train') for i in indices[:n_train]]
    valid_data = [process_item(all_data[i], i, 'valid') for i in indices[n_train:n_train+n_valid]]
    test_data = [process_item(all_data[i], i, 'test') for i in indices[n_train+n_valid:]]

    gpqa_dir = f"{output_dir}/gpqa"
    os.makedirs(gpqa_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{gpqa_dir}/train.parquet")
    valid_df.to_parquet(f"{gpqa_dir}/valid.parquet")
    test_df.to_parquet(f"{gpqa_dir}/test.parquet")

    print(f"✓ GPQA: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df


def prepare_commonsenseqa_dataset(output_dir: str):
    """
    Prepare CommonsenseQA dataset.
    """
    print("Loading CommonsenseQA dataset...")
    dataset = load_dataset("tau/commonsense_qa")

    print(f"  Raw: {len(dataset['train'])} train, {len(dataset['validation'])} valid")

    def format_choices(item):
        labels = item['choices']['label']
        texts = item['choices']['text']
        return "\n".join([f"{l}. {t}" for l, t in zip(labels, texts)])

    def process_item(item, idx, split):
        question = item['question']
        choices = format_choices(item)
        answer = item['answerKey']

        prompt = create_qa_prompt(question, choices)

        return {
            'data_source': 'commonsenseqa',
            'prompt': prompt,
            'ability': 'commonsense',
            'reward_model': {'ground_truth': answer},
            'extra_info': {
                'answer': answer,
                'question': question,
                'choices': item['choices'],
                'index': idx,
                'split': split
            }
        }

    train_data = [process_item(item, i, 'train') for i, item in enumerate(dataset['train'])]
    valid_data = [process_item(item, i, 'valid') for i, item in enumerate(dataset['validation'])]

    np.random.seed(42)
    valid_items = list(dataset['validation'])
    np.random.shuffle(valid_items)
    test_data = [process_item(item, i, 'test') for i, item in enumerate(valid_items[:500])]

    cqa_dir = f"{output_dir}/commonsenseqa"
    os.makedirs(cqa_dir, exist_ok=True)

    train_df = pd.DataFrame(train_data)
    valid_df = pd.DataFrame(valid_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_parquet(f"{cqa_dir}/train.parquet")
    valid_df.to_parquet(f"{cqa_dir}/valid.parquet")
    test_df.to_parquet(f"{cqa_dir}/test.parquet")

    print(f"✓ CommonsenseQA: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
    return train_df, valid_df, test_df










if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description="Download datasets for training")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "data"),
                        help="Output directory")
    parser.add_argument("--datasets", type=str, nargs='+',
                        default=['gsm8k', 'gsm_symbolic', 'humaneval_plus', 'mbpp_plus', 'obqa', 'mmlu',
                                 'arc_c', 'commonsenseqa'],
                        help="Datasets to download (GPQA uses separate script)")
    parser.add_argument("--gsm_symbolic_variant", type=str, default="main",
                        choices=["main", "p1", "p2"],
                        help="GSM-Symbolic variant: main, p1 (1 extra clause), p2 (2 extra clauses)")

    args = parser.parse_args()

    print(f"Downloading datasets to: {args.output_dir}")
    print(f"Datasets: {args.datasets}")
    print()

    for ds_name in args.datasets:
        try:
            if ds_name == 'gsm8k':
                prepare_gsm8k_dataset(args.output_dir)
            elif ds_name == 'gsm_symbolic':
                prepare_gsm_symbolic_dataset(args.output_dir, variant=args.gsm_symbolic_variant)
            elif ds_name == 'humaneval_plus':
                prepare_humaneval_plus_dataset(args.output_dir)
            elif ds_name == 'mbpp_plus':
                prepare_mbpp_plus_dataset(args.output_dir)
            elif ds_name == 'obqa':
                prepare_obqa_dataset(args.output_dir)
            elif ds_name == 'mmlu':
                prepare_mmlu_dataset(args.output_dir)
            elif ds_name == 'arc_c':
                prepare_arc_c_dataset(args.output_dir)
            elif ds_name == 'commonsenseqa':
                prepare_commonsenseqa_dataset(args.output_dir)
            else:
                print(f"Unknown dataset: {ds_name}")
                print(f"Available datasets: gsm8k, gsm_symbolic, humaneval_plus, mbpp_plus, obqa, mmlu, arc_c, commonsenseqa")
        except Exception as e:
            print(f"Error downloading {ds_name}: {e}")
            import traceback
            traceback.print_exc()
        print()

    print("\n✓ All datasets downloaded!")
