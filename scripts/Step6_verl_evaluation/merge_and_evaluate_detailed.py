#!/usr/bin/env python
'''
Merge FSDP checkpoints and evaluate on standard (10 benchmarks) or generalization datasets.
Saves detailed per-sample results (prompts, responses, token counts, correctness) in JSONL.
Related: find_best_checkpoint.py for auto-selecting best step, evaluate_standard_models.py for lightweight eval.
'''

import os
os.environ.setdefault('VLLM_USE_V1', '0')

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jsonlines
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

from utils.evaluate_standard_models import (
    extract_answer,
    evaluate_response,
    NumpyEncoder
)
from utils.find_best_checkpoint import find_best_checkpoint_dir


def generalization_extract_answer(response: str, data_source: str) -> Optional[str]:
    """Extract answer based on data source for generalization datasets."""
    ds = data_source.lower()

    if 'numina' in ds or 'math' in ds:
        if '####' in response:
            match = re.search(r'####\s*([^#\n][^\n]*?)(?:\s*####|\s*$)', response)
            if match:
                answer = match.group(1).strip().rstrip('.')
                if answer:
                    return answer
        boxed_match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', response)
        if boxed_match:
            return boxed_match.group(1).strip()
        boxed_match = re.search(r'\\boxed\{(.+?)\}', response)
        if boxed_match:
            return boxed_match.group(1).strip()
        return None

    elif 'piqa' in ds or 'siqa' in ds:
        response_upper = response.upper()

        match = re.search(r'####\s*([A-C])\b', response_upper)
        if match:
            return match.group(1)

        match = re.search(r'(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+IS\s*:?\s*([A-C])\b', response_upper)
        if match:
            return match.group(1)

        match = re.search(r'\b([A-B])\b\s*$', response_upper.strip())
        if match:
            return match.group(1)

        return None

    return None


def normalize_math_answer(ans: str) -> str:
    """Normalize mathematical answer for comparison."""
    if not ans:
        return ""

    ans = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]+)\}', r'\1', ans)
    ans = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', ans)

    ans = ans.replace('\\', '')

    ans = ans.replace(' ', '').lower()

    ans = re.sub(r'(?:dollars?|cents?|\$|%|degrees?|°)', '', ans)

    if '/' in ans:
        try:
            parts = ans.split('/')
            if len(parts) == 2:
                numerator = float(parts[0].strip('()'))
                denominator = float(parts[1].strip('()'))
                if denominator != 0:
                    result = numerator / denominator
                    return str(int(result) if result.is_integer() else result)
        except:
            pass

    number_match = re.search(r'-?\d+(?:\.\d+)?', ans)
    if number_match:
        num = float(number_match.group())
        return str(int(num) if num.is_integer() else num)

    return ans


def generalization_evaluate_response(response: str, data_source: str, extra_info: dict) -> bool:
    """Evaluate a single response for generalization datasets."""
    answer = generalization_extract_answer(response, data_source)
    ds = data_source.lower()

    if 'numina' in ds or 'math' in ds:
        if answer is None:
            return False

        gt = str(extra_info.get('answer', ''))
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', gt)
        if boxed_match:
            gt = boxed_match.group(1)

        pred_norm = normalize_math_answer(answer)
        gt_norm = normalize_math_answer(gt)
        return pred_norm == gt_norm

    elif 'piqa' in ds or 'siqa' in ds:
        gt = str(extra_info.get('answer', '')).strip().upper()
        gt_letter = re.search(r'[A-C]', gt)
        if not gt_letter or not answer:
            return False
        return answer == gt_letter.group()

    return False


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object potentially containing numpy types

    Returns:
        Object with all numpy types converted to Python native types
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def merge_checkpoint(checkpoint_dir: str) -> str:
    """
    Merge FSDP checkpoint shards into HuggingFace format.

    Args:
        checkpoint_dir: Path to checkpoint directory (containing actor/ folder)

    Returns:
        Path to merged model directory
    """
    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    checkpoint_path = Path(checkpoint_dir)
    actor_dir = checkpoint_path / "actor"
    output_dir = checkpoint_path / "merged_hf_model"

    if not actor_dir.exists():
        raise ValueError(f"Actor directory not found: {actor_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MERGING FSDP CHECKPOINT")
    print(f"{'='*60}")
    print(f"Input:  {actor_dir}")
    print(f"Output: {output_dir}")

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(actor_dir),
        target_dir=str(output_dir),
        hf_model_config_path=str(actor_dir / "huggingface"),
        trust_remote_code=True,
    )

    merger = FSDPModelMerger(config)
    merger.merge_and_save()

    _ensure_chat_template(output_dir)

    print(f"\n✓ Successfully merged checkpoint to: {output_dir}")
    return str(output_dir)


def _ensure_chat_template(model_dir: Path):
    """Ensure tokenizer has chat_template field."""
    tokenizer_config_path = model_dir / "tokenizer_config.json"

    if not tokenizer_config_path.exists():
        print("\n⚠ Warning: tokenizer_config.json not found")
        return

    with open(tokenizer_config_path, 'r') as f:
        tokenizer_config = json.load(f)

    if 'chat_template' in tokenizer_config:
        print("✓ chat_template already exists in tokenizer_config.json")
        return

    print("\n→ Adding missing chat_template to tokenizer_config.json...")

    qwen_chat_template = (
        "{% for message in messages %}"
        "{% if loop.first and messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
        "{% endif %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )

    tokenizer_config['chat_template'] = qwen_chat_template

    with open(tokenizer_config_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    print("✓ chat_template added successfully!")


def evaluate_with_details(
    model_path: str,
    test_dir: str,
    checkpoint_name: str = "checkpoint",
    dataset_type: str = "standard"
) -> Tuple[List[Dict], Dict]:
    """
    Run evaluation and collect detailed information for each sample.

    Args:
        model_path: Path to merged HuggingFace model
        test_dir: Directory containing test parquet files
        checkpoint_name: Name of checkpoint for logging
        dataset_type: 'standard' (10 benchmarks) or 'generalization'

    Returns:
        Tuple of (all_details, summary):
            - all_details: List of dicts with per-sample information
            - summary: Dict with aggregate statistics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL WITH DETAILED OUTPUT")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Test data: {test_dir}")
    print(f"Dataset type: {dataset_type}")

    print(f"\n→ Loading model with vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.4,
        trust_remote_code=True,
        enforce_eager=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["<|endoftext|>", "<|im_end|>"]
    )

    test_path = Path(test_dir)
    test_files = sorted(test_path.glob("*_test.parquet"))

    if dataset_type == "generalization":
        gen_datasets = ['numina_math', 'piqa', 'siqa']
        test_files = [f for f in test_files if any(ds in f.stem.lower() for ds in gen_datasets)]
    else:
        exclude_patterns = ['pharma', 'tdc', 'dti', 'numina', 'piqa', 'siqa']
        test_files = [f for f in test_files if not any(p in f.stem.lower() for p in exclude_patterns)]

    if not test_files:
        raise ValueError(f"No {dataset_type} test files found in {test_dir}")

    print(f"✓ Found {len(test_files)} test datasets")

    all_details = []
    dataset_stats = {}

    for test_file in test_files:
        dataset = test_file.stem.replace("_sampled_test", "")
        df = pd.read_parquet(test_file)

        print(f"\n→ Processing {dataset}: {len(df)} samples")

        prompts = []
        for _, row in df.iterrows():
            prompt_data = row['prompt']

            if isinstance(prompt_data, np.ndarray):
                prompt_data = prompt_data.tolist()
            elif isinstance(prompt_data, str):
                prompt_data = eval(prompt_data)

            prompt_text = tokenizer.apply_chat_template(
                prompt_data,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt_text)

        outputs = llm.generate(prompts, sampling_params)

        correct_count = 0
        for idx, (output, (_, row)) in enumerate(zip(outputs, df.iterrows())):
            prompt_data = row['prompt']
            if isinstance(prompt_data, np.ndarray):
                prompt_data = prompt_data.tolist()
            elif isinstance(prompt_data, str):
                prompt_data = eval(prompt_data)

            question = ""
            for msg in prompt_data:
                if msg.get('role') == 'user':
                    question = msg['content']
                    break

            response = output.outputs[0].text

            extra_info = row['extra_info']
            if isinstance(extra_info, str):
                extra_info = eval(extra_info)

            if dataset_type == "generalization":
                extracted = generalization_extract_answer(response, row['data_source'])
                is_correct = generalization_evaluate_response(
                    response,
                    row['data_source'],
                    extra_info
                )
            else:
                extracted = extract_answer(response, row['data_source'])
                is_correct = evaluate_response(
                    response,
                    row['data_source'],
                    extra_info
                )

            if is_correct:
                correct_count += 1

            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)

            detail = {
                'dataset': dataset,
                'sample_idx': idx,
                'data_source': row['data_source'],
                'prompt_chat': convert_numpy_types(prompt_data),
                'prompt_text': prompts[idx],
                'question': question,
                'response_full': response,
                'extracted_answer': str(extracted) if extracted is not None else "",
                'ground_truth': str(extra_info.get('answer', '')),
                'correct': bool(is_correct),
                'prompt_tokens': int(prompt_tokens),
                'completion_tokens': int(completion_tokens),
                'total_tokens': int(prompt_tokens + completion_tokens),
                'extra_info': convert_numpy_types(extra_info),
                'ability': str(row.get('ability', '')),
            }
            all_details.append(detail)

        dataset_details = [d for d in all_details if d['dataset'] == dataset]
        dataset_prompt_tokens = [d['prompt_tokens'] for d in dataset_details]
        dataset_completion_tokens = [d['completion_tokens'] for d in dataset_details]
        dataset_total_tokens = [d['total_tokens'] for d in dataset_details]

        accuracy = round(100 * correct_count / len(df), 2)
        dataset_stats[dataset] = {
            'correct': correct_count,
            'total': len(df),
            'accuracy': accuracy,
            'avg_prompt_tokens': round(np.mean(dataset_prompt_tokens), 2),
            'avg_completion_tokens': round(np.mean(dataset_completion_tokens), 2),
            'avg_total_tokens': round(np.mean(dataset_total_tokens), 2),
        }

        print(f"  → {dataset}: {accuracy:.1f}% ({correct_count}/{len(df)}), "
              f"avg_tokens: {dataset_stats[dataset]['avg_total_tokens']:.1f}")

    total_correct = sum(s['correct'] for s in dataset_stats.values())
    total_samples = sum(s['total'] for s in dataset_stats.values())

    all_prompt_tokens = [d['prompt_tokens'] for d in all_details]
    all_completion_tokens = [d['completion_tokens'] for d in all_details]
    all_total_tokens = [d['total_tokens'] for d in all_details]

    summary = {
        'checkpoint': checkpoint_name,
        'dataset_type': dataset_type,
        'timestamp': datetime.now().isoformat(),
        'total_samples': total_samples,
        'overall': {
            'correct': total_correct,
            'total': total_samples,
            'accuracy': round(100 * total_correct / total_samples, 2) if total_samples > 0 else 0,
            'avg_prompt_tokens': round(np.mean(all_prompt_tokens), 2) if all_prompt_tokens else 0,
            'avg_completion_tokens': round(np.mean(all_completion_tokens), 2) if all_completion_tokens else 0,
            'avg_total_tokens': round(np.mean(all_total_tokens), 2) if all_total_tokens else 0,
        },
        'per_dataset': dataset_stats
    }

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {summary['overall']['accuracy']:.2f}%")
    print(f"Correct: {total_correct} / {total_samples}")
    print(f"Avg Tokens: {summary['overall']['avg_total_tokens']:.1f}")

    return all_details, summary


def save_detailed_results(
    all_details: List[Dict],
    summary: Dict,
    output_dir: str
):
    """
    Save evaluation results to JSONL and JSON formats.

    Args:
        all_details: List of per-sample detail dicts
        summary: Summary statistics dict
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SAVING DETAILED RESULTS")
    print(f"{'='*60}")
    print(f"Output directory: {output_path}")

    details_file = output_path / "detailed_results.jsonl"
    print(f"\n→ Saving all {len(all_details)} samples...")
    with jsonlines.open(details_file, mode='w') as writer:
        for detail in all_details:
            writer.write(detail)
    print(f"  ✓ Saved to: {details_file}")

    correct_samples = [d for d in all_details if d['correct']]
    correct_file = output_path / "correct_samples.jsonl"
    print(f"\n→ Saving {len(correct_samples)} correct samples...")
    with jsonlines.open(correct_file, mode='w') as writer:
        for detail in correct_samples:
            writer.write(detail)
    print(f"  ✓ Saved to: {correct_file}")

    summary_file = output_path / "summary.json"
    print(f"\n→ Saving summary statistics...")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    print(f"  ✓ Saved to: {summary_file}")

    stats_file = output_path / "statistics.txt"
    print(f"\n→ Saving human-readable statistics...")
    dataset_type = summary.get('dataset_type', 'standard')
    with open(stats_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"EVALUATION RESULTS SUMMARY ({dataset_type.upper()})\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint: {summary['checkpoint']}\n")
        f.write(f"Dataset Type: {dataset_type}\n")
        f.write(f"Timestamp: {summary['timestamp']}\n")
        f.write(f"Total Samples: {summary['total_samples']}\n\n")

        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"  Accuracy: {summary['overall']['accuracy']:.2f}%\n")
        f.write(f"  Correct: {summary['overall']['correct']} / {summary['overall']['total']}\n")
        f.write(f"  Avg Prompt Tokens: {summary['overall']['avg_prompt_tokens']:.1f}\n")
        f.write(f"  Avg Completion Tokens: {summary['overall']['avg_completion_tokens']:.1f}\n")
        f.write(f"  Avg Total Tokens: {summary['overall']['avg_total_tokens']:.1f}\n\n")

        f.write("PER-DATASET BREAKDOWN:\n")
        f.write(f"  {'Dataset':20s} {'Acc':>7s} {'Correct':>10s} {'Avg Tokens':>12s}\n")
        f.write(f"  {'-'*20} {'-'*7} {'-'*10} {'-'*12}\n")
        for dataset, stats in sorted(summary['per_dataset'].items()):
            avg_tokens = stats.get('avg_total_tokens', 'N/A')
            avg_tokens_str = f"{avg_tokens:.1f}" if isinstance(avg_tokens, (int, float)) else avg_tokens
            f.write(f"  {dataset:20s} {stats['accuracy']:6.1f}% "
                   f"({stats['correct']:3d}/{stats['total']:3d}) "
                   f"{avg_tokens_str:>12s}\n")
    print(f"  ✓ Saved to: {stats_file}")

    print(f"\n{'='*60}")
    print(f"✓ All results saved successfully!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  - detailed_results.jsonl: {len(all_details)} samples")
    print(f"  - correct_samples.jsonl: {len(correct_samples)} samples")
    print(f"  - summary.json: Statistics")
    print(f"  - statistics.txt: Human-readable summary")


def main():
    parser = argparse.ArgumentParser(
        description="Merge FSDP checkpoint and evaluate with detailed output"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Path to FSDP checkpoint (e.g., outputs/.../checkpoint)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <checkpoint_dir>/evaluation_results)"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=None,
        help="Directory with test parquet files (default: depends on dataset_type)"
    )
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="Only merge checkpoint, skip evaluation"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only evaluate (requires --merged_model_path)"
    )
    parser.add_argument(
        "--merged_model_path",
        type=str,
        help="Path to already-merged model (for --eval_only)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID (sets CUDA_VISIBLE_DEVICES)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Experiment name for logging"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="standard",
        choices=["standard", "generalization"],
        help="Dataset type: 'standard' (10 benchmarks) or 'generalization' (numina_math, piqa, siqa)"
    )
    parser.add_argument(
        "--find_best",
        type=str,
        default=None,
        metavar="TRAINING_DIR",
        help="Find best checkpoint from training log, then merge+evaluate it. "
             "Pass the training output dir (e.g., outputs/.../verl_grpo_100perc_gt)"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Explicit path to training log file (used with --find_best). "
             "If not set, auto-discovers from training_dir/training.log or PROJECT_ROOT/logs/"
    )

    args = parser.parse_args()

    if args.find_best:
        best_dir = find_best_checkpoint_dir(args.find_best, log_file=args.log_file)
        if best_dir is None:
            print("ERROR: Could not find best checkpoint")
            sys.exit(1)
        args.checkpoint_dir = str(best_dir)
        print(f"Auto-selected best checkpoint: {args.checkpoint_dir}")

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(f"\n{'='*60}")
    print(f"MERGE AND EVALUATE DETAILED")
    print(f"{'='*60}")
    print(f"GPU: {args.gpu}")
    print(f"Dataset type: {args.dataset_type}")

    if args.test_dir is None:
        if args.dataset_type == "generalization":
            args.test_dir = str(PROJECT_ROOT / "data" / "generalization")
        else:
            args.test_dir = str(PROJECT_ROOT / "data" / "sampled_1500")

    if args.checkpoint_dir:
        checkpoint_name = Path(args.checkpoint_dir).name
    elif args.merged_model_path:
        checkpoint_name = Path(args.merged_model_path).parent.name
    else:
        checkpoint_name = "checkpoint"

    if args.eval_only:
        if not args.merged_model_path:
            raise ValueError("--merged_model_path required with --eval_only")
        merged_model_path = args.merged_model_path
        print(f"Mode: Evaluation only")
        print(f"Model: {merged_model_path}")
    else:
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint_dir required (or use --eval_only with --merged_model_path)")

        print(f"Mode: Merge + Evaluate")
        print(f"Checkpoint: {args.checkpoint_dir}")
        merged_model_path = merge_checkpoint(args.checkpoint_dir)

        if args.merge_only:
            print(f"\n{'='*60}")
            print(f"✓ Merge complete (--merge_only flag set)")
            print(f"{'='*60}")
            return

    all_details, summary = evaluate_with_details(
        merged_model_path,
        args.test_dir,
        checkpoint_name=checkpoint_name,
        dataset_type=args.dataset_type
    )

    if args.output_dir is None:
        suffix = "" if args.dataset_type == "standard" else f"_{args.dataset_type}"
        if args.checkpoint_dir:
            args.output_dir = str(Path(args.checkpoint_dir) / f"evaluation_results{suffix}")
        elif args.merged_model_path:
            args.output_dir = str(Path(args.merged_model_path).parent / f"evaluation_results{suffix}")
        else:
            raise ValueError("Cannot determine output directory")

    save_detailed_results(all_details, summary, args.output_dir)

    print(f"\n{'='*60}")
    print(f"✓ PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
