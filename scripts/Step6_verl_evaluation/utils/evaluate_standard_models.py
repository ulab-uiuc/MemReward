#!/usr/bin/env python
'''
Evaluate model on all 10 standard benchmarks using vLLM batch inference.
Covers math (GSM8K, MATH, GSM-Symbolic), QA (MMLU, CSQA, OBQA, ARC-C, GPQA), code (HumanEval+, MBPP+).
Related: merge_and_evaluate_detailed.py for full pipeline with FSDP merge.
'''

import os
os.environ['VLLM_USE_V1'] = '0'

import argparse
import io
import json
import multiprocessing
import os
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _run_code_in_process(code: str, test_cases: list, result_queue):
    """Run code with tests in isolated process. Used by multiprocessing."""
    sys.setrecursionlimit(500)

    try:
        namespace = {}
        exec(code, namespace)

        for test in test_cases:
            exec(test, namespace)

        result_queue.put(1)
    except:
        result_queue.put(0)


def _safe_exec_with_timeout(code: str, test_cases: list, timeout_seconds: float = 5.0) -> bool:
    """Execute code with tests in isolated process with timeout.

    IMPORTANT: timeout_seconds=5.0 matches training code (code_reward in multi_domain_reward.py)
    to ensure train/test consistency.

    This prevents crashes from:
    - Segmentation faults
    - Stack overflows
    - Infinite loops
    - Malicious code

    Returns True if all tests pass, False otherwise.
    """
    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_code_in_process,
        args=(code, test_cases, result_queue)
    )
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.5)
        if proc.is_alive():
            proc.kill()
        return False

    try:
        result = result_queue.get_nowait()
        return result == 1
    except:
        return False


def extract_answer(response: str, data_source: str) -> Optional[str]:
    """Extract answer based on data source.

    IMPORTANT: This function must match the training code logic in multi_domain_reward.py
    to ensure train/test consistency.
    """
    ds = data_source.lower()

    if any(keyword in ds for keyword in ['gsm8k', 'gsm_symbolic', 'math']):
        if '####' in response:
            match = re.search(r'####\s*([^#\n][^\n]*?)(?:\s*####|\s*$)', response)
            if match:
                answer = match.group(1).strip().rstrip('.')
                if answer:
                    return answer
            parts = response.split('####')
            for part in parts[1:]:
                answer = part.strip().split('\n')[0].strip()
                if answer and not answer.startswith('#'):
                    answer = answer.rstrip('.')
                    return answer

        boxed_match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', response)
        if boxed_match:
            return boxed_match.group(1).strip()
        boxed_match = re.search(r'\\boxed\{(.+?)\}', response)
        if boxed_match:
            return boxed_match.group(1).strip()

        return None

    elif any(keyword in ds for keyword in ['mmlu', 'commonsenseqa', 'obqa', 'arc_c', 'arc-c', 'gpqa']):
        response_upper = response.upper()

        match = re.search(r'####\s*([A-E])\b', response_upper)
        if match:
            return match.group(1)

        match = re.search(r'(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+IS\s*:?\s*([A-E])\b', response_upper)
        if match:
            return match.group(1)

        match = re.search(r'\b([A-D])\b\s*$', response_upper.strip())
        if match:
            return match.group(1)

        return None

    elif 'humaneval' in ds or 'mbpp' in ds:
        code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        def_match = re.search(r'(def\s+\w+.*?)(?=\n\n|\Z)', response, re.DOTALL)
        if def_match:
            return def_match.group(1).strip()

        return response.strip()

    return None


def evaluate_response(response: str, data_source: str, extra_info: dict) -> bool:
    """Evaluate a single response."""
    answer = extract_answer(response, data_source)
    ds = data_source.lower()

    if any(keyword in ds for keyword in ['gsm8k', 'gsm_symbolic', 'math']):
        if answer is None:
            return False

        gt = str(extra_info.get('answer', ''))
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', gt)
        if boxed_match:
            gt = boxed_match.group(1)

        def normalize_math_answer(ans: str) -> str:
            """Normalize mathematical answer for comparison."""
            if not ans:
                return ""

            ans = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]+)\}', r'\1', ans)
            ans = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', ans)

            ans = ans.replace('\\', '').replace('$', '').strip()

            ans = ans.replace(',', '').replace(' ', '').lower()

            ans = re.sub(r'(dollars?|cents?|%|degrees?|°)$', '', ans)

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

        pred_norm = normalize_math_answer(answer)
        gt_norm = normalize_math_answer(gt)
        return pred_norm == gt_norm

    elif any(keyword in ds for keyword in ['mmlu', 'commonsenseqa', 'obqa', 'arc_c', 'arc-c', 'gpqa']):
        gt = str(extra_info.get('answer', '')).strip().upper()
        gt_letter = re.search(r'[A-E]', gt)
        if not gt_letter or not answer:
            return False
        return answer == gt_letter.group()

    elif 'humaneval' in ds or 'mbpp' in ds:
        if answer is None:
            return False
        test_cases = extra_info.get('test_list', extra_info.get('test', []))
        if isinstance(test_cases, str):
            test_cases = [test_cases]
        if isinstance(test_cases, np.ndarray):
            test_cases = test_cases.tolist()
        if not test_cases or len(test_cases) == 0:
            return False
        return _safe_exec_with_timeout(answer, test_cases, timeout_seconds=5.0)

    return False


def main():
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--test_dir", type=str,
                        default=str(PROJECT_ROOT / "data" / "sampled_1500"))
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n{'#'*60}")
    print(f"# Evaluating: {args.name or args.model_path}")
    print(f"{'#'*60}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        stop=["<|endoftext|>", "<|im_end|>"]
    )

    test_files = [f for f in os.listdir(args.test_dir) if f.endswith('_test.parquet')]

    results = {}
    all_details = {}

    for test_file in sorted(test_files):
        dataset = test_file.replace('_sampled_test.parquet', '')
        print(f"\n{'='*50}")
        print(f"Evaluating: {dataset}")
        print(f"{'='*50}")

        df = pd.read_parquet(os.path.join(args.test_dir, test_file))
        print(f"Samples: {len(df)}")

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

        print("Generating responses...")
        outputs = llm.generate(prompts, sampling_params)

        correct = 0
        details = []

        for idx, (output, (_, row)) in enumerate(zip(outputs, df.iterrows())):
            response = output.outputs[0].text
            extra_info = row['extra_info']
            if isinstance(extra_info, str):
                extra_info = eval(extra_info)
            data_source = row['data_source']

            is_correct = evaluate_response(response, data_source, extra_info)
            if is_correct:
                correct += 1

            details.append({
                'idx': idx,
                'correct': is_correct,
                'response': response[:500],
            })

        accuracy = correct / len(df) * 100
        print(f"Accuracy: {correct}/{len(df)} = {accuracy:.2f}%")

        results[dataset] = {
            'correct': correct,
            'total': len(df),
            'accuracy': accuracy
        }
        all_details[dataset] = details

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    total_correct = sum(r['correct'] for r in results.values())
    total_samples = sum(r['total'] for r in results.values())
    overall = total_correct / total_samples * 100 if total_samples > 0 else 0

    for ds, r in sorted(results.items()):
        print(f"{ds:20s}: {r['correct']:3d}/{r['total']:3d} = {r['accuracy']:6.2f}%")
    print(f"{'Overall':20s}: {total_correct:3d}/{total_samples:3d} = {overall:6.2f}%")

    results['overall'] = {'correct': total_correct, 'total': total_samples, 'accuracy': overall}

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nResults saved to {args.output_dir}")

    print(f"\n{args.name}: {overall:.2f}%")
    print("Done!")


if __name__ == "__main__":
    main()
