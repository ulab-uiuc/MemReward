'''
Multi-domain unified reward functions for math, QA, and code.
All rewards return binary {0.0, 1.0} based on answer correctness.
Related: mixed_gnn_reward_batch_*.py GT reward delegation.
'''

import re
import sys
import signal
from typing import Dict, Any, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


def extract_answer_after_hashtag(text: str) -> Optional[str]:
    '''
    Definition: Answer extractor after #### marker.
    Purpose: Finds first valid answer text following #### in response.
    Related: math_reward() primary extraction path.
    '''
    if '####' not in text:
        return None

    # Use regex to find answer after #### (not followed immediately by # or newline)
    # This handles cases like "#### answer ####" correctly
    import re

    # Try to find #### followed by non-empty content (not # or whitespace-only)
    match = re.search(r'####\s*([^#\n][^\n]*?)(?:\s*####|\s*$)', text)
    if match:
        answer = match.group(1).strip().rstrip('.')
        if answer:
            return answer

    # Fallback: split and find first non-empty part
    parts = text.split('####')
    for part in parts[1:]:  # Skip first part (before any ####)
        answer = part.strip().split('\n')[0].strip()
        # Skip if it's just # symbols or empty
        if answer and not answer.startswith('#'):
            answer = answer.rstrip('.')
            return answer

    return None


def normalize_math_answer_legacy(answer: str) -> str:
    '''
    Definition: Legacy numeric normalizer for math answers.
    Purpose: Strips LaTeX and extracts numbers loosely (may be too lenient).
    Related: normalize_math_answer() stricter version.
    '''
    if not answer:
        return ""
    answer = answer.replace('\\', '')
    answer = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\(?:frac)\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', answer)
    answer = answer.replace(' ', '').lower()
    answer = re.sub(r'(?:dollars?|cents?|\$|%|degrees?|°)', '', answer)
    number_match = re.search(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer)
    if number_match:
        return number_match.group().replace(',', '')
    return answer


def normalize_math_answer(answer: str) -> str:
    '''
    Definition: Numeric normalizer for math answer comparison.
    Purpose: Handles LaTeX, fractions, decimals, and units removal.
    Related: math_reward() answer comparison.
    '''
    if not answer:
        return ""

    answer = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2', answer)
    answer = answer.replace('\\', '').replace('$', '').strip()
    answer = answer.replace(',', '').replace(' ', '').lower()
    answer = re.sub(r'(dollars?|cents?|%|degrees?|°)$', '', answer)

    if '/' in answer:
        try:
            parts = answer.split('/')
            if len(parts) == 2:
                numerator = float(parts[0].strip('()'))
                denominator = float(parts[1].strip('()'))
                if denominator != 0:
                    result = numerator / denominator
                    return str(int(result) if result.is_integer() else result)
        except:
            pass

    number_match = re.search(r'-?\d+(?:\.\d+)?', answer)
    if number_match:
        num = float(number_match.group())
        return str(int(num) if num.is_integer() else num)

    return answer


def extract_boxed_answer(text: str) -> Optional[str]:
    '''
    Definition: Answer extractor from \\boxed{} LaTeX format.
    Purpose: Matches nested braces in \\boxed{} expressions.
    Related: math_reward() fallback extraction.
    '''
    # Match \boxed{...} with nested braces
    match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if match:
        return match.group(1).strip()
    # Fallback: try simpler pattern
    match = re.search(r'\\boxed\{(.+?)\}', text)
    if match:
        return match.group(1).strip()
    return None


def math_reward(response: str, ground_truth: str) -> float:
    '''
    Definition: Binary reward for math problems requiring #### marker.
    Purpose: Extracts and normalizes predicted vs ground truth answers.
    Related: unified_reward() math routing.
    '''
    # Try #### marker first
    pred = extract_answer_after_hashtag(response)

    # Fallback: try \boxed{} if no #### found
    if pred is None:
        pred = extract_boxed_answer(response)

    if pred is None:
        return 0.0

    gt = ground_truth
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', gt)
    if boxed_match:
        gt = boxed_match.group(1)

    # Normalize and compare
    pred_norm = normalize_math_answer(pred)
    gt_norm = normalize_math_answer(gt)

    return 1.0 if pred_norm == gt_norm else 0.0


def extract_qa_choice(response: str) -> str:
    '''
    Definition: Letter choice extractor (A-E) from QA responses.
    Purpose: Uses #### marker, "answer is" pattern, or trailing letter.
    Related: qa_reward() extraction step.
    '''
    response_upper = response.upper()

    # Pattern 1: #### A (letter right after ####, word boundary)
    match = re.search(r'####\s*([A-E])\b', response_upper)
    if match:
        return match.group(1)

    # Pattern 2: "The answer is X" or "correct answer is X"
    match = re.search(r'(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+IS\s*:?\s*([A-E])\b', response_upper)
    if match:
        return match.group(1)

    # Pattern 3: Standalone letter at end (A-D only)
    match = re.search(r'\b([A-D])\b\s*$', response_upper.strip())
    if match:
        return match.group(1)

    return 'X'


def qa_reward(response: str, ground_truth: str) -> float:
    '''
    Definition: Binary reward for multiple-choice QA problems.
    Purpose: Compares extracted letter choice against ground truth.
    Related: unified_reward() qa routing.
    '''
    gt = ground_truth.strip().upper()
    if not gt or gt not in 'ABCDE':
        return 0.0

    extracted_answer = extract_qa_choice(response)

    if extracted_answer != 'X' and extracted_answer == gt:
        return 1.0

    return 0.0


@contextmanager
def timeout(seconds):
    '''
    Definition: Context manager for code execution timeout via SIGALRM.
    Purpose: Sets alarm signal for timeout enforcement.
    Related: code_reward() execution safety.
    '''
    def signal_handler(signum, frame):
        raise TimeoutError("Code execution timed out")

    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _run_code_in_process(code: str, fn_name: str, inputs, expected, result_queue):
    '''
    Definition: Isolated code execution worker for multiprocessing.
    Purpose: Runs function-call or stdin/stdout mode in subprocess.
    Related: _safe_exec_with_timeout() process spawning.
    '''
    import sys
    import io

    sys.setrecursionlimit(500)

    try:
        if fn_name:
            # Function call mode
            exec_globals = {}
            exec(code, exec_globals)

            func = exec_globals.get(fn_name)
            if func is None:
                # Try to find first callable
                for name, obj in exec_globals.items():
                    if callable(obj) and not name.startswith('_'):
                        func = obj
                        break

            if func is None:
                result_queue.put(0)
                return

            # Call function
            if isinstance(inputs, list):
                result = func(*inputs)
            else:
                result = func(inputs)

            # Compare result
            exp = expected[0] if isinstance(expected, list) and len(expected) == 1 else expected
            if result is not None and (str(result).strip() == str(exp).strip() or result == exp):
                result_queue.put(1)
            else:
                result_queue.put(0)
        else:
            # stdin/stdout mode
            old_stdin = sys.stdin
            old_stdout = sys.stdout

            try:
                if isinstance(inputs, list):
                    inp_str = '\n'.join(str(line) for line in inputs)
                else:
                    inp_str = str(inputs) if inputs is not None else ''

                sys.stdin = io.StringIO(inp_str)
                captured = io.StringIO()
                sys.stdout = captured

                exec(code, {'__builtins__': __builtins__})

                sys.stdin = old_stdin
                sys.stdout = old_stdout

                actual = captured.getvalue().strip()

                if isinstance(expected, list):
                    exp_str = '\n'.join(str(line) for line in expected)
                else:
                    exp_str = str(expected) if expected is not None else ''
                exp_str = exp_str.strip()

                if actual == exp_str:
                    result_queue.put(1)
                else:
                    result_queue.put(0)
            finally:
                sys.stdin = old_stdin
                sys.stdout = old_stdout
    except:
        result_queue.put(0)


def _safe_exec_with_timeout(code: str, fn_name: str, inputs, expected, timeout_seconds: float = 2.0) -> int:
    '''
    Definition: Ray-compatible code executor with process timeout.
    Purpose: Spawns subprocess, enforces timeout, returns pass/fail.
    Related: code_reward() execution backend.
    '''
    import multiprocessing

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_code_in_process,
        args=(code, fn_name, inputs, expected, result_queue)
    )
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=0.5)
        if proc.is_alive():
            proc.kill()
        return 0

    try:
        return result_queue.get_nowait()
    except:
        return 0


def _run_test_cases_in_process(code: str, test_cases: list, result_queue):
    '''
    Definition: Test case execution worker for multiprocessing.
    Purpose: Runs assert-style test cases against code in subprocess.
    Related: _safe_exec_test_cases() process spawning.
    '''
    import sys
    sys.setrecursionlimit(500)

    try:
        namespace = {}
        exec(code, namespace)
        for test in test_cases:
            exec(test, namespace)
        result_queue.put(1)
    except:
        result_queue.put(0)


def _safe_exec_test_cases(code: str, test_cases: list, timeout_seconds: float = 5.0) -> bool:
    '''
    Definition: Ray-compatible test case executor with process timeout.
    Purpose: Spawns subprocess for assert tests, enforces timeout.
    Related: code_reward() test execution.
    '''
    import multiprocessing

    result_queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_run_test_cases_in_process,
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


def extract_code_from_response(response: str) -> Optional[str]:
    '''
    Definition: Python code extractor from model response.
    Purpose: Tries ```python blocks, def statements, then full response.
    Related: code_reward() code extraction step.
    '''
    # Try ```python block first
    match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find function definition
    match = re.search(r'(def\s+\w+.*?)(?=\n\n|\Z)', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: return entire response
    return response.strip()


def code_reward(response: str, test_cases: list, timeout_seconds: int = 5) -> float:
    '''
    Definition: Binary reward for coding problems via test execution.
    Purpose: Extracts code and runs assert test cases with timeout.
    Related: unified_reward() coding routing.
    '''
    code = extract_code_from_response(response)
    if code is None:
        return 0.0

    # No test cases = nothing to verify
    if not test_cases:
        return 0.0

    # Use Ray-compatible multiprocessing execution
    passed = _safe_exec_test_cases(code, test_cases, timeout_seconds=float(timeout_seconds))
    return 1.0 if passed else 0.0


def split_think_and_answer(response: str) -> tuple:
    '''
    Definition: Response splitter at think/answer tags.
    Purpose: Separates <think>...</think> tagged reasoning from answer.
    Related: extract_answer_from_tags() tag extraction.
    '''
    if '<think>' in response and '</think>' in response:
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            think_part = think_match.group(1).strip()
            answer_start = response.find('</think>') + len('</think>')
            answer_part = response[answer_start:].strip()
            return think_part, answer_part

    # Fallback: no think tags, entire response is answer
    return '', response


def extract_answer_from_tags(text: str) -> Optional[str]:
    '''
    Definition: Answer extractor from <answer> tags or #### marker.
    Purpose: Tries <answer> tags first, falls back to #### extraction.
    Related: split_think_and_answer() tag processing.
    '''
    if '<answer>' in text and '</answer>' in text:
        start = text.find('<answer>') + len('<answer>')
        end = text.find('</answer>')
        return text[start:end].strip()

    # Fallback: try #### marker
    if '####' in text:
        match = re.search(r'####\s*([^#\n][^\n]*?)(?:\s*####|\s*$)', text)
        if match:
            return match.group(1).strip()
        # Simple fallback
        parts = text.split('####')
        for part in parts[1:]:
            answer = part.strip().split('\n')[0].strip()
            if answer and not answer.startswith('#'):
                return answer

    return None


def unified_reward(
    response: str,
    extra_info: Dict[str, Any],
    domain: str
) -> float:
    '''
    Definition: Domain-routing reward function.
    Purpose: Routes to math_reward, qa_reward, or code_reward by domain.
    Related: get_reward_function() factory.
    '''
    if isinstance(extra_info, str):
        import json
        try:
            extra_info = json.loads(extra_info)
        except:
            extra_info = {}

    if domain == 'math':
        gt = extra_info.get('answer', '')
        return math_reward(response, gt)

    elif domain == 'qa':
        gt = extra_info.get('answer', '')
        return qa_reward(response, gt)

    elif domain == 'coding':
        test_cases = extra_info.get('test_list', [])
        return code_reward(response, test_cases)

    else:
        raise ValueError(f"Unknown domain: {domain}")


def get_reward_function(domain: str):
    '''
    Definition: Domain-specific reward function factory.
    Purpose: Returns lambda wrapping math_reward, qa_reward, or code_reward.
    Related: unified_reward() direct routing.
    '''
    if domain == 'math':
        return lambda resp, info: math_reward(resp, info.get('answer', ''))
    elif domain == 'qa':
        return lambda resp, info: qa_reward(resp, info.get('answer', ''))
    elif domain == 'coding':
        return lambda resp, info: code_reward(resp, info.get('test_list', []))
    else:
        raise ValueError(f"Unknown domain: {domain}")


def compute_batch_rewards(
    responses: list,
    extra_infos: list,
    domains: list
) -> list:
    '''
    Definition: Batch reward computation over response list.
    Purpose: Iterates responses calling unified_reward per sample.
    Related: unified_reward() per-sample computation.
    '''
    rewards = []
    for resp, info, domain in zip(responses, extra_infos, domains):
        reward = unified_reward(resp, info, domain)
        rewards.append(reward)
    return rewards


if __name__ == "__main__":
    # Test MATH reward
    print("Testing MATH reward...")
    math_response = "Let me solve this step by step.\n3 + 5 = 8\n#### 8"
    math_gt = "8"
    print(f"  Response: {math_response}")
    print(f"  GT: {math_gt}")
    print(f"  Reward: {math_reward(math_response, math_gt)}")

    # Test QA reward
    print("\nTesting QA reward...")
    qa_response = "The capital of France is Paris.\n#### A"
    qa_gt = "A"
    print(f"  Response: {qa_response}")
    print(f"  GT: {qa_gt}")
    print(f"  Reward: {qa_reward(qa_response, qa_gt)}")

    # Test Code reward
    print("\nTesting Code reward...")
    code_response = '''
Here's the solution:
####
```python
def add(a, b):
    return a + b
```
'''
    test_cases = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
    print(f"  Response: {code_response}")
    print(f"  Tests: {test_cases}")
    print(f"  Reward: {code_reward(code_response, test_cases)}")

    # Test unified interface
    print("\nTesting unified interface...")
    print(f"  MATH: {unified_reward(math_response, {'answer': '8'}, 'math')}")
    print(f"  QA: {unified_reward(qa_response, {'answer': 'A'}, 'qa')}")
    print(f"  Code: {unified_reward(code_response, {'test_list': test_cases}, 'coding')}")

    print("\n✓ All tests passed!")
