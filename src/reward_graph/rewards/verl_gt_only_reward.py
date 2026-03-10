'''
VERL GT-only reward baseline (no GNN prediction).
All queries use ground truth reward functions.
Related: verl_mixed_reward_*.py mixed GT/GNN alternatives.
'''

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional, Union

sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from reward_graph.rewards.utils.multi_domain_reward import (
    math_reward, qa_reward, code_reward
)

logger = logging.getLogger(__name__)


def _map_data_source_to_domain(data_source: str) -> str:
    '''
    Definition: Data source to domain mapper.
    Purpose: Maps dataset name to math, qa, or coding domain.
    Related: _compute_gt_reward() domain routing.
    '''
    ds = data_source.lower()
    # Math domain
    if 'math' in ds or 'gsm' in ds:
        return 'math'
    # QA domain (multiple-choice, letter-based answers)
    elif any(keyword in ds for keyword in ['mmlu', 'commonsenseqa', 'obqa', 'arc_c', 'arc-c', 'gpqa']):
        return 'qa'
    # Code domain
    elif 'humaneval' in ds or 'mbpp' in ds or 'code' in ds or 'apps' in ds:
        return 'coding'
    logger.warning(f"[VERLGTOnlyReward] Unknown data_source: {data_source}, defaulting to 'qa'")
    return 'qa'


def _compute_gt_reward(
    response: str,
    domain: str,
    ground_truth: str,
    extra_info: Dict[str, Any]
) -> float:
    '''
    Definition: Ground truth reward calculator per domain.
    Purpose: Delegates to math_reward, qa_reward, or code_reward.
    Related: multi_domain_reward.py domain functions.
    '''
    if domain == 'math':
        return math_reward(response, ground_truth)
    elif domain == 'qa':
        return qa_reward(response, ground_truth)
    elif domain == 'coding':
        # Code uses 'test_list' with list of test case strings
        test_list = extra_info.get('test_list', [])
        return code_reward(response, test_list, timeout_seconds=5)
    logger.warning(f"[VERLGTOnlyReward] Unknown domain in _compute_gt_reward: {domain}")
    return 0.0


def _parse_extra_info(extra_info: Any) -> Dict[str, Any]:
    '''
    Definition: Extra info parser to dict.
    Purpose: Converts None, JSON string, or dict to standardized dict.
    Related: compute_score() input normalization.
    '''
    if extra_info is None:
        return {}
    if isinstance(extra_info, str):
        try:
            return json.loads(extra_info)
        except:
            return {}
    return extra_info if isinstance(extra_info, dict) else {}


def compute_score(
    data_source: Union[str, List[str]] = None,
    solution_str: Union[str, List[str]] = None,
    ground_truth: Union[str, List[str]] = None,
    extra_info: Union[Dict[str, Any], List[Dict[str, Any]]] = None,
    # Batch mode parameters (VERL BatchRewardManager uses these)
    data_sources: List[str] = None,
    solution_strs: List[str] = None,
    ground_truths: List[str] = None,
    extra_infos: List[Dict[str, Any]] = None,
    **kwargs,
) -> Union[float, List[float]]:
    '''
    Definition: VERL-compatible GT-only reward entry point.
    Purpose: Supports single and batch modes, all using GT reward.
    Related: _compute_score_batch() batch processing.
    '''
    import numpy as np

    def to_list(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x.tolist()
        return list(x) if hasattr(x, '__iter__') and not isinstance(x, (str, dict)) else x

    # Determine if this is batch mode
    is_batch = data_sources is not None and (
        isinstance(data_sources, (list, np.ndarray)) and len(data_sources) > 0
    )

    if is_batch:
        # Convert numpy arrays to lists
        data_sources = to_list(data_sources)
        solution_strs = to_list(solution_strs)
        ground_truths = to_list(ground_truths)
        extra_infos = to_list(extra_infos)

        # BATCH MODE - All GT
        return _compute_score_batch(
            data_sources=data_sources,
            solution_strs=solution_strs,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )
    else:
        # SINGLE MODE
        return _compute_score_single(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )


def _compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Dict[str, Any]],
) -> List[float]:
    '''
    Definition: Batch GT reward computation.
    Purpose: Computes GT reward for all samples without GNN routing.
    Related: _compute_gt_reward() per-sample calculation.
    '''
    N = len(solution_strs)

    parsed_extras = [_parse_extra_info(e) for e in (extra_infos or [{}] * N)]
    scores = []
    for i in range(N):
        domain = _map_data_source_to_domain(data_sources[i])
        reward = _compute_gt_reward(
            solution_strs[i], domain, ground_truths[i], parsed_extras[i]
        )
        scores.append(reward)

    return scores


def _compute_score_single(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any],
) -> float:
    '''
    Definition: Single sample GT reward computation.
    Purpose: Computes GT reward for one sample.
    Related: _compute_gt_reward() GT calculation.
    '''
    extra_info = _parse_extra_info(extra_info)
    domain = _map_data_source_to_domain(data_source)
    return _compute_gt_reward(solution_str, domain, ground_truth, extra_info)


if __name__ == "__main__":
    print("Testing VERL GT-Only Reward Function...")

    # Test batch mode
    print("\n=== Batch Mode Test ===")
    batch_result = compute_score(
        data_sources=['math'] * 4 + ['musique'] * 2 + ['apps'] * 2,
        solution_strs=[
            "2+2=4\n#### 4",
            "The answer is 4\n#### 4",
            "I think 5\n#### 5",  # Wrong
            "Sum is 4\n#### 4",
            "The answer is Paris",
            "I don't know",  # Wrong
            "```python\ndef solution():\n    return 42\n```",
            "print(42)",
        ],
        ground_truths=['4', '4', '4', '4', 'Paris', 'London', '42', '42'],
        extra_infos=[
            {'answer': '4'}, {'answer': '4'}, {'answer': '4'}, {'answer': '4'},
            {'answer': 'Paris'}, {'answer': 'London'},
            {'test_cases': json.dumps({'fn_name': 'solution', 'inputs': [[]], 'outputs': [42]})},
            {'test_cases': json.dumps({'inputs': [[]], 'outputs': ['42']})},
        ],
    )
    print(f"Batch scores: {batch_result}")
    print(f"Expected: [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, ?, ?]")

    # Test single mode
    print("\n=== Single Mode Test ===")
    single_result = compute_score(
        data_source='math',
        solution_str='The answer is 42\n#### 42',
        ground_truth='42',
        extra_info={'answer': '42'},
    )
    print(f"Single score: {single_result}")
