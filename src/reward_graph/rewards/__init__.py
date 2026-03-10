'''
Reward functions for RLHF training (math, QA, code).
Provides multi-domain GT rewards and mixed GNN+GT VERL rewards.
Related: utils/multi_domain_reward.py, mixed_gnn_reward_batch_*.py.
'''

from .utils.multi_domain_reward import math_reward, qa_reward, code_reward

__all__ = [
    'math_reward',
    'qa_reward',
    'code_reward',
]
