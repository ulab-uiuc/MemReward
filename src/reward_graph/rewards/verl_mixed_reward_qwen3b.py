'''
VERL mixed reward entry point for Qwen 3B.
Re-exports compute_score from batch variant.
Related: mixed_gnn_reward_batch_qwen3b.py batch reward engine.
'''

from reward_graph.rewards.mixed_gnn_reward_batch_qwen3b import (
    compute_score,
    get_batch_mixed_reward_function,
)


def get_reward_stats():
    '''
    GT vs GNN usage statistics reporter.
    Returns counts and percentages from batch GNN function.
    Related: BaseBatchMixedGNNRewardWithWarmup.get_stats().
    '''
    fn = get_batch_mixed_reward_function()
    return fn.get_stats() if fn else {'gt_count': 0, 'gnn_count': 0}
