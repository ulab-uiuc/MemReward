'''
Mixed GNN Reward for VERL - Qwen 2.5-3B variant.
Supports all 3 GNN architectures with env var overrides.
Related: mixed_gnn_reward_base.py base class.
'''

from pathlib import Path

from reward_graph.rewards.mixed_gnn_reward_base import (
    BaseBatchMixedGNNRewardWithWarmup,
    extract_math_answer,
    extract_qa_answer,
    split_think_and_answer,
    make_get_batch_mixed_reward_function,
    make_compute_score,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DEFAULT_GNN_CHECKPOINT = str(_REPO_ROOT / "outputs/gnn_standard_domains/qwen3b/unified_gnn_qwen3b_hard_train20.pt")
DEFAULT_WARMUP_EMBEDDINGS = str(_REPO_ROOT / "outputs/gnn_standard_domains/qwen3b/qwen3b_cache_unified_train20/embeddings.pt")
DEFAULT_GT_IDENTIFIERS = str(_REPO_ROOT / "configs/gt_identifiers_train20.json")


class BatchMixedGNNRewardWithWarmup(BaseBatchMixedGNNRewardWithWarmup):
    '''
    Batch mixed reward with warmup graph for VERL (Qwen 2.5-3B).
    Routes queries to GT or GNN reward via gt_identifiers.
    Related: compute_score() VERL entry point.
    '''

    DEFAULT_CHECKPOINT = DEFAULT_GNN_CHECKPOINT
    DEFAULT_WARMUP = DEFAULT_WARMUP_EMBEDDINGS
    DEFAULT_GT = DEFAULT_GT_IDENTIFIERS
    CACHE_DIR_NAME = 'qwen3b_cache_unified_train20'
    CACHE_PREFIX = 'qwen3b_cache_'
    SUPPORTS_MULTI_ARCH = True
    ENV_KEYS = ('GNN_CHECKPOINT_PATH', 'GT_IDENTIFIERS_PATH', 'WARMUP_EMBEDDINGS_PATH')
    DOMAIN_FILTER_QQ_EDGES = False


BatchMixedGNNReward = BatchMixedGNNRewardWithWarmup

get_batch_mixed_reward_function = make_get_batch_mixed_reward_function(
    BatchMixedGNNRewardWithWarmup, DEFAULT_WARMUP_EMBEDDINGS, DEFAULT_GT_IDENTIFIERS
)

compute_score = make_compute_score(get_batch_mixed_reward_function)


if __name__ == "__main__":
    print("Testing Batch Mixed GNN Reward with Warmup Graph...")

    reward_fn = get_batch_mixed_reward_function()

    test_prompt = "What is 2+2? Let's think step by step."
    test_responses = [
        "Let me calculate: 2+2=4\n#### 4",
        "The answer is 4\n#### 4",
        "2 plus 2 equals 4\n#### 4",
        "I think it's 5\n#### 5",
        "Let me see... 2+2=4\n#### 4",
        "The sum is 4\n#### 4",
        "It's obviously 4\n#### 4",
        "Hmm, I believe it's 3\n#### 3",
    ]

    data_sources = ['math'] * 8
    ground_truths = ['4'] * 8
    extra_infos = [{'prompt': test_prompt, 'answer': '4'}] * 8

    scores = reward_fn.compute_rewards_batch(
        data_sources=data_sources,
        solution_strs=test_responses,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
    )

    print(f"\nResponses and scores:")
    for i, (resp, score) in enumerate(zip(test_responses, scores)):
        print(f"  {i+1}. Score={score:.1f}: {resp[:50]}...")

    print(f"\nStats: {reward_fn.get_stats()}")
