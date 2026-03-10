# Reward Functions for VERL Training

## Training Flow

```
Training Step:
  1. VERL generates 8 responses per prompt (rollout.n=8)
  2. For each (prompt, response), call compute_score()
  3. verl_mixed_reward_qwen3b.py:
     - Check if query index is in gt_identifiers
     - Yes → Call math_reward/qa_reward/code_reward (GT)
     - No  → Call mixed_gnn_reward_batch_qwen (GNN prediction)
  4. Return reward to VERL for GRPO update
```
