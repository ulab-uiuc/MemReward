# Step 5: VERL Training

GRPO training scripts organized by model size. Each script includes a `# Run with:` comment at the top showing the exact nohup command.

## Qwen2.5-3B (`qwen2.5-3b/`)

### Baselines
- `verl_grpo_100perc_gt.sh` — 100% ground-truth reward (all 5358 train queries).
- `verl_grpo_20perc_gt_only.sh` — 20% GT only (~1104 train queries, no GNN).

### Mixed GT + GNN (DotProduct with answer features)
- `verl_grpo_20gt_80gnn_dot_product.sh` — 20% GT + 80% GNN.
- `verl_grpo_30gt_70gnn_dot_product.sh` — 30% GT + 70% GNN.
- `verl_grpo_40gt_60gnn_dot_product.sh` — 40% GT + 60% GNN.
- `verl_grpo_50gt_50gnn_dot_product.sh` — 50% GT + 50% GNN.
- `verl_grpo_60gt_40gnn_dot_product.sh` — 60% GT + 40% GNN.
- `verl_grpo_70gt_30gnn_dot_product.sh` — 70% GT + 30% GNN.

### Generalization (numina_math, siqa, piqa)
- `verl_grpo_generalization_100perc_gt.sh` — 100% GT baseline.
- `verl_grpo_generalization_20perc_gt_only.sh` — 20% GT only.
- `verl_grpo_generalization_20gt_80gnn_dot_product.sh` — 20% GT + 80% GNN.

## Qwen2.5-1.5B (`qwen2.5-1.5b/`)

### Baselines
- `verl_grpo_100perc_gt.sh` — 100% GT baseline.
- `verl_grpo_20perc_gt_only.sh` — 20% GT only.

### Mixed GT + GNN (DotProduct with answer features)
- `verl_grpo_20gt_80gnn_dot_product.sh` — 20% GT + 80% GNN.

### Generalization
- `verl_grpo_generalization_100perc_gt.sh` — 100% GT baseline.
- `verl_grpo_generalization_20perc_gt_only.sh` — 20% GT only.
- `verl_grpo_generalization_20gt_80gnn_dot_product.sh` — 20% GT + 80% GNN.

## Utilities (`utils/`)

**Command:** `python scripts/Step5_verl_training/utils/fix_reward_model_format.py <parquet_file>`
**Function:** Fix reward_model field from JSON string to Python dict (VERL format requirement).

**Command:** `python scripts/Step5_verl_training/utils/fix_validation_is_train.py --dataset qwen2.5_3b_standard`
**Function:** Ensure training data has `is_train=True` and validation data has `is_train=False`.

**Command:** `python scripts/Step5_verl_training/utils/verify_is_train_fields.py`
**Function:** Verify `is_train` field consistency across GNN and VERL data sources.
