# Step 4: GNN Training & Evaluation

## Best-of-N Training (recommended)

**Command:** `bash scripts/Step4_gnn_training_eval/train_gnn_best_of_n_dotproduct.sh --model-type qwen3b --hard-label`
**Function:** Train N GNN models (DotProduct + answer features) in parallel across GPUs, select the best (lowest FP rate), deploy, and generate unified cache.

Common options: `--gpus 0,1,2,3` (GPU list), `--num-runs 40` (total runs), `--train-ratio 20` (training data percentage), `--soft-label` (use soft labels instead of hard).

## Individual Training

**Command:** `python scripts/Step4_gnn_training_eval/train_gnn_from_cache_dotproduct.py --model-type qwen3b --hard-label`
**Function:** Train a single DotProduct-based UnifiedGNN model from cached embeddings. Uses scaled dot-product scoring with answer features.

## Output

Models and results are saved to `outputs/gnn_standard_domains/`. The best-of-N script also generates a unified embedding cache for VERL inference.
