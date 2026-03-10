# Step 6: VERL Evaluation

## Main Script

**Command:** `python scripts/Step6_verl_evaluation/merge_and_evaluate_detailed.py --checkpoint_dir <path> --gpu 0`
**Function:** Merge FSDP checkpoint into HF model and evaluate on all test sets (10 standard or 3 generalization). Saves detailed JSONL with prompts, responses, correctness, and token usage.

**Command:** `python scripts/Step6_verl_evaluation/merge_and_evaluate_detailed.py --checkpoint_dir <path> --merge_only`
**Function:** Merge FSDP checkpoint only (skip evaluation).

**Command:** `python scripts/Step6_verl_evaluation/merge_and_evaluate_detailed.py --find_best <training_dir> --gpu 0`
**Function:** Auto-find best checkpoint (highest val score from training log), then merge and evaluate it. Use `--log_file <path>` to specify a custom log file.

**Command:** `python scripts/Step6_verl_evaluation/merge_and_evaluate_detailed.py --eval_only --merged_model_path <path> --gpu 0`
**Function:** Evaluate an already-merged model (skip merge step).

Use `--dataset_type generalization` for generalization benchmarks (numina_math, piqa, siqa).

## Utilities

**Command:** `python scripts/Step6_verl_evaluation/utils/evaluate_standard_models.py --model_path <path> --name <exp_name> --gpu 0`
**Function:** Standalone evaluation on all 10 standard benchmarks using vLLM with greedy decoding.

**Command:** `python scripts/Step6_verl_evaluation/utils/find_best_checkpoint.py --training_dir <path>`
**Function:** Parse training log to find the best checkpoint step based on arithmetic average of validation scores.
