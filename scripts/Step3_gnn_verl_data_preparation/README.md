# Step 3: GNN & VERL Data Preparation

## Pipeline (one-click)

**Command:** `bash scripts/Step3_gnn_verl_data_preparation/run_standard_pipeline.sh --gpus 0,1,2,3`
**Function:** Run full standard pipeline (10 datasets): sample -> splits -> GPU responses -> GT identifiers -> 3-mode VERL data -> verify.

**Command:** `bash scripts/Step3_gnn_verl_data_preparation/run_generalization_pipeline.sh --gpus 0,1,2`
**Function:** Run full generalization pipeline (numina_math, siqa, piqa): sample -> GPU responses -> combine + partial GT + verify.

Omit `--gpus` to pause before the GPU step and run it manually. Use `--start-step N` to resume from step N.

## Individual Scripts

**Command:** `python scripts/Step3_gnn_verl_data_preparation/sample_1500/sample_1500_datasets.py`
**Function:** Sample 1500 queries per standard dataset with 50/20/30 train/valid/test split.

**Command:** `python scripts/Step3_gnn_verl_data_preparation/sample_1500/sample_1500_generalization.py`
**Function:** Sample 1500 queries per generalization dataset (NuminaMath, SIQA, PIQA) with 50/20/30 split.

**Command:** `python scripts/Step3_gnn_verl_data_preparation/sample_1500/generate_multi_ratio_splits.py`
**Function:** Generate {10,30,40,50,60,70}% training splits from train_full parquet files.

**Command:** `bash scripts/Step3_gnn_verl_data_preparation/generate_response/generate_train_parallel.sh --gpus 0,1,2,3`
**Function:** Generate train-split LLM responses in parallel across specified GPUs (round-robin). Outputs to `outputs/gnn_standard_domains/{model_type}/`.

**Command:** `bash scripts/Step3_gnn_verl_data_preparation/generate_response/generate_valid_parallel.sh --gpus 0,1,2,3`
**Function:** Generate valid-split LLM responses in parallel across specified GPUs (round-robin).

**Command:** `python scripts/Step3_gnn_verl_data_preparation/generate_and_verify_gt_identifier/generate_gt_identifiers.py --train-ratio 20`
**Function:** Generate `configs/gt_identifiers_train{ratio}.json` marking which queries use ground-truth reward (default 20%).

**Command:** `python scripts/Step3_gnn_verl_data_preparation/generate_and_verify_gt_identifier/verify_gt_alignment.py`
**Function:** Verify GT identifier alignment between `gt_identifiers_train20.json` and VERL parquet data.

**Command:** `python scripts/Step3_gnn_verl_data_preparation/generate_verl_data/generate_standard_verl_3modes_data.py --model qwen2.5`
**Function:** Generate 3 VERL training modes (Full GT / Partial GT / Mix) for standard datasets.

**Command:** `python scripts/Step3_gnn_verl_data_preparation/generate_verl_data/generate_generalization_verl_3modes_data.py`
**Function:** Combine generalization datasets and generate 3 VERL training modes with built-in verification.
