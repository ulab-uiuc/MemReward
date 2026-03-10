# Step 2: Original Data Download

## One-click

**Command:** `bash scripts/Step2_original_data_download/download_all.sh`
**Function:** Download all 13 datasets (10 standard + 3 generalization).

**Command:** `bash scripts/Step2_original_data_download/download_all.sh --only standard`
**Function:** Download only the 10 standard datasets.

**Command:** `bash scripts/Step2_original_data_download/download_all.sh --only generalization`
**Function:** Download only the 3 generalization datasets.

## Individual Scripts

**Command:** `python scripts/Step2_original_data_download/download_datasets.py`
**Function:** Download 8 standard datasets: GSM8K, GSM-Symbolic, HumanEval+, MBPP+, OBQA, MMLU, ARC-C, CommonsenseQA. Output: `data/{dataset}/`.

**Command:** `python scripts/Step2_original_data_download/download_math.py`
**Function:** Download MATH dataset (hendrycks/math). Separate script due to special `math_reward` format. Output: `data/math/`.

**Command:** `python scripts/Step2_original_data_download/download_gpqa.py`
**Function:** Download GPQA dataset (requires HuggingFace token for gated access). Output: `data/gpqa/`.

**Command:** `python scripts/Step2_original_data_download/download_datasets_generalization.py`
**Function:** Download 3 generalization datasets: NuminaMath, SIQA, PIQA. Output: `data/generalization/{dataset}/`.

## Output Format

All datasets are saved as Parquet files (`train.parquet`, `valid.parquet`, `test.parquet`) with VERL-compatible schema: `data_source`, `prompt`, `ability`, `reward_model`, `extra_info`.
