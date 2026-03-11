#!/bin/bash
# =============================================================================
# Standard Data Preparation Pipeline (10 datasets)
#
# Runs all CPU-based data preparation steps for standard domains:
#   Step 1: Sample 1500 queries per dataset (70/20/10 split)
#   Step 2: Generate multi-ratio splits (10/30/40/50/60/70%)
#   Step 3: Generate GT identifiers (configs/gt_identifiers_train20.json)
#   Step 4: Generate VERL 3-mode data (Full GT / Partial GT / Mix)
#   Step 5: Verify GT alignment across all components
#
# NOTE: Between Step 2 and Step 3, GPU-heavy response generation is needed.
#   Use --gpus to auto-run it, or run separately:
#     bash scripts/Step3_gnn_verl_data_preparation/generate_response/generate_train_parallel.sh
#     bash scripts/Step3_gnn_verl_data_preparation/generate_response/generate_valid_parallel.sh
#
# Usage:
#   bash scripts/Step3_gnn_verl_data_preparation/run_standard_pipeline.sh
#   bash scripts/Step3_gnn_verl_data_preparation/run_standard_pipeline.sh --start-step 3
#   bash scripts/Step3_gnn_verl_data_preparation/run_standard_pipeline.sh --train-ratio 30
#   bash scripts/Step3_gnn_verl_data_preparation/run_standard_pipeline.sh --gpus 0,1,2,3  # auto-run GPU step
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"

# Defaults
START_STEP=1
TRAIN_RATIO=20
MODEL=qwen2.5
GPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-step)   START_STEP="$2"; shift 2;;
        --train-ratio)  TRAIN_RATIO="$2"; shift 2;;
        --model)        MODEL="$2"; shift 2;;
        --python)       PYTHON="$2"; shift 2;;
        --gpus)         GPUS="$2"; shift 2;;
        *)              echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "============================================================"
echo "  Standard Data Preparation Pipeline"
echo "============================================================"
echo "  Project root:  $PROJECT_ROOT"
echo "  Python:        $PYTHON"
echo "  Start step:    $START_STEP"
echo "  Train ratio:   ${TRAIN_RATIO}%"
echo "  Model:         $MODEL"
echo "  GPUs:          ${GPUS:-not set (will pause for manual GPU step)}"
echo "============================================================"

# ---- Step 1: Sample 1500 ----
if [[ $START_STEP -le 1 ]]; then
    echo ""
    echo "[Step 1/5] Sampling 1500 queries per dataset..."
    $PYTHON scripts/Step3_gnn_verl_data_preparation/sample_1500/sample_1500_datasets.py
    echo "[Step 1/5] Done."
fi

# ---- Step 2: Multi-ratio splits ----
if [[ $START_STEP -le 2 ]]; then
    echo ""
    echo "[Step 2/5] Generating multi-ratio splits..."
    $PYTHON scripts/Step3_gnn_verl_data_preparation/sample_1500/generate_multi_ratio_splits.py
    echo "[Step 2/5] Done."
fi

# ---- GPU step: response generation ----
if [[ $START_STEP -le 2 ]]; then
    if [[ -n "$GPUS" ]]; then
        echo ""
        echo "[GPU] Generating train responses (GPUs: $GPUS)..."
        bash "$SCRIPT_DIR/generate_response/generate_train_parallel.sh" --gpus "$GPUS"
        echo "[GPU] Generating valid responses (GPUs: $GPUS)..."
        bash "$SCRIPT_DIR/generate_response/generate_valid_parallel.sh" --gpus "$GPUS"
        echo "[GPU] Response generation done."
    else
        echo ""
        echo "============================================================"
        echo "  IMPORTANT: Run response generation (GPU) before Step 3:"
        echo "    bash scripts/Step3_gnn_verl_data_preparation/generate_response/generate_train_parallel.sh"
        echo "    bash scripts/Step3_gnn_verl_data_preparation/generate_response/generate_valid_parallel.sh"
        echo ""
        echo "  Or re-run with --gpus to auto-run: --gpus 0,1,2,3"
        echo "  Then re-run with: --start-step 3"
        echo "============================================================"
        if [[ $START_STEP -lt 3 ]]; then
            exit 0
        fi
    fi
fi

# ---- Step 3: Generate GT identifiers ----
if [[ $START_STEP -le 3 ]]; then
    echo ""
    echo "[Step 3/5] Generating GT identifiers (ratio=${TRAIN_RATIO}%)..."
    $PYTHON scripts/Step3_gnn_verl_data_preparation/generate_and_verify_gt_identifier/generate_gt_identifiers.py \
        --train-ratio "$TRAIN_RATIO"
    echo "[Step 3/5] Done."
fi

# ---- Step 4: Generate 3-mode VERL data ----
if [[ $START_STEP -le 4 ]]; then
    echo ""
    echo "[Step 4/5] Generating VERL 3-mode data (model=$MODEL)..."
    $PYTHON scripts/Step3_gnn_verl_data_preparation/generate_verl_data/generate_standard_verl_3modes_data.py \
        --model "$MODEL"
    echo "[Step 4/5] Done."
fi

# ---- Step 5: Verify alignment ----
if [[ $START_STEP -le 5 ]]; then
    echo ""
    echo "[Step 5/5] Verifying GT alignment..."
    $PYTHON scripts/Step3_gnn_verl_data_preparation/generate_and_verify_gt_identifier/verify_gt_alignment.py
    echo "[Step 5/5] Done."
fi

echo ""
echo "============================================================"
echo "  Standard Pipeline Complete"
echo "============================================================"
