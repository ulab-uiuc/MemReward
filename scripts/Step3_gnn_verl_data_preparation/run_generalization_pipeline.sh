#!/bin/bash
# =============================================================================
# Generalization Data Preparation Pipeline (numina_math, siqa, piqa)
#
# Runs all CPU-based data preparation steps for generalization domains:
#   Step 1: Sample 1500 queries per dataset (50/20/30 split)
#   Step 2: Generate VERL 3-mode data (combine + partial GT + verify)
#           - Combines 3 datasets into verl_train/
#           - Creates verl_train_partial_gt/ from generalization_gt_identifiers.json
#           - Verifies index alignment
#
# NOTE: Between Step 1 and Step 2, GPU-heavy response generation is needed.
#   Use --gpus to auto-run it, or run separately:
#     bash scripts/Step3_gnn_verl_data_preparation/generate_response/generate_train_parallel.sh \
#       --datasets "numina_math siqa piqa"
#     bash scripts/Step3_gnn_verl_data_preparation/generate_response/generate_valid_parallel.sh \
#       --datasets "numina_math siqa piqa"
#
# NOTE: generalization_gt_identifiers.json uses a fixed pattern (every 5th
#   sample per dataset = 20% GT). No generation script needed — edit
#   configs/generalization_gt_identifiers.json directly if the ratio changes.
#
# Usage:
#   bash scripts/Step3_gnn_verl_data_preparation/run_generalization_pipeline.sh
#   bash scripts/Step3_gnn_verl_data_preparation/run_generalization_pipeline.sh --start-step 2
#   bash scripts/Step3_gnn_verl_data_preparation/run_generalization_pipeline.sh --gpus 0,1,2  # auto-run GPU step
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"

# Defaults
START_STEP=1
GPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --start-step)   START_STEP="$2"; shift 2;;
        --python)       PYTHON="$2"; shift 2;;
        --gpus)         GPUS="$2"; shift 2;;
        *)              echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "============================================================"
echo "  Generalization Data Preparation Pipeline"
echo "============================================================"
echo "  Project root:  $PROJECT_ROOT"
echo "  Python:        $PYTHON"
echo "  Start step:    $START_STEP"
echo "  Datasets:      numina_math, siqa, piqa"
echo "  GPUs:          ${GPUS:-not set (will pause for manual GPU step)}"
echo "============================================================"

# ---- Step 1: Sample 1500 ----
if [[ $START_STEP -le 1 ]]; then
    echo ""
    echo "[Step 1/2] Sampling 1500 queries per generalization dataset..."
    $PYTHON scripts/Step3_gnn_verl_data_preparation/sample_1500/sample_1500_generalization.py
    echo "[Step 1/2] Done."

    if [[ -n "$GPUS" ]]; then
        echo ""
        echo "[GPU] Generating train responses for generalization datasets (GPUs: $GPUS)..."
        bash "$SCRIPT_DIR/generate_response/generate_train_parallel.sh" \
            --datasets "numina_math siqa piqa" --gpus "$GPUS"
        echo "[GPU] Generating valid responses for generalization datasets (GPUs: $GPUS)..."
        bash "$SCRIPT_DIR/generate_response/generate_valid_parallel.sh" \
            --datasets "numina_math siqa piqa" --gpus "$GPUS"
        echo "[GPU] Response generation done."
    else
        echo ""
        echo "============================================================"
        echo "  IMPORTANT: Run response generation (GPU) before Step 2:"
        echo "    bash scripts/Step3_gnn_verl_data_preparation/generate_response/generate_train_parallel.sh \\"
        echo "      --datasets \"numina_math siqa piqa\""
        echo "    bash scripts/Step3_gnn_verl_data_preparation/generate_response/generate_valid_parallel.sh \\"
        echo "      --datasets \"numina_math siqa piqa\""
        echo ""
        echo "  Or re-run with --gpus to auto-run: --gpus 0,1,2"
        echo "  Then re-run with: --start-step 2"
        echo "============================================================"
        if [[ $START_STEP -lt 2 ]]; then
            exit 0
        fi
    fi
fi

# ---- Step 2: Combine datasets + Generate 3-mode data + Verify ----
if [[ $START_STEP -le 2 ]]; then
    echo ""
    echo "[Step 2/2] Combining datasets + generating 3-mode data + verifying..."
    $PYTHON scripts/Step3_gnn_verl_data_preparation/generate_verl_data/generate_generalization_verl_3modes_data.py
    echo "[Step 2/2] Done."
fi

echo ""
echo "============================================================"
echo "  Generalization Pipeline Complete"
echo "============================================================"
