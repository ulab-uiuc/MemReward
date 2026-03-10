#!/bin/bash
# =============================================================================
# Download All Datasets (one-click)
#
# Downloads all 13 datasets used in this project:
#   Standard (10):  gsm8k, math, gsm_symbolic, mmlu, commonsenseqa,
#                   obqa, arc_c, gpqa, humaneval_plus, mbpp_plus
#   Generalization (3): numina_math, siqa, piqa
#
# Usage:
#   bash scripts/Step2_original_data_download/download_all.sh
#   bash scripts/Step2_original_data_download/download_all.sh --only standard
#   bash scripts/Step2_original_data_download/download_all.sh --only generalization
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON=${PYTHON:-/home/taofeng2/.conda/envs/graphrouter/bin/python}

# Defaults
ONLY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --only)     ONLY="$2"; shift 2;;
        --python)   PYTHON="$2"; shift 2;;
        *)          echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "============================================================"
echo "  Step 2: Download All Datasets"
echo "============================================================"
echo "  Project root:  $PROJECT_ROOT"
echo "  Python:        $PYTHON"
echo "  Scope:         ${ONLY:-all (standard + generalization)}"
echo "============================================================"

FAILED=()

run_script() {
    local label="$1"
    local script="$2"
    echo ""
    echo "------------------------------------------------------------"
    echo "  $label"
    echo "------------------------------------------------------------"
    if $PYTHON "$script"; then
        echo "[OK] $label"
    else
        echo "[FAILED] $label"
        FAILED+=("$label")
    fi
}

# ---- Standard datasets ----
if [[ -z "$ONLY" || "$ONLY" == "standard" ]]; then
    run_script "Standard datasets (8): gsm8k, gsm_symbolic, humaneval_plus, mbpp_plus, obqa, mmlu, arc_c, commonsenseqa" \
        "$SCRIPT_DIR/download_datasets.py"

    run_script "MATH dataset (hendrycks/math)" \
        "$SCRIPT_DIR/download_math.py"

    run_script "GPQA dataset (requires HF token)" \
        "$SCRIPT_DIR/download_gpqa.py"
fi

# ---- Generalization datasets ----
if [[ -z "$ONLY" || "$ONLY" == "generalization" ]]; then
    run_script "Generalization datasets (3): numina_math, siqa, piqa" \
        "$SCRIPT_DIR/download_datasets_generalization.py"
fi

# ---- Summary ----
echo ""
echo "============================================================"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "  All downloads completed successfully!"
else
    echo "  Downloads completed with ${#FAILED[@]} failure(s):"
    for f in "${FAILED[@]}"; do
        echo "    - $f"
    done
fi
echo "============================================================"
