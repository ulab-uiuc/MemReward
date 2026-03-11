#!/bin/bash
# Generate valid responses for all 10 standard datasets in parallel
# Uses 10 GPUs (0-9), 1 dataset per GPU
#
# Usage:
#   ./generate_valid_parallel.sh [--model-type qwen3b|qwen1.5b] [--datasets "ds1 ds2 ..."] [--gpus "0,1,2,3"]
#
# Examples:
#   ./generate_valid_parallel.sh                           # All 10 datasets with qwen3b
#   ./generate_valid_parallel.sh --model-type qwen1.5b     # All 10 datasets with qwen1.5b
#   ./generate_valid_parallel.sh --datasets "gsm8k math"   # Only gsm8k and math
#   ./generate_valid_parallel.sh --gpus "0,1,2,3"         # Use specific GPUs (round-robin)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../../.."

export VLLM_USE_V1=0
PYTHON="${PYTHON:-python3}"

# Default values
MODEL_TYPE="qwen3b"
DATASETS="gsm8k math gsm_symbolic mmlu commonsenseqa obqa arc_c gpqa humaneval_plus mbpp_plus"
TEMPERATURE=""
GPUS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Parse GPU list (comma-separated → array)
if [[ -n "$GPUS" ]]; then
    IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
else
    GPU_ARRAY=()
fi

# Build temperature flag
TEMP_FLAG=""
if [[ -n "$TEMPERATURE" ]]; then
    TEMP_FLAG="--temperature $TEMPERATURE"
fi

# Convert to array
read -ra DS_ARRAY <<< "$DATASETS"

echo "=========================================="
echo "Generating VALID responses (parallel)"
echo "Model: $MODEL_TYPE"
echo "Temperature: ${TEMPERATURE:-default}"
echo "Datasets: ${DS_ARRAY[*]}"
if [[ ${#GPU_ARRAY[@]} -gt 0 ]]; then
    echo "GPUs: ${GPUS}"
else
    echo "GPUs: 0-$((${#DS_ARRAY[@]}-1))"
fi
echo "=========================================="

# Run each dataset on a separate GPU
for i in "${!DS_ARRAY[@]}"; do
    ds="${DS_ARRAY[$i]}"
    if [[ ${#GPU_ARRAY[@]} -gt 0 ]]; then
        gpu=${GPU_ARRAY[$((i % ${#GPU_ARRAY[@]}))]}
    else
        gpu=$((i % 10))
    fi
    echo "[GPU $gpu] Processing: ${ds}_valid"
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON scripts/Step3_gnn_verl_data_preparation/generate_response/generate_responses.py \
        --dataset "${ds}_valid" \
        --output_name "cache_${ds}" \
        --model-type "$MODEL_TYPE" \
        --split valid \
        $TEMP_FLAG \
        --force > /tmp/${MODEL_TYPE}_valid_gpu${gpu}_${ds}.log 2>&1 &
done

echo ""
echo "All jobs started!"
echo "Monitor: tail -f /tmp/${MODEL_TYPE}_valid_gpu*.log"
echo "Wait for completion..."

wait

echo ""
echo "=========================================="
echo "All valid responses generated!"
echo "=========================================="
