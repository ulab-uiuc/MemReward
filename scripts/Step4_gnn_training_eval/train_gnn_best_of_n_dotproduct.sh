#!/bin/bash
# =============================================================================
# Train N GNN models in parallel, auto-select best (lowest FP rate),
# deploy to correct location, generate unified cache, and cleanup.
#
# Usage:
#   bash scripts/Step4_gnn_training_eval/train_gnn_best_of_n_dotproduct.sh --model-type qwen1.5b --hard-label
#   bash scripts/Step4_gnn_training_eval/train_gnn_best_of_n_dotproduct.sh --model-type qwen3b --hard-label --gpus 0,1,2,3 --num-runs 10
#   bash scripts/Step4_gnn_training_eval/train_gnn_best_of_n_dotproduct.sh --model-type qwen1.5b --hard-label --num-runs 30
# =============================================================================

set -e

# ==================== Parse Arguments ====================
MODEL_TYPE=""
LABEL_FLAG=""
NUM_RUNS=40
GPUS="0,1,3,4,5"
TRAIN_RATIO=20

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-type)   MODEL_TYPE="$2"; shift 2;;
        --hard-label)   LABEL_FLAG="--hard-label"; shift;;
        --num-runs)     NUM_RUNS="$2"; shift 2;;
        --gpus)         GPUS="$2"; shift 2;;
        --train-ratio)  TRAIN_RATIO="$2"; shift 2;;
        *)              echo "Unknown arg: $1"; exit 1;;
    esac
done

if [[ -z "$MODEL_TYPE" || -z "$LABEL_FLAG" ]]; then
    echo "Usage: $0 --model-type {qwen3b,qwen1.5b} --hard-label [--num-runs N] [--gpus 0,1,2,3] [--train-ratio 20]"
    exit 1
fi

# ==================== Configuration ====================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
TRAIN_SCRIPT=scripts/Step4_gnn_training_eval/train_gnn_from_cache_dotproduct.py
OUTPUT_DIR=outputs/gnn_standard_domains/${MODEL_TYPE}

RATIO_SUFFIX="_train${TRAIN_RATIO}"

RESULTS_DIR="outputs/gnn_multiple_runs_dotproduct${RATIO_SUFFIX}"
# Internal names: what the Python script actually saves as (with --save-dir)
INTERNAL_MODEL="unified_gnn_${MODEL_TYPE}_hard${RATIO_SUFFIX}.pt"
INTERNAL_RESULTS="gnn_results_${MODEL_TYPE}_hard${RATIO_SUFFIX}.json"
# External names: what we deploy as (avoid collision with MLP version)
MODEL_FILENAME="unified_gnn_${MODEL_TYPE}_hard_dotproduct${RATIO_SUFFIX}.pt"
RESULTS_FILENAME="gnn_results_${MODEL_TYPE}_hard_dotproduct${RATIO_SUFFIX}.json"

# Parse GPU list into array
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}

# Calculate runs per GPU
RUNS_PER_GPU=$(( (NUM_RUNS + NUM_GPUS - 1) / NUM_GPUS ))

mkdir -p $RESULTS_DIR logs

echo "============================================================"
echo "  GNN Best-of-$NUM_RUNS Training Pipeline"
echo "============================================================"
echo "  Model type:    $MODEL_TYPE"
echo "  Train ratio:   ${TRAIN_RATIO}%"
echo "  Total runs:    $NUM_RUNS"
echo "  GPUs:          ${GPU_ARRAY[*]} ($NUM_GPUS GPUs)"
echo "  Runs per GPU:  ~$RUNS_PER_GPU"
echo "  Output:        $OUTPUT_DIR/$MODEL_FILENAME"
echo "============================================================"

# ==================== Phase 1: Parallel Training ====================
echo ""
echo "[Phase 1/$4] Training $NUM_RUNS GNN models in parallel..."

# Clear old run results
rm -f $RESULTS_DIR/${MODEL_FILENAME/.pt/_run*.pt}
rm -f $RESULTS_DIR/${RESULTS_FILENAME/.json/_run*.json}

train_on_gpu() {
    local GPU=$1
    local START_RUN=$2
    local END_RUN=$3
    local GPU_OUTPUT_DIR="outputs/gnn_train_gpu${GPU}"

    mkdir -p $GPU_OUTPUT_DIR

    for i in $(seq $START_RUN $END_RUN); do
        [[ $i -gt $NUM_RUNS ]] && break
        echo "[GPU $GPU] Starting Run $i/$NUM_RUNS..."

        CUDA_VISIBLE_DEVICES=$GPU $PYTHON $TRAIN_SCRIPT \
            --model-type $MODEL_TYPE $LABEL_FLAG \
            --train-ratio $TRAIN_RATIO \
            --save-dir $GPU_OUTPUT_DIR \
            2>&1 | tee logs/gnn_run_${i}.log

        cp $GPU_OUTPUT_DIR/$INTERNAL_MODEL $RESULTS_DIR/${MODEL_FILENAME/.pt/_run${i}.pt}
        cp $GPU_OUTPUT_DIR/$INTERNAL_RESULTS $RESULTS_DIR/${RESULTS_FILENAME/.json/_run${i}.json}

        FP_RATE=$(python3 -c "import json; d=json.load(open('$RESULTS_DIR/${RESULTS_FILENAME/.json/_run${i}.json}')); print(f\"{d['overall']['fp_rate']*100:.2f}%\")" 2>/dev/null || echo "N/A")
        echo "[GPU $GPU] Run $i completed - FP Rate: $FP_RATE"
    done

    rm -rf $GPU_OUTPUT_DIR
}

PIDS=()
RUN_IDX=1
for gpu_idx in "${!GPU_ARRAY[@]}"; do
    GPU=${GPU_ARRAY[$gpu_idx]}
    START=$RUN_IDX
    END=$((RUN_IDX + RUNS_PER_GPU - 1))
    train_on_gpu $GPU $START $END &
    PIDS+=($!)
    RUN_IDX=$((END + 1))
done

echo "Launched ${#PIDS[@]} GPU workers, waiting..."
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "[Phase 1 complete] All $NUM_RUNS training runs finished."

# ==================== Phase 2: Select Best ====================
echo ""
echo "[Phase 2/4] Selecting best model..."

BEST_RUN=$($PYTHON << PYEOF
import json, os, statistics

results_dir = "$RESULTS_DIR"
model_type = "$MODEL_TYPE"
label_mode = "hard"
num_runs = $NUM_RUNS
results_fn = "$RESULTS_FILENAME"

runs = []
for i in range(1, num_runs + 1):
    jf = f"{results_dir}/{results_fn.replace('.json', f'_run{i}.json')}"
    if os.path.exists(jf):
        with open(jf) as f:
            data = json.load(f)
            runs.append({
                'run': i,
                'fp_rate': data['overall']['fp_rate'] * 100,
                'accuracy': data['overall']['accuracy'] * 100,
                'f1': data['overall']['f1'] * 100,
                'recall': data['overall']['recall'] * 100,
                'precision': data['overall']['precision'] * 100
            })

runs_sorted = sorted(runs, key=lambda x: x['fp_rate'])

print("", flush=True)
print("=" * 80)
print(f"GNN Training Results ({len(runs)} Runs, Sorted by FP Rate)")
print("=" * 80)
print(f"{'Run':<5} {'FP Rate':<10} {'Accuracy':<10} {'F1':<10} {'Recall':<10} {'Precision':<10}")
print("-" * 80)
for run in runs_sorted:
    m = " <-- BEST" if run == runs_sorted[0] else ""
    print(f"{run['run']:<5} {run['fp_rate']:>7.2f}%  {run['accuracy']:>7.2f}%  {run['f1']:>7.2f}%  {run['recall']:>7.2f}%  {run['precision']:>7.2f}%{m}")
print("-" * 80)

best = runs_sorted[0]
fp_rates = [r['fp_rate'] for r in runs]
print(f"Best: Run #{best['run']}, FP Rate = {best['fp_rate']:.2f}%")
print(f"Stats: mean={statistics.mean(fp_rates):.2f}%, std={statistics.stdev(fp_rates):.2f}%, min={min(fp_rates):.2f}%, max={max(fp_rates):.2f}%")
print("=" * 80)

# Output best run number (last line, consumed by bash)
print(best['run'])
PYEOF
)

# Last line of Python output is the best run number
BEST_RUN_NUM=$(echo "$BEST_RUN" | tail -1)
# Print the table (all lines except last)
echo "$BEST_RUN" | head -n -1

echo ""
echo "[Phase 2 complete] Best run: #$BEST_RUN_NUM"

# ==================== Phase 3: Deploy Best Model ====================
echo ""
echo "[Phase 3/4] Deploying best model to $OUTPUT_DIR/..."

cp "$RESULTS_DIR/${MODEL_FILENAME/.pt/_run${BEST_RUN_NUM}.pt}" "$OUTPUT_DIR/$MODEL_FILENAME"
cp "$RESULTS_DIR/${RESULTS_FILENAME/.json/_run${BEST_RUN_NUM}.json}" "$OUTPUT_DIR/$RESULTS_FILENAME"

FINAL_FP=$(python3 -c "import json; d=json.load(open('$OUTPUT_DIR/$RESULTS_FILENAME')); print(f\"{d['overall']['fp_rate']*100:.2f}%\")")
echo "Deployed: $OUTPUT_DIR/$MODEL_FILENAME (FP Rate: $FINAL_FP)"

# Generate unified cache for VERL inference
UNIFIED_CACHE_NAME="${MODEL_TYPE}_cache_unified_train${TRAIN_RATIO}"
if [[ "$TRAIN_RATIO" -ne 20 ]]; then
    MAX_PER_DATASET=$(( 750 * TRAIN_RATIO / 100 ))
    MAX_EXPECTED=$(( MAX_PER_DATASET * 10 ))
else
    MAX_PER_DATASET=0  # 0 means no limit (20% uses all cache entries)
    MAX_EXPECTED=1200
fi
echo "Generating unified cache ($UNIFIED_CACHE_NAME, max_per_dataset=${MAX_PER_DATASET:-unlimited})..."
$PYTHON -c "
import sys; sys.path.insert(0, 'src')
from reward_graph.utils.cache_utils import load_or_create_unified_cache
max_pd = $MAX_PER_DATASET if $MAX_PER_DATASET > 0 else None
cd = load_or_create_unified_cache(
    cache_dir='$OUTPUT_DIR', prefix='${MODEL_TYPE}_cache_',
    unified_name='$UNIFIED_CACHE_NAME', force=True,
    max_per_dataset=max_pd)
n_q = cd['query_embeddings'].shape[0]
n_r = cd['think_embeddings'].shape[0]
print(f'Unified cache: {n_q} queries, {n_r} responses')
# Sanity check: unified cache must only contain training queries (not validation)
max_expected = $MAX_EXPECTED
assert n_q <= max_expected, f'ERROR: Unified cache has {n_q} queries, expected <= {max_expected} (train only). Validation data may have leaked in!'
print(f'Sanity check passed: {n_q} queries (train-only, limit={max_expected})')
"

echo "[Phase 3 complete] Model and unified cache deployed."

# ==================== Phase 4: Cleanup ====================
echo ""
echo "[Phase 4/4] Cleaning up temporary files..."

rm -rf $RESULTS_DIR
rm -rf outputs/gnn_train_gpu*

echo "[Phase 4 complete] Cleanup done."

echo ""
echo "============================================================"
echo "  GNN Training Pipeline Complete"
echo "  Final model: $OUTPUT_DIR/$MODEL_FILENAME"
echo "  FP Rate:     $FINAL_FP"
echo "============================================================"
