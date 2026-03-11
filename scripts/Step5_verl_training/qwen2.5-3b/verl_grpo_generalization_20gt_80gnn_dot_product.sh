#!/usr/bin/env bash
# Ignore SIGTERM and SIGHUP to prevent accidental termination
trap '' SIGTERM SIGHUP
# Run with: nohup bash scripts/Step5_verl_training/qwen2.5-3b/verl_grpo_generalization_20gt_80gnn_dot_product.sh > outputs/qwen2.5-3b/verl_grpo_generalization_20gt_80gnn_dot_product/training.log 2>&1 &
# VERL GRPO - Generalization Mixed 20% GT + 80% GNN Dot Product (Qwen 2.5 3B)
# Training samples: 2249, Steps per epoch: 17, Total: 10 epochs (170 steps)

set -x

# Get project root directory (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Ray memory and temp directory config
source "$PROJECT_ROOT/configs/ray_memory_config.sh"

# Environment configuration
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=6,7,8,9
export VLLM_USE_V1=0

# GNN device - use the first GPU from CUDA_VISIBLE_DEVICES for GNN inference
export GNN_CUDA_DEVICE=6

# CRITICAL: Prevent Ray from clearing CUDA_VISIBLE_DEVICES for workers with num_gpus=0
# This allows GNN reward function to access GPU even without explicit GPU allocation
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# ==================== [CRASH PREVENTION] ====================
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TORCH_DIST_INIT_BARRIER=0

# Use different port to avoid conflicts
export MASTER_PORT=44105

# GNN checkpoint override: use dot product model
export GNN_CHECKPOINT_PATH="$PROJECT_ROOT/outputs/gnn_standard_domains/qwen3b/unified_gnn_qwen3b_hard_dotproduct_train20.pt"
echo "GNN checkpoint: $GNN_CHECKPOINT_PATH"

# WandB config
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_PROJECT=verl_grpo_reward_comparison
export WANDB_NAME=generalization_20gt_80gnn_dot_product
export WANDB_INIT_TIMEOUT=300
export WANDB__SERVICE_WAIT=300

# Data and model paths
DATA_DIR="$PROJECT_ROOT/data/generalization/verl_train"
MODEL_PATH="$PROJECT_ROOT/llm/qwen2.5_3b_instruct"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen2.5-3b/verl_grpo_generalization_20gt_80gnn_dot_product"

# Auto-fix and verify is_train fields before training
echo "============================================================"
echo "Step 1: Fixing is_train fields in training/validation data..."
echo "============================================================"
python3 "$PROJECT_ROOT/scripts/Step5_verl_training/utils/fix_validation_is_train.py" --dataset qwen2.5_3b_generalization --no-backup
if [ $? -ne 0 ]; then
    echo "❌ Failed to fix is_train fields. Please check the error above."
    exit 1
fi
echo "✅ is_train fields fixed"
echo ""

echo "============================================================"
echo "Step 2: Verifying is_train fields consistency..."
echo "============================================================"
python3 "$PROJECT_ROOT/scripts/Step5_verl_training/utils/verify_is_train_fields.py" 2>&1 | grep -A 20 "Generalization"
VERIFY_EXIT_CODE=${PIPESTATUS[0]}
if [ $VERIFY_EXIT_CODE -ne 0 ]; then
    echo "❌ Verification failed. Please check the error above."
    exit 1
fi
echo "✅ Verification passed - all is_train fields are correct"
echo ""

# Fix reward_model format (JSON string -> dict)
echo "============================================================"
echo "Step 3: Fixing reward_model format..."
echo "============================================================"
python3 "$PROJECT_ROOT/scripts/Step5_verl_training/utils/fix_reward_model_format.py" "$DATA_DIR/train.parquet"
python3 "$PROJECT_ROOT/scripts/Step5_verl_training/utils/fix_reward_model_format.py" "$DATA_DIR/valid.parquet"
echo "✅ reward_model format fixed"
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/valid.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    custom_reward_function.path=$PROJECT_ROOT/src/reward_graph/rewards/verl_mixed_reward_qwen3b.py \
    custom_reward_function.name=compute_score \
    reward_model.use_reward_loop=False \
    reward_model.reward_manager=batch \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_reward_comparison' \
    trainer.experiment_name='generalization_20gt_80gnn_dot_product' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=41 \
    trainer.test_freq=41 \
    trainer.total_training_steps=410 \
    trainer.default_local_dir=$OUTPUT_DIR $@
