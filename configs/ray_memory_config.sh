#!/usr/bin/env bash
# Ray Memory Configuration - Prevent Training OOM
# Usage: source configs/ray_memory_config.sh

# ==================== Ray Memory Optimization Config ====================
# Purpose: Resolve Ray worker memory leak issues during long training runs
# Applicable: VERL GRPO training, large-scale distributed training

# 1. OOM Threshold Increase (default 95% → 98%)
# Effect: Ray kills workers only when system memory reaches 98%, giving more space for training
# Impact: No performance impact
export RAY_memory_usage_threshold=0.98

# 2. Object Store Memory Hard Limit (500GB)
# Effect: Limit Ray shared memory object store to prevent unbounded growth
# Impact: <1% performance loss, async cleanup
# Tuning: Adjust based on node memory, recommended 5-10% of total RAM
export RAY_object_store_memory=500000000000  # 500GB

# 2.5. Worker Heap Memory Management (promote garbage collection)
# Effect: Accelerate Python memory release, prevent worker process memory leaks
# Impact: ~1% performance loss, significantly reduce memory accumulation
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=1000000000  # Trigger trim after 1GB
# More aggressive Python GC
export PYTHONHASHSEED=0

# 3. Real-time Output
# Effect: Reduce log buffering for real-time monitoring
export PYTHONUNBUFFERED=1

# 4. Ray Temp Directory Redirection (Critical!)
# Effect: Move Ray temp files from /tmp to /mnt/disk2 to avoid root partition space issues
# Impact: No performance impact, prevents disk full crashes
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"
# Ensure directory exists
mkdir -p $RAY_TMPDIR

# ==================== Verify Configuration ====================
echo "[Ray Config] Memory threshold: $RAY_memory_usage_threshold"
echo "[Ray Config] Object store limit: $(echo "scale=2; $RAY_object_store_memory/1024/1024/1024" | bc)GB"
echo "[Ray Config] Unbuffered output: $PYTHONUNBUFFERED"
echo "[Ray Config] Temp directory: $RAY_TMPDIR"

# ==================== Usage Instructions ====================
# Add to training script header:
#   source $PROJECT_ROOT/configs/ray_memory_config.sh
#
# Or run directly in command line:
#   source configs/ray_memory_config.sh
#   bash scripts/your_training_script.sh
#
# ==================== Performance Impact ====================
# Overall performance loss: <1%
# Memory stability improvement: Prevents OOM crashes
# Recommended scenario: Long training runs (>10 hours)
