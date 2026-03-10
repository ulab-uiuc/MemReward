# MemReward


<!-- Overview Section -->
<h3 align="center">📌 Overview</h3>

<p align="center">
  MemReward is a graph-based experience memory framework for LLM reward prediction with limited labels. It covers 10 standard benchmarks across math (GSM8K, MATH, GSM-Symbolic), QA (MMLU, CommonsenseQA, OBQA, ARC-C, GPQA), and code (HumanEval+, MBPP+), plus 3 generalization domains (NuminaMath, PIQA, SIQA). With only 20% reward labels, MemReward achieves 97.3% of Oracle performance on Qwen-3B and 96.6% on Qwen-1.5B.
</p>

<!-- Method Section -->
<h3 align="center">🧠 Method</h3>

<p align="center">
  An initial LLM policy generates rollouts for each query, each comprising a thinking process and a final answer, and these rollouts are stored as experience memory. Queries, thinking processes, and answers form nodes in a heterogeneous graph with similarity and structural edges; a GNN trained on labeled nodes propagates rewards to unlabeled rollouts during online optimization.
</p>

<p align="center">
  <img src="figure/Architecture.png" width="90%">
</p>


## 📂 Project Structure

```
scripts/
├── Step1_llm_download/              # Download Qwen-3B and 1.5B models
├── Step2_original_data_download/    # Download 13 benchmark datasets
├── Step3_gnn_verl_data_preparation/ # Sample, generate responses, create VERL data
│   ├── sample_1500/                 #   Subsample 1500 queries per dataset
│   ├── generate_response/           #   Generate LLM rollouts with vLLM
│   ├── generate_and_verify_gt_identifier/  #   Create GT/GNN query routing configs
│   └── generate_verl_data/          #   Format data for VERL training (3 modes)
├── Step4_gnn_training_eval/         # Train and evaluate GNN reward models
├── Step5_verl_training/             # GRPO training scripts
│   ├── qwen2.5-3b/                  #   8 standard + 3 generalization configs
│   └── qwen2.5-1.5b/               #   3 standard + 3 generalization configs
└── Step6_verl_evaluation/           # Merge FSDP checkpoints and evaluate

src/reward_graph/
├── rewards/                         # GT and GNN reward functions for VERL
│   └── utils/                       #   GNN model architecture and multi-domain scoring
├── heterogeneous_gnn/               # Heterogeneous graph construction and GNN training strategies
└── utils/                           # Embedding cache management and merging
```


## 📌 Preliminary

### Environment Setup

```shell
# Create virtual environment
python3.12 -m venv /path/to/venv
source /path/to/venv/bin/activate

# Install PyTorch 2.9.0 with CUDA 12.8
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install VERL from source
cd /tmp
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 3b1c139607f377f599b60792fa51a54d7bc42897
pip install -e .

# Install remaining packages
pip install -r environment_installation/requirements.txt

# Install the project package
cd src && pip install -e . && cd ..

# Verify installation
python -c "import torch, verl, vllm; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
```


## 🔄 Reproduce Paper Results

Download the complete project (code, data, and trained checkpoints) directly from [HuggingFace](https://huggingface.co/datasets/ulab-ai/memreward):

### Step 1: Download from HuggingFace

```bash
# Install git-lfs if needed
git lfs install

# Clone the complete repository
git clone https://huggingface.co/datasets/ulab-ai/memreward
cd memreward
```

The repository contains everything needed for reproduction:

| Folder | Contents | Size |
|--------|----------|------|
| `configs/` | GT identifier JSONs for query routing (20%-70% ratios) | 212K |
| `data/` | Sampled datasets, VERL-formatted training data, generalization data | 56M |
| `outputs/` | GNN embeddings + trained VERL checkpoints (Qwen-3B and Qwen-1.5B) | ~93G |
| `scripts/` | Full pipeline scripts (data prep, GNN training, VERL training, evaluation) | — |
| `src/` | Core reward_graph library | — |

### Step 2: Setup Environment and Download LLMs

```bash
# Setup environment (see Preliminary section above)

# Download LLMs
python scripts/Step1_llm_download/download_models.py
```

This downloads `Qwen2.5-3B-Instruct` and `Qwen2.5-1.5B-Instruct` to `llm/`.

### Step 3: Evaluate

```bash
# Evaluate Qwen-3B MemReward (20% GT + 80% GNN) on standard benchmarks
python scripts/Step6_verl_evaluation/merge_and_evaluate_detailed.py \
    --find_best outputs/qwen2.5-3b/verl_grpo_20gt_80gnn_dot_product_hard --gpu 0

# Evaluate on generalization benchmarks
python scripts/Step6_verl_evaluation/merge_and_evaluate_detailed.py \
    --find_best outputs/qwen2.5-3b/verl_grpo_generalization_20gt_80gnn_dot_product \
    --dataset_type generalization --gpu 0
```


## ⭐ Train from Scratch

> **Tip:** We recommend downloading `configs/` and `data/` from [HuggingFace](https://huggingface.co/datasets/ulab-ai/memreward) to ensure consistent data splits and GT routing configurations for stable reproduction.

### Step 1: Download LLMs and Datasets

```shell
# Download LLMs (Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct)
python scripts/Step1_llm_download/download_models.py

# Download all 13 datasets (10 standard + 3 generalization)
bash scripts/Step2_original_data_download/download_all.sh
```

### Step 2: Data Preparation

```shell
# Full data preparation pipeline (sample → responses → GT identifiers → VERL data)
bash scripts/Step3_gnn_verl_data_preparation/run_standard_pipeline.sh --gpus 0,1,2,3
bash scripts/Step3_gnn_verl_data_preparation/run_generalization_pipeline.sh --gpus 0,1,2
```

### Step 3: GNN Training

```bash
bash scripts/Step4_gnn_training_eval/train_gnn_best_of_n_dotproduct.sh \
    --model-type qwen3b --hard-label --gpus 0,1,2,3 --num-runs 40
```

### Step 4: VERL Training

GRPO training scripts are in `scripts/Step5_verl_training/`, organized by model size:

```bash
# Baseline: 100% ground-truth reward
nohup bash scripts/Step5_verl_training/qwen2.5-3b/verl_grpo_100perc_gt.sh \
    > outputs/qwen2.5-3b/verl_grpo_100perc_gt/training.log 2>&1 &

# Sparse baseline: 20% GT only
nohup bash scripts/Step5_verl_training/qwen2.5-3b/verl_grpo_20perc_gt_only.sh \
    > outputs/qwen2.5-3b/verl_grpo_20perc_gt_only/training.log 2>&1 &

# MemReward: 20% GT + 80% GNN
nohup bash scripts/Step5_verl_training/qwen2.5-3b/verl_grpo_20gt_80gnn_dot_product.sh \
    > outputs/qwen2.5-3b/verl_grpo_20gt_80gnn_dot_product_hard/training.log 2>&1 &
```

Additional GT/GNN ratio variants (30/70, 40/60, 50/50, 60/40, 70/30) and generalization scripts are also available. See `scripts/Step5_verl_training/README.md` for the full list.

### Step 5: Evaluation

Merge FSDP checkpoints and evaluate on all test benchmarks:

```bash
# Auto-find best checkpoint, merge, and evaluate
python scripts/Step6_verl_evaluation/merge_and_evaluate_detailed.py \
    --find_best outputs/qwen2.5-3b/verl_grpo_20gt_80gnn_dot_product_hard --gpu 0

# Evaluate on generalization benchmarks
python scripts/Step6_verl_evaluation/merge_and_evaluate_detailed.py \
    --find_best outputs/qwen2.5-3b/verl_grpo_generalization_20gt_80gnn_dot_product \
    --dataset_type generalization --gpu 0
```


## 📝 Acknowledgement

The implementation of **MemReward** is built upon [VERL](https://github.com/volcengine/verl), [vLLM](https://github.com/vllm-project/vllm), [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric), and [Qwen](https://github.com/QwenLM/Qwen2.5).

We sincerely appreciate the efforts of these teams for their contributions to open-source research and development.


## Citation

Coming soon.
