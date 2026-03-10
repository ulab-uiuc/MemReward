# MemReward - Environment Setup

## System Requirements

- Python 3.12.7
- CUDA 12.8 (CRITICAL: NOT cu126 or other versions)
- GPU: NVIDIA RTX A6000 or compatible (Ampere architecture)
- Linux OS (tested on Ubuntu 22.04)

## Installation Steps

1. **Create virtual environment:**
   ```bash
   python3.12 -m venv /path/to/venv
   source /path/to/venv/bin/activate
   ```

2. **Install PyTorch 2.9.0 with CUDA 12.8:**
   ```bash
   pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
       --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Install VERL from source:**
   ```bash
   cd /tmp
   git clone https://github.com/volcengine/verl.git
   cd verl
   git checkout 3b1c139607f377f599b60792fa51a54d7bc42897
   pip install -e .
   ```

4. **Install remaining packages:**
   ```bash
   pip install -r environment_installation/requirements.txt
   ```

5. **Install the project package:**
   ```bash
   cd src && pip install -e . && cd ..
   ```

6. **Verify installation:**
   ```bash
   python -c "import torch, verl, vllm; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"
   ```

## Important Version Notes

- **PyTorch 2.9.0+cu128 is REQUIRED** — Do NOT use cu126 or other CUDA versions
- **vLLM 0.13.0** is the tested version — ensure compatibility with PyTorch 2.9.0
- **Accelerate 1.12.0** is required for VERL distributed training
- **VERL** must be installed from the specific git commit for batch reward manager support
