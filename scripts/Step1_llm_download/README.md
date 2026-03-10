# Step 1: LLM Download

**Command:** `python scripts/Step1_llm_download/download_models.py`
**Function:** Download Qwen2.5-3B-Instruct and Qwen2.5-1.5B-Instruct from HuggingFace. Output: `llm/qwen2.5_3b_instruct/`, `llm/qwen2.5_1.5b_instruct/`.

Set `HF_TOKEN` environment variable for gated model access. Already-downloaded models are automatically skipped.
