#!/usr/bin/env python3
'''
Download LLM backbone models from HuggingFace.
Supports Qwen2.5-3B-Instruct, Qwen2.5-1.5B-Instruct.
Related: Step2 scripts for dataset download.
'''

import os
from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

OUTPUT_DIR = str(PROJECT_ROOT / "llm")

MODELS = {
    "qwen2.5_3b_instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5_1.5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for local_name, hf_repo in MODELS.items():
        local_path = os.path.join(OUTPUT_DIR, local_name)
        print(f"\n{'='*60}")
        print(f"Downloading: {hf_repo}")
        print(f"To: {local_path}")
        print(f"{'='*60}")

        if os.path.exists(local_path) and os.listdir(local_path):
            print(f"Already exists, skipping...")
            continue

        try:
            snapshot_download(
                repo_id=hf_repo,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                token=HF_TOKEN,
            )
            print(f"Downloaded successfully!")
        except Exception as e:
            print(f"Error downloading {hf_repo}: {e}")

if __name__ == "__main__":
    main()
