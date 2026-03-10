'''
Generate LLM responses and embeddings with vLLM batch inference.
Produces 8 responses per query with query/think/answer embeddings for GNN.
Related: train_gnn_from_cache*.py for GNN training on cached responses.

Note: num_responses=8 must match mixed_gnn_reward_batch_*.py and VERL rollout.n.
'''

import os
import sys
import json
import random
import logging
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

SEED = 42

DEFAULT_NUM_RESPONSES = 8


def set_seed(seed: int = SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# Output directories based on domain type
OUTPUT_DIR_STANDARD = BASE_DIR / "outputs/gnn_standard_domains"  # 10 standard datasets
OUTPUT_DIR_GENERALIZATION = BASE_DIR / "outputs/gnn_generalization"  # Generalization test datasets

# Dataset names for auto-detection
GENERALIZATION_DATASETS = {'numina_math', 'siqa', 'piqa',
                           'numina_math_valid', 'siqa_valid', 'piqa_valid'}

# Dataset paths mapping - 10 standard GNN datasets from sampled_1500
# train_20: 150 queries (small datasets use all available)
# valid: 300 queries (small datasets use all available)
SAMPLED_DIR = DATA_DIR / "sampled_1500"           # 10 standard datasets (math/qa/code)
SAMPLED_DIR_GENERALIZATION = DATA_DIR / "generalization"  # 4 generalization test datasets

DATASET_PATHS = {
    # Math domain (3 datasets)
    'gsm8k_train': SAMPLED_DIR / "gsm8k_sampled_train_20.parquet",
    'gsm8k_valid': SAMPLED_DIR / "gsm8k_sampled_valid.parquet",
    'math_train': SAMPLED_DIR / "math_sampled_train_20.parquet",
    'math_valid': SAMPLED_DIR / "math_sampled_valid.parquet",
    'gsm_symbolic_train': SAMPLED_DIR / "gsm_symbolic_sampled_train_20.parquet",
    'gsm_symbolic_valid': SAMPLED_DIR / "gsm_symbolic_sampled_valid.parquet",
    # QA domain (5 datasets)
    'mmlu_train': SAMPLED_DIR / "mmlu_sampled_train_20.parquet",
    'mmlu_valid': SAMPLED_DIR / "mmlu_sampled_valid.parquet",
    'commonsenseqa_train': SAMPLED_DIR / "commonsenseqa_sampled_train_20.parquet",
    'commonsenseqa_valid': SAMPLED_DIR / "commonsenseqa_sampled_valid.parquet",
    'obqa_train': SAMPLED_DIR / "obqa_sampled_train_20.parquet",
    'obqa_valid': SAMPLED_DIR / "obqa_sampled_valid.parquet",
    'arc_c_train': SAMPLED_DIR / "arc_c_sampled_train_20.parquet",
    'arc_c_valid': SAMPLED_DIR / "arc_c_sampled_valid.parquet",
    'gpqa_train': SAMPLED_DIR / "gpqa_sampled_train_20.parquet",
    'gpqa_valid': SAMPLED_DIR / "gpqa_sampled_valid.parquet",
    # Code domain (2 datasets)
    'humaneval_plus_train': SAMPLED_DIR / "humaneval_plus_sampled_train_20.parquet",
    'humaneval_plus_valid': SAMPLED_DIR / "humaneval_plus_sampled_valid.parquet",
    'mbpp_plus_train': SAMPLED_DIR / "mbpp_plus_sampled_train_20.parquet",
    'mbpp_plus_valid': SAMPLED_DIR / "mbpp_plus_sampled_valid.parquet",
    # Generalization test datasets (3 datasets) - for GNN generalization testing only
    'numina_math_valid': SAMPLED_DIR_GENERALIZATION / "numina_math_sampled_valid.parquet",
    'siqa_valid': SAMPLED_DIR_GENERALIZATION / "siqa_sampled_valid.parquet",
    'piqa_valid': SAMPLED_DIR_GENERALIZATION / "piqa_sampled_valid.parquet",
}

MERGED_DATASETS = {}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_system_prompt(domain: str) -> str:
    """Get domain-specific system prompt."""
    if domain in ['apps', 'humaneval', 'mbpp', 'code']:
        return "You are a Python programmer. Write clean, working code. Put your final code in a python code block."
    elif domain in ['math', 'gsm8k']:
        return ("You are a helpful assistant that solves problems step by step.\n"
                "CRITICAL: End your response with #### followed by just the final answer value.")
    else:
        return ("You are a helpful assistant. Think step by step and end your response with "
                "#### followed by your final answer.")


def build_prompt(row, domain: str):
    """Build prompt from parquet row based on domain."""
    # Check if prompt column exists and is not empty (VERL format)
    if 'prompt' in row and row['prompt'] is not None:
        prompt_messages = row['prompt']
        if isinstance(prompt_messages, str):
            prompt_messages = json.loads(prompt_messages)
        # Convert numpy array to list if needed (fix for parquet numpy array bug)
        if isinstance(prompt_messages, np.ndarray):
            prompt_messages = prompt_messages.tolist()
        # Only return if prompt is not empty (fix for musique empty prompt bug)
        if isinstance(prompt_messages, list) and len(prompt_messages) > 0:
            return prompt_messages

    # Build prompt from question/problem field
    question = row.get('question', row.get('problem', ''))
    if not question and 'extra_info' in row:
        extra = row['extra_info']
        if isinstance(extra, str):
            extra = json.loads(extra)
        question = extra.get('question', extra.get('problem', ''))

    system_prompt = get_system_prompt(domain)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]


def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable types to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def get_extra_info(row, domain: str) -> dict:
    """Extract extra_info from parquet row."""
    if 'extra_info' in row and row['extra_info'] is not None:
        extra = row['extra_info']
        if isinstance(extra, str):
            extra = json.loads(extra)
        return convert_to_serializable(extra)

    # Build from available columns
    extra = {}
    if 'answer' in row:
        extra['answer'] = convert_to_serializable(row['answer'])
    if 'answerKey' in row:
        extra['answer'] = row['answerKey']
    if 'input_output' in row:
        extra['test_list'] = convert_to_serializable(row['input_output'])
    return extra


def split_think_and_answer(response: str, domain: str) -> tuple:
    """Split response into think and answer parts based on domain.

    Supports multiple formats (priority order):
    1. <think>...</think> tags (VERL-style)
    2. ```python blocks (code domain)
    3. #### delimiter (math/qa domain)
    4. Fallback: split in half
    """
    import re

    if '<think>' in response and '</think>' in response:
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            think_part = think_match.group(1).strip()
            # Everything after </think> is the answer
            answer_start = response.find('</think>') + len('</think>')
            answer_part = response[answer_start:].strip()
            # If no content after </think>, use a placeholder
            if not answer_part:
                answer_part = "No explicit answer provided"
            return think_part, answer_part

    # Priority 2: Code domain - split by ```python block
    if domain in ['apps', 'humaneval', 'mbpp', 'code']:
        if '```python' in response:
            parts = response.split('```python')
            think_part = parts[0].strip()
            answer_part = '```python' + parts[1] if len(parts) > 1 else response
            return think_part, answer_part

    # Priority 3: Math/QA domain - split by #### delimiter
    else:
        if '####' in response:
            parts = response.split('####')
            think_part = parts[0].strip()
            answer_part = '####' + parts[1] if len(parts) > 1 else ''
            return think_part, answer_part

    # Fallback: split in half (only for malformed responses)
    mid = len(response) // 2
    return response[:mid], response[mid:]


def generate_responses_vllm(
    data_path,  # Path or None
    domain: str,
    llm,
    sampling_params,
    embed_model,
    tokenizer,
    num_responses: int = 8,
    dataframe=None,  # Optional: pass dataframe directly (for merged datasets)
    no_morgan: bool = False,  # NEW: disable Morgan Fingerprints for baseline
):
    """
    Generate responses using vLLM (batch generation for speed).

    IMPORTANT: Embeddings are stored with correct 1:N ratio:
        - query_embeddings: [N, dim] - one per query (unique)
        - think_embeddings: [N*8, dim] - 8 per query (flattened)
        - answer_embeddings: [N*8, dim] - 8 per query (flattened)
    """
    import pandas as pd

    logger.info(f"\n{'='*60}")
    logger.info(f"Generating responses for {domain.upper()} (vLLM accelerated)")
    logger.info(f"Data: {data_path if data_path else 'merged dataframe'}")
    logger.info(f"Temperature: {sampling_params.temperature}, Responses per query: {num_responses}")
    logger.info(f"{'='*60}")

    if dataframe is not None:
        df = dataframe
    else:
        df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} queries")

    # Prepare all prompts
    all_prompts = []
    all_extra_info = []
    query_texts = []

    for idx, row in df.iterrows():
        prompt_messages = build_prompt(row, domain)
        extra_info = get_extra_info(row, domain)

        # Format prompt for vLLM
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Each query needs num_responses generations
        for _ in range(num_responses):
            all_prompts.append(prompt_text)

        all_extra_info.append(extra_info)
        query_text = prompt_messages[-1]['content'] if isinstance(prompt_messages, list) else str(prompt_messages)
        query_texts.append(query_text)

    logger.info(f"Total prompts to generate: {len(all_prompts)} ({len(df)} queries × {num_responses} responses)")

    # Batch generate with vLLM
    logger.info("Generating with vLLM (this may take a few minutes)...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Collect responses
    all_responses = []
    all_query_emb = []
    all_think_emb = []
    all_answer_emb = []
    domains_list = []

    for q_idx in tqdm(range(len(df)), desc="Processing embeddings"):
        responses = []
        for r_idx in range(num_responses):
            output_idx = q_idx * num_responses + r_idx
            response_text = outputs[output_idx].outputs[0].text
            responses.append(response_text)

        # Query embedding: ONE per query
        # Use query text for embedding
        query_text_to_encode = query_texts[q_idx]

        if domain in ['dti', 'TDC']:
            # Extract core content: SMILES and/or Protein Sequence
            import re
            smiles_match = re.search(r'SMILES:\s*([^\n]+)', query_text_to_encode)
            seq_match = re.search(r'(?:Protein )?Sequence:\s*([^\n]+)', query_text_to_encode)

            # Use RDKit Morgan fingerprint for SMILES (molecular structure)
            # Skip if --no-morgan flag is set (for baseline comparison)
            if smiles_match and domain == 'TDC' and not no_morgan:
                try:
                    from rdkit import Chem
                    from rdkit.Chem import AllChem
                    import numpy as np

                    smiles = smiles_match.group(1).strip()
                    mol = Chem.MolFromSmiles(smiles)

                    if mol:
                        # Morgan fingerprint (1024-bit)
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                        fp_array = np.array(fp, dtype=np.float32)
                        query_emb = torch.tensor(fp_array, dtype=torch.float32)
                    else:
                        # Fallback to text embedding if SMILES invalid
                        logger.warning(f"Invalid SMILES at query {q_idx}, using text embedding")
                        query_emb = embed_model.encode(smiles, convert_to_tensor=True)
                except Exception as e:
                    logger.warning(f"RDKit encoding failed for query {q_idx}: {e}, using text embedding")
                    query_emb = embed_model.encode(smiles_match.group(1), convert_to_tensor=True)
            else:
                # DTI or fallback: use text embedding with core content only
                core_parts = []
                if smiles_match:
                    core_parts.append(f"SMILES: {smiles_match.group(1)}")
                if seq_match:
                    core_parts.append(f"Sequence: {seq_match.group(1)}")

                if core_parts:
                    query_text_to_encode = " | ".join(core_parts)

                query_emb = embed_model.encode(query_text_to_encode[:1000], convert_to_tensor=True)
        else:
            # Standard datasets: use text embedding
            query_emb = embed_model.encode(query_text_to_encode[:1000], convert_to_tensor=True)

        all_query_emb.append(query_emb)
        domains_list.append(domain)

        # Response embeddings: N per query
        for response in responses:
            think_part, answer_part = split_think_and_answer(response, domain)
            think_emb = embed_model.encode(think_part[:1000], convert_to_tensor=True)
            answer_emb = embed_model.encode(answer_part[:500], convert_to_tensor=True)
            all_think_emb.append(think_emb)
            all_answer_emb.append(answer_emb)

        all_responses.append({
            'query_idx': q_idx,
            'domain': domain,
            'responses': responses,
            'extra_info': all_extra_info[q_idx]
        })

    # Stack embeddings
    query_emb_tensor = torch.stack(all_query_emb)
    think_emb_tensor = torch.stack(all_think_emb)
    answer_emb_tensor = torch.stack(all_answer_emb)

    # Verify correct ratio
    ratio = len(all_think_emb) // len(all_query_emb)
    logger.info(f"Generated {len(all_responses)} queries × {num_responses} responses")
    logger.info(f"Embeddings: query={query_emb_tensor.shape}, think={think_emb_tensor.shape}")
    logger.info(f"Ratio: 1:{ratio} {'✓' if ratio == num_responses else '✗ ERROR!'}")

    if ratio != num_responses:
        raise ValueError(f"Embedding ratio is 1:{ratio}, expected 1:{num_responses}. Bug in code!")

    return all_responses, query_emb_tensor, think_emb_tensor, answer_emb_tensor, domains_list


def list_datasets():
    """List all available datasets."""
    print("\n" + "=" * 60)
    print("Available Datasets")
    print("=" * 60)
    for name, path in sorted(DATASET_PATHS.items()):
        exists = "✓" if path.exists() else "✗"
        print(f"  {exists} {name:20s} -> {path}")
    print("\n  Merged Datasets (dynamically created):")
    for name, sources in MERGED_DATASETS.items():
        all_exist = all(DATASET_PATHS[s].exists() for s in sources)
        exists = "✓" if all_exist else "✗"
        print(f"  {exists} {name:20s} -> {' + '.join(sources)}")
    print("\nUsage: python scripts/Step3_gnn_verl_data_preparation/generate_response/generate_responses.py --dataset <name>")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate responses and embeddings with vLLM acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--dataset', type=str, help='Dataset name (see --list)')
    parser.add_argument('--data_path', type=str, help='Direct path to parquet file')
    parser.add_argument('--output_name', type=str, help='Output cache name (default: cache_{dataset})')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature (default: 1.0, use 1.0 for Qwen to avoid encoding issues)')
    parser.add_argument('--num_responses', type=int, default=DEFAULT_NUM_RESPONSES,
                        help=f'Responses per query (default: {DEFAULT_NUM_RESPONSES}). '
                             'MUST match mixed_gnn_reward_batch_qwen3b.py and GNN training!')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Max tokens per response')
    parser.add_argument('--force', action='store_true', help='Overwrite existing cache')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--model', type=str, default=None,
                        help='Model path (overrides --model-type if specified)')
    parser.add_argument('--model-type', type=str, choices=['qwen3b', 'qwen1.5b'], default='qwen3b',
                        help='Model type: qwen3b (Qwen2.5-3B) or qwen1.5b (Qwen2.5-1.5B) (default: qwen3b). Uses local llm/ directory.')
    parser.add_argument('--split', type=str, choices=['train', 'valid'],
                        help='Split name for output files (e.g., responses_train.json)')
    parser.add_argument('--domain-type', type=str, choices=['standard', 'generalization', 'auto'], default='auto',
                        help='Domain type: standard (10 datasets), generalization (numina_math/apps_intro/siqa/piqa), or auto-detect (default: auto)')
    parser.add_argument('--no-morgan', action='store_true',
                        help='Disable Morgan Fingerprints for TDC (use text embeddings instead, for baseline comparison)')
    args = parser.parse_args()

    # Model path configuration - use local llm/ directory by default
    MODEL_PATHS = {
        'qwen3b': BASE_DIR / "llm/qwen2.5_3b_instruct",
        'qwen1.5b': BASE_DIR / "llm/qwen2.5_1.5b_instruct",
    }

    # Determine model path: explicit --model overrides --model-type
    if args.model:
        model_path = args.model
    else:
        model_path = str(MODEL_PATHS[args.model_type])

    # Store model_type for cache prefix
    model_type = args.model_type

    if args.list:
        list_datasets()
        return

    if not args.dataset and not args.data_path:
        parser.print_help()
        print("\nError: Please specify --dataset or --data_path")
        return

    set_seed(SEED)

    # Determine data path and domain
    import pandas as pd
    merged_df = None

    if args.data_path:
        data_path = Path(args.data_path)
        domain = args.output_name or data_path.stem
    elif args.dataset in MERGED_DATASETS:
        # Handle merged datasets (e.g., 'code' = humaneval + mbpp)
        sub_datasets = MERGED_DATASETS[args.dataset]
        dfs = []
        for sub in sub_datasets:
            sub_path = DATASET_PATHS[sub]
            if not sub_path.exists():
                print(f"Error: Sub-dataset file not found: {sub_path}")
                return
            dfs.append(pd.read_parquet(sub_path))
        merged_df = pd.concat(dfs, ignore_index=True)
        domain = args.dataset
        data_path = None  # Will use merged_df instead
        logger.info(f"Merged {sub_datasets} -> {len(merged_df)} samples")
    elif args.dataset in DATASET_PATHS:
        data_path = DATASET_PATHS[args.dataset]
        domain = args.dataset
    else:
        print(f"Error: Unknown dataset '{args.dataset}'")
        list_datasets()
        return

    if data_path is not None and not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    # Determine output name (include model type prefix)
    base_output_name = args.output_name or f"cache_{domain}"
    output_name = f"{model_type}_{base_output_name}"

    # Determine domain type and output directory
    domain_type = args.domain_type
    if domain_type == 'auto':
        # Auto-detect based on dataset name
        base_dataset = domain.split('_')[0]  # Remove _train/_valid suffix
        # Also check full domain name for generalization datasets like numina_math_valid
        if base_dataset in GENERALIZATION_DATASETS or domain in GENERALIZATION_DATASETS:
            domain_type = 'generalization'
        else:
            domain_type = 'standard'
        logger.info(f"Auto-detected domain-type: {domain_type}")

    # Set output directory based on domain type
    if domain_type == 'generalization':
        output_dir = OUTPUT_DIR_GENERALIZATION
    else:
        output_dir = OUTPUT_DIR_STANDARD / model_type
    logger.info(f"Output directory: {output_dir}")

    # Set default temperature based on model type
    if args.temperature is None:
        if model_type == 'qwen1.5b':
            args.temperature = 1.3  # Qwen 1.5B default temperature for Math/Code
        else:
            args.temperature = 1.5  # Qwen 3B default temperature
        logger.info(f"Using default temperature for {model_type}: {args.temperature}")

    # Temperature experiment: testing temp=1.5 for all datasets with qwen1.5b
    if model_type == 'qwen1.5b':
        args.temperature = 1.5
        logger.info(f"Override temperature for {domain} with qwen1.5b: {args.temperature}")

    cache_dir = output_dir / output_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check if already exists (with split suffix if specified)
    split_suffix = f"_{args.split}" if args.split else ""
    resp_file = f"responses{split_suffix}.json"
    emb_file = f"embeddings{split_suffix}.pt"
    if not args.force and (cache_dir / resp_file).exists() and (cache_dir / emb_file).exists():
        logger.info(f"Cache already exists at {cache_dir}/{resp_file}")
        logger.info("Use --force to overwrite")
        return

    # Load vLLM
    logger.info(f"Loading {model_path} with vLLM...")
    logger.info(f"Model type: {model_type}")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from huggingface_hub import login

    # Login to HuggingFace for gated models
    HF_TOKEN = ""
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        logger.warning(f"HF login failed (may not be needed for local models): {e}")

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype='bfloat16',
        gpu_memory_utilization=0.4,
        enforce_eager=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=0.95,
    )

    logger.info("Loading embedding model...")
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate
    responses, query_emb, think_emb, answer_emb, domains = generate_responses_vllm(
        data_path=data_path,
        domain=domain.replace('_train', '').replace('_valid', ''),  # Remove only train/valid suffixes
        llm=llm,
        sampling_params=sampling_params,
        embed_model=embed_model,
        tokenizer=tokenizer,
        num_responses=args.num_responses,
        dataframe=merged_df,  # Pass merged dataframe if available
        no_morgan=args.no_morgan,  # Pass no_morgan flag
    )

    # Save (split_suffix, resp_file, emb_file already defined above)
    with open(cache_dir / resp_file, 'w') as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

    torch.save({
        'query_embeddings': query_emb.cpu(),
        'think_embeddings': think_emb.cpu(),
        'answer_embeddings': answer_emb.cpu(),
        'domains': domains,
    }, cache_dir / emb_file)

    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Saved to {cache_dir}")
    logger.info(f"  - {resp_file}: {len(responses)} queries")
    logger.info(f"  - {emb_file}: query={query_emb.shape}, think={think_emb.shape}")
    logger.info(f"  - Ratio: 1:{think_emb.shape[0] // query_emb.shape[0]}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
