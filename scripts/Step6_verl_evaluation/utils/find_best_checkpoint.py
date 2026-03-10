'''
Find best checkpoint by sample-weighted average validation score from training logs.
Supports auto-discovery of log files from training output directories.
Related: merge_and_evaluate_detailed.py for --find_best integration.
'''

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

DATASET_NAMES = [
    'gsm8k', 'mmlu', 'math', 'humaneval_plus', 'mbpp_plus',
    'commonsenseqa', 'obqa', 'arc_c', 'gpqa', 'gsm_symbolic'
]

DATASET_VAL_SIZES = {
    'gsm8k': 300, 'mmlu': 300, 'math': 300,
    'humaneval_plus': 26, 'mbpp_plus': 52,
    'commonsenseqa': 300, 'obqa': 300, 'arc_c': 300,
    'gpqa': 39, 'gsm_symbolic': 300,
}


def extract_validation_metrics(log_file: Path) -> Dict[int, Dict[str, float]]:
    """
    Extract validation metrics from training log file.

    Returns:
        Dict mapping step number to dict of dataset accuracies
        Example: {41: {'gsm8k': 0.85, 'math': 0.72, ...}, ...}
    """
    log_file = Path(log_file)
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")

    content = log_file.read_text()
    metrics_by_step = {}

    val_pattern = r'val-core/(\w+)/(?:acc|reward)/mean@1:([\d.]+)'

    for line in content.split('\n'):
        step_match = re.search(r'step:(\d+)\s', line)
        if not step_match:
            continue

        step_num = int(step_match.group(1))
        val_metrics = {}

        for val_match in re.finditer(val_pattern, line):
            dataset = val_match.group(1)
            score = float(val_match.group(2))
            if dataset == 'gsm_symbolic_main':
                dataset = 'gsm_symbolic'
            if dataset in DATASET_NAMES:
                val_metrics[dataset] = score

        if val_metrics:
            metrics_by_step[step_num] = val_metrics

    return metrics_by_step


def _weighted_avg(metrics: Dict[str, float]) -> float:
    """Compute sample-weighted average across datasets."""
    total_correct = sum(metrics.get(d, 0) * DATASET_VAL_SIZES.get(d, 0) for d in metrics)
    total_samples = sum(DATASET_VAL_SIZES.get(d, 0) for d in metrics)
    return total_correct / total_samples if total_samples > 0 else 0.0


def find_best_step(log_file, verbose: bool = True) -> Tuple[Optional[int], float, Dict]:
    """
    Find the step with highest sample-weighted average validation score.

    Args:
        log_file: Path to training log file
        verbose: Print progress table

    Returns:
        (best_step, best_score, metrics_by_step)
    """
    log_file = Path(log_file)
    metrics_by_step = extract_validation_metrics(log_file)

    if not metrics_by_step:
        if verbose:
            print("WARNING: No validation metrics found in log!")
        return None, 0.0, {}

    best_step = None
    best_score = -1.0

    if verbose:
        short = {'gsm8k': 'gsm8k', 'mmlu': 'mmlu', 'math': 'math',
                 'humaneval_plus': 'heval', 'mbpp_plus': 'mbpp',
                 'commonsenseqa': 'csqa', 'obqa': 'obqa', 'arc_c': 'arc_c',
                 'gpqa': 'gpqa', 'gsm_symbolic': 'gsym'}
        header = f'{"Step":<6} {"WAvg":<7} ' + ' '.join(f'{short[d]:<6}' for d in DATASET_NAMES)
        print(header)
        print('-' * len(header))

    for step in sorted(metrics_by_step.keys()):
        m = metrics_by_step[step]
        avg = _weighted_avg(m)
        if avg > best_score:
            best_score = avg
            best_step = step
        if verbose:
            scores = ' '.join(f'{m.get(d, 0):<6.3f}' for d in DATASET_NAMES)
            marker = ' *' if avg == best_score else ''
            print(f'{step:<6} {avg:<7.4f} {scores}{marker}')

    if verbose:
        print(f'\nBest: Step {best_step}, Weighted Avg: {best_score:.4f} ({best_score * 100:.2f}%)')

    return best_step, best_score, metrics_by_step


def _find_log_file(training_dir: Path) -> Optional[Path]:
    """
    Find the training log file for a given training output directory.

    Search order:
        1. training_dir/training.log (conventional location)
        2. PROJECT_ROOT/logs/ - search for log files whose val metrics
           match the checkpoint steps in training_dir
    """
    log_file = training_dir / 'training.log'
    if log_file.exists():
        return log_file

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    logs_dir = project_root / 'logs'
    if not logs_dir.exists():
        return None

    existing_steps = set()
    for d in training_dir.glob('global_step_*'):
        if d.is_dir():
            try:
                step = int(d.name.split('_')[-1])
                existing_steps.add(step)
            except ValueError:
                pass

    if not existing_steps:
        return None

    best_match = None
    best_match_count = 0

    for candidate in sorted(logs_dir.glob('*.log'), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            metrics = extract_validation_metrics(candidate)
            if not metrics:
                continue
            log_steps = set(metrics.keys())
            overlap = log_steps & existing_steps
            if len(overlap) >= len(existing_steps) * 0.5 and len(overlap) > best_match_count:
                best_match = candidate
                best_match_count = len(overlap)
                if overlap == existing_steps:
                    break
        except Exception:
            continue

    return best_match


def find_best_checkpoint_dir(training_dir, log_file=None) -> Optional[Path]:
    """
    Find the best checkpoint directory from a training output directory.

    Supports two directory layouts:
        - global_step_* dirs (multiple checkpoints saved during training)
        - Single checkpoint/ dir (pre-selected best checkpoint)

    Args:
        training_dir: Path to training output (parent of global_step_* or checkpoint/ dirs)
        log_file: Optional explicit path to training log file.
                  If None, auto-discovers from training_dir/training.log
                  or PROJECT_ROOT/logs/.

    Returns:
        Path to best checkpoint directory, or None
    """
    training_dir = Path(training_dir)

    # If there's a single checkpoint/ dir (pre-selected best checkpoint),
    # return it directly without needing a training log.
    single_ckpt = training_dir / 'checkpoint'
    has_global_steps = any(training_dir.glob('global_step_*'))
    if single_ckpt.is_dir() and not has_global_steps:
        print(f"Found single checkpoint directory: {single_ckpt}")
        return single_ckpt

    if log_file is not None:
        log_file = Path(log_file)
        if not log_file.exists():
            raise FileNotFoundError(f"Specified log file not found: {log_file}")
    else:
        log_file = _find_log_file(training_dir)
        if log_file is None:
            raise FileNotFoundError(
                f"Training log not found. Searched:\n"
                f"  1. {training_dir / 'training.log'}\n"
                f"  2. PROJECT_ROOT/logs/*.log (auto-match by step numbers)\n"
                f"Use --log_file to specify the log path explicitly."
            )

    print(f"Finding best checkpoint from: {log_file}")
    best_step, best_score, _ = find_best_step(log_file)

    if best_step is None:
        return None

    checkpoint_dir = training_dir / f'global_step_{best_step}'
    if not checkpoint_dir.exists():
        print(f"WARNING: Best step {best_step} checkpoint dir not found: {checkpoint_dir}")
        return None

    return checkpoint_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Find best checkpoint from training log validation scores"
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        required=True,
        help="Training output directory containing checkpoint/ or global_step_* dirs"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Explicit path to training log file (auto-discovered if not set)"
    )
    args = parser.parse_args()

    best_dir = find_best_checkpoint_dir(args.training_dir, log_file=args.log_file)
    if best_dir:
        print(f"\nBest checkpoint: {best_dir}")
