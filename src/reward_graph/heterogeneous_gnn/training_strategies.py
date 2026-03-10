'''
GNN training strategies with BCE + ranking loss.
Implements training loop with early stopping on F1 or ROC-AUC.
Related: graph_builders.py graph construction.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score


def compute_ranking_loss(
    logits: torch.Tensor,
    rewards: torch.Tensor,
    num_responses_per_query: int = 8,
    margin: float = 0.5,
) -> torch.Tensor:
    '''
    Definition: Pairwise ranking loss for response ordering.
    Purpose: Enforces positive responses score higher than negative ones per query.
    Related: train_with_ranking_loss() combined loss.
    '''
    device = logits.device
    total_loss = torch.tensor(0.0, device=device)
    count = 0

    N_resp = logits.shape[0]
    N_query = N_resp // num_responses_per_query

    for q_idx in range(N_query):
        start = q_idx * num_responses_per_query
        end = start + num_responses_per_query

        q_logits = logits[start:end]
        q_rewards = rewards[start:end]

        pos_mask = q_rewards > 0.5
        neg_mask = q_rewards <= 0.5

        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_logits = q_logits[pos_mask]
            neg_logits = q_logits[neg_mask]

            for p in pos_logits:
                for n in neg_logits:
                    loss = F.relu(margin - (p - n))
                    total_loss = total_loss + loss
                    count += 1

    if count > 0:
        return total_loss / count
    return torch.tensor(0.0, device=device, requires_grad=True)


def compute_precision(preds: torch.Tensor, targets: torch.Tensor) -> float:
    '''
    Definition: Precision metric calculator (TP / (TP + FP)).
    Purpose: Computes precision from binary predictions and targets.
    Related: evaluate_domain() detailed metrics.
    '''
    TP = ((preds == 1) & (targets == 1)).sum().item()
    FP = ((preds == 1) & (targets == 0)).sum().item()
    if (TP + FP) > 0:
        return TP / (TP + FP)
    return 0.0


def compute_roc_auc(probs: torch.Tensor, targets: torch.Tensor) -> float:
    '''
    Definition: ROC-AUC score calculator.
    Purpose: Computes area under ROC curve from probabilities and targets.
    Related: train_with_ranking_loss() early stopping metric.
    '''
    probs_np = probs.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Need both classes present for ROC AUC
    if len(set(targets_np)) < 2:
        return 0.5

    try:
        return roc_auc_score(targets_np, probs_np)
    except Exception:
        return 0.5


def compute_f1(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    '''
    Definition: F1 score calculator from probabilities.
    Purpose: Computes F1 from thresholded predictions and targets.
    Related: train_with_ranking_loss() early stopping metric.
    '''
    preds = (probs > threshold).float()
    TP = ((preds == 1) & (targets == 1)).sum().item()
    FP = ((preds == 1) & (targets == 0)).sum().item()
    FN = ((preds == 0) & (targets == 1)).sum().item()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def train_with_ranking_loss(
    model: nn.Module,
    data,
    rewards_tensor: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    num_responses_per_query: int = 8,
    num_epochs: int = 150,
    patience: int = 20,
    learning_rate: float = 1e-3,
    ranking_weight: float = 0.5,
    early_stop_metric: str = 'f1',
    device: str = 'cuda',
    use_class_weight: bool = True,
) -> Tuple[nn.Module, Dict]:
    '''
    Definition: GNN trainer with BCE + ranking loss and early stopping.
    Purpose: Trains model, tracks best validation metric, restores best weights.
    Related: evaluate_domain() post-training evaluation.
    '''
    model = model.to(device)
    data = data.to(device)
    rewards_tensor = rewards_tensor.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if use_class_weight:
        train_labels = rewards_tensor[train_mask]
        pos_count = (train_labels > 0.5).sum().float()
        neg_count = (train_labels <= 0.5).sum().float()
        if pos_count > 0 and neg_count > 0:
            # pos_weight = neg_count / pos_count makes the loss treat
            # positive and negative samples equally important
            pos_weight = neg_count / pos_count
            bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            bce_criterion = nn.BCEWithLogitsLoss()
    else:
        bce_criterion = nn.BCEWithLogitsLoss()

    best_val_metric = 0.0
    best_state = None
    best_epoch = 0
    no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(data).squeeze(-1)

        bce_loss = bce_criterion(logits[train_mask], rewards_tensor[train_mask])

        rank_loss = compute_ranking_loss(
            logits[train_mask],
            rewards_tensor[train_mask],
            num_responses_per_query,
            margin=0.5
        )

        loss = bce_loss + ranking_weight * rank_loss

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(data).squeeze(-1)
            val_probs = torch.sigmoid(val_logits)

            if early_stop_metric == 'f1':
                val_metric = compute_f1(val_probs[val_mask], rewards_tensor[val_mask])
            elif early_stop_metric == 'roc_auc':
                val_metric = compute_roc_auc(val_probs[val_mask], rewards_tensor[val_mask])
            elif early_stop_metric == 'precision':
                preds = (val_probs[val_mask] > 0.5).float()
                val_metric = compute_precision(preds, rewards_tensor[val_mask])
            else:
                val_metric = compute_f1(val_probs[val_mask], rewards_tensor[val_mask])

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    training_info = {
        'best_val_metric': best_val_metric,
        'best_val_auc': best_val_metric if early_stop_metric == 'roc_auc' else 0.0,
        'early_stop_metric': early_stop_metric,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1
    }

    return model, training_info


def evaluate_domain(
    model: nn.Module,
    data,
    rewards_tensor: torch.Tensor,
    val_mask: torch.Tensor,
    device: str = 'cuda'
) -> Dict:
    '''
    Definition: GNN evaluator with detailed classification metrics.
    Purpose: Computes precision, recall, F1, ROC-AUC, FP/FN rates on validation set.
    Related: train_with_ranking_loss() training loop.
    '''
    model.eval()
    data = data.to(device)
    rewards_tensor = rewards_tensor.to(device)

    with torch.no_grad():
        logits = model(data).squeeze(-1)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    probs_val = probs[val_mask].cpu()
    preds_val = preds[val_mask].cpu()
    rewards = rewards_tensor[val_mask].cpu()

    # CRITICAL: Use rewards > 0.5 as positive class (match training threshold)
    # This handles partial credit rewards (0.3, 0.33, 0.67, etc.)
    # reward > 0.5 = positive (1.0, 0.67), reward <= 0.5 = negative (0.0, 0.3, 0.33)
    targets = (rewards > 0.5).float()

    TP = ((preds_val == 1) & (targets == 1)).sum().item()
    TN = ((preds_val == 0) & (targets == 0)).sum().item()
    FP = ((preds_val == 1) & (targets == 0)).sum().item()
    FN = ((preds_val == 0) & (targets == 1)).sum().item()

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    positive_rate = rewards.mean().item()
    improvement = accuracy - positive_rate

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    fp_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    fn_rate = FN / (FN + TP) if (FN + TP) > 0 else 0

    roc_auc = compute_roc_auc(probs_val, rewards)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fp_rate': fp_rate,
        'fn_rate': fn_rate,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'positive_rate': positive_rate,
        'improvement': improvement,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }
