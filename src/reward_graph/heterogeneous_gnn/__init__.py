'''
Cross-domain GNN module for graph construction and training.
Provides graph builders, ranking loss training, and evaluation metrics.
Related: graph_builders.py, training_strategies.py.
'''

from .graph_builders import (
    build_independent_domain_graph,
    compute_similarity_matrix
)
from .training_strategies import (
    train_with_ranking_loss,
    evaluate_domain,
    compute_precision,
    compute_ranking_loss,
    compute_roc_auc
)

__all__ = [
    'build_independent_domain_graph',
    'compute_similarity_matrix',
    'train_with_ranking_loss',
    'evaluate_domain',
    'compute_precision',
    'compute_ranking_loss',
    'compute_roc_auc',
]
