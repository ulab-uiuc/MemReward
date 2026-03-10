'''
GNN model classes for reward prediction.
Provides UnifiedGNNDotProduct with answer features.
Related: mixed_gnn_reward_base.py for inference wrapper.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear


EDGE_TYPES = [
    ('query', 'generates_reasoning', 'think'),
    ('think', 'rev_generates_reasoning', 'query'),
    ('think', 'leads_to', 'answer'),
    ('answer', 'rev_leads_to', 'think'),
    ('query', 'similar_to', 'query'),
    ('think', 'competes_with', 'think'),
]

SELF_LOOP_EDGES = {('query', 'similar_to', 'query')}


def build_hetero_convs(hidden_dim, num_layers, num_heads, dropout):
    '''
    Shared HeteroConv layer builder for all GNN variants.
    Constructs GATv2Conv layers for 6 edge types.
    Related: UnifiedGNNDotProduct.
    '''
    convs = nn.ModuleList()
    for _ in range(num_layers):
        conv_dict = {}
        for edge_type in EDGE_TYPES:
            conv_dict[edge_type] = GATv2Conv(
                hidden_dim, hidden_dim // num_heads, heads=num_heads,
                dropout=dropout, add_self_loops=(edge_type in SELF_LOOP_EDGES),
            )
        convs.append(HeteroConv(conv_dict, aggr='mean'))
    return convs


def run_message_passing(convs, x_dict, data):
    '''
    Shared message passing loop for all GNN variants.
    Filters edge types present in data, runs conv + ReLU.
    Related: build_hetero_convs() layer construction.
    '''
    for conv in convs:
        filtered_edge_index = {}
        for edge_type in conv.convs.keys():
            if edge_type in data.edge_index_dict:
                filtered_edge_index[edge_type] = data.edge_index_dict[edge_type]
        if filtered_edge_index:
            x_dict = conv(x_dict, filtered_edge_index)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
    return x_dict


def extract_qta_embeddings(x_dict, data):
    '''
    Extract query/think/answer embeddings aligned by generates edge.
    Returns (query_emb, think_emb, answer_emb) indexed by edge pairs.
    Related: UnifiedGNNDotProduct.forward().
    '''
    edge_index = data[('query', 'generates', 'answer')].edge_index
    query_emb = x_dict['query'][edge_index[0]]
    answer_emb = x_dict['answer'][edge_index[1]]
    think_emb = x_dict['think'][edge_index[1]]
    return query_emb, think_emb, answer_emb


class UnifiedGNNDotProduct(nn.Module):
    '''
    HeteroGNN with GATv2Conv and dot-product scorer.
    Scores query-response pairs via scaled dot product with answer features.
    Related: mixed_gnn_reward_base.py for inference wrapper.
    '''

    def __init__(
        self,
        query_dim: int = 384,
        think_dim: int = 384,
        answer_dim: int = 384,
        answer_feat_dim: int = 6,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        dot_dim: int = 256,
    ):
        super().__init__()

        self.dot_dim = dot_dim
        self.query_proj = Linear(query_dim, hidden_dim)
        self.think_proj = Linear(think_dim, hidden_dim)
        self.answer_proj = Linear(answer_dim, hidden_dim)

        self.answer_feat_proj = nn.Sequential(
            nn.Linear(answer_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

        self.convs = build_hetero_convs(hidden_dim, num_layers, num_heads, dropout)

        self.query_head = nn.Sequential(
            nn.Linear(hidden_dim, dot_dim),
            nn.LayerNorm(dot_dim),
        )
        self.response_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, dot_dim),
            nn.LayerNorm(dot_dim),
        )
        self.score_bias = nn.Parameter(torch.zeros(1))

    def forward(self, data: HeteroData, answer_features: torch.Tensor) -> torch.Tensor:
        '''
        GNN forward pass with dot-product scoring.
        Runs message passing and computes q*r scaled dot product.
        Related: compute_rewards_batch() inference call.
        '''
        x_dict = {
            'query': self.query_proj(data['query'].x),
            'think': self.think_proj(data['think'].x),
            'answer': self.answer_proj(data['answer'].x),
        }

        x_dict = run_message_passing(self.convs, x_dict, data)

        query_emb, think_emb, answer_emb = extract_qta_embeddings(x_dict, data)

        answer_feat_emb = self.answer_feat_proj(answer_features)
        q = self.query_head(query_emb)
        r = self.response_head(torch.cat([think_emb, answer_emb, answer_feat_emb], dim=-1))
        score = (q * r).sum(dim=-1) / (self.dot_dim ** 0.5) + self.score_bias
        return score
