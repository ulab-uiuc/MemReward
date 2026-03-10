'''
Graph construction for independent domain GNN training.
Builds HeteroData graphs with kNN edges per domain.
Related: training_strategies.py GNN training loop.
'''

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from collections import defaultdict
from typing import List, Tuple


def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    '''
    Definition: Cosine similarity matrix calculator.
    Purpose: Computes pairwise cosine similarity for embedding tensor.
    Related: build_independent_domain_graph() kNN edge construction.
    '''
    emb_norm = F.normalize(embeddings, p=2, dim=-1)
    return torch.mm(emb_norm, emb_norm.t())


def build_independent_domain_graph(
    query_emb: torch.Tensor,
    think_emb: torch.Tensor,
    answer_emb: torch.Tensor,
    domains: List[str],
    target_domain: str,
    knn_k: int = 15,
    num_responses_per_query: int = 8,
    think_cross_k: int = 0,  # Disabled by default
    intra_think_k: int = 2,  # Intra-query think-think connections (0=disabled, -1=full, >0=kNN)
    use_aa_edges: bool = True,
) -> Tuple[HeteroData, List[int], List[int]]:
    '''
    Definition: Single-domain HeteroData graph builder.
    Purpose: Filters to target domain and builds query-think-answer edges with kNN.
    Related: training_strategies.py train_with_ranking_loss().
    '''
    domain_query_indices = [i for i, d in enumerate(domains) if d == target_domain]
    domain_query_emb = query_emb[domain_query_indices]

    response_indices = []
    for q_idx in domain_query_indices:
        for r in range(num_responses_per_query):
            response_indices.append(q_idx * num_responses_per_query + r)

    domain_think_emb = think_emb[response_indices]
    domain_answer_emb = answer_emb[response_indices]

    N_query = len(domain_query_indices)
    N_resp = len(response_indices)

    data = HeteroData()
    data['query'].x = domain_query_emb
    data['think'].x = domain_think_emb
    data['answer'].x = domain_answer_emb

    # Query -> Answer edges
    qa_edges = []
    for q_idx in range(N_query):
        for r_offset in range(num_responses_per_query):
            r_idx = q_idx * num_responses_per_query + r_offset
            if r_idx < N_resp:
                qa_edges.append([q_idx, r_idx])

    data[('query', 'generates', 'answer')].edge_index = torch.tensor(qa_edges).t().contiguous()
    data[('answer', 'rev_generates', 'query')].edge_index = data[('query', 'generates', 'answer')].edge_index.flip(0)

    # Query -> Think edges
    data[('query', 'generates_reasoning', 'think')].edge_index = torch.tensor(qa_edges).t().contiguous()
    data[('think', 'rev_generates_reasoning', 'query')].edge_index = data[('query', 'generates_reasoning', 'think')].edge_index.flip(0)

    # Think -> Answer edges (one-to-one)
    ta_edges = [[i, i] for i in range(N_resp)]
    data[('think', 'leads_to', 'answer')].edge_index = torch.tensor(ta_edges).t().contiguous()
    data[('answer', 'rev_leads_to', 'think')].edge_index = data[('think', 'leads_to', 'answer')].edge_index.flip(0)

    # Query-Query edges (kNN within domain)
    sim = compute_similarity_matrix(domain_query_emb)
    qq_edges = []

    for i in range(N_query):
        other_idx = [j for j in range(N_query) if j != i]
        if other_idx:
            other_sim = sim[i, other_idx]
            k_w = min(knn_k, len(other_idx))
            _, top = other_sim.topk(k_w)
            for j in top.tolist():
                qq_edges.append([i, other_idx[j]])

    data[('query', 'similar_to', 'query')].edge_index = torch.tensor(qq_edges).t().contiguous()

    # Think-Think edges (intra-query kNN + cross-query kNN)
    tt_edges = []

    # Intra-query think-think edges
    if intra_think_k != 0:
        if intra_think_k == -1:
            # Full connection (original behavior)
            for q_idx in range(N_query):
                start = q_idx * num_responses_per_query
                for i in range(num_responses_per_query):
                    for j in range(i + 1, num_responses_per_query):
                        t_i = start + i
                        t_j = start + j
                        if t_i < N_resp and t_j < N_resp:
                            tt_edges.append([t_i, t_j])
                            tt_edges.append([t_j, t_i])
        else:
            # Top-k similar thinks within same query
            think_sim = compute_similarity_matrix(domain_think_emb)
            for q_idx in range(N_query):
                start = q_idx * num_responses_per_query
                end = min(start + num_responses_per_query, N_resp)
                query_thinks = list(range(start, end))

                for i, think_idx in enumerate(query_thinks):
                    other_thinks = [t for t in query_thinks if t != think_idx]
                    if other_thinks:
                        other_sim = think_sim[think_idx, other_thinks]
                        k = min(intra_think_k, len(other_thinks))
                        _, top_k_local = other_sim.topk(k)

                        for local_j in top_k_local.tolist():
                            j = other_thinks[local_j]
                            tt_edges.append([think_idx, j])

    # Cross-query think-think edges
    if think_cross_k > 0 and N_resp > num_responses_per_query:
        if intra_think_k <= 0:
            think_sim = compute_similarity_matrix(domain_think_emb)
        for i in range(N_resp):
            my_query = i // num_responses_per_query
            mask = torch.ones(N_resp, dtype=torch.bool, device=domain_think_emb.device)
            start = my_query * num_responses_per_query
            end = min(start + num_responses_per_query, N_resp)
            mask[start:end] = False

            other_sim = think_sim[i].clone()
            other_sim[~mask] = -float('inf')

            if (other_sim > -float('inf')).sum() > 0:
                k = min(think_cross_k, (other_sim > -float('inf')).sum().item())
                _, topk = other_sim.topk(k)
                for j in topk.tolist():
                    tt_edges.append([i, j])

    if tt_edges:
        data[('think', 'competes_with', 'think')].edge_index = torch.tensor(tt_edges).t().contiguous()
    else:
        data[('think', 'competes_with', 'think')].edge_index = torch.empty((2, 0), dtype=torch.long)

    # Answer-Answer edges (same-query competition)
    if use_aa_edges:
        aa_edges = []
        for q_idx in range(N_query):
            start = q_idx * num_responses_per_query
            for i in range(num_responses_per_query):
                for j in range(i + 1, num_responses_per_query):
                    a_i = start + i
                    a_j = start + j
                    if a_i < N_resp and a_j < N_resp:
                        aa_edges.append([a_i, a_j])
                        aa_edges.append([a_j, a_i])

        if aa_edges:
            data[('answer', 'competes_with', 'answer')].edge_index = torch.tensor(aa_edges).t().contiguous()

    return data, response_indices, domain_query_indices
