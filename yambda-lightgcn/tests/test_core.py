import numpy as np
import torch
from yambda_gcn.graph import BipartiteGraph
from yambda_gcn.metrics import rank_metrics
from yambda_gcn.sampling import BPRSampler


def test_perfect_ranking_scores_one():
    m = rank_metrics(np.array([[1, 2, 3]]), [{1, 2, 3}], 3)
    assert m["ndcg@3"] == 1.0 and m["recall@3"] == 1.0 and m["map@3"] == 1.0


def test_miss_scores_zero():
    m = rank_metrics(np.array([[7, 8, 9]]), [{1}], 3)
    assert all(v == 0 for v in m.values())


def test_norm_adj_symmetric():
    g = BipartiteGraph(np.array([0, 0, 1]), np.array([0, 1, 1]), 2, 2)
    A = g.norm_adj().to_dense()
    assert torch.allclose(A, A.T)


def test_sampler_never_returns_positives():
    u, i = np.array([0, 0, 1]), np.array([0, 1, 2])
    s = BPRSampler(u, i, 5)
    for uu, _, nn in s.epoch(3):
        assert not s._observed(uu, nn).any()
