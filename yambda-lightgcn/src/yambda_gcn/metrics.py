import numpy as np


def rank_metrics(topk, truth, k):
    """topk: (n, k) item indices; truth: list[set[int]] aligned by row. Returns means over users."""
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.cumsum(disc)
    P = R = N = M = 0.0
    n = 0
    for row, t in zip(topk, truth):
        if not t:
            continue
        hits = np.fromiter((int(i) in t for i in row[:k]), dtype=bool, count=k)
        h = hits.sum()
        P += h / k
        R += h / len(t)
        N += (hits * disc).sum() / ideal[min(len(t), k) - 1]
        if h:
            prec = np.cumsum(hits) / np.arange(1, k + 1)
            M += (prec * hits).sum() / min(len(t), k)
        n += 1
    n = max(n, 1)
    return {f"precision@{k}": P / n, f"recall@{k}": R / n, f"ndcg@{k}": N / n, f"map@{k}": M / n}
