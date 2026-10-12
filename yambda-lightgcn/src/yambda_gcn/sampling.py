import numpy as np


class BPRSampler:
    """Yields (user, positive, negative) triplets; negatives are uniform and rejected if observed."""

    def __init__(self, users, items, n_items, seed=0):
        self.u, self.i, self.n_items = users, items, n_items
        self.rng = np.random.default_rng(seed)
        self.keys = np.unique(users.astype(np.int64) * n_items + items)   # sorted pair codes

    def _observed(self, u, j):
        k = u.astype(np.int64) * self.n_items + j
        pos = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
        return self.keys[pos] == k

    def epoch(self, batch_size):
        perm = self.rng.permutation(len(self.u))
        for s in range(0, len(perm), batch_size):
            idx = perm[s: s + batch_size]
            u, p = self.u[idx], self.i[idx]
            n = self.rng.integers(0, self.n_items, len(idx))
            for _ in range(10):
                bad = self._observed(u, n)
                if not bad.any():
                    break
                n[bad] = self.rng.integers(0, self.n_items, int(bad.sum()))
            yield u, p, n
