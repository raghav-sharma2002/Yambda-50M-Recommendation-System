"""BipartiteGraph: interaction matrix, adjacency, symmetric normalisation D^-1/2 A D^-1/2."""
import numpy as np
import scipy.sparse as sp
import torch


class BipartiteGraph:
    def __init__(self, users, items, n_users, n_items):
        self.n_users, self.n_items = n_users, n_items
        R = sp.coo_matrix((np.ones(len(users), dtype=np.float32), (users, items)),
                          shape=(n_users, n_items)).tocsr()
        R.sum_duplicates()
        R.data[:] = 1.0                      # binary implicit feedback
        self.R = R                           # (n_users, n_items), CSR, deduped

    def adjacency(self) -> sp.csr_matrix:
        return sp.bmat([[None, self.R], [self.R.T, None]], format="csr")

    def norm_adj(self, device="cpu") -> torch.Tensor:
        A = self.adjacency()
        deg = np.asarray(A.sum(1)).ravel().astype(np.float64)
        d_inv = np.zeros_like(deg)
        np.power(deg, -0.5, out=d_inv, where=deg > 0)
        D = sp.diags(d_inv)
        L = (D @ A @ D).tocoo()
        idx = torch.as_tensor(np.vstack([L.row, L.col]), dtype=torch.long)
        val = torch.as_tensor(L.data, dtype=torch.float32)
        n = self.n_users + self.n_items
        return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce().to(device)
