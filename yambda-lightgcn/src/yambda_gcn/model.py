"""LightGCN with optional audio-content fusion on the item side (fusion="add")."""
import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, dim=64, n_layers=3, content=None, alpha=1.0):
        super().__init__()
        self.n_users, self.n_items, self.n_layers, self.alpha = n_users, n_items, n_layers, alpha
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)
        self.content = None
        if content is not None:
            self.register_buffer("content", torch.as_tensor(content, dtype=torch.float32))
            self.content_proj = nn.Linear(content.shape[1], dim, bias=False)

    def item_base(self):
        """Layer-0 item representation: ID embedding (+ projected audio embedding)."""
        w = self.item_emb.weight
        return w if self.content is None else w + self.alpha * self.content_proj(self.content)

    def forward(self, adj):
        x = torch.cat([self.user_emb.weight, self.item_base()])
        outs = [x]
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            outs.append(x)
        x = torch.stack(outs).mean(0)        # layer-mean readout
        return x[: self.n_users], x[self.n_users:]

    def ego(self, u, p, n):
        """Layer-0 ID embeddings of a batch, used for L2 regularisation."""
        return self.user_emb(u), self.item_emb(p), self.item_emb(n)
