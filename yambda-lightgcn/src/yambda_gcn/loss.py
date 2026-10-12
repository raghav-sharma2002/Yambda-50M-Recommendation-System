import torch.nn.functional as F


def bpr_loss(u, pos, neg, u0, p0, n0, reg=1e-4):
    """BPR ranking loss + L2 on layer-0 embeddings. u/pos/neg are propagated embeddings."""
    s_pos, s_neg = (u * pos).sum(-1), (u * neg).sum(-1)
    rank = F.softplus(s_neg - s_pos).mean()          # -log sigmoid(s_pos - s_neg)
    l2 = 0.5 * (u0.pow(2).sum() + p0.pow(2).sum() + n0.pow(2).sum()) / u0.shape[0]
    return rank + reg * l2
