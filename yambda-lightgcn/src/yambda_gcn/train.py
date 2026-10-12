"""Train LightGCN(+BPR[+audio fusion]); evaluate against a popularity baseline; export artifacts."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from .data import YambdaLoader
from .graph import BipartiteGraph
from .model import LightGCN
from .loss import bpr_loss
from .sampling import BPRSampler
from .metrics import rank_metrics


@torch.no_grad()
def evaluate(score_fn, train_csr, test_u, test_i, k, device, batch=1024):
    truth = {}
    for u, i in zip(test_u.tolist(), test_i.tolist()):
        truth.setdefault(u, set()).add(i)
    users = np.array(sorted(truth))
    tops = []
    for s in range(0, len(users), batch):
        b = users[s: s + batch]
        scores = score_fn(torch.as_tensor(b, device=device)).clone()
        for r, u in enumerate(b):                                  # never recommend seen items
            seen = train_csr.indices[train_csr.indptr[u]: train_csr.indptr[u + 1]]
            scores[r, torch.as_tensor(seen, device=device, dtype=torch.long)] = float("-inf")
        tops.append(torch.topk(scores, k, dim=1).indices.cpu().numpy())
    return rank_metrics(np.concatenate(tops), [truth[int(u)] for u in users], k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--likes", required=True)
    p.add_argument("--embeddings", default=None)
    p.add_argument("--fusion", choices=["none", "add"], default="add")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--vocab", choices=["train", "all"], default="train")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg", type=float, default=1e-4)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--out", default="artifacts")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    d = YambdaLoader(a.likes, a.embeddings if a.fusion == "add" else None, vocab=a.vocab).load()
    print(f"users={d.n_users} items={d.n_items} train={len(d.train_u)} test={len(d.test_u)} "
          f"cold_test_dropped={d.n_test_dropped_cold}")
    g = BipartiteGraph(d.train_u, d.train_i, d.n_users, d.n_items)
    adj = g.norm_adj(dev)
    model = LightGCN(d.n_users, d.n_items, a.dim, a.layers, d.content, a.alpha).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sampler = BPRSampler(d.train_u, d.train_i, d.n_items, a.seed)

    # Sanity floor: if the GCN can't beat "recommend what's popular", something is wrong.
    pop = torch.as_tensor(np.asarray(g.R.sum(0)).ravel(), dtype=torch.float32, device=dev)
    results = {"popularity": evaluate(lambda b: pop.expand(len(b), -1), g.R, d.test_u, d.test_i, a.k, dev)}
    print("popularity:", results["popularity"])

    history = []
    for ep in range(1, a.epochs + 1):
        model.train(); total, steps, t0 = 0.0, 0, time.time()
        for u, pos, neg in sampler.epoch(a.batch_size):
            u, pos, neg = (torch.as_tensor(x, device=dev) for x in (u, pos, neg))
            U, I = model(adj)                                      # full-graph propagation per step
            loss = bpr_loss(U[u], I[pos], I[neg], *model.ego(u, pos, neg), reg=a.reg)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); steps += 1
        msg = f"epoch {ep:03d} loss={total / steps:.4f} ({time.time() - t0:.0f}s)"
        if ep % a.eval_every == 0 or ep == a.epochs:
            model.eval()
            with torch.no_grad():
                U, I = model(adj)
            m = evaluate(lambda b: U[b] @ I.T, g.R, d.test_u, d.test_i, a.k, dev)
            history.append({"epoch": ep, **m}); msg += f" {m}"
        print(msg)

    results["lightgcn"] = history[-1] if history else {}
    results["config"] = vars(a)
    (out / "metrics.json").write_text(json.dumps({"results": results, "history": history}, indent=2))
    with torch.no_grad():
        model.eval(); U, I = model(adj)
    np.savez_compressed(out / "embeddings.npz", user_emb=U.cpu().numpy(), item_emb=I.cpu().numpy(),
                        uids=d.uids, iids=d.iids, train_u=d.train_u, train_i=d.train_i)
    print("saved to", out)


if __name__ == "__main__":
    main()
