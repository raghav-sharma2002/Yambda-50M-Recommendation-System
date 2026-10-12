# Yambda × LightGCN — a systems-design experiment

An end-to-end graph recommender on Yandex's [Yambda](https://huggingface.co/datasets/yandex/yambda) music-listening dataset:
interaction history → user–item graph → LightGCN → BPR training → Top-K → FastAPI → React demo.

**Honest summary up front:** the model trains, the pipeline runs end to end, and the ranking metrics are
**close to zero**. I'm publishing it anyway, because the plumbing is the part I wanted to learn, and
the failure is more instructive than a polished number would have been.

## Why I built this

I wanted to understand what it takes to take a recommender from raw logs to something a person can click on —
not just the model, but the graph construction, negative sampling, evaluation, serving, and UI.
Yambda is large, real, and messy, which made it a good teacher.

## What's in the pipeline

```
likes.parquet ──► temporal split ──► BipartiteGraph (D^-½ A D^-½) ──► LightGCN (+ audio fusion)
                                                                          │  BPR loss, sampled negatives
audio embeddings.parquet ──► projected & added to item ID embeddings ─────┘
                                                                          ▼
                          embeddings.npz ──► FastAPI /recommend/{uid}?k=10 ──► React UI
```

| Module | Role |
|---|---|
| `data.py` — `YambdaLoader` | Global temporal split, contiguous ids, aligned audio embeddings |
| `graph.py` — `BipartiteGraph` | Binary interaction matrix, symmetric-normalised adjacency |
| `model.py` — `LightGCN` | Parameter-free propagation, layer-mean readout, optional content fusion |
| `loss.py`, `sampling.py` | BPR loss; vectorised rejection-sampled negatives |
| `metrics.py` | Precision / Recall / NDCG / MAP @ k |
| `train.py` | Training, evaluation, popularity baseline, artifact export |
| `api.py` | `GET /health`, `GET /recommend/{user_id}?k=10` |

## Results

| Model (k=10) | Precision | Recall | NDCG | MAP |
|---|---|---|---|---|
| Popularity baseline | _fill from `artifacts/metrics.json`_ | | | |
| LightGCN + BPR | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| LightGCN + BPR + audio fusion | ≈ 0 | ≈ 0 | ≈ 0 | ≈ 0 |

_Replace the ≈ 0 entries with your actual numbers before pushing; precise zeros are more credible than a tilde._

## What I think went wrong (hypotheses, not findings)

I haven't proven any of these. They're ordered by how suspicious I am of each.

1. **Content fusion swamped the ID signal.** The audio projection is added to item embeddings at full strength
   from step one, so early propagation is dominated by a feature that wasn't trained for this task.
   `--alpha` and `--fusion none` exist so this can be ablated cleanly.
2. **Extreme sparsity vs. a temporal split.** Future likes land on items with few or no training edges;
   there's little for graph propagation to latch onto. `data.py` reports how many test likes were cold.
3. **Full-graph propagation per step on a huge graph** forces small epochs/short training, so the model may
   simply be under-trained. The loss curve, not the final metric, would show this.
4. **Evaluation design.** Test items outside the training vocabulary are dropped (`--vocab train`); `--vocab all`
   keeps them and makes cold-start the real test of content fusion.
5. **Dataset limits.** No song/artist metadata, so I can't eyeball whether recommendations "feel" right.

## What I'd do next

- Run `--fusion none` first and confirm it beats the popularity baseline. If it doesn't, the bug is upstream of fusion.
- Overfit a tiny subset (a few hundred users) — a correct pipeline should memorise it.
- Add a zero-initialised fusion gate so content starts at 0 and earns its weight.
- Move from full-graph to neighbour-sampled training for the larger splits.

## Run it

```bash
pip install -e . && pytest -q
python -m yambda_gcn.train --likes data/likes.parquet --embeddings data/embeddings.parquet --fusion add
python -m yambda_gcn.train --likes data/likes.parquet --fusion none          # ablation
uvicorn yambda_gcn.api:app --reload                                          # serve artifacts/
```

Check parquet column names against your Yambda download (`uid`, `item_id`, `timestamp`, `embed`/`normalized_embed`).

## Not deployed, on purpose

Since the results aren't good enough to stand behind, I'm not hosting this as a live demo. The repo is
meant to be read as a worked example of the system around a recommender, and of debugging one that doesn't work yet.

## Layout

`src/yambda_gcn/` library · `tests/` unit tests · `frontend/` React demo · `artifacts/` (git-ignored) trained outputs
