# Yambda LightGCN

A graph-based music recommender built on Yandex's [Yambda](https://huggingface.co/datasets/yandex/yambda) dataset. It covers the full path from raw listening logs to a served Top-K recommendation list: graph construction, LightGCN training with BPR loss, audio-content fusion and evaluation
## Overview

User likes are modelled as a bipartite user–item graph. A LightGCN learns user and item embeddings by propagating information across that graph, and recommendations are produced by scoring candidate items against a user's embedding. Yambda's precomputed audio embeddings can optionally be fused into the item representations before propagation.

Yambda contains no human-readable song, artist, or album metadata, only `user_id`, `item_id`, timestamps, and embeddings, so the demo works with anonymised item IDs and scores.

```
likes → temporal split → user–item graph → LightGCN (+ audio fusion)
      → BPR training → Top-K scoring → FastAPI → React frontend
```

## Features

- **Graph construction:** binary interaction matrix and symmetric-normalised bipartite adjacency (`D^-1/2 A D^-1/2`)
- **LightGCN:** parameter-free neighbourhood propagation with layer-mean readout
- **BPR loss:** pairwise ranking objective with L2 regularisation on layer-0 embeddings
- **Negative sampling:** vectorised uniform sampling that rejects items the user has already liked
- **Audio-content fusion:** projected audio embeddings added to item ID embeddings before propagation (toggleable)
- **Evaluation:** Precision, Recall, NDCG, and MAP @ k on a global temporal split, with already-seen items masked out


## Repository structure

```
yambda-lightgcn/
├── src/yambda_gcn/
│   ├── data.py        # YambdaLoader: temporal split, id mapping, audio embedding alignment
│   ├── graph.py       # BipartiteGraph: interaction matrix and normalised adjacency
│   ├── model.py       # LightGCN with optional content fusion
│   ├── loss.py        # BPR loss
│   ├── sampling.py    # BPR triplet sampler
│   ├── metrics.py     # Precision / Recall / NDCG / MAP @ k
│   ├── train.py       # Training, evaluation, artifact export
│   └── api.py         # FastAPI service
├── tests/             # Unit tests for metrics, graph, and sampler
├── requirements.txt
└── pyproject.toml
```

## Getting started

**Install**

```bash
pip install -e .
pytest -q
```

**Train**

```bash
python -m yambda_gcn.train \
  --likes data/likes.parquet \
  --embeddings data/embeddings.parquet \
  --fusion add
```

Useful options:

| Flag | Description |
|---|---|
| `--fusion {none,add}` | Disable or enable audio-content fusion |
| `--alpha` | Scale of the fused audio projection |
| `--vocab {train,all}` | Restrict items to the training set or include all items |
| `--dim`, `--layers` | Embedding size and number of propagation layers |
| `--epochs`, `--batch-size`, `--lr`, `--reg` | Training hyperparameters |
| `--k` | Cutoff for evaluation metrics |

Training writes `metrics.json` and `embeddings.npz` to `artifacts/`.


## Data

Download Yambda from Hugging Face and point `--likes` and `--embeddings` at the parquet files. The loader expects the columns `uid`, `item_id`, `timestamp` (likes) and `item_id` with `embed` or `normalized_embed` (embeddings). Check 

## Acknowledgements

Built on the Yambda dataset by Yandex. Model design follows *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* (He et al., 2020) and BPR (Rendle et al., 2009).
