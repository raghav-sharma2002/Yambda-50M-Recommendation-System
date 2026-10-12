"""YambdaLoader: likes -> temporal train/test split -> contiguous ids + aligned audio embeddings.

Expected parquet columns (verify against your download of yandex/yambda):
  likes:      uid, item_id, timestamp
  embeddings: item_id, embed | normalized_embed
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class Split:
    n_users: int
    n_items: int
    train_u: np.ndarray
    train_i: np.ndarray
    test_u: np.ndarray
    test_i: np.ndarray
    content: np.ndarray | None      # (n_items, d) or None; zero rows = no embedding
    uids: np.ndarray                # idx -> raw uid
    iids: np.ndarray                # idx -> raw item_id
    n_test_dropped_cold: int        # test likes on items unseen in train (vocab="train")


class YambdaLoader:
    def __init__(self, likes_path, embeddings_path=None, test_frac=0.1,
                 min_user_train=5, vocab="train"):
        assert vocab in {"train", "all"}
        self.likes_path, self.emb_path = Path(likes_path), embeddings_path
        self.test_frac, self.min_user_train, self.vocab = test_frac, min_user_train, vocab

    def load(self) -> Split:
        df = pd.read_parquet(self.likes_path, columns=["uid", "item_id", "timestamp"])
        cutoff = df["timestamp"].quantile(1 - self.test_frac)       # global temporal split
        train, test = df[df.timestamp <= cutoff], df[df.timestamp > cutoff]

        counts = train.uid.value_counts()
        keep = counts[counts >= self.min_user_train].index
        train, test = train[train.uid.isin(keep)], test[test.uid.isin(keep)]

        uids = np.sort(train.uid.unique())
        item_pool = train.item_id if self.vocab == "train" else pd.concat([train.item_id, test.item_id])
        iids = np.sort(item_pool.unique())
        umap = {u: k for k, u in enumerate(uids)}
        imap = {i: k for k, i in enumerate(iids)}

        n_before = len(test)
        test = test[test.item_id.isin(imap)]
        content = self._content(iids, imap)
        return Split(
            n_users=len(uids), n_items=len(iids),
            train_u=train.uid.map(umap).to_numpy(), train_i=train.item_id.map(imap).to_numpy(),
            test_u=test.uid.map(umap).to_numpy(), test_i=test.item_id.map(imap).to_numpy(),
            content=content, uids=uids, iids=iids, n_test_dropped_cold=n_before - len(test))

    def _content(self, iids, imap):
        if self.emb_path is None:
            return None
        emb = pd.read_parquet(self.emb_path)
        emb = emb[emb.item_id.isin(imap)]
        col = "normalized_embed" if "normalized_embed" in emb else "embed"
        mat = np.stack(emb[col].to_numpy()).astype(np.float32)
        out = np.zeros((len(iids), mat.shape[1]), dtype=np.float32)
        out[emb.item_id.map(imap).to_numpy()] = mat
        return out
