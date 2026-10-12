"""FastAPI service: GET /health, GET /recommend/{user_id}?k=10  (user_id = raw Yambda uid)."""
import os
from pathlib import Path
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

ART = Path(os.environ.get("ARTIFACTS", "artifacts")) / "embeddings.npz"
app = FastAPI(title="Yambda LightGCN demo")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["GET"])

_z = np.load(ART, allow_pickle=False) if ART.exists() else None
if _z is not None:
    U, I, UIDS, IIDS = _z["user_emb"], _z["item_emb"], _z["uids"], _z["iids"]
    UMAP = {int(u): k for k, u in enumerate(UIDS)}
    SEEN = {}
    for u, i in zip(_z["train_u"], _z["train_i"]):
        SEEN.setdefault(int(u), []).append(int(i))


@app.get("/health")
def health():
    return {"status": "ok" if _z is not None else "no_artifacts", "users": len(UMAP) if _z else 0}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=100)):
    if _z is None:
        raise HTTPException(503, "artifacts not found; run training first")
    idx = UMAP.get(user_id)
    if idx is None:
        raise HTTPException(404, f"unknown user {user_id}")
    s = I @ U[idx]
    s[SEEN.get(idx, [])] = -np.inf                                 # drop already-liked items
    top = np.argpartition(-s, k)[:k]
    top = top[np.argsort(-s[top])]
    return {"user_id": user_id, "k": k,
            "items": [{"item_id": int(IIDS[j]), "score": float(s[j])} for j in top]}
