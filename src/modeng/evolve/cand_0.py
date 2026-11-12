"""cand_0: transfer_jaccard baseline (top-50 Jaccard-similar context diseases, sum of sim x label row)."""
import numpy as np


def score(tasks, B, k=50):
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t)
        sz = L.getnnz(1).astype(np.float32)
        inter = np.asarray(L[:, t.seeds].sum(1)).ravel()
        sim = inter / np.maximum(len(t.seeds) + sz - inter, 1)
        own = B.own_row(t)
        if own >= 0:
            sim[own] = 0
        cand = np.flatnonzero(sim > 0)
        if len(cand) == 0:
            continue
        top = cand[np.argsort(-sim[cand])[:k]]
        out[i] = np.asarray(L[top].T @ sim[top].astype(np.float32)).ravel()
    return out
