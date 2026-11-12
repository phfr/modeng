"""cand_4 (<- 0): IDF-weighted Jaccard (seed genes weighted log(N/df), df = masked column count), k=50."""
import numpy as np
from . import ops

def score(tasks, B, k=50):
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t)
        df = np.asarray(L.sum(0)).ravel()
        idf = np.log((L.shape[0] + 1) / (df + 1))
        winter = np.asarray(L[:, t.seeds] @ idf[t.seeds]).ravel()
        wrow = np.asarray(L @ idf).ravel()
        sim = winter / np.maximum(idf[t.seeds].sum() + wrow - winter, 1e-9)
        rows, w = ops.topk(sim, B.own_row(t), k)
        out[i] = ops.propagate(L, rows, w)
    return out
