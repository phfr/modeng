"""cand_10 (<- 5): cand_5 plus a weak fallback for genes no neighbour carries: 1e-3 x global transfer over ALL rows with sim>0
(no top-k cut), so the zero-score tail is ordered instead of tied."""
import numpy as np
from . import ops

def score(tasks, B, k=50, a=0.3, eps=1e-3):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = ops.jaccard(inter, sz, len(t.seeds))
        rows, w = ops.topk(sim, B.own_row(t), k)
        allr, allw = ops.topk(sim, B.own_row(t), L.shape[0])
        s = ops.propagate(L, rows, w)
        tail = ops.propagate(L, allr, allw)
        tail = tail / max(tail.max(), 1e-12)
        out[i] = (s + eps * tail) * dw
    return out
