"""cand_16 (<- 11): Jaccard^2 k=200 transfer (max-normalised) + 0.3 x item-item CF (max-normalised), x deg^-0.45."""
import numpy as np
from . import ops

def score(tasks, B, k=200, p=2.0, a=0.45, lam=0.3):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        rows, w = ops.topk(ops.jaccard(inter, sz, len(t.seeds)), B.own_row(t), k)
        s = ops.propagate(L, rows, w ** p); s /= max(s.max(), 1e-12)
        c = ops.item_cf(L, t.seeds); c[t.seeds] = 0; c /= max(c.max(), 1e-12)
        out[i] = (s + lam * c) * dw
    return out
