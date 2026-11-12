"""cand_11 (<- 5 x 2): Jaccard^2, k=200, x (degree+1)^-0.45."""
import numpy as np
from . import ops

def score(tasks, B, k=200, p=2.0, a=0.45):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        rows, w = ops.topk(ops.jaccard(inter, sz, len(t.seeds)), B.own_row(t), k)
        out[i] = ops.propagate(L, rows, w ** p) * dw
    return out
