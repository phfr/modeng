"""cand_7 (<- 5): transfer_jaccard x (degree+1)^-0.6 (stronger degree discount)."""
import numpy as np
from . import ops

def score(tasks, B, k=50, a=0.6):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        rows, w = ops.topk(ops.jaccard(inter, sz, len(t.seeds)), B.own_row(t), k)
        out[i] = ops.propagate(L, rows, w) * dw
    return out
