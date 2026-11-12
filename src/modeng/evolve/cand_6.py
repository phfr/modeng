"""cand_6 (<- 0): transfer_jaccard / (1 + gene popularity)^0.3 (popularity = masked column count)."""
import numpy as np
from . import ops

def score(tasks, B, k=50, b=0.3):
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        pop = np.asarray(L.sum(0)).ravel()
        rows, w = ops.topk(ops.jaccard(inter, sz, len(t.seeds)), B.own_row(t), k)
        out[i] = ops.propagate(L, rows, w) / (1.0 + pop) ** b
    return out
