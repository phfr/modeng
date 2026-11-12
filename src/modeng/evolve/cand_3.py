"""cand_3 (<- 0): Jaccard top-50, label rows scaled by |row|^-0.5 (large diseases spread less mass per gene)."""
import numpy as np
from . import ops

def score(tasks, B, k=50, a=0.5):
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        rows, w = ops.topk(ops.jaccard(inter, sz, len(t.seeds)), B.own_row(t), k)
        out[i] = ops.propagate(L, rows, w, np.maximum(sz, 1) ** -a)
    return out
