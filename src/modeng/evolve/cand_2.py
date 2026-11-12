"""cand_2 (<- 0): Jaccard^2 weights (sharper), k=100."""
import numpy as np
from . import ops

def score(tasks, B, k=100, p=2.0):
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        rows, w = ops.topk(ops.jaccard(inter, sz, len(t.seeds)), B.own_row(t), k)
        out[i] = ops.propagate(L, rows, w ** p)
    return out
