"""cand_1 (<- 0): cosine similarity instead of Jaccard, k=50."""
import numpy as np
from . import ops

def score(tasks, B, k=50):
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        rows, w = ops.topk(ops.cosine(inter, sz, len(t.seeds)), B.own_row(t), k)
        out[i] = ops.propagate(L, rows, w)
    return out
