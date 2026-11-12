"""cand_15 (<- 11): Jaccard^2 k=200 x deg^-0.45, ontology relatives' rows get +0.05 similarity before ranking."""
import numpy as np
from . import ops

def score(tasks, B, k=200, p=2.0, a=0.45, boost=0.05):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = ops.jaccard(inter, sz, len(t.seeds))
        rel = B.relatives(t)
        sim[rel] += boost
        rows, w = ops.topk(sim, B.own_row(t), k)
        out[i] = ops.propagate(L, rows, w ** p) * dw
    return out
