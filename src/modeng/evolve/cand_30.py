"""cand_30 (<- 17 x 15): cosine^2 k=200 x deg^-0.6 + tail; ontology ANCESTOR/descendant rows' weights doubled (relatives
carry sibling genes)."""
import numpy as np
from . import ops

def score(tasks, B, k=200, p=2.0, a=0.6, eps=1e-3, mult=2.0):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = ops.cosine(inter, sz, len(t.seeds))
        rel = B.relatives(t)
        sim[rel] *= mult
        rows, w = ops.topk(sim, B.own_row(t), k)
        out[i] = (ops.propagate(L, rows, w ** p) + eps * ops.tail(L, sim, B.own_row(t))) * dw
    return out
