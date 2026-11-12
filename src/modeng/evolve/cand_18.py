"""cand_18 (<- 12): cosine^3 over ALL rows with overlap (no k cut), x (degree+1)^-0.45, plus 1e-3 tail."""
import numpy as np
from . import ops

def score(tasks, B, p=3.0, a=0.45, eps=1e-3):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = ops.cosine(inter, sz, len(t.seeds))
        rows, w = ops.topk(sim, B.own_row(t), L.shape[0])
        out[i] = (ops.propagate(L, rows, w ** p) + eps * ops.tail(L, sim, B.own_row(t))) * dw
    return out
