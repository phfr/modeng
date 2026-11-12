"""cand_23 (<- 17): softmax neighbour weights exp(cos / 0.05) over top-200 cosine rows, + 1e-3 tail, x deg^-0.6."""
import numpy as np
from . import ops

def score(tasks, B, k=200, tau=0.05, a=0.6, eps=1e-3):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = ops.cosine(inter, sz, len(t.seeds))
        rows, w = ops.topk(sim, B.own_row(t), k)
        if len(rows):
            w = np.exp((w - w.max()) / tau)
        out[i] = (ops.propagate(L, rows, w) + eps * ops.tail(L, sim, B.own_row(t))) * dw
    return out
