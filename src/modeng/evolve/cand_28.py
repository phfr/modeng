"""cand_28 (<- 17): seed-count-adaptive neighbourhood k = clip(8 x n_seeds, 30, 300), cosine^2, x deg^-0.6, + tail."""
import numpy as np
from . import ops

def score(tasks, B, p=2.0, a=0.6, eps=1e-3):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = ops.cosine(inter, sz, len(t.seeds))
        k = int(np.clip(8 * len(t.seeds), 30, 300))
        rows, w = ops.topk(sim, B.own_row(t), k)
        out[i] = (ops.propagate(L, rows, w ** p) + eps * ops.tail(L, sim, B.own_row(t))) * dw
    return out
