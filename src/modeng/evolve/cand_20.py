"""cand_20 (<- 12): weights inter^2 / |row|^1.5 (stronger large-row penalty than cosine^2), k=200, x deg^-0.45, plus tail."""
import numpy as np
from . import ops

def score(tasks, B, k=200, q=1.5, a=0.45, eps=1e-3):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = inter ** 2 / np.maximum(sz, 1) ** q / max(len(t.seeds), 1)
        rows, w = ops.topk(sim, B.own_row(t), k)
        out[i] = (ops.propagate(L, rows, w) + eps * ops.tail(L, sim, B.own_row(t))) * dw
    return out
