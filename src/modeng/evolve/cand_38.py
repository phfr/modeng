"""cand_38 (<- 32): cosine^1.5 k=200 transfer + 0.05 tail, x deg^-0.6."""
import numpy as np
from . import ops

def score(tasks, B, k=200, p=1.5, a=0.6, eps=0.05):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = ops.cosine(inter, sz, len(t.seeds))
        rows, w = ops.topk(sim, B.own_row(t), k)
        s = ops.propagate(L, rows, w ** p); s /= max(s.max(), 1e-12)
        out[i] = (s + eps * ops.tail(L, sim, B.own_row(t))) * dw
    return out
