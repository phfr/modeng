"""cand_19 (<- 12 x 16): cand_12 (max-normalised) + 0.1 x item-item CF (max-normalised), x deg^-0.45."""
import numpy as np
from . import ops

def score(tasks, B, k=200, p=2.0, a=0.45, eps=1e-3, lam=0.1):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = ops.cosine(inter, sz, len(t.seeds))
        rows, w = ops.topk(sim, B.own_row(t), k)
        s = ops.propagate(L, rows, w ** p); s /= max(s.max(), 1e-12)
        c = ops.item_cf(L, t.seeds); c[t.seeds] = 0; c /= max(c.max(), 1e-12)
        out[i] = (s + lam * c + eps * ops.tail(L, sim, B.own_row(t))) * dw
    return out
