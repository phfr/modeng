"""cand_31 (<- 17): max aggregation: score = max_r cos_r^2 L[r,g] + 0.1 x sum (k=200), + 1e-3 tail, x deg^-0.6."""
import numpy as np
from . import ops

def score(tasks, B, k=200, p=2.0, a=0.6, eps=1e-3, lam=0.1):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = ops.cosine(inter, sz, len(t.seeds))
        rows, w = ops.topk(sim, B.own_row(t), k)
        s = ops.propagate(L, rows, w ** p)
        mx = np.zeros(B.n_genes)
        if len(rows):
            M = L[rows].multiply((w ** p)[:, None]).tocsc()
            mx = np.asarray(M.max(0).todense()).ravel()
        out[i] = (mx + lam * s + eps * ops.tail(L, sim, B.own_row(t))) * dw
    return out
