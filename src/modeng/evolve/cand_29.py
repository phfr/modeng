"""cand_29 (<- 17): degree-weighted cosine: seed overlaps weighted (deg+1)^-0.5 (a shared rare/low-degree gene is stronger
evidence of relatedness), row norm with the same weights; cosine^2, k=200, x deg^-0.6, + tail."""
import numpy as np
from . import ops

def score(tasks, B, k=200, p=2.0, a=0.6, c=0.5, eps=1e-3):
    dw = (B.degree + 1.0) ** -a
    gw = (B.degree + 1.0) ** -c
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t)
        winter = np.asarray(L[:, t.seeds] @ gw[t.seeds]).ravel()
        wsz = np.asarray(L @ gw).ravel()
        sim = winter / np.sqrt(np.maximum(wsz * gw[t.seeds].sum(), 1e-12))
        rows, w = ops.topk(sim, B.own_row(t), k)
        out[i] = (ops.propagate(L, rows, w ** p) + eps * ops.tail(L, sim, B.own_row(t))) * dw
    return out
