"""cand_24 (<- 17): two-hop query expansion (Rocchio): first pass cosine^2 k=200 transfer; the query becomes
seeds + 0.3 x (first-pass scores, max-normalised, seeds removed); cosine of the weighted query with each row; second
transfer pass with those weights; + 1e-3 tail, x deg^-0.6."""
import numpy as np
from . import ops

def score(tasks, B, k=200, p=2.0, a=0.6, eps=1e-3, beta=0.3):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = ops.cosine(inter, sz, len(t.seeds)); own = B.own_row(t)
        rows, w = ops.topk(sim, own, k)
        s1 = ops.propagate(L, rows, w ** p); s1[t.seeds] = 0; s1 /= max(s1.max(), 1e-12)
        q = beta * s1; q[t.seeds] = 1.0
        sim2 = np.asarray(L @ q).ravel() / np.sqrt(np.maximum(sz, 1) * (q ** 2).sum())
        rows2, w2 = ops.topk(sim2, own, k)
        out[i] = (ops.propagate(L, rows2, w2 ** p) + eps * ops.tail(L, sim, own)) * dw
    return out
