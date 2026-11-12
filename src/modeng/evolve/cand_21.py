"""cand_21 (<- 12): hypergeometric similarity -log10 P(X >= inter) (gene universe = genes in any context row), weight = that^2,
k=200, x deg^-0.45, plus tail."""
import numpy as np
from scipy.stats import hypergeom
from . import ops

def score(tasks, B, k=200, p=2.0, a=0.45, eps=1e-3):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        N = int((np.asarray(L.sum(0)).ravel() > 0).sum())
        ns = len(t.seeds)
        lp = -hypergeom.logsf(inter - 1, N, ns, sz) / np.log(10)
        sim = np.where(inter > 0, np.nan_to_num(lp, posinf=300.0), 0.0)
        rows, w = ops.topk(sim, B.own_row(t), k)
        out[i] = (ops.propagate(L, rows, w ** p) + eps * ops.tail(L, sim, B.own_row(t))) * dw
    return out
