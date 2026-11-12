"""cand_34 (<- 28 x 19): adaptive k = clip(8 n_seeds, 30, 300), cosine^2 transfer + 0.1 item-item CF + 1e-3 tail, x deg^-0.5."""
import numpy as np
from . import ops

def score(tasks, B, p=2.0, a=0.5, eps=1e-3, lam=0.1):
    dw = (B.degree + 1.0) ** -a
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        L = B.L(t); inter, sz = ops.overlap(L, t.seeds)
        sim = ops.cosine(inter, sz, len(t.seeds))
        rows, w = ops.topk(sim, B.own_row(t), int(np.clip(8 * len(t.seeds), 30, 300)))
        s = ops.propagate(L, rows, w ** p); s /= max(s.max(), 1e-12)
        c = ops.item_cf(L, t.seeds); c[t.seeds] = 0; c /= max(c.max(), 1e-12)
        out[i] = (s + lam * c + eps * ops.tail(L, sim, B.own_row(t))) * dw
    return out
