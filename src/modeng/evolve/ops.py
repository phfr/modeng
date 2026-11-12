"""Shared building blocks for evolved candidates (pure functions of the masked context and seeds)."""
import numpy as np


def overlap(L, seeds):
    """inter[r] = |seeds & row r|, sz[r] = |row r| (masked context)."""
    sz = L.getnnz(1).astype(np.float64)
    inter = np.asarray(L[:, seeds].sum(1)).ravel().astype(np.float64)
    return inter, sz


def jaccard(inter, sz, ns):
    return inter / np.maximum(ns + sz - inter, 1)


def cosine(inter, sz, ns):
    return inter / np.sqrt(np.maximum(ns * sz, 1))


def overlap_coef(inter, sz, ns):
    return inter / np.maximum(np.minimum(ns, sz), 1)


def topk(sim, own, k):
    sim = sim.copy()
    if own >= 0:
        sim[own] = 0
    cand = np.flatnonzero(sim > 0)
    if len(cand) == 0:
        return cand, sim[cand]
    top = cand[np.argsort(-sim[cand], kind="stable")[:k]]
    return top, sim[top]


def propagate(L, rows, w, rowscale=None):
    """sum_r w_r * rowscale_r * L[r]."""
    if len(rows) == 0:
        return np.zeros(L.shape[1])
    ww = w if rowscale is None else w * rowscale[rows]
    return np.asarray(L[rows].T @ ww.astype(np.float64)).ravel()


def tail(L, sim, own):
    """Transfer over ALL context rows with sim > 0 (no cut), max-normalised: orders the genes no top-k neighbour carries."""
    rows, w = topk(sim, own, L.shape[0])
    s = propagate(L, rows, w)
    return s / max(s.max(), 1e-12)


def item_cf(L, seeds):
    """Item-item collaborative filtering: sum over seeds of cosine(column g, column s) in the masked context."""
    df = np.asarray(L.sum(0)).ravel().astype(np.float64)
    inv = 1.0 / np.sqrt(np.maximum(df, 1))
    u = np.asarray(L[:, seeds] @ inv[seeds]).ravel()
    return np.asarray(L.T @ u).ravel() * inv
