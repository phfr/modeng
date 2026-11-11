"""Baseline methods. Each is `score(tasks, B) -> np.ndarray [n_tasks, n_genes]` (higher = more likely a target).

`B` is the `tasks.Benchmark` (context matrix, network names, leakage masks). Methods must only use
`B.context_for(task)` (or an equivalent exact correction, as done here for speed) when touching context labels.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .. import data as D


def _indicator(sets, n):
    r = np.concatenate([np.full(len(s), i) for i, s in enumerate(sets)])
    c = np.concatenate(sets)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(sets), n))


# ------------------------------------------------------------------------------------------------ trivial
def random(tasks, B):
    return np.random.default_rng(0).random((len(tasks), B.n_genes), dtype=np.float32)


def degree(tasks, B):
    d = D.degree(B.network).astype(np.float32)
    return np.broadcast_to(d, (len(tasks), B.n_genes)).copy()


def popularity(tasks, B):
    """Number of context diseases a gene is associated with (hidden pairs subtracted per task)."""
    col = np.asarray(B.context.sum(0)).ravel().astype(np.float32)
    out = np.empty((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        out[i] = col - B.hidden_counts(t)
    return out


# ------------------------------------------------------------------------------------------------ RWR
def rwr(tasks, A: sp.csr_matrix, restart=0.5, tol=1e-6, max_iter=100, chunk=512):
    """Random walk with restart from the seeds (uniform restart vector), column-stochastic weighted transition matrix.
    Batched: P <- (1 - r) W P + r P0 on [n_genes, chunk] dense blocks."""
    n = A.shape[0]
    deg = np.asarray(A.sum(0)).ravel()
    W = (A @ sp.diags(1.0 / np.maximum(deg, 1e-12))).astype(np.float32).tocsr()
    out = np.empty((len(tasks), n), np.float32)
    for s in range(0, len(tasks), chunk):
        sub = tasks[s:s + chunk]
        P0 = _indicator([t.seeds for t in sub], n).T.toarray()
        P0 /= np.maximum(P0.sum(0, keepdims=True), 1)
        P = P0.copy()
        for _ in range(max_iter):
            Pn = (1 - restart) * (W @ P) + restart * P0
            if np.abs(Pn - P).sum(0).max() < tol:
                P = Pn
                break
            P = Pn
        out[s:s + chunk] = P.T
    return out


def rwr_string(tasks, B):
    return rwr(tasks, D.network(B.network))


def rwr_union(tasks, B):
    return rwr(tasks, D.network(B.union_network))


# ------------------------------------------------------------------------------------------------ transfer
def _transfer(tasks, B, k=50, onto=False, floor=0.0):
    """score(g) = sum_{d' in N(d)} sim(d, d') * L_t[d', g], sim = Jaccard(seeds, L_t[d']), N = top-k by sim (all context
    rows, or ontology relatives only), own row excluded; L_t = context with the task's hidden pairs masked (applied
    exactly via a correction on the family rows)."""
    L = B.context
    sizes = L.getnnz(1).astype(np.float32)
    S = _indicator([t.seeds for t in tasks], B.n_genes)
    inter = (S @ L.T).toarray()                       # seeds and targets are disjoint -> unaffected by masking
    pos = {d: i for i, d in enumerate(B.context_ids)}
    out = np.zeros((len(tasks), B.n_genes), np.float32)
    for i, t in enumerate(tasks):
        sz = sizes - B.row_masked_counts(t)
        jac = inter[i] / np.maximum(len(t.seeds) + sz - inter[i], 1)
        if onto:
            allowed = np.array(sorted(pos[d] for d in D.relatives(t.disease_id) if d in pos), dtype=np.int64)
            sim = np.zeros_like(jac)
            if len(allowed):
                sim[allowed] = jac[allowed] + floor
        else:
            sim = jac
        own = pos.get(t.disease_id)
        if own is not None:
            sim[own] = 0
        cand = np.flatnonzero(sim > 0)
        if len(cand) == 0:
            continue
        top = cand[np.argsort(-sim[cand])[:k]]
        w = sim[top].astype(np.float32)
        out[i] = np.asarray(L[top].T @ w).ravel()
        # exact masking correction: subtract the masked pairs that fall in the chosen neighbour rows
        wfull = np.zeros(len(sim), np.float32)
        wfull[top] = w
        np.subtract.at(out[i], t.mask_c, wfull[t.mask_r])
    return out


def transfer_jaccard(tasks, B):
    return _transfer(tasks, B, k=50)


def transfer_onto(tasks, B):
    return _transfer(tasks, B, k=50, onto=True, floor=0.01)


# ------------------------------------------------------------------------------------------------ modet headline
MODET_HEADLINE = (D.MODET / "results/cache/runs/ucov_parity_string_s0/items/temporal__t0_string115_phys_notm_400/"
                  "components_t0_string115_phys_notm_400.npz")


def modet_headline(tasks, B):
    """modet-U-cov-Lb (head `lgbm_bin_cov_bcv`) scores from modet's devall STRING run `ucov_parity_string_s0`
    (fold-cross-fitted, T0 inputs). Temporal benchmark only. Off-network genes (not in the T0 STRING nodes) get the
    lowest score; diseases missing from the run get NaN (metrics NaN)."""
    if B.name != "temporal":
        raise ValueError("modet_headline exists only for the temporal benchmark")
    z = np.load(MODET_HEADLINE, allow_pickle=True)
    nodes = D.universe().get_indexer(z["nodes"].astype(str))
    col = {d: j for j, d in enumerate(z["disease_ids"])}
    M = z["lgbm_bin_cov_bcv"]
    out = np.full((len(tasks), B.n_genes), np.nan, np.float32)
    ok = nodes >= 0
    for i, t in enumerate(tasks):
        j = col.get(t.disease_id)
        if j is None:
            continue
        v = M[:, j]
        out[i] = np.nanmin(v) - 1.0
        out[i, nodes[ok]] = v[ok]
    return out
