"""E-A4 backlog transfer (round 2): donor weights applied to the donors' GWAS Catalog backlog.

Matrix C [context rows x genes]: C[r, g] = 1 if the GWAS Catalog has an association p <= 1e-6 with MAPPED_GENE g for a trait
mapped to context disease r or one of its ontology descendants, added to the catalog on or before the benchmark cutoff
(gwas_d.CUTOFF: past 2019-04-30, temporal 2021-04-30, forward 2025-09-30, holdout none). Same association table, p filter, gene
parsing and date filter as `gwas_catlag` (gwas_d._assoc), but for every context disease instead of only the task's own family.

Per task, donor weights w over context rows; score = C^T w (quantised to 1e-5). Donors removed (w = 0): the task's own row,
fam rows, every row carrying a masked pair (as in disease_c._ridge_w) and every row whose catalog trait set intersects the
task's own trait set (own id + descendants), so the task's own catalog hits (= gwas_catlag) never enter through an ancestor.

Channels
  catxfer_ridge      w = clip(dc_ridge weights, 0)       (EASE ridge of the seed vector on unit-norm donor rows, lambda 1)
  catxfer_jac        w = Jaccard(seeds, masked donor row), top-50 donors
  catxfer_ridge_new  catxfer_ridge on C_new = C minus the donor's own context label (catalog backlog not yet in OT at the seed date)
  catxfer_jac_new    catxfer_jac on C_new
  catxfer_ridge_perm / catxfer_jac_perm / catxfer_ridge_new_perm   negative controls: gene axis of C permuted (fixed seed 7)

Leakage class = gwas_catlag: clean by catalog date-added, residual route = 2026-10 trait/gene mapping of the catalog release; the
context is the benchmark's (masked) context only.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .. import data as D
from . import gwas_d as G

K = 50
_C = {}
_W = {}


def _trait_matrix(B):
    """T [traits x genes] binary (catalog hits <= cutoff), trait index, M [context rows x traits] (own + descendants)."""
    key = B.name
    if key in _C:
        return _C[key]
    a = G._assoc()
    a = a[a.added <= G.CUTOFF.get(B.name, "9999-12-31")]
    a = a.drop_duplicates(["trait", "gene"])
    tr = {t: i for i, t in enumerate(sorted(a.trait.unique()))}
    T = sp.csr_matrix((np.ones(len(a), np.float32), (a.trait.map(tr).to_numpy(), a.gene.to_numpy())),
                      shape=(len(tr), B.n_genes))
    rows, cols = [], []
    for r, d in enumerate(B.context_ids):
        for x in ({d} | set(D.descendants(d))):
            j = tr.get(x)
            if j is not None:
                rows.append(r); cols.append(j)
    M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(B.context_ids), len(tr)))
    C = (M @ T).tocsr()
    C.data[:] = 1.0
    _C[key] = (C, M.tocsc(), tr)
    return _C[key]


def _blocked(B, t, M, tr):
    """donor rows that may not be used for task t."""
    F = set(t.mask_r.tolist()) | set(t.fam_rows.tolist())
    own = B.own_row(t)
    if own >= 0:
        F.add(own)
    oj = [tr[x] for x in ({t.disease_id} | set(D.descendants(t.disease_id))) if x in tr]
    if oj:
        F |= set(np.flatnonzero(np.asarray(M[:, oj].sum(1)).ravel() > 0).tolist())
    return np.array(sorted(F), np.int64)


def _weights(B, kind):
    key = (B.name, kind, tuple(t.disease_id for t in B.tasks))
    if key in _W:
        return _W[key]
    _, M, tr = _trait_matrix(B)
    if kind == "ridge":
        from .disease_c import _ridge_w
        W, _ = _ridge_w(B)
        W = np.clip(W, 0, None)
    else:
        from .disease_c import _jaccard
        W, _, _ = _jaccard(B)
        W = W.astype(np.float32)
    W = W.copy()
    for i, t in enumerate(B.tasks):
        W[i, _blocked(B, t, M, tr)] = 0
        if kind == "jac":
            w = W[i]
            cand = np.flatnonzero(w > 0)
            keep = cand[np.argsort(-w[cand])[:K]]
            m = np.zeros_like(w); m[keep] = w[keep]
            W[i] = m
    _W[key] = W
    return W


def _score(B, kind, new=False, perm=False):
    C, _, _ = _trait_matrix(B)
    if new:
        C = (C - C.multiply(B.context > 0)).tocsr()
        C.eliminate_zeros()
    if perm:
        C = C[:, np.random.default_rng(7).permutation(B.n_genes)]
    W = _weights(B, kind)
    S = np.asarray((C.T @ sp.csr_matrix(W.T.astype(np.float64))).toarray()).T
    return np.round(S, 5).astype(np.float32)


CHANNELS = {
    "catxfer_ridge": lambda B, s: _score(B, "ridge"),
    "catxfer_jac": lambda B, s: _score(B, "jac"),
    "catxfer_ridge_new": lambda B, s: _score(B, "ridge", new=True),
    "catxfer_jac_new": lambda B, s: _score(B, "jac", new=True),
    "catxfer_ridge_perm": lambda B, s: _score(B, "ridge", perm=True),
    "catxfer_jac_perm": lambda B, s: _score(B, "jac", perm=True),
    "catxfer_ridge_new_perm": lambda B, s: _score(B, "ridge", new=True, perm=True),
}
