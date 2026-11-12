"""Smarter disease-side similarity for transfer (implementer C, round 1: IR-6, CI-04, ML-8, EASE/SLIM).

Channels (all masked exactly: either the generic quantised correction of baselines._transfer, or masked rows dropped):
  dc_trend      IR-6: transfer with donor matrix V = L + 1 * Delta, Delta = context pairs absent in the previous release
                (temporal: 21.04 vs 19.04; forward: 25.09 vs 21.04). Jaccard top-50 weights (as transfer_jaccard).
  dc_delta      transfer with V = Delta only (the "trending" part alone; for the stacker).
  dc_directed   CI-04: w(a->d) = jaccard * exp(asym), asym = log((n_ad+1)/(n_da+1)) clipped [-3,3], n_ad = genes of donor a in
                the previous release that are new in d's current (masked) row; top-50 by w.
  dc_learned    ML-8 (text-free): LightGBM regression of donor precision (share of a donor's non-seed genes that arrive in the
                window) from pair features; trained on `past` (19.04 -> 21.04) episodes, 2-fold cluster cross-fit on `past`
                itself; transfer over top-50 donors by predicted precision.
  dc_ridge      EASE/SLIM-style: seed vector regressed on unit-norm donor rows (ridge lambda = 1); every row carrying a masked
                pair (and the fam rows) is removed as a donor -> exactly masked. score = X_K^T w.
  dc_ridge_pos  same, w clipped >= 0, top-50 donors.
  dc_negperm_<ch>  negative control: channel <ch> with gene columns permuted by a fixed random permutation.

Previous-release channels (dc_trend, dc_delta, dc_directed) are NaN on `past` (no release before 19.04 on disk) and on
holdout (the 25.09 own rows would contain hidden 26.09 genes).
Leakage class: temporal-clean on temporal (19.04/21.04 labels, 21.04 ontology only) and on forward (21.04/25.09 labels);
dc_learned / dc_ridge also run on holdout (holdout-current).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .. import data as D
from ..methods.baselines import _indicator

K = 50
PREV = {"temporal": "1904", "forward": "2104", "temporal_wide": "1904"}   # temporal_wide = same 21.04 seed release
ONTO = ("2104",)


# ------------------------------------------------------------------------------------------------ helpers
def _jaccard(B):
    L = B.context
    sizes = L.getnnz(1).astype(np.float32)
    S = _indicator([t.seeds for t in B.tasks], B.n_genes)
    inter = (S @ L.T).toarray().astype(np.float32)   # seeds and targets disjoint -> unaffected by masking
    jac = np.zeros_like(inter)
    szm = np.zeros_like(inter)
    for i, t in enumerate(B.tasks):
        sz = sizes - B.row_masked_counts(t)
        szm[i] = sz
        jac[i] = inter[i] / np.maximum(len(t.seeds) + sz - inter[i], 1)
    return jac, inter, szm


def _own(B, t):
    i = B.context_ids.get_indexer([t.disease_id])[0]
    return int(i)


def _weighted_transfer(B, sims, V=None, k=K):
    """score = sum over top-k donors (by sims, > 0, own row excluded) of sims * V[donor]; V's nonzeros must be a subset of
    the context's; masked pairs removed exactly, then quantised to 1e-5 (tie-leak rule)."""
    L = B.context
    V = L if V is None else V.tocsr()
    out = np.zeros((len(B.tasks), B.n_genes), np.float32)
    for i, t in enumerate(B.tasks):
        sim = np.array(sims[i], np.float32, copy=True)
        own = _own(B, t)
        if own >= 0:
            sim[own] = 0
        cand = np.flatnonzero(sim > 0)
        if len(cand) == 0:
            continue
        top = cand[np.argsort(-sim[cand])[:k]]
        w = sim[top]
        out[i] = np.asarray(V[top].T @ w).ravel()
        wfull = np.zeros(len(sim), np.float32)
        wfull[top] = w
        if len(t.mask_r):
            vals = np.asarray(V[np.array(t.mask_r), np.array(t.mask_c)]).ravel().astype(np.float32)
            np.subtract.at(out[i], t.mask_c, wfull[t.mask_r] * vals)
        out[i] = np.round(out[i], 5)
    return out


def _prev_aligned(B):
    """previous-release matrix aligned to B.context_ids (rows absent in prev = 0) and a presence flag."""
    Lp, idp = D.label_matrix(PREV[B.name])
    ix = idp.get_indexer(B.context_ids)
    present = ix >= 0
    rows = np.where(present, ix, 0)
    P = Lp[rows].tocsr()
    P = sp.diags(present.astype(np.float32)) @ P
    P = P.tocsr(); P.eliminate_zeros()
    return P, present


def _delta(B):
    L = B.context.tocsr()
    P, present = _prev_aligned(B)
    Dl = (L - L.multiply(P)).tocsr()                  # in L, not in prev
    Dl = sp.diags(present.astype(np.float32)) @ Dl    # rows absent from prev: no delta information
    Dl = Dl.tocsr(); Dl.eliminate_zeros()
    return Dl, P, present


def _nan(B):
    return np.full((len(B.tasks), B.n_genes), np.nan, np.float32)


# ------------------------------------------------------------------------------------------------ IR-6
def dc_trend(B, split_tag, beta=1.0):
    if B.name not in PREV:
        return _nan(B)
    Dl, _, _ = _delta(B)
    jac, _, _ = _jaccard(B)
    return _weighted_transfer(B, jac, (B.context + beta * Dl).tocsr())


def dc_delta(B, split_tag):
    if B.name not in PREV:
        return _nan(B)
    Dl, _, _ = _delta(B)
    jac, _, _ = _jaccard(B)
    return _weighted_transfer(B, jac, Dl)


# ------------------------------------------------------------------------------------------------ CI-04
def dc_directed(B, split_tag, theta=1.0):
    if B.name not in PREV:
        return _nan(B)
    Dl, P, present = _delta(B)
    jac, _, _ = _jaccard(B)
    Dc = Dl.tocsc()
    sims = np.zeros_like(jac)
    for i, t in enumerate(B.tasks):
        own = _own(B, t)
        if own < 0 or not present[own]:
            sims[i] = jac[i]
            continue
        # masked delta row of d: drop masked genes of the own row
        dd = Dl[own].toarray().ravel()
        if len(t.mask_r):
            m = t.mask_r == own
            dd[t.mask_c[m]] = 0
        n_ad = np.asarray(P @ dd).ravel()                          # prev genes of a that are new in d
        pd_ = P[own].toarray().ravel()
        n_da = np.asarray(Dl @ pd_).ravel()                        # prev genes of d that are new in a
        if len(t.mask_r):                                          # remove masked pairs of donor rows a
            dv = np.asarray(Dl[np.array(t.mask_r), np.array(t.mask_c)]).ravel()
            np.subtract.at(n_da, t.mask_r, dv * pd_[t.mask_c])
        asym = np.clip(np.log((n_ad + 1) / (np.maximum(n_da, 0) + 1)), -3, 3)
        sims[i] = jac[i] * np.exp(theta * asym)
    return _weighted_transfer(B, sims)


# ------------------------------------------------------------------------------------------------ ML-8 (learned donor weights)
NCAND = 200


def _pair_feats(B):
    """per task: candidate donors (top-NCAND by Jaccard, inter > 0, own excluded) and features."""
    jac, inter, szm = _jaccard(B)
    rel = D.relatives_map(ONTO)
    desc = D.descendants_map(ONTO)
    pos = {d: i for i, d in enumerate(B.context_ids)}
    out = []
    for i, t in enumerate(B.tasks):
        j = jac[i].copy()
        own = _own(B, t)
        if own >= 0:
            j[own] = 0
        cand = np.flatnonzero(inter[i] > 0)
        if own >= 0:
            cand = cand[cand != own]
        cand = cand[np.argsort(-j[cand])[:NCAND]]
        ns = len(t.seeds)
        relset = {pos[x] for x in rel.get(t.disease_id, ()) if x in pos}
        descset = {pos[x] for x in desc.get(t.disease_id, ()) if x in pos}
        isrel = np.array([c in relset for c in cand], np.float32)
        isdesc = np.array([c in descset for c in cand], np.float32)
        it, sz = inter[i, cand], szm[i, cand]
        F = np.stack([j[cand], it, np.log1p(sz), np.full(len(cand), np.log1p(ns), np.float32), it / max(ns, 1),
                      it / np.maximum(sz, 1), isrel, isdesc, isrel * (1 - isdesc), np.arange(len(cand), dtype=np.float32),
                      np.log1p(np.maximum(sz - it, 0))], 1)
        out.append((cand, F))
    return out


def _pair_targets(B, feats):
    """donor precision: |L_t[a] & targets| / |L_t[a] \\ seeds| (masked context)."""
    L = B.context
    ys = []
    for i, t in enumerate(B.tasks):
        cand, F = feats[i]
        hit = np.asarray(L[cand][:, t.targets].sum(1)).ravel().astype(np.float32)
        if len(t.mask_r):
            mc = np.bincount(t.mask_r, minlength=L.shape[0]).astype(np.float32)
            hit -= mc[cand]
        novel = np.maximum(np.expm1(F[:, 10]), 1)   # |L_t[a]| - inter (masked size minus seed overlap)
        ys.append(np.clip(hit, 0, None) / novel)
    return ys


_LGB = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=0)


def _fit(Xs, ys):
    import lightgbm as lgb
    return lgb.train(_LGB, lgb.Dataset(np.vstack(Xs), np.concatenate(ys)), 300)


_PAST_MODELS = {}


def _past_models():
    """(full model, {disease_id: fold}, fold models) trained on all `past` episodes."""
    if _PAST_MODELS:
        return _PAST_MODELS
    from .. import tasks as T
    import hashlib
    P = T.load("past")
    feats = _pair_feats(P)
    ys = _pair_targets(P, feats)
    fold = np.array([int(hashlib.sha1((t.cluster or t.disease_id).encode()).hexdigest(), 16) % 2 for t in P.tasks])
    X = [f[1] for f in feats]
    _PAST_MODELS["full"] = _fit(X, ys)
    for k in (0, 1):
        idx = np.flatnonzero(fold != k)
        _PAST_MODELS[k] = _fit([X[i] for i in idx], [ys[i] for i in idx])
    _PAST_MODELS["fold"] = {t.disease_id: int(f) for t, f in zip(P.tasks, fold)}
    return _PAST_MODELS


def dc_learned(B, split_tag):
    M = _past_models()
    feats = _pair_feats(B)
    sims = np.zeros((len(B.tasks), B.context.shape[0]), np.float32)
    for i, t in enumerate(B.tasks):
        cand, F = feats[i]
        if len(cand) == 0:
            continue
        m = M[M["fold"][t.disease_id]] if B.name == "past" else M["full"]   # past: cross-fitted (other fold's model)
        p = np.clip(m.predict(F), 0, None)
        sims[i, cand] = p + 1e-6 * (p > 0)
    return _weighted_transfer(B, sims)


# ------------------------------------------------------------------------------------------------ EASE / SLIM ridge
LAMBDA = 1.0


def _ridge_w(B, drop_relatives=False):
    """per task donor weights w (n_tasks x n_rows) of the ridge regression s ~ X_K^T w, X = unit-norm context rows,
    K = rows without any masked pair and outside fam rows (own row included in fam)."""
    L = B.context.tocsr()
    nrm = np.sqrt(np.maximum(L.getnnz(1), 1)).astype(np.float64)
    X = sp.diags(1.0 / nrm) @ L
    G = (X @ X.T).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += LAMBDA
    Pinv = np.linalg.inv(G)
    S = _indicator([t.seeds for t in B.tasks], B.n_genes)
    Bm = (X @ S.T).toarray()                         # [rows, tasks] (seeds never masked)
    W = np.zeros((len(B.tasks), L.shape[0]), np.float32)
    pos = {d: i for i, d in enumerate(B.context_ids)}
    for i, t in enumerate(B.tasks):
        F = set(t.mask_r.tolist()) | set(t.fam_rows.tolist())
        if drop_relatives:                           # target-independent donor removal (all ontology relatives)
            F |= {pos[x] for x in D.relatives(t.disease_id) if x in pos}
        own = _own(B, t)
        if own >= 0:
            F.add(own)
        F = np.array(sorted(F), np.int64)
        b = Bm[:, i].copy()
        b[F] = 0
        y = Pinv @ b
        if len(F):
            corr = Pinv[:, F] @ np.linalg.solve(Pinv[np.ix_(F, F)], y[F])
            w = y - corr
            w[F] = 0
        else:
            w = y
        W[i] = w
    return W, X.tocsr()


def dc_ridge(B, split_tag):
    W, X = _ridge_w(B)
    return np.round(np.asarray((X.T @ W.T.astype(np.float64))).T, 6).astype(np.float32)


def dc_ridge_pos(B, split_tag):
    W, X = _ridge_w(B)
    return _weighted_transfer(B, np.clip(W, 0, None), X)   # masked rows have w = 0: correction is a no-op


def dc_ridge_rel(B, split_tag):
    """dc_ridge_pos with every ontology relative also removed as a donor (holdout leak diagnostic: on holdout the rows that
    carry masked pairs depend on the hidden genes, so removing exactly those rows is a target-dependent selection)."""
    W, X = _ridge_w(B, drop_relatives=True)
    return _weighted_transfer(B, np.clip(W, 0, None), X)


# ------------------------------------------------------------------------------------------------ negative control
def _negperm(name):
    def f(B, split_tag):
        from .. import channels as CH
        S = CH.compute(B, name, split_tag)
        perm = np.random.default_rng(12345).permutation(B.n_genes)
        return S[:, perm]
    return f


CHANNELS = {"dc_trend": dc_trend, "dc_delta": dc_delta, "dc_directed": dc_directed, "dc_learned": dc_learned,
            "dc_ridge": dc_ridge, "dc_ridge_pos": dc_ridge_pos, "dc_ridge_rel": dc_ridge_rel}
for _c in list(CHANNELS):
    CHANNELS["dc_negperm_" + _c] = _negperm(_c)
