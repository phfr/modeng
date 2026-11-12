"""E-A3 (round-2 evolution A, implemented by the lead): ridge regression of the seed vector on PubTator3 MeSH-disease literature
sets (dA_pt family, PMID-date-cut per benchmark: past<=2019-03, temporal<=2021-03, forward<=2025-08; holdout NaN), i.e. the dc_ridge
solver applied to the literature matrix. Per task: top-500 sets by seed overlap, unit-norm rows, w = (X X^T + lam I)^-1 X s,
score = X^T w (unclipped) and X^T max(w,0). Literature sets are not OT labels, so no masking is needed; leakage class as dA_pt."""
import numpy as np
import scipy.sparse as sp
from modeng.ext.donors_a import load_any, PT_BY_BENCH
from modeng.methods.baselines import _indicator

LAM, TOP = 1.0, 500
_C = {}


def _ptr(B, split_tag, pos=False):
    fam = PT_BY_BENCH.get(B.name)
    if fam is None:
        return np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    key = (B.name, split_tag, len(B.tasks))
    if key not in _C:
        M = load_any(fam).tocsr().astype(np.float64)
        X = sp.diags(1.0 / np.sqrt(np.maximum(M.getnnz(1), 1))) @ M
        X = X.tocsr()
        S = _indicator([t.seeds for t in B.tasks], B.n_genes)
        ov = (M @ S.T).toarray()                       # [sets, tasks]
        out_u = np.zeros((len(B.tasks), B.n_genes), np.float32); out_p = np.zeros_like(out_u)
        for i, t in enumerate(B.tasks):
            cand = np.flatnonzero(ov[:, i] > 0)
            if len(cand) == 0:
                continue
            cand = cand[np.argsort(-ov[cand, i])[:TOP]]
            Xc = X[cand]
            G = (Xc @ Xc.T).toarray(); G[np.diag_indices_from(G)] += LAM
            s = np.zeros(B.n_genes); s[t.seeds] = 1.0
            w = np.linalg.solve(G, Xc @ s)
            out_u[i] = np.asarray(Xc.T @ w).ravel()
            out_p[i] = np.asarray(Xc.T @ np.clip(w, 0, None)).ravel()
        _C[key] = (out_u, out_p)
    return _C[key][1] if pos else _C[key][0]


CHANNELS = {"pt_ridge": lambda B, s: _ptr(B, s), "pt_ridge_pos": lambda B, s: _ptr(B, s, True)}
