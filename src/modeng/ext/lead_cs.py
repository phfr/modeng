"""F1-CSDONOR (round-2 island 1, implemented by the lead): joint sparse non-negative reconstruction of the seed set over donor
rows instead of independent per-donor Jaccard weights. For each task: candidate donors = top-300 context rows by Jaccard with the
seeds (masked context via B.context_for), solve  min ||s - X b||^2 + alpha*|b|_1, b >= 0  with X = donor rows (genes x donors),
s = seed indicator; score = X b (genes covered by the donors that jointly explain the seeds). Leakage class = that of the context
(temporal-clean on temporal/past/forward). Exact masking (context_for), so no residue issue.
"""
import numpy as np
import scipy.sparse as sp


def _cs(B, split_tag, alpha=1e-5, top=300):
    from sklearn.linear_model import Lasso
    out = np.zeros((len(B.tasks), B.n_genes), np.float32)
    for i, t in enumerate(B.tasks):
        L = B.context_for(t)
        own = B.own_row(t)
        sz = L.getnnz(1).astype(np.float32)
        inter = np.asarray(L[:, t.seeds].sum(1)).ravel()
        jac = inter / np.maximum(len(t.seeds) + sz - inter, 1)
        if own >= 0:
            jac[own] = 0
        cand = np.flatnonzero(jac > 0)
        if len(cand) == 0:
            continue
        cand = cand[np.argsort(-jac[cand])[:top]]
        X = L[cand].T.tocsc().astype(np.float64)           # genes x donors
        X = X.multiply(1.0 / np.sqrt(np.maximum(sz[cand], 1))).tocsc()
        s = np.zeros(B.n_genes); s[t.seeds] = 1.0
        m = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=2000)
        m.fit(X, s)
        out[i] = np.asarray(X @ m.coef_).ravel()
    return out


CHANNELS = {"cs_donor": _cs}
