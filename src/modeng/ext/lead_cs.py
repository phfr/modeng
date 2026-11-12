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


_DON = {}


def _donor_stack(families):
    import scipy.sparse as sp
    from modeng.channels import load_family
    key = tuple(families)
    if key not in _DON:
        _DON[key] = sp.vstack([load_family(f) for f in families]).tocsr()
    return _DON[key]


# virtual-donor families admissible on every benchmark incl. past/temporal (dated <= 2021-04 or sequence/label-free);
# hpo2101 is dated 2021-01 (> 19.04): admissible on temporal, transductive on past training episodes (only biases the learned weight)
UNI_FAMILIES = ["msig_go", "msig_reactome", "msig_wiki", "hpo2101", "impc13", "complexportal", "paralog30",
                "km_esm2_15b", "km_depmap20q4", "km_phylo_npp"]


def _cs_universal(B, split_tag, alpha=1e-5, top=300, top_don=300):
    """F1-CSDONOR extended: one joint non-negative lasso of the seed indicator over the top disease rows AND the top
    virtual-donor sets (Jaccard-preselected), so diseases and gene-set families compete to explain the seeds."""
    from sklearn.linear_model import Lasso
    D = _donor_stack(UNI_FAMILIES)
    dsz = D.getnnz(1).astype(np.float32)
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
        cand = cand[np.argsort(-jac[cand])[:top]]
        dint = np.asarray(D[:, t.seeds].sum(1)).ravel()
        djac = dint / np.maximum(len(t.seeds) + dsz - dint, 1)
        dc = np.flatnonzero(djac > 0)
        dc = dc[np.argsort(-djac[dc])[:top_don]]
        X = sp.vstack([L[cand].multiply(1.0 / np.sqrt(np.maximum(sz[cand], 1))[:, None]),
                       D[dc].multiply(1.0 / np.sqrt(np.maximum(dsz[dc], 1))[:, None])]).T.tocsc().astype(np.float64)
        if X.shape[1] == 0:
            continue
        s = np.zeros(B.n_genes); s[t.seeds] = 1.0
        m = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=2000).fit(X, s)
        out[i] = np.asarray(X @ m.coef_).ravel()
    return out


CHANNELS["cs_universal"] = _cs_universal
