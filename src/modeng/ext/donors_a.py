"""Virtual-donor transfer, implementer A (round 1: SC-2 experiments as donors, LT-2 literature donors, EV-1 phenolog donors).

Donor families: data/donors/*.npz (old, MANIFEST.tsv) and data/donors/a_*.npz (new, MANIFEST_A.tsv; scripts/build_donors_a.py).

Scorer `hyper` (channels dA_<family>): for task seeds S (restricted to genes covered by the family, n = |S|, universe N =
genes covered by the family) and donor set d (size m_d, overlap k_d = |S & d|):
    w_d = -log10 P_hypergeom(X >= k_d | N, m_d, n)        if k_d >= KMIN, else 0
    keep the top-K donors by w_d;  score(g) = idf(g) * sum_d w_d * M[d, g] / m_d ** ALPHA
    idf(g) = log((1 + n_sets) / (1 + df(g)))  (df = number of sets containing g; damps hub genes in many sets)
K / ALPHA / IDF / KMIN chosen on temporal dev only (scripts/tune_donors_a.py, results/donors_a/tune.tsv).
Genes not covered by the family: NaN.

Channels
  dA_<fam>          hyper scorer on family <fam> (any family in either manifest)
  dA_pt             PubTator3 gene x MeSH co-mention family, cut by benchmark: past -> a_pt19 (PMID <= 2019-03),
                    temporal -> a_pt21 (<= 2021-03), forward -> a_pt25 (<= 2025-08); holdout -> NaN (literature of any date
                    would be transductive on a current-label CV).
  dA_perm_<fam>     negative control: the family with gene identities permuted (fixed seed), same scorer.
  dJ_perm_<fam>     negative control with the stock Jaccard scorer (channels.donor_transfer).

Leakage class: pre21 families (IMPC 13 + MGI orthology classes, DepMap 20Q4, PubTator cut, MSigDB 7.4, HPO 2021-01) are
temporal-clean; Perturb-seq K562/RPE1 (2022) is transductive on temporal (label-free cell data, fair on forward);
CD4 GWCD4i (2025) is transductive on temporal and treated as transductive on forward as well. The disease-derived literature
family (dA_pt) is dated by PMID per benchmark (temporal-clean when the cut precedes the seed release).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom

from .. import channels as CH

DON = CH.DON
PARAMS = dict(K=50, ALPHA=0.5, IDF=True, KMIN=1)   # set from scripts/tune_donors_a.py (temporal dev only)
PT_BY_BENCH = {"past": "a_pt19", "temporal": "a_pt21", "forward": "a_pt25"}


def families_a():
    f = DON / "MANIFEST_A.tsv"
    return [l.split("\t")[0] for l in open(f)] if f.exists() else []


def load_any(name):
    z = np.load(DON / f"{name}.npz")
    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))


def permuted(M, seed=0):
    p = np.random.default_rng(seed).permutation(M.shape[1])
    return sp.csr_matrix(M.toarray()[:, p]) if M.shape[0] * M.shape[1] < 3e8 else M[:, np.argsort(p)]


def hyper_transfer(tasks, M, K=50, ALPHA=0.5, IDF=True, KMIN=2):
    M = sp.csr_matrix(M, dtype=np.float32)
    M.data[:] = 1.0
    covered = np.asarray(M.getnnz(0) > 0).ravel()
    Ncov = int(covered.sum())
    m = M.getnnz(1).astype(np.float64)
    df = M.getnnz(0).astype(np.float64)
    idf = np.log((1 + M.shape[0]) / (1 + df)) if IDF else np.ones(M.shape[1])
    MT = M.T.tocsr()
    out = np.full((len(tasks), M.shape[1]), np.nan, np.float32)
    for i, t in enumerate(tasks):
        s = np.asarray(t.seeds)
        s = s[covered[s]]
        row = np.zeros(M.shape[1], np.float32)
        if len(s) == 0:
            out[i, covered] = 0.0
            continue
        k = np.asarray(MT[s].sum(0)).ravel()            # overlap per set
        ok = k >= KMIN
        w = np.zeros(M.shape[0])
        if ok.any():
            w[ok] = -hypergeom.logsf(k[ok] - 1, Ncov, m[ok], len(s)) / np.log(10)
        w[~np.isfinite(w)] = 300.0
        if (w > 0).any():
            top = np.argsort(-w)[:K]
            top = top[w[top] > 0]
            ww = (w[top] / m[top] ** ALPHA).astype(np.float32)
            row = np.asarray(M[top].T @ ww).ravel() * idf
        row = np.round(row, 5)
        out[i] = np.where(covered, row, np.nan)
    return out


def _make(fam, perm=False, jac=False):
    def fn(B, split_tag):
        M = load_any(fam)
        if perm:
            M = permuted(M)
        if jac:
            return CH.donor_transfer(B.tasks, M)
        return hyper_transfer(B.tasks, M, **PARAMS)
    return fn


def _pt(B, split_tag):
    fam = PT_BY_BENCH.get(B.name)
    if fam is None:
        return np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    return hyper_transfer(B.tasks, load_any(fam), **PARAMS)


CHANNELS = {}
for _f in CH.donor_families() + [f for f in families_a() if not f.startswith("a_pt")]:
    CHANNELS[f"dA_{_f}"] = _make(_f)
    CHANNELS[f"dA_perm_{_f}"] = _make(_f, perm=True)
    CHANNELS[f"dJ_perm_{_f}"] = _make(_f, perm=True, jac=True)
    CHANNELS[f"dJ_{_f}"] = _make(_f, jac=True)
CHANNELS["dA_pt"] = _pt
