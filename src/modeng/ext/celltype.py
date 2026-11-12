"""R5 cell-type witnesses (sweep 03 rows 1-2): CT-ACT (cell type of action) and SC-3 cell-address donors.

Substrates (both temporal-clean: data dated before 2021-04-30, so clean on past, temporal and forward):
  hpa  HPA v20.1 rna_single_cell_type.tsv (51 cell types, NX; file dated 2020-11-16; /mnt/msadb/subex/raw/hpa_v20).
       Specificity Z = log1p(NX), column-centred, then z across cell types per gene.
  pops PoPS pathway-naive feature matrix (Weeks et al. 2023; Zenodo 12635505), only the `*_diffexprs_tstat_clusters*`
       columns (2,583 human + mouse single-cell cluster DE t-statistics from pre-2020 atlases), each column z-scored over genes.
       Genes with a constant PoPS row (unmapped / filled) are missing.
Expression matching covariate: mean log1p NX over HPA v20.1 tissue consensus (10 quantile bins + a 'missing' bin).

Enrichment of a gene set s over cluster c (competitive, analytic expression-matched null = random sets drawing each member
from the member's expression bin, with replacement):  e_c = (mean_{g in s} Z_gc - mu0_c) / sd0_c,
  mu0_c = mean_{g in s} mu_{bin(g),c},  sd0_c^2 = sum_{g in s} var_{bin(g),c} / |s|^2   (sets with < 3 covered genes: NaN).

Channels
  ct_act        per substrate: w_c = e_c if e_c > 2 else 0 (seed enrichment); score_g = sum_c w_c Z_gc / sum_c w_c;
                NaN if no cluster passes or the gene is missing; ct_act = nanmean over the two substrates.
  ct_donor      address of a set = [max(e_hpa,0)/sqrt(C_hpa), max(e_pops,0)/sqrt(C_pops)] (NaN -> 0). Per task: cosine of the
                seed address to the address of every context row computed from its MASKED genes (rows carrying masked pairs
                recomputed from B.context_for-equivalent rows); own row and fam_rows removed; top-50 donors by cosine > 0;
                score = sum_donors cos * masked donor row (masked pairs subtracted exactly, quantised to 1e-5).
  ct_act_hpa, ct_act_pops   diagnostics (single substrate).
  ct_act_perm, ct_donor_perm   negative controls: gene axis of both substrates (Z, presence, expression bin) permuted (seed 7).
Uses only task seeds and the task's masked context. No label-process input.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import data as D

HPA = Path("/mnt/msadb/subex/raw/hpa_v20")
POPS = Path("/mnt/msadb/modet/downloads/pops_features/extract/pops_features_pathway_naive/munged_features")
CACHE = Path(__file__).resolve().parents[3] / "data/cache/celltype"
K = 50
E_MIN = 2.0
NB = 10
_S = {}


def _read_hpa(fname, col, val):
    with zipfile.ZipFile(HPA / f"{fname}.tsv.zip") as z:
        df = pd.read_csv(z.open(f"{fname}.tsv"), sep="\t", usecols=["Gene", col, val])
    return df.pivot_table(index="Gene", columns=col, values=val, aggfunc="mean")


def _hpa_Z():
    f = CACHE / "hpa_Z.npy"
    if f.exists():
        return np.load(f)
    u = D.universe()
    M = _read_hpa("rna_single_cell_type", "Cell type", "NX").reindex(u)
    L = np.log1p(M.to_numpy(np.float64))
    L = L - np.nanmean(L, 0, keepdims=True)
    sd = np.nanstd(L, 1, keepdims=True)
    Z = (L - np.nanmean(L, 1, keepdims=True)) / np.where(sd > 0, sd, np.nan)
    Z = Z.astype(np.float32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, Z)
    return Z


def _pops_Z():
    f = CACHE / "pops_Z.npy"
    if f.exists():
        return np.load(f)
    rows = pd.Index([l.strip() for l in open(POPS / "pops_features.rows.txt")])
    blocks = []
    i = 0
    while (POPS / f"pops_features.cols.{i}.txt").exists():
        cols = [l.strip() for l in open(POPS / f"pops_features.cols.{i}.txt")]
        idx = [j for j, c in enumerate(cols) if "_diffexprs_tstat_clusters" in c]
        if idx:
            m = np.load(POPS / f"pops_features.mat.{i}.npy", mmap_mode="r")
            blocks.append(np.asarray(m[:, idx], np.float32))
        i += 1
    X = np.hstack(blocks)
    const = X.std(1) == 0
    X[const] = np.nan
    X = (X - np.nanmean(X, 0)) / np.nanstd(X, 0)
    u = D.universe()
    ix = rows.get_indexer(u)
    Z = np.full((len(u), X.shape[1]), np.nan, np.float32)
    Z[ix >= 0] = X[ix[ix >= 0]]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, Z)
    return Z


def _expr_bins():
    f = CACHE / "expr_bin.npy"
    if f.exists():
        return np.load(f)
    T = _read_hpa("rna_tissue_consensus", "Tissue", "NX").reindex(D.universe())
    m = np.log1p(T.to_numpy(np.float64)).mean(1)
    b = np.full(len(m), NB, np.int64)
    ok = np.isfinite(m)
    q = np.quantile(m[ok], np.linspace(0, 1, NB + 1)[1:-1])
    b[ok] = np.searchsorted(q, m[ok])
    np.save(f, b)
    return b


def _substrate(name, perm=False):
    key = (name, perm)
    if key in _S:
        return _S[key]
    Z = _hpa_Z() if name == "hpa" else _pops_Z()
    bins = _expr_bins()
    if perm:
        p = np.random.default_rng(7).permutation(Z.shape[0])
        Z, bins = Z[p], bins[p]
    present = np.isfinite(Z).all(1)
    Z0 = np.where(present[:, None], Z, 0).astype(np.float64)
    Bm = sp.csr_matrix((np.ones(present.sum()), (np.flatnonzero(present), bins[present])), shape=(len(bins), NB + 1))
    mu = np.zeros((NB + 1, Z.shape[1])); var = np.ones_like(mu)
    for b in range(NB + 1):
        sel = present & (bins == b)
        if sel.sum() > 1:
            mu[b] = Z0[sel].mean(0); var[b] = Z0[sel].var(0)
    _S[key] = dict(Z=Z.astype(np.float32), Z0=Z0, present=present, Bm=Bm, mu=mu, var=var)
    return _S[key]


def enrich(Ind, sub):
    """Ind: sparse 0/1 [n_sets, n_genes] -> e [n_sets, C] (NaN for sets with < 3 covered genes)."""
    Ind = (sp.csr_matrix(Ind) @ sp.diags(sub["present"].astype(np.float64))).tocsr()
    n = np.asarray(Ind.sum(1)).ravel()
    nn = np.maximum(n, 1)[:, None]
    obs = (Ind @ sub["Z0"]) / nn
    cnt = (Ind @ sub["Bm"]).toarray()
    mu0 = cnt @ sub["mu"] / nn
    sd0 = np.sqrt(cnt @ sub["var"]) / nn
    e = (obs - mu0) / np.maximum(sd0, 1e-9)
    e[n < 3] = np.nan
    return e


def _ind(sets, n):
    r = np.concatenate([np.full(len(s), i) for i, s in enumerate(sets)]) if sets else np.array([], int)
    c = np.concatenate([np.asarray(s, np.int64) for s in sets]) if sets else np.array([], int)
    return sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(sets), n))


def _act(B, name, perm=False):
    sub = _substrate(name, perm)
    E = enrich(_ind([t.seeds for t in B.tasks], B.n_genes), sub)
    W = np.where(np.nan_to_num(E, nan=-1) > E_MIN, E, 0)
    out = np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    for i in range(len(B.tasks)):
        if W[i].sum() > 0:
            out[i] = (sub["Z"] @ W[i] / W[i].sum()).astype(np.float32)
    return out


def ct_act(B, perm=False):
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack([_act(B, "hpa", perm), _act(B, "pops", perm)]), 0).astype(np.float32)


def _address(E_list, subs):
    parts = [np.nan_to_num(np.clip(E, 0, None)) / np.sqrt(s["Z"].shape[1]) for E, s in zip(E_list, subs)]
    A = np.hstack(parts)
    return A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)


def ct_donor(B, perm=False):
    subs = [_substrate("hpa", perm), _substrate("pops", perm)]
    L = B.context.tocsr()
    A_ctx = _address([enrich(L, s) for s in subs], subs)                              # unmasked rows
    A_seed = _address([enrich(_ind([t.seeds for t in B.tasks], B.n_genes), s) for s in subs], subs)
    out = np.zeros((len(B.tasks), B.n_genes), np.float32)
    for i, t in enumerate(B.tasks):
        A = A_ctx
        if t.mask_r is not None and len(t.mask_r):
            rows = np.unique(t.mask_r)
            Lm = (L[rows] - sp.csr_matrix((np.ones(len(t.mask_r)), (np.searchsorted(rows, t.mask_r), t.mask_c)),
                                          shape=(len(rows), B.n_genes))).tocsr()
            Lm.eliminate_zeros()
            A = A_ctx.copy()
            A[rows] = _address([enrich(Lm, s) for s in subs], subs)
        sim = A @ A_seed[i]
        own = B.own_row(t)
        if own >= 0:
            sim[own] = 0
        if t.fam_rows is not None and len(t.fam_rows):
            sim[t.fam_rows] = 0
        cand = np.flatnonzero(sim > 0)
        if len(cand) == 0:
            continue
        top = cand[np.argsort(-sim[cand])[:K]]
        w = np.zeros(L.shape[0]); w[top] = sim[top]
        s = np.asarray(L[top].T @ sim[top]).ravel()
        if t.mask_r is not None and len(t.mask_r):
            np.subtract.at(s, t.mask_c, w[t.mask_r])
        out[i] = np.round(s, 5)
    return out


CHANNELS = {
    "ct_act": lambda B, s: ct_act(B),
    "ct_donor": lambda B, s: ct_donor(B),
    "ct_act_hpa": lambda B, s: _act(B, "hpa"),
    "ct_act_pops": lambda B, s: _act(B, "pops"),
    "ct_act_perm": lambda B, s: ct_act(B, perm=True),
    "ct_donor_perm": lambda B, s: ct_donor(B, perm=True),
}
