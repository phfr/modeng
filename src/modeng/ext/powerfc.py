"""ML-4: learned GWAS power-growth forecaster ("which sub-threshold gene signals cross significance as studies grow"), round 4.

Training pairs (all dated <= the benchmark cutoff; temporal: 2021-04-30, past: 2019-04-30):
  small = a MAGMA study of `gwas_d.studies()` (gene z, window 10 kb) with publication DATE <= cutoff;
  large = any GWAS Catalog study with DATE <= cutoff that shares a MAPPED_TRAIT_URI id with the small study, has effective N
          (cases if the sample text names cases, else total) >= 1.2 x the small study's, and >= 1 association p <= 5e-8.
  label(gene) = gene is a MAPPED_GENE of an association p <= 5e-8 of the large study (catalog-hit label; this is the evidence route
          Open Targets' GWAS channel reads, not MAGMA p < 2.5e-6, which would need summary statistics of every large study).
  Features per (pair, gene): small-study z, z percentile, max z of the 10 position-neighbours on each side, # neighbours with
          z > 3, MAGMA nsnps, gene length, local gene density, gnomAD LOEUF / pLI, pleiotropy (# other available studies with
          z > 4.56, studies sharing a trait id excluded), small-study log effective N, # genes z > 4.56, lambda (median z^2 / 0.455),
          log N ratio.
  Model: LightGBM (8 threads), negatives subsampled 1:10 (weight 10). Validation: GroupKFold(5) by small study (=trait).

Channels (fn(B, split_tag) -> float32 [n_tasks, n_genes], NaN = disease without an own study):
  pfc_own       max_s w_s * P(cross | features of study s, N ratio = K_GROW) over the studies gwas_own matches (ontology self /
                descendants w=1, specific ancestors w=0.7). Same coverage as gwas_own.
  pfc_sem       the same over the gwas_sem matches (ontology + TF-IDF semantic proxies, weight = match weight).
  pfc_own_perm  negative control: pfc_own with the gene axis permuted (fixed seed), same width / coverage.
Past (cutoff 2019-04-30): model trained on pairs with both studies <= 2019-04-30 and applied to studies <= 2019-04-30; NaN if fewer
than MIN_SMALL small studies have a pair.
Leakage class on temporal: temporal-clean by publication date (pairs, labels and inputs <= 2021-04-30); residual routes: GWAS
Catalog 2026 trait mapping / gene mapping of old studies (same caveat as gwas_catlag) and modeng ontology releases for matching.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from .. import data as D
from . import gwas_d as G

K_GROW = 2.5
HIT_P = 5e-8
MIN_RATIO = 1.2
MIN_SMALL = 8
NB = 10
NEG_FRAC = 0.1


def _parse_n(s):
    s = str(s).replace(",", "")
    tot = sum(int(x) for x in re.findall(r"(\d+)\s", s + " "))
    cas = sum(int(m) for m in re.findall(r"(\d+)[^\d]*?cases", s))
    return cas if cas > 0 else tot


@lru_cache(None)
def _cat():
    st = G.catalog_studies().copy()
    st["ids"] = st.MAPPED_TRAIT_URI.map(G._ids)
    st["neff"] = st["INITIAL SAMPLE SIZE"].map(_parse_n)
    return st


@lru_cache(None)
def _sym2i():
    u = D.universe(); pos = {g: i for i, g in enumerate(u)}; s2i = {}
    for g, s in D.symbols().items():
        if isinstance(s, str) and g in pos:
            s2i.setdefault(s, pos[g])
    return s2i


@lru_cache(None)
def _hits():
    """study accession -> sorted gene indices of MAPPED_GENE at p <= HIT_P."""
    a = pd.read_parquet(G.CAT / "associations_subset.parquet", columns=["STUDY ACCESSION", "MAPPED_GENE", "P-VALUE"])
    a = a[a["P-VALUE"] <= HIT_P].dropna(subset=["MAPPED_GENE"])
    s2i = _sym2i(); out = {}
    for acc, genes in zip(a["STUDY ACCESSION"], a.MAPPED_GENE):
        gi = {s2i[x] for x in re.split(r"\s*[-,;]\s*|\s+x\s+", genes) if x in s2i}
        if gi:
            out.setdefault(acc, set()).update(gi)
    return {k: np.array(sorted(v)) for k, v in out.items()}


@lru_cache(None)
def pairs(cutoff: str) -> pd.DataFrame:
    S = G.studies(); st = _cat(); hits = _hits()
    st = st[st.DATE <= cutoff]
    idx = {}
    for acc, ids in zip(st.index, st.ids):
        for i in ids:
            idx.setdefault(i, []).append(acc)
    rows = []
    for si, (g, d) in enumerate(zip(S.gcst, S.date)):
        if d > cutoff or g not in _cat().index:
            continue
        ids = _cat().ids[g]; e0 = _cat().neff[g]
        for c in {x for i in ids for x in idx.get(i, [])} - {g}:
            e1 = st.neff[c]
            if e0 > 0 and e1 >= MIN_RATIO * e0 and c in hits:
                rows.append(dict(si=si, small=g, large=c, e0=e0, e1=e1, lratio=np.log(e1 / e0), nhit=len(hits[c])))
    return pd.DataFrame(rows)


@lru_cache(None)
def _static():
    """per-gene static features + neighbour order: log gene length, density (#genes within 500 kb), loeuf, pli; order arrays."""
    u = D.universe()
    z = pd.read_csv("/home/ph/code/modet/data/raw/gnomad_v2.1.1/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz", sep="\t",
                    compression="gzip", usecols=["gene_id", "chromosome", "start_position", "end_position", "oe_lof_upper", "pLI"]
                    ).drop_duplicates("gene_id").set_index("gene_id").reindex(u)
    chrom = z.chromosome.fillna("").astype(str).to_numpy(); pos = z.start_position.to_numpy(float)
    ln = np.log10(np.maximum(z.end_position.to_numpy(float) - pos, 1.0))
    dens = np.full(len(u), np.nan)
    order = []
    for c in np.unique(chrom):
        if c == "":
            continue
        g = np.flatnonzero((chrom == c) & ~np.isnan(pos))
        g = g[np.argsort(pos[g])]
        order.append(g)
        p = pos[g]
        dens[g] = np.searchsorted(p, p + 5e5) - np.searchsorted(p, p - 5e5)
    return dict(lnlen=ln, dens=dens, loeuf=z.oe_lof_upper.to_numpy(float), pli=z.pLI.to_numpy(float), order=order)


@lru_cache(None)
def _nsnps():
    S = G.studies(); u = D.universe()
    M = np.full((len(S), len(u)), np.nan, np.float32)
    for i, p in enumerate(S.path):
        d = pd.read_parquet(p, columns=["gene", "nsnps"]).dropna().drop_duplicates("gene")
        j = u.get_indexer(d.gene); ok = j >= 0
        M[i, j[ok]] = np.log10(d.nsnps.to_numpy(np.float32)[ok] + 1)
    return M


def _nb(z, order):
    """max z and # z > 3 among the NB position-neighbours on each side (self excluded)."""
    mx = np.full_like(z, np.nan); cnt = np.full_like(z, np.nan)
    for g in order:
        x = np.nan_to_num(z[g], nan=0.0); n = len(x)
        pad = np.concatenate([np.full(NB, -np.inf), x, np.full(NB, -np.inf)])
        pz = np.concatenate([np.zeros(NB), (x > 3).astype(float), np.zeros(NB)])
        m = np.full(n, -np.inf); c = np.zeros(n)
        for k in list(range(-NB, 0)) + list(range(1, NB + 1)):
            m = np.maximum(m, pad[NB + k: NB + k + n]); c += pz[NB + k: NB + k + n]
        mx[g] = m; cnt[g] = c
    return mx, cnt


FEATS = ["z", "zpct", "nbmax", "nbcnt", "nsnps", "lnlen", "dens", "loeuf", "pli", "pleio", "le0", "nsig", "lam", "lratio"]


@lru_cache(None)
def study_feats(si: int, cutoff: str) -> np.ndarray:
    """[n_genes, len(FEATS)] for MAGMA study si; pleiotropy over studies available at cutoff not sharing a trait id. lratio = 0."""
    Z, _ = G.zmatrix(); S = G.studies(); stc = _static()
    z = Z[si].astype(float)
    ok = ~np.isnan(z)
    zp = np.full_like(z, np.nan); zp[ok] = pd.Series(z[ok]).rank(pct=True).to_numpy()
    mx, cnt = _nb(z, stc["order"])
    av = (S.date <= cutoff).to_numpy().copy(); av[si] = False
    ids = S.ids[si]
    av &= np.array([len(ids & x) == 0 for x in S.ids])
    pleio = np.nansum(Z[av] > G.Z_THR, 0).astype(float)
    g = S.gcst[si]
    e0 = _cat().neff.get(g, np.nan) if g in _cat().index else np.nan
    zz = z[ok]
    nsig = float((zz > G.Z_THR).sum()); lam = float(np.median(zz ** 2) / 0.455)
    F = np.column_stack([z, zp, mx, cnt, _nsnps()[si], stc["lnlen"], stc["dens"], stc["loeuf"], stc["pli"], pleio,
                         np.full_like(z, np.log10(max(e0, 1)) if np.isfinite(e0) else np.nan), np.full_like(z, nsig),
                         np.full_like(z, lam), np.zeros_like(z)])
    F[~ok] = np.nan
    return F.astype(np.float32)


def training_table(cutoff: str, seed=0):
    P = pairs(cutoff); hits = _hits(); rng = np.random.default_rng(seed)
    Xs, ys, ws, gs, ps = [], [], [], [], []
    for k, r in enumerate(P.itertuples()):
        F = study_feats(r.si, cutoff).copy(); F[:, -1] = r.lratio
        y = np.zeros(len(F), np.int8); y[hits[r.large]] = 1
        ok = ~np.isnan(F[:, 0])
        keep = ok & ((y == 1) | (rng.random(len(F)) < NEG_FRAC))
        Xs.append(F[keep]); ys.append(y[keep]); ws.append(np.where(y[keep] == 1, 1.0, 1 / NEG_FRAC))
        gs.append(np.full(keep.sum(), r.si)); ps.append(np.full(keep.sum(), k))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws), np.concatenate(gs), np.concatenate(ps)


PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=0)
ROUNDS = 300


@lru_cache(None)
def model(cutoff: str):
    import lightgbm as lgb
    P = pairs(cutoff)
    if len(P) == 0 or P.si.nunique() < MIN_SMALL:
        return None
    X, y, w, _, _ = training_table(cutoff)
    return lgb.train(PARAMS, lgb.Dataset(X, y, weight=w, feature_name=FEATS), ROUNDS)


@lru_cache(None)
def crossprob(si: int, cutoff: str, k: float = K_GROW):
    m = model(cutoff)
    F = study_feats(si, cutoff).copy(); F[:, -1] = np.log(k)
    p = m.predict(F).astype(np.float32)
    p[np.isnan(F[:, 0])] = np.nan
    return p


def _chan(B, sem):
    cutoff = G.CUTOFF.get(B.name, "9999-12-31")
    out = np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    if model(cutoff) is None:
        return out
    for t, m in enumerate(G._matches(B, sem=sem)):
        if m:
            out[t] = np.nanmax(np.stack([w * crossprob(i, cutoff) for i, w in m.items()]), 0)
    return out


def pfc_own(B, split_tag):
    return _chan(B, False)


def pfc_sem(B, split_tag):
    return _chan(B, True)


def pfc_own_perm(B, split_tag):
    S = _chan(B, False)
    return S[:, np.random.default_rng(0).permutation(S.shape[1])]


CHANNELS = {"pfc_own": pfc_own, "pfc_sem": pfc_sem, "pfc_own_perm": pfc_own_perm}
