"""GWAS statistics before they cross the threshold (implementer D, round 1: SG-7, CI-08 / WC-XPOWER, LT-9).

Inputs (all read-only from modet):
  * MAGMA gene z (window 10 kb) for 108 GWAS Catalog studies <= T0 (`modet/data/processed/gwas_magma_t0/<GCST>_w10.parquet`)
    and 92 post-T0 studies (`gwas_magma/<disease>.parquet`, one study per modet panel disease).
  * GWAS Catalog studies + associations (release 2026-10-22, `modet/data/raw/gwas_catalog`), for study dates, trait text,
    mapped trait ids (EFO/MONDO/...) and the SG-7 association table.

Study availability per benchmark (publication DATE of the study <= cutoff):
    past 2019-04-30 | temporal 2021-04-30 | forward 2025-09-30 | holdout: all (holdout-current).
  SG-7 (catalog associations) instead filters on DATE ADDED TO CATALOG <= cutoff (closest to an archived snapshot; trait
  re-mapping done later is still a residual back-fill route).

Channels (fn(B, split_tag) -> float32 [n_tasks, n_genes], NaN = disease not covered):
  gwas_own      max_s w_s * z_s(g) over studies mapped to the disease itself or an ontology descendant (w=1) or a
                specific ancestor (w=0.7, ancestors with <= 100 descendants). NaN if no study.
  gwas_sem      own + semantic proxies (LT-9): studies whose trait text has TF-IDF (char 3-5) cosine >= SEM_MIN with the
                disease name (top 5), weighted mean of standardised z with weight = cosine (ontology matches weight 1).
  gwas_xpow     WC-XPOWER crossing transform of the gwas_sem studies: per study a 2-component normal mixture
                z ~ pi0 N(0,1) + (1-pi0) N(0,1+tau2) (EM), posterior noncentrality, P(z' > z_thr at N'=3N);
                max over matched studies, times match weight.
  gwas_seedw    CI-08 / WC-XPOWER linear arm, works for every disease: per study an enrichment z of the task's seeds
                (sum of standardised z over seeds / sqrt(n)); weights = max(e_s - 2, 0); score = weighted mean z over studies.
                Uses seeds only (task-visible). NaN if no study is enriched.
  gwas_seedw_far  gwas_seedw with genes within 1 Mb of a seed set to 0 (borrowing at new loci only).
  gwas_catlag   SG-7: GWAS Catalog associations (p <= 1e-6) mapped to the disease or descendants (w=1) / specific ancestors
                (w=0.7), gene = MAPPED_GENE symbols, score = max w * -log10 p (capped 50); 0 for other genes, NaN if the
                disease has no association.
  gwas_catlag12 the same with associations added >= 12 months before the cutoff (ingestion-lag sensitivity).
  gwas_sem_perm / gwas_seedw_perm   negative controls: gene axis of Z permuted (fixed seed).

Leakage class: temporal: temporal-clean for MAGMA studies (all dated <= 2021-04; 92 of the 108 are the SAIGE UKB PheCode
GWAS of 2018 whose catalog accession was created later), task->trait mapping uses the modeng ontology releases (2104, 2509,
2609: mild transductive route), TF-IDF fitted on names only. forward: fair by publication date (<= 2025-09) but many
post-T0 PheCode studies were ADDED to the catalog on 2026-09-08 (summary statistics public since 2024-07). holdout: current.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from .. import data as D

MOD = Path("/home/ph/code/modet/data/processed")
CAT = Path("/home/ph/code/modet/data/raw/gwas_catalog")
CUTOFF = {"past": "2019-04-30", "temporal": "2021-04-30", "forward": "2025-09-30", "holdout": "9999-12-31"}
ANC_W, ANC_MAXDESC = 0.7, 100
SEM_MIN, SEM_TOP = 0.55, 5
Z_THR = norm.isf(2.5e-6)          # MAGMA gene-level genome-wide threshold
K_N = 3.0                         # N' = 3N


def _ids(uri) -> set:
    if not isinstance(uri, str):
        return set()
    return {u.strip().rsplit("/", 1)[-1] for u in uri.split(",") if u.strip()}


def _clean(txt: str) -> str:
    txt = re.sub(r"\((PheCode|UKB data field)[^)]*\)", " ", str(txt))
    txt = re.sub(r"ICD10 [A-Z]\d+(\.\d+)?:", " ", txt)
    return txt.lower().strip()


@lru_cache(None)
def catalog_studies() -> pd.DataFrame:
    st = pd.read_csv(CAT / "gwas-catalog-download-studies-v1.0.3.1.tsv", sep="\t", low_memory=False,
                     usecols=["STUDY ACCESSION", "DATE", "DATE ADDED TO CATALOG", "DISEASE/TRAIT", "MAPPED_TRAIT_URI",
                              "INITIAL SAMPLE SIZE"])
    return st.drop_duplicates("STUDY ACCESSION").set_index("STUDY ACCESSION")


@lru_cache(None)
def studies() -> pd.DataFrame:
    """One row per MAGMA study: gcst, path, date, added, trait text, mapped ids, modet disease ids."""
    st = catalog_studies()
    rows = []
    t0sel = pd.read_csv(MOD / "gwas_magma/studies_selected_t0.tsv", sep="\t")
    t0dis = t0sel.groupby("gcst")["disease_id"].apply(set).to_dict()
    runs = pd.read_csv(MOD / "gwas_magma_t0/magma_runs_t0.tsv", sep="\t")
    for g in runs.gcst:
        p = MOD / f"gwas_magma_t0/{g}_w10.parquet"
        if p.exists():
            rows.append(dict(gcst=g, path=str(p), src="t0", modet_ids=t0dis.get(g, set())))
    seen = {r["gcst"] for r in rows}
    post = {}
    for p in sorted((MOD / "gwas_magma").glob("*.parquet")):
        g = pd.read_parquet(p, columns=["gcst"]).gcst.iloc[0]
        post.setdefault(g, [str(p), set()])[1].add(p.stem)
    for g, (p, ids) in post.items():
        if g not in seen:
            rows.append(dict(gcst=g, path=p, src="post", modet_ids=ids))
    df = pd.DataFrame(rows)
    df["date"] = df.gcst.map(st["DATE"]).fillna("9999-12-31")
    df["added"] = df.gcst.map(st["DATE ADDED TO CATALOG"]).fillna("9999-12-31")
    df["trait"] = df.gcst.map(st["DISEASE/TRAIT"]).fillna("")
    df["ids"] = [(_ids(u) | m) for u, m in zip(df.gcst.map(st["MAPPED_TRAIT_URI"]), df.modet_ids)]
    return df.reset_index(drop=True)


@lru_cache(None)
def zmatrix():
    """Z [n_studies, n_genes] (raw MAGMA z, NaN where the gene is missing) and the standardised version."""
    S = studies()
    u = D.universe()
    Z = np.full((len(S), len(u)), np.nan, np.float32)
    for i, p in enumerate(S.path):
        d = pd.read_parquet(p, columns=["gene", "z"]).dropna().drop_duplicates("gene")
        j = u.get_indexer(d.gene)
        ok = j >= 0
        Z[i, j[ok]] = d.z.to_numpy(np.float32)[ok]
    med = np.nanmedian(Z, 1, keepdims=True)
    mad = np.nanmedian(np.abs(Z - med), 1, keepdims=True) * 1.4826
    Zs = (Z - med) / np.maximum(mad, 1e-3)
    return Z, Zs


def _avail(B) -> np.ndarray:
    S = studies()
    return (S.date <= CUTOFF.get(B.name, "9999-12-31")).to_numpy()


@lru_cache(None)
def _ndesc():
    return {k: len(v) for k, v in D.descendants_map().items()}


def _onto_w(d: str) -> dict:
    """disease id -> weight for trait ids: self/descendants 1, specific ancestors ANC_W."""
    own = {d} | set(D.descendants(d))
    nd = _ndesc()
    w = {x: 1.0 for x in own}
    for a in set(D.relatives(d)) - own:
        if nd.get(a, 0) <= ANC_MAXDESC:
            w.setdefault(a, ANC_W)
    return w


def _onto_match(d: str, avail: np.ndarray) -> dict:
    S = studies()
    w = _onto_w(d)
    out = {}
    for i in np.flatnonzero(avail):
        ws = [w[x] for x in S.ids[i] if x in w]
        if ws:
            out[i] = max(ws)
    return out


@lru_cache(None)
def _tfidf():
    from sklearn.feature_extraction.text import TfidfVectorizer
    S = studies()
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    v.fit([_clean(t) for t in S.trait] + [_clean(n) for n in D.disease_names("2609").astype(str).tolist()[:20000]])
    return v, v.transform([_clean(t) for t in S.trait])


def _sem_match(name: str, avail: np.ndarray) -> dict:
    v, M = _tfidf()
    q = v.transform([_clean(name)])
    c = (M @ q.T).toarray().ravel()
    c[~avail] = 0
    top = np.argsort(-c)[:SEM_TOP]
    return {int(i): float(c[i]) for i in top if c[i] >= SEM_MIN}


def _matches(B, sem=True):
    return _matches_c(B.name, tuple((t.disease_id, t.name) for t in B.tasks), sem)


@lru_cache(None)
def _matches_c(bname, dn, sem):
    av = (studies().date <= CUTOFF.get(bname, "9999-12-31")).to_numpy()
    out = []
    for did, name in dn:
        m = _sem_match(name, av) if sem else {}
        for i, w in _onto_match(did, av).items():
            m[i] = max(w, m.get(i, 0.0))
        out.append(m)
    return out


def _perm(Z, seed=0):
    p = np.random.default_rng(seed).permutation(Z.shape[1])
    return Z[:, p]


# ---------------------------------------------------------------------------------------------------------- channels
def gwas_own(B, split_tag):
    Z, _ = zmatrix()
    out = np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    for k, m in enumerate(_matches(B, sem=False)):
        if m:
            out[k] = np.nanmax(np.stack([w * Z[i] for i, w in m.items()]), 0)
    return out


def _sem(B, Zs):
    out = np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    for k, m in enumerate(_matches(B)):
        if m:
            idx = np.array(list(m)); w = np.array([m[i] for i in idx], np.float32)
            X = Zs[idx]; ok = ~np.isnan(X)
            num = np.nansum(X * w[:, None], 0); den = (ok * w[:, None]).sum(0)
            out[k] = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)
    return out


def gwas_sem(B, split_tag):
    return _sem(B, zmatrix()[1])


def gwas_sem_perm(B, split_tag):
    return _sem(B, _perm(zmatrix()[1]))


@lru_cache(None)
def _mixture():
    """Per study EM for z ~ pi0 N(0,1) + (1-pi0) N(0, 1+tau2) on the standardised z."""
    _, Zs = zmatrix()
    pars = []
    for z in Zs:
        z = z[~np.isnan(z)]
        pi0, tau2 = 0.9, 4.0
        for _ in range(100):
            f0 = pi0 * norm.pdf(z); f1 = (1 - pi0) * norm.pdf(z, scale=np.sqrt(1 + tau2))
            r = f1 / (f0 + f1 + 1e-300)
            pi0 = float(np.clip(1 - r.mean(), 0.5, 0.9999))
            tau2 = float(max((r * z * z).sum() / max(r.sum(), 1e-9) - 1, 0.05))
        pars.append((pi0, tau2))
    return np.array(pars)


def _crossprob(Zs):
    P = _mixture()
    out = np.full_like(Zs, np.nan)
    for s, z in enumerate(Zs):
        pi0, tau2 = P[s]
        f0 = pi0 * norm.pdf(z); f1 = (1 - pi0) * norm.pdf(z, scale=np.sqrt(1 + tau2))
        palt = f1 / (f0 + f1 + 1e-300)
        mu = z * tau2 / (1 + tau2); sd = np.sqrt(tau2 / (1 + tau2))
        # z' = sqrt(K) mu + e, mu ~ N(mu_post, sd^2)  ->  z' ~ N(sqrt(K) mu, 1 + K sd^2)
        out[s] = palt * norm.sf(Z_THR, loc=np.sqrt(K_N) * mu, scale=np.sqrt(1 + K_N * sd * sd))
    return out.astype(np.float32)


def gwas_xpow(B, split_tag):
    _, Zs = zmatrix()
    C = _crossprob(np.nan_to_num(Zs, nan=0.0))
    C[np.isnan(Zs)] = np.nan
    out = np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    for k, m in enumerate(_matches(B)):
        if m:
            out[k] = np.nanmax(np.stack([w * C[i] for i, w in m.items()]), 0)
    return out


def _seedw(B, Zs, thr=2.0):
    av = np.flatnonzero(_avail(B))
    out = np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    if len(av) == 0:
        return out
    X = Zs[av]
    for k, t in enumerate(B.tasks):
        s = X[:, t.seeds]
        n = (~np.isnan(s)).sum(1)
        e = np.nansum(s, 1) / np.sqrt(np.maximum(n, 1))
        w = np.maximum(e - thr, 0) * (n >= 3)
        if w.sum() <= 0:
            continue
        ok = ~np.isnan(X)
        num = np.nansum(X * w[:, None], 0); den = (ok * w[:, None]).sum(0)
        out[k] = np.where(den > 0, num / np.maximum(den, 1e-6), np.nan)
    return out


def gwas_seedw(B, split_tag):
    return _seedw(B, zmatrix()[1])


@lru_cache(None)
def _loc():
    from .kernels_b import _gene_loc
    return _gene_loc()


def gwas_seedw_far(B, split_tag, dist=1_000_000):
    """gwas_seedw with genes within `dist` of any seed set to 0 (neutral): tests borrowing at NEW loci only."""
    S = _seedw(B, zmatrix()[1])
    chrom, pos = _loc()
    for k, t in enumerate(B.tasks):
        if np.all(np.isnan(S[k])):
            continue
        near = np.zeros(B.n_genes, bool)
        for c in np.unique(chrom[t.seeds]):
            if c == "":
                continue
            sp_ = np.sort(pos[t.seeds][(chrom[t.seeds] == c) & ~np.isnan(pos[t.seeds])])
            g = np.flatnonzero(chrom == c)
            j = np.searchsorted(sp_, pos[g])
            d = np.minimum(np.abs(pos[g] - sp_[np.clip(j - 1, 0, len(sp_) - 1)]), np.abs(pos[g] - sp_[np.clip(j, 0, len(sp_) - 1)]))
            near[g[d < dist]] = True
        S[k, near & ~np.isnan(S[k])] = 0.0
    return S


def gwas_seedw_perm(B, split_tag):
    return _seedw(B, _perm(zmatrix()[1]))


@lru_cache(None)
def _assoc():
    a = pd.read_parquet(CAT / "associations_subset.parquet",
                        columns=["DATE ADDED TO CATALOG", "MAPPED_GENE", "P-VALUE", "MAPPED_TRAIT_URI"])
    a = a[a["P-VALUE"] <= 1e-6].dropna(subset=["MAPPED_GENE", "MAPPED_TRAIT_URI"])
    sym = D.symbols()
    s2i = {}
    u = D.universe()
    pos = {g: i for i, g in enumerate(u)}
    for g, s in sym.items():
        if isinstance(s, str) and g in pos:
            s2i.setdefault(s, pos[g])
    rows = []
    lp = np.minimum(-np.log10(np.maximum(a["P-VALUE"].to_numpy(), 1e-300)), 50.0)
    for added, genes, l, uri in zip(a["DATE ADDED TO CATALOG"], a.MAPPED_GENE, lp, a.MAPPED_TRAIT_URI):
        gi = {s2i[x] for x in re.split(r"\s*[-,;]\s*|\s+x\s+", genes) if x in s2i}
        if not gi:
            continue
        for tid in _ids(uri):
            for g in gi:
                rows.append((added, tid, g, l))
    df = pd.DataFrame(rows, columns=["added", "trait", "gene", "lp"])
    return df


def _shift(date: str, months: int) -> str:
    if date.startswith("9999"):
        return date
    return (pd.Timestamp(date) - pd.DateOffset(months=months)).strftime("%Y-%m-%d")


def gwas_catlag(B, split_tag, lag_months=0):
    a = _assoc()
    a = a[a.added <= _shift(CUTOFF.get(B.name, "9999-12-31"), lag_months)]
    best = a.groupby(["trait", "gene"]).lp.max()
    by_trait = {t: g for t, g in best.groupby(level=0)}
    out = np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    for k, t in enumerate(B.tasks):
        w = _onto_w(t.disease_id)
        hit = False
        row = np.zeros(B.n_genes, np.float32)
        for tid, wt in w.items():
            s = by_trait.get(tid)
            if s is None:
                continue
            hit = True
            gi = s.index.get_level_values(1).to_numpy()
            np.maximum.at(row, gi, (wt * s.to_numpy()).astype(np.float32))
        if hit:
            out[k] = row
    return out


def gwas_catlag12(B, split_tag):
    """Sensitivity: catalog associations added >= 12 months before the seed release only."""
    return gwas_catlag(B, split_tag, 12)


CHANNELS = {"gwas_catlag12": gwas_catlag12, "gwas_own": gwas_own, "gwas_sem": gwas_sem, "gwas_xpow": gwas_xpow, "gwas_seedw": gwas_seedw,
            "gwas_catlag": gwas_catlag, "gwas_sem_perm": gwas_sem_perm, "gwas_seedw_perm": gwas_seedw_perm,
            "gwas_seedw_far": gwas_seedw_far}
