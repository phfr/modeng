"""Channel priors, channel-typed transfer and phenotype matching (implementer H, round 1: critic_bio CI-06/SG-8, SG-4/B1, EV-2 note).

Motivation (critic_bio M3 / M7, re-verified here on modet arrival_devall_t0_string115_phys_notm_400): 86.4 % of gwas_only arrival
genes were already an L2G >= 0.5 winner for some trait at T0 (universe base 40.6 %); 74 % of ClinVar / curated / mixed non-GWAS
arrival genes had an HPO 2021-01 gene-phenotype profile (base 22.0 %).

Channels (fn(B, split_tag) -> float32 [n_tasks, n_genes]; NaN = missing; B.name == 'past' -> all NaN, no 19.04 evidence):
  pri_l2g_n      log1p(# evidence diseases where the gene is a GWAS winner: ot_genetics_portal L2G >= 0.5 (21.04) /
                 gwas_credible_sets score >= 0.5 (25.09)), task family (own disease + ontology descendants) excluded.
  pri_l2g_max    max GWAS score of the gene over all evidence diseases outside the task family.
  pri_mend_n     log1p(# diseases with ClinVar (eva >= 0.5) / curated (gene2phenotype, genomics_england, clingen, uniprot_*)
                 evidence, family excluded). Temporal only (evidence_2509 on disk holds only gwas_credible_sets): NaN on forward.
  pri_hpo_n      log1p(# HPO terms of the gene) (HPO 2021-01-15 temporal; HPO 2025-09-01 forward; current HPO holdout). NaN = no HPO.
  pop_<src>      per-datasource popularity: log1p(# evidence diseases of the gene in source src), family excluded
                 (temporal: europepmc, phenodigm, chembl, expression_atlas, eva, ot_genetics_portal (any score); forward: gwas_credible_sets).
  hz_gwas        pri_l2g_n * mix_g, mix_g = share of the seeds with GWAS-winner evidence in the task family (seed channel mix).
  hz_mend        pri_hpo_n(filled 0) * (1 - mix_g).
  tr_gwas        transfer_jaccard (top-50 Jaccard donors, own row excluded, exact masking + np.round(.,5)) where each donor row
                 keeps only its genes with GWAS-winner support (E_g = direct evidence of the row's disease or any descendant).
  tr_nongwas     same with the genes WITHOUT GWAS-winner support (ClinVar / curated / burden part of the donor rows).
                 Donor weights = Jaccard on the full masked rows (as transfer_jaccard).
  ph_mean        phenotype matching: gene x HPO-term matrix, IDF-weighted (idf = log(n_annot_genes / df)), rows L2-normalised;
                 disease profile = normalised sum of its seeds' rows; score = cosine(gene, profile). NaN for genes without HPO
                 or tasks without HPO-annotated seeds.
  ph_top3        mean of the 3 largest cosines of the gene to individual seeds.
  Negative controls: ph_mean_perm (HPO gene identities permuted), pri_l2g_n_perm (gene identities permuted).

Masking: priors use evidence of diseases outside the task family only (uniform over genes; no target-specific correction).
tr_*: donor rows are the masked context rows (B.context_for semantics, applied as an exact integer correction then quantised).
HPO gene annotations are gene-level (no disease id), seeds only define the profile.
Leakage class: temporal: OT evidence 21.04 + HPO 2021-01 -> temporal-clean. forward: evidence 25.09 + HPO 2025-09-01 ->
temporal-clean (inputs <= seed release). holdout: current HPO (2025-10) is holdout-current/transductive.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import data as D
from ..methods.baselines import _indicator

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/cache/priors_h"
EVDIR = Path("/mnt/msadb/modet/downloads/ot_evidence")
HPO = {"temporal": Path("/mnt/msadb/modet/downloads/hpo_annotations_2021-01_wayback/genes_to_phenotype_ALL_SOURCES_20210115.txt"),
       "forward": ROOT / "data/raw/hpo_2025-09-01/genes_to_phenotype.txt",
       "holdout": Path.home() / "biotools/data/evidence/raw/hpo/genes_to_phenotype.txt"}
REL = {"temporal": "2104", "forward": "2509", "holdout": "2509"}   # never 26.09 evidence as an input
HPO["temporal_wide"], REL["temporal_wide"] = HPO["temporal"], REL["temporal"]   # temporal_wide = same 21.04 seed release
GWAS_SRC = {"2104": "ot_genetics_portal", "2509": "gwas_credible_sets"}
MEND_SRC = ("eva", "gene2phenotype", "genomics_england", "clingen", "uniprot_literature", "uniprot_variants", "orphanet")
POP_SRC = {"2104": ("europepmc", "phenodigm", "chembl", "expression_atlas", "eva", "ot_genetics_portal"),
           "2509": ("gwas_credible_sets",)}
ALL_POP = sorted({s for v in POP_SRC.values() for s in v})
_MEMO: dict = {}


# ------------------------------------------------------------------------------------------------ inputs
def _evidence(rel):
    """Aggregated evidence (src, diseaseId, targetId, s = max score, n) of an OT release (built once with duckdb)."""
    f = CACHE / f"ev_{rel}.parquet"
    if not f.exists():
        import duckdb
        CACHE.mkdir(parents=True, exist_ok=True)
        p = EVDIR / f"evidence_{rel}" / "*" / "*.parquet"
        duckdb.connect().execute(
            f"COPY (SELECT datasourceId src, diseaseId, targetId, max(score) s, count(*) n FROM read_parquet('{p}', "
            f"hive_partitioning=true, union_by_name=true) GROUP BY ALL) TO '{f}'")
    return pd.read_parquet(f)


def _ev_mats(rel):
    """{name: (disease x gene csr of values, disease ids)} for the sources used here (universe order)."""
    key = ("ev", rel)
    if key in _MEMO:
        return _MEMO[key]
    ev = _evidence(rel)
    U = D.universe()
    ev = ev.assign(g=U.get_indexer(ev.targetId))
    ev = ev[ev.g >= 0]
    dids = pd.Index(sorted(ev.diseaseId.unique()))
    ev = ev.assign(r=dids.get_indexer(ev.diseaseId))

    def mat(sub, val):
        M = sp.csr_matrix((val.astype(np.float32), (sub.r.to_numpy(), sub.g.to_numpy())), shape=(len(dids), len(U)))
        M.sum_duplicates()
        return M

    out = {}
    gw = ev[ev.src == GWAS_SRC[rel]]
    out["gwas_score"] = _max_dup(gw, len(dids), len(U))
    w = gw[gw.s >= 0.5]
    out["gwas_win"] = (mat(w, np.ones(len(w))) > 0).astype(np.float32)
    me = ev[ev.src.isin(MEND_SRC) & ~((ev.src == "eva") & (ev.s < 0.5))]
    out["mend"] = (mat(me, np.ones(len(me))) > 0).astype(np.float32) if len(me) else None
    for s in ALL_POP:
        sub = ev[ev.src == s]
        out["pop_" + s] = (mat(sub, np.ones(len(sub))) > 0).astype(np.float32) if len(sub) else None
    _MEMO[key] = (out, dids)
    return out, dids


def _max_dup(sub, nr, nc):
    sub = sub.groupby(["r", "g"], as_index=False)["s"].max()
    return sp.csr_matrix((sub.s.to_numpy(np.float32), (sub.r.to_numpy(), sub.g.to_numpy())), shape=(nr, nc))


def _family_rows(task, dids):
    fam = set(D.descendants(task.disease_id)) | {task.disease_id}
    ix = dids.get_indexer(sorted(fam))
    return ix[ix >= 0]


def _hpo(bench):
    key = ("hpo", bench)
    if key in _MEMO:
        return _MEMO[key]
    f = HPO[bench]
    if bench in ("temporal", "temporal_wide"):
        h = pd.read_csv(f, sep="\t", comment="#", header=None, usecols=[0, 3], names=["entrez", "term"])
    else:
        h = pd.read_csv(f, sep="\t", usecols=["ncbi_gene_id", "hpo_id"]).rename(columns={"ncbi_gene_id": "entrez", "hpo_id": "term"})
    uh = pd.read_parquet(D.MODET / "data/processed/hgnc_protein_coding.parquet")
    U = D.universe()
    e2g = {int(e): g for e, g in zip(uh.entrez_id, uh.ensembl_gene_id) if pd.notna(e)}
    h = h.assign(g=U.get_indexer(h.entrez.map(e2g))).query("g >= 0").drop_duplicates(["g", "term"])
    terms = pd.Index(sorted(h.term.unique()))
    G = sp.csr_matrix((np.ones(len(h), np.float32), (h.g.to_numpy(), terms.get_indexer(h.term))), shape=(len(U), len(terms)))
    _MEMO[key] = G
    return G


# ------------------------------------------------------------------------------------------------ channels
def _nan(B):
    return np.full((len(B.tasks), B.n_genes), np.nan, np.float32)


def _count_excl(B, M, log=True):
    """per task: column count of M over evidence diseases outside the task family."""
    _, dids = _ev_mats(REL[B.name])
    col = np.asarray(M.sum(0)).ravel().astype(np.float32)
    out = np.empty((len(B.tasks), B.n_genes), np.float32)
    for i, t in enumerate(B.tasks):
        F = _family_rows(t, dids)
        out[i] = col - (np.asarray(M[F].sum(0)).ravel() if len(F) else 0)
    return np.log1p(out) if log else out


def _max_excl(B, M):
    _, dids = _ev_mats(REL[B.name])
    gmax = M.max(0).toarray().ravel().astype(np.float32)
    C = M.tocsc()
    out = np.empty((len(B.tasks), B.n_genes), np.float32)
    for i, t in enumerate(B.tasks):
        F = _family_rows(t, dids)
        v = gmax.copy()
        if len(F):
            touched = np.unique(M[F].indices)
            if len(touched):
                sub = C[:, touched].tocoo()
                keep = ~np.isin(sub.row, F)
                m = np.zeros(len(touched), np.float32)
                np.maximum.at(m, sub.col[keep], sub.data[keep])
                v[touched] = m
        out[i] = v
    return out


def _mix_g(B):
    mats, dids = _ev_mats(REL[B.name])
    W = mats["gwas_win"]
    mix = np.zeros(len(B.tasks), np.float32)
    for i, t in enumerate(B.tasks):
        F = _family_rows(t, dids)
        if len(F) and len(t.seeds):
            has = np.asarray(W[F].sum(0)).ravel() > 0
            mix[i] = has[t.seeds].mean()
    return mix


def pri_l2g_n(B, tag):
    if B.name == "past":
        return _nan(B)
    return _count_excl(B, _ev_mats(REL[B.name])[0]["gwas_win"])


def pri_l2g_n_perm(B, tag):
    S = pri_l2g_n(B, tag)
    p = np.random.default_rng(1).permutation(B.n_genes)
    return S[:, p]


def pri_l2g_max(B, tag):
    if B.name == "past":
        return _nan(B)
    return _max_excl(B, _ev_mats(REL[B.name])[0]["gwas_score"])


def pri_mend_n(B, tag):
    if B.name == "past":
        return _nan(B)
    M = _ev_mats(REL[B.name])[0]["mend"]
    return _nan(B) if M is None else _count_excl(B, M)


def _hpo_n(B):
    G = _hpo(B.name)
    n = np.asarray(G.sum(1)).ravel().astype(np.float32)
    v = np.where(n > 0, np.log1p(n), np.nan).astype(np.float32)
    return v


def pri_hpo_n(B, tag):
    if B.name == "past":
        return _nan(B)
    return np.tile(_hpo_n(B), (len(B.tasks), 1))


def _pop(src):
    def fn(B, tag):
        if B.name == "past":
            return _nan(B)
        M = _ev_mats(REL[B.name])[0]["pop_" + src]
        return _nan(B) if M is None else _count_excl(B, M)
    return fn


def hz_gwas(B, tag):
    if B.name == "past":
        return _nan(B)
    return pri_l2g_n(B, tag) * _mix_g(B)[:, None]


def hz_mend(B, tag):
    if B.name == "past":
        return _nan(B)
    v = np.nan_to_num(_hpo_n(B), nan=0.0)
    return (1 - _mix_g(B))[:, None] * v[None, :]


# ------------------------------------------------------------------------------------- channel-typed transfer
def _support(B):
    """E_g on the context rows: 1 where gene has GWAS-winner evidence for the row's disease or one of its descendants."""
    key = ("support", B.name)
    if key in _MEMO:
        return _MEMO[key]
    mats, dids = _ev_mats(REL[B.name])
    W = mats["gwas_win"]
    pos = {d: i for i, d in enumerate(dids)}
    rows, cols = [], []
    for r, d in enumerate(B.context_ids):
        ix = [pos[x] for x in ({d} | set(D.descendants(d))) if x in pos]
        if ix:
            g = np.unique(W[ix].indices)
            rows.append(np.full(len(g), r)); cols.append(g)
    r = np.concatenate(rows); c = np.concatenate(cols)
    E = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=B.context.shape)
    E.data[:] = 1
    _MEMO[key] = E
    return E


def _typed_transfer(B, part, k=50):
    """transfer_jaccard with donor rows restricted to the GWAS (part='g') or non-GWAS (part='n') genes of each row."""
    L = B.context
    E = _support(B)
    Lg = L.multiply(E).tocsr()
    Lp = Lg if part == "g" else (L - Lg).tocsr()
    Lp.eliminate_zeros()
    sizes = L.getnnz(1).astype(np.float32)
    S = _indicator([t.seeds for t in B.tasks], B.n_genes)
    inter = (S @ L.T).toarray()
    pos = {d: i for i, d in enumerate(B.context_ids)}
    out = np.zeros((len(B.tasks), B.n_genes), np.float32)
    for i, t in enumerate(B.tasks):
        sz = sizes - B.row_masked_counts(t)
        sim = inter[i] / np.maximum(len(t.seeds) + sz - inter[i], 1)
        own = pos.get(t.disease_id)
        if own is not None:
            sim[own] = 0
        cand = np.flatnonzero(sim > 0)
        if len(cand) == 0:
            continue
        top = cand[np.argsort(-sim[cand])[:k]]
        w = sim[top].astype(np.float32)
        out[i] = np.asarray(Lp[top].T @ w).ravel()
        wfull = np.zeros(len(sim), np.float32)
        wfull[top] = w
        # masked pairs that are present in the typed part (the masked pairs are all 1 in L)
        inpart = np.asarray(Lp[np.array(t.mask_r), np.array(t.mask_c)]).ravel() > 0 if len(t.mask_r) else np.zeros(0, bool)
        np.subtract.at(out[i], t.mask_c[inpart], wfull[t.mask_r[inpart]])
        out[i] = np.round(out[i], 5)   # leakage rule 1: quantise after the correction
    return out


def tr_gwas(B, tag):
    return _nan(B) if B.name == "past" else _typed_transfer(B, "g")


def tr_nongwas(B, tag):
    return _nan(B) if B.name == "past" else _typed_transfer(B, "n")


# ------------------------------------------------------------------------------------------- phenotype matching
def _pheno(B, perm=False, topk=3):
    G = _hpo(B.name).astype(np.float32)
    if perm:
        G = G[np.random.default_rng(2).permutation(G.shape[0])]
    has = np.asarray(G.getnnz(1)) > 0
    df = np.asarray((G > 0).sum(0)).ravel()
    idf = np.log(has.sum() / np.maximum(df, 1)).astype(np.float32)
    Gw = (G @ sp.diags(idf)).tocsr()
    nr = np.sqrt(np.asarray(Gw.multiply(Gw).sum(1)).ravel())
    Gw = (sp.diags(1 / np.maximum(nr, 1e-9)) @ Gw).tocsr()
    mean = _nan(B); top = _nan(B)
    for i, t in enumerate(B.tasks):
        s = t.seeds[has[t.seeds]]
        if len(s) == 0:
            continue
        p = np.asarray(Gw[s].sum(0)).ravel()
        p /= max(np.linalg.norm(p), 1e-9)
        mean[i] = Gw @ p
        C = (Gw @ Gw[s].T).toarray()
        kk = min(topk, C.shape[1])
        top[i] = np.sort(C, 1)[:, -kk:].mean(1)
    mean[:, ~has] = np.nan
    top[:, ~has] = np.nan
    return mean, top


def ph_mean(B, tag):
    return _nan(B) if B.name == "past" else _pheno(B)[0]


def ph_top3(B, tag):
    return _nan(B) if B.name == "past" else _pheno(B)[1]


def ph_mean_perm(B, tag):
    return _nan(B) if B.name == "past" else _pheno(B, perm=True)[0]


CHANNELS = {"pri_l2g_n": pri_l2g_n, "pri_l2g_max": pri_l2g_max, "pri_mend_n": pri_mend_n, "pri_hpo_n": pri_hpo_n,
            "hz_gwas": hz_gwas, "hz_mend": hz_mend, "tr_gwas": tr_gwas, "tr_nongwas": tr_nongwas,
            "ph_mean": ph_mean, "ph_top3": ph_top3, "ph_mean_perm": ph_mean_perm, "pri_l2g_n_perm": pri_l2g_n_perm}
CHANNELS.update({"pop_" + s: _pop(s) for s in ALL_POP})
