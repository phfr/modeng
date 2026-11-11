"""Data loaders for modeng (read-only views of modet's processed data, cached under `data/cache/`).

Everything is indexed on ONE gene universe: the 19,255 approved protein-coding HGNC genes (sorted Ensembl ids), identical
to modet's `modet_bench.universe.allgene_nodes`.

Releases (verified against `modet/src/modet_data/groundtruth.py` and `production.py`):
    "1904" -> ot_t19_sets.parquet          (OT 19.04 genetic sets, 21.04 ontology; modet's "past" transfer sets)
    "2104" -> ot_t0_sets.parquet           (OT 21.04 = modet T0, 21.04 ontology)
    "2509" -> ot_current_sets.parquet      (OT 25.09 = modet T1, 25.09 ontology; `production.current_sets("2509")`)
    "2609" -> ot_current_sets_2609.parquet (OT 26.09, 26.09 ontology)
All sets are Open Targets *genetic* associations (GWAS L2G >= 0.5, ClinVar >= 0.5, curated Mendelian/burden) propagated
to ancestors (a disease's set = union of its own and all descendants' direct genes).

Sealed set: modet's `folds_v1.json` diseases with `sealed: true` (single-use confirmation fold). `excluded_task_ids()` =
sealed ids + all their ontology ancestors/descendants (OT 21.04/25.09/26.09 union) that are not themselves non-sealed
modet panel diseases. Those ids are NEVER evaluation tasks here (see `tasks.py` for the context-matrix policy).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
MODET = Path("/home/ph/code/modet")
PROC = MODET / "data" / "processed"
GT = PROC / "groundtruth"
NETS = PROC / "networks"
FEAT = PROC / "features"
OT_RAW = MODET / "data" / "raw" / "ot"
FOLDS = MODET / "src" / "modet_bench" / "manifests" / "folds_v1.json"

SET_FILES = {"1904": "ot_t19_sets.parquet", "2104": "ot_t0_sets.parquet",
             "2509": "ot_current_sets.parquet", "2609": "ot_current_sets_2609.parquet"}
ONTOLOGY_RELEASES = ("2104", "2509", "2609")


def _cache(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / name


# ----------------------------------------------------------------------------------------------------------- genes
@lru_cache(None)
def universe() -> pd.Index:
    """Sorted unique Ensembl ids of the approved protein-coding HGNC genes (n = 19,255)."""
    h = pd.read_parquet(PROC / "hgnc_protein_coding.parquet")
    return pd.Index(np.sort(h["ensembl_gene_id"].dropna().astype(str).unique()), name="ensembl_gene_id")


@lru_cache(None)
def symbols() -> pd.Series:
    h = pd.read_parquet(PROC / "hgnc_protein_coding.parquet").dropna(subset=["ensembl_gene_id"])
    return h.drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")["symbol"].reindex(universe())


def gene_idx(genes) -> np.ndarray:
    """Universe indices of `genes` (genes outside the universe are dropped)."""
    ix = universe().get_indexer(list(genes))
    return np.unique(ix[ix >= 0])


# -------------------------------------------------------------------------------------------------------- networks
def list_networks() -> list[str]:
    return sorted(p.stem for p in NETS.glob("*.parquet"))


@lru_cache(None)
def network(name: str) -> sp.csr_matrix:
    """Symmetric weighted adjacency (float32 CSR, n_genes x n_genes) of modet network `name` on the universe.
    Edges with an endpoint outside the universe are dropped; duplicate edges keep the max weight; no self loops.
    `name` may also be "a+b+c" = union (max weight) of several networks."""
    if "+" in name:
        A = None
        for part in name.split("+"):
            B = network(part)
            A = B if A is None else A.maximum(B)
        return A.tocsr()
    f = _cache(f"net_{name}.npz")
    if f.exists():
        return sp.load_npz(f).tocsr()
    df = pd.read_parquet(NETS / f"{name}.parquet")
    U = universe()
    a, b = U.get_indexer(df["a"].astype(str)), U.get_indexer(df["b"].astype(str))
    w = df["weight"].to_numpy(np.float32) if "weight" in df else np.ones(len(df), np.float32)
    ok = (a >= 0) & (b >= 0) & (a != b)
    a, b, w = a[ok], b[ok], w[ok]
    n = len(U)
    # duplicates (a,b listed twice / in both directions) keep the max weight: max of the two directed halves
    A1 = sp.coo_matrix((w, (a, b)), shape=(n, n)).tocsr()
    A2 = sp.coo_matrix((w, (b, a)), shape=(n, n)).tocsr()
    A1.sum_duplicates(); A2.sum_duplicates()
    A = A1.maximum(A2).astype(np.float32).tocsr()
    A.eliminate_zeros()
    sp.save_npz(f, A)
    return A


@lru_cache(None)
def degree(name: str = "string_phys_notm_400") -> np.ndarray:
    """Unweighted degree (number of neighbours, as modet's `Graph.degree`); isolated / off-network genes have 0.
    The nuAP weight is 1 / max(degree, 1) (modet `universe.nu_weight`)."""
    return np.asarray((network(name) > 0).sum(1)).ravel().astype(np.int64)


# ---------------------------------------------------------------------------------------------------------- labels
@lru_cache(None)
def label_sets(release: str) -> pd.DataFrame:
    """disease_id, name, genes (Ensembl list) of an OT release (see module docstring for the release map)."""
    return pd.read_parquet(GT / SET_FILES[release])


@lru_cache(None)
def label_matrix(release: str) -> tuple[sp.csr_matrix, pd.Index]:
    """Disease x gene binary matrix (float32 CSR) of an OT release on the universe, and its disease ids (row order)."""
    f = _cache(f"labels_{release}.npz")
    fi = _cache(f"labels_{release}_ids.parquet")
    if f.exists() and fi.exists():
        return sp.load_npz(f).tocsr(), pd.Index(pd.read_parquet(fi)["disease_id"])
    df = label_sets(release)
    U = universe()
    rows, cols = [], []
    for i, g in enumerate(df["genes"]):
        ix = U.get_indexer(list(g))
        ix = np.unique(ix[ix >= 0])
        rows.append(np.full(len(ix), i)); cols.append(ix)
    r, c = np.concatenate(rows), np.concatenate(cols)
    L = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(df), len(U)))
    ids = pd.Index(df["disease_id"].astype(str))
    sp.save_npz(f, L)
    pd.DataFrame({"disease_id": ids}).to_parquet(fi)
    return L, ids


def disease_names(release: str = "2609") -> pd.Series:
    names = {}
    for r in ("2104", "2509", "2609"):
        if r == release:
            continue
        df = label_sets(r); names.update(zip(df.disease_id, df.name))
    df = label_sets(release); names.update(zip(df.disease_id, df.name))
    return pd.Series(names)


# -------------------------------------------------------------------------------------------------------- ontology
@lru_cache(None)
def ontology(release: str) -> pd.DataFrame:
    """OT disease table of a release: id, parents, ancestors, descendants (lists)."""
    f = _cache(f"onto_{release}.parquet")
    if f.exists():
        return pd.read_parquet(f)
    df = pd.read_parquet(OT_RAW / f"disease_{release}", columns=["id", "parents", "ancestors", "descendants"])
    for c in ("parents", "ancestors", "descendants"):
        df[c] = df[c].map(lambda v: [str(x) for x in v] if v is not None else [])
    df.to_parquet(f)
    return df


@lru_cache(None)
def relatives_map(releases: tuple = ONTOLOGY_RELEASES) -> dict[str, frozenset]:
    """id -> set of ontology ancestors and descendants in ANY of `releases` (not including the id itself)."""
    rel: dict[str, set] = {}
    for r in releases:
        for row in ontology(r).itertuples():
            s = rel.setdefault(row.id, set())
            s.update(row.ancestors); s.update(row.descendants)
            for a in row.ancestors:            # make symmetric even if a release lists only one direction
                rel.setdefault(a, set()).add(row.id)
            for d in row.descendants:
                rel.setdefault(d, set()).add(row.id)
    return {k: frozenset(v - {k}) for k, v in rel.items()}


def relatives(disease_id: str) -> frozenset:
    return relatives_map().get(disease_id, frozenset())


@lru_cache(None)
def descendants_map(releases: tuple = ONTOLOGY_RELEASES) -> dict[str, frozenset]:
    """id -> set of ontology descendants in ANY of `releases` (not including the id itself)."""
    out: dict[str, set] = {}
    for r in releases:
        for row in ontology(r).itertuples():
            out.setdefault(row.id, set()).update(row.descendants)
            for a in row.ancestors:
                out.setdefault(a, set()).add(row.id)
    return {k: frozenset(v - {k}) for k, v in out.items()}


def descendants(disease_id: str, releases: tuple = ONTOLOGY_RELEASES) -> frozenset:
    return descendants_map(releases).get(disease_id, frozenset())


# ----------------------------------------------------------------------------------------------- modet folds / seal
@lru_cache(None)
def modet_folds() -> pd.DataFrame:
    """modet folds_v1 manifest: disease_id -> cluster, family, fold, sealed, old_split (420 diseases, 3 networks)."""
    j = json.load(open(FOLDS))["diseases"]
    df = pd.DataFrame.from_dict(j, orient="index")
    df.index.name = "disease_id"
    return df


@lru_cache(None)
def sealed_ids() -> frozenset:
    f = modet_folds()
    return frozenset(f.index[f["sealed"].astype(bool)])


@lru_cache(None)
def excluded_task_ids() -> frozenset:
    """Sealed ids + their ontology relatives (21.04/25.09/26.09 union) that are not non-sealed modet panel diseases.
    Never used as evaluation tasks anywhere in modeng."""
    f = modet_folds()
    devall = set(f.index[~f["sealed"].astype(bool)])
    out = set(sealed_ids())
    for s in sealed_ids():
        out |= set(relatives(s)) - devall
    return frozenset(out)


# --------------------------------------------------------------------------------------------------------- features
def feature_table(name: str) -> pd.DataFrame:
    """Gene feature tables from modet (`esmc`, `gtex_v8`: dense, indexed by ENSG -> reindexed to the universe, NaN for
    missing genes; `go_t0`, `reactome_t0`: long tables ensembl_gene_id x term). All dated <= 2021 (modet MANIFEST.tsv)."""
    df = pd.read_parquet(FEAT / f"{name}.parquet")
    if name in ("esmc", "gtex_v8"):
        return df.reindex(universe())
    return df


def feature_sparse(name: str) -> tuple[sp.csr_matrix, pd.Index]:
    """`go_t0` / `reactome_t0` as a binary gene x term CSR on the universe."""
    df = feature_table(name)
    term = "go_id" if "go_id" in df else "pathway_id"
    gi = universe().get_indexer(df["ensembl_gene_id"].astype(str))
    ok = gi >= 0
    terms = pd.Index(np.sort(df.loc[ok, term].astype(str).unique()))
    ti = terms.get_indexer(df.loc[ok, term].astype(str))
    X = sp.csr_matrix((np.ones(ok.sum(), np.float32), (gi[ok], ti)), shape=(len(universe()), len(terms)))
    X.data[:] = 1
    return X, terms


def covariates(name: str = "string_phys_notm_400") -> pd.DataFrame:
    """modet's per-gene covariate block (`data/processed/covariates/<name>.parquet`), reindexed to the universe."""
    df = pd.read_parquet(PROC / "covariates" / f"{name}.parquet")
    if "ensembl_gene_id" in df.columns:
        df = df.set_index("ensembl_gene_id")
    return df.reindex(universe())
