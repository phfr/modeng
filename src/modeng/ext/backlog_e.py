"""Curation-backlog channels (implementer E, round 1: IR-3 identity linking, LT-3 / WC-BACKLOG curation debt, IR-7 weak-evidence
funnel, SG-7 ingestion lag). These channels forecast what the Open Targets LABEL PIPELINE attaches next, from evidence that is
already in the seed release but does not (yet) count as a label. They predict the label process, not biology.

Input: OT `association_by_datasource_direct` of the benchmark's SEED release only:
  forward  -> OT 25.09 (modet/data/raw/ot/assoc_2509), ontology 25.09   [temporal-clean on forward: inputs = seed release]
  temporal -> OT 21.04 (assoc_2104), ontology 21.04                     [temporal-clean]
  past     -> OT 19.04 (assoc_1904: genetic sources only), ontology 21.04 [<= T0; non-genetic channels are NaN]
  holdout  -> all NaN (per-datasource OT scores are label re-encodings on holdout; barred)
OT 26.09 files are never read. On temporal / past the evidence rows of `data.excluded_task_ids()` (sealed fold + relatives) are
dropped, as the temporal context does.

Family F(d) = task disease + its descendants in the seed-release ontology. Per gene, channel = max datasource score over F(d):
  bl_genweak  genetic sources that the label counts only above 0.5 (GWAS L2G / credible sets, ClinVar eva) + phewas_catalog
              (the sub-threshold part: strong ones are seeds; on temporal own-family weak genes are 'unknown' = not ranked)
  bl_lit      europepmc text mining
  bl_animal   phenodigm / impc
  bl_drug     chembl
  bl_somatic  cancer_gene_census, intogen, eva_somatic, cancer_biomarkers
  bl_func     expression_atlas, reactome, slapenrich, progeny, sysbio, crispr, crispr_screen
  bl_any      max over all of the above; bl_nsrc = number of these groups with evidence
  bl_xid      IR-3 exact-match arm: evidence (any source incl. strong genetic) filed under OTHER disease ids that are not in F(d) and
              not ancestors of d, whose normalised name / exact synonym equals the task's name or one of its exact synonyms
              (seed-release disease table). Score = max over matched ids.
Negative control: bl_any_perm = bl_any with gene identities permuted (one fixed permutation): must add ~0.
Masking: none of these inputs are label rows (the strong genetic family evidence = seeds, excluded from ranking), so the task
masks (own row + descendants in the context label matrix) are unaffected. Exception: bl_xid includes strong genetic evidence of
matched non-family ids; those genes are also visible in the (unmasked) context rows of those ids, i.e. no extra leak.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import data as D

RAW = Path("/home/ph/code/modet/data/raw/ot")
ROOT = Path(__file__).resolve().parents[3]
CC = ROOT / "data/cache/backlog_e"
RELEASE = {"forward": ("2509", "2509"), "temporal": ("2104", "2104"), "past": ("1904", "2104")}  # bench -> (assoc, ontology)
GROUPS = {
    "genweak": ["ot_genetics_portal", "gwas_credible_sets", "eva", "phewas_catalog"],
    "lit": ["europepmc"],
    "animal": ["phenodigm", "impc"],
    "drug": ["chembl"],
    "somatic": ["cancer_gene_census", "intogen", "eva_somatic", "cancer_biomarkers"],
    "func": ["expression_atlas", "reactome", "slapenrich", "progeny", "sysbio", "crispr", "crispr_screen"],
}
LABEL_ANYSCORE = ["genomics_england", "gene2phenotype", "clingen", "orphanet", "uniprot_variants", "uniprot_literature",
                  "gene_burden"]
STRONG_GEN = ["ot_genetics_portal", "gwas_credible_sets", "eva"]


@lru_cache(None)
def _assoc(rel: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW / f"assoc_{rel}")
    if "datasourceHarmonicScore" in df:
        df = df.rename(columns={"datasourceHarmonicScore": "score"})
    df = df[["diseaseId", "targetId", "datasourceId", "score"]]
    g = D.universe().get_indexer(df["targetId"].astype(str))
    df = df[g >= 0].assign(g=g[g >= 0])
    return df


@lru_cache(None)
def _group_mats(rel: str, drop_excluded: bool):
    """{group: (CSR disease x gene of max score, disease index)} for the release; 'xid' group = all sources (label sources too)."""
    df = _assoc(rel)
    if drop_excluded:
        df = df[~df["diseaseId"].isin(D.excluded_task_ids())]
    ids = pd.Index(np.sort(df["diseaseId"].astype(str).unique()))
    r_all = ids.get_indexer(df["diseaseId"].astype(str))
    n = len(D.universe())
    out = {}
    for name, srcs in list(GROUPS.items()) + [("xid", None)]:
        m = df["datasourceId"].isin(srcs).to_numpy() if srcs else np.ones(len(df), bool)
        if not m.any():
            out[name] = None
            continue
        s = df["score"].to_numpy(np.float32)[m]
        if srcs is None:   # label sources count as 1 (any score) for the cross-id arm
            s = np.where(df["datasourceId"].isin(LABEL_ANYSCORE).to_numpy()[m], 1.0, s).astype(np.float32)
        s = np.round(s, 5)
        # several sources per (disease, gene) in a group -> keep the max
        t = pd.DataFrame({"r": r_all[m], "g": df["g"].to_numpy()[m], "s": s}).groupby(["r", "g"], sort=False)["s"].max()
        M = sp.csr_matrix((t.to_numpy(np.float32), (t.index.get_level_values(0), t.index.get_level_values(1))),
                          shape=(len(ids), n))
        out[name] = M
    return out, ids


def _norm(s: str) -> str:
    s = s.lower().replace("'s", "s")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


@lru_cache(None)
def _onto(rel: str):
    df = pd.read_parquet(RAW / f"disease_{rel}", columns=["id", "name", "synonyms", "descendants", "ancestors"])
    desc = {r.id: set(r.descendants) if r.descendants is not None else set() for r in df.itertuples()}
    anc = {r.id: set(r.ancestors) if r.ancestors is not None else set() for r in df.itertuples()}
    keys = {}
    for r in df.itertuples():
        k = {_norm(r.name)} if isinstance(r.name, str) else set()
        syn = r.synonyms
        if syn is not None and syn.get("hasExactSynonym") is not None:
            k |= {_norm(x) for x in syn["hasExactSynonym"]}
        keys[r.id] = {x for x in k if len(x) >= 4}
    inv = {}
    for i, ks in keys.items():
        for k in ks:
            inv.setdefault(k, set()).add(i)
    return desc, anc, keys, inv


def _family_rows(task, ids, desc):
    fam = {task.disease_id} | desc.get(task.disease_id, set())
    r = ids.get_indexer(sorted(fam))
    return r[r >= 0]


def _xid_rows(task, ids, desc, anc, keys, inv):
    d = task.disease_id
    k = set(keys.get(d, set()))
    if task.name:
        k.add(_norm(task.name))
    match = set()
    for x in k:
        match |= inv.get(x, set())
    fam = {d} | desc.get(d, set()) | anc.get(d, set())
    match -= fam
    r = ids.get_indexer(sorted(match))
    return r[r >= 0]


def _rowmax(M, rows):
    if M is None:
        return None
    if len(rows) == 0:
        return np.zeros(M.shape[1], np.float32)
    return np.asarray(M[rows].max(0).todense()).ravel().astype(np.float32)


_MEMO = {}


def _compute_all(B, split_tag):
    key = (B.name, split_tag, len(B.tasks))
    if key in _MEMO:
        return _MEMO[key]
    n = B.n_genes
    names = ["bl_" + g for g in GROUPS] + ["bl_any", "bl_nsrc", "bl_xid", "bl_any_perm"]
    if B.name not in RELEASE:
        out = {k: np.full((len(B.tasks), n), np.nan, np.float32) for k in names}
        _MEMO[key] = out
        return out
    arel, orel = RELEASE[B.name]
    mats, ids = _group_mats(arel, B.name in ("temporal", "past"))
    desc, anc, keys, inv = _onto(orel)
    out = {k: np.full((len(B.tasks), n), np.nan, np.float32) for k in names}
    perm = np.random.default_rng(12345).permutation(n)
    for i, t in enumerate(B.tasks):
        rows = _family_rows(t, ids, desc)
        anyv = np.zeros(n, np.float32); nsrc = np.zeros(n, np.float32); have = False
        for g in GROUPS:
            v = _rowmax(mats[g], rows)
            if v is None:
                continue
            have = True
            out["bl_" + g][i] = v
            anyv = np.maximum(anyv, v); nsrc += (v > 0)
        if have:
            out["bl_any"][i] = anyv; out["bl_nsrc"][i] = nsrc
            out["bl_any_perm"][i] = anyv[perm]
        xr = _xid_rows(t, ids, desc, anc, keys, inv)
        out["bl_xid"][i] = _rowmax(mats["xid"], xr)
    _MEMO[key] = out
    return out


def _make(name):
    def fn(B, split_tag):
        return _compute_all(B, split_tag)[name]
    return fn


CHANNELS = {k: _make(k) for k in ["bl_" + g for g in GROUPS] + ["bl_any", "bl_nsrc", "bl_xid", "bl_any_perm"]}
