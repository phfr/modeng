"""r6 (R4 plan P2): FinnGen R4 genome-wide hits as a genetic witness for strict-a.

Data: FinnGen R4 (DF4, public 2020-11-30; anonymous download of gs://finngen-public-data-r4, see docs/reports/r6_finngen.md).
  * `data/raw/finngen_r4/R4_manifest.tsv` (2,264 endpoints with summary statistics), `finngen_R4_pheno_n.tsv`.
  * SuSiE credible sets (fine-mapped regions around genome-wide significant loci) -> `data/finngen/cs.parquet`
    (scripts/finngen_build.py: hg38 -> hg19 liftover, 1000G EUR rsIDs). 4,400 credible sets in 1,035 endpoints.
Never used: `coding_variants/` (2021-11), R5+.

Endpoint -> disease id (`endpoint_map()`), two routes, union:
  ot2509_code  phenocode == FinnGen R12 study code in the OT 25.09 study table (`FINNGEN_R12_<code>`) -> its diseaseIds
               (mapping vintage post-T0, like gwas_catlag's trait mapping; the codes themselves are FinnGen's).
  name21       normalised endpoint name (parentheticals and ", definitions combined"/", NAS"-type suffixes removed) == OT 21.04
               disease name or exact synonym (T0 vintage).
Disease -> endpoints: endpoint mapped to the disease or an ontology descendant (w = 1) or a specific ancestor (w = 0.7, <= 100
descendants), as gwas_d._onto_w.

Channels (float32 [n_tasks, n_genes]; NaN = disease without a mapped endpoint that has credible sets, or a benchmark other than
temporal / temporal_wide (R4 is post the `past` cutoff 2019-04; forward / holdout labels already contain later FinnGen freezes)):
  fg_hit        w * (-log10 p of the credible-set lead) for the gene with the nearest TSS (hg19, <= 500 kb), max over sets/endpoints.
  fg_hit_cs2g   w * (-log10 p lead) * max over credible-set variants of PIP * cS2G(variant, gene) (Gazal 2022, hg19).
  *_perm        same-width negative control (gene axis permuted, fixed seed).
Leakage: temporal-clean by date (R4 2020-11-30 <= T0 2021-04-30); mechanism partly label-process (OT ingested later FinnGen
freezes after T0); mapping route ot2509_code is post-T0 vintage.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data as D
from . import gwas_d as G
from . import subthr as ST

ROOT = Path("/home/ph/code/modeng")
RAW = ROOT / "data/raw/finngen_r4"
FG = ROOT / "data/finngen"
STUDY2509 = Path("/mnt/msadb/modet/downloads/ot_evidence/study_2509")
OK_BENCH = ("temporal", "temporal_wide")
WIN = 500_000


def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r",\s*(definitions combined|nas|strict|wide definition|all|mixed|not elsewhere classified|unspecified|"
               r"other/unspecified|including.*|excluding.*)$", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


@lru_cache(None)
def manifest() -> pd.DataFrame:
    m = pd.read_csv(RAW / "R4_manifest.tsv", sep="\t")
    n = pd.read_csv(RAW / "finngen_R4_pheno_n.tsv", sep="\t")[["phenocode", "num_gw_significant"]]
    return m.merge(n, on="phenocode", how="left")


@lru_cache(None)
def _ot2509_codes() -> dict:
    s = pd.concat([pd.read_parquet(f, columns=["studyId", "diseaseIds", "traitFromSource"])
                   for f in sorted(STUDY2509.glob("*.parquet"))])
    s = s[s.studyId.str.startswith("FINNGEN_R12_")]
    return {r.studyId[len("FINNGEN_R12_"):]: (set(r.diseaseIds) if r.diseaseIds is not None else set(), r.traitFromSource)
            for r in s.itertuples()}


@lru_cache(None)
def _names2104() -> dict:
    d = pd.read_parquet(D.OT_RAW / "disease_2104", columns=["id", "name", "synonyms"])
    out = {}
    for r in d.itertuples():
        out.setdefault(_norm(r.name), set()).add(r.id)
    names = set(out)
    for r in d.itertuples():           # exact synonyms only where no disease name already claims the string
        syn = r.synonyms.get("hasExactSynonym") if isinstance(r.synonyms, dict) else None
        if syn is not None:
            for s in syn:
                k = _norm(s)
                if len(k) >= 4 and k not in names:
                    out.setdefault(k, set()).add(r.id)
    return out


@lru_cache(None)
def endpoint_map() -> pd.DataFrame:
    """phenocode, name, disease_id, route (one row per pair)."""
    f = FG / "endpoint_map.tsv"
    if f.exists():
        return pd.read_csv(f, sep="\t")
    m = manifest(); ot = _ot2509_codes(); nm = _names2104()
    rows = []
    for r in m.itertuples():
        if r.phenocode in ot:
            for d in ot[r.phenocode][0]:
                rows.append((r.phenocode, r.name, d, "ot2509_code"))
        for d in nm.get(_norm(r.name), ()):
            rows.append((r.phenocode, r.name, d, "name21"))
    df = pd.DataFrame(rows, columns=["phenocode", "name", "disease_id", "route"])
    df = df.groupby(["phenocode", "name", "disease_id"], as_index=False)["route"].agg(lambda x: "+".join(sorted(set(x))))
    FG.mkdir(parents=True, exist_ok=True)
    df.to_csv(f, sep="\t", index=False)
    return df


@lru_cache(None)
def credible_sets() -> pd.DataFrame:
    C = pd.read_parquet(FG / "cs.parquet")
    lead = C[C.lead].drop_duplicates(["trait", "region", "cs"])[["trait", "region", "cs", "chr19", "pos19", "p"]]
    lead = lead.rename(columns={"p": "p_lead", "pos19": "pos_lead", "chr19": "chr_lead"})
    C = C.merge(lead, on=["trait", "region", "cs"])
    C["lp"] = -np.log10(C.p_lead.clip(lower=1e-300)).clip(upper=50)
    return C


@lru_cache(None)
def _cs2g() -> pd.DataFrame:
    f = FG / "cs2g.parquet"
    if f.exists():
        return pd.read_parquet(f)
    snps = set(credible_sets().rsid1kg.dropna())
    sym = D.symbols(); u = D.universe(); idx = {g: i for i, g in enumerate(u)}
    s2i = {}
    for g, s in sym.items():
        if isinstance(s, str) and g in idx:
            s2i.setdefault(s, idx[g])
    out = []
    for p in sorted(ST.CS2G.glob("cS2G.*.SGscore.gz")):
        for d in pd.read_csv(p, sep="\t", usecols=["SNP", "GENE", "cS2G"], chunksize=2_000_000):
            d = d[d.SNP.isin(snps)]
            if len(d):
                out.append(d)
    d = pd.concat(out)
    d["gi"] = d.GENE.map(s2i)
    d = d.dropna(subset=["gi"]).astype({"gi": int})
    d.to_parquet(f, index=False)
    return d


@lru_cache(None)
def _endpoint_vec(trait: str, kind: str, n_genes: int) -> np.ndarray:
    C = credible_sets(); C = C[C.trait == trait]
    v = np.zeros(n_genes, np.float32)
    if C.empty:
        return v
    if kind == "cs2g":
        X = _cs2g()[["SNP", "gi", "cS2G"]].merge(C[["rsid1kg", "prob", "lp"]], left_on="SNP", right_on="rsid1kg")
        np.maximum.at(v, X.gi.to_numpy(), (X.prob * X.cS2G * X.lp).to_numpy(np.float32))
        return v
    L = C[C.lead].drop_duplicates(["region", "cs"]).dropna(subset=["pos_lead"])
    chrom, pos = ST.tss()
    for c, part in L.groupby("chr_lead"):
        g = np.flatnonzero((chrom == str(c)) & ~np.isnan(pos))
        if len(g) == 0:
            continue
        d = np.abs(part.pos_lead.to_numpy()[:, None] - pos[g][None, :])
        j = d.argmin(1); ok = d[np.arange(len(j)), j] <= WIN
        np.maximum.at(v, g[j[ok]], part.lp.to_numpy()[ok].astype(np.float32))
    return v


def task_endpoints(disease_id: str, have: frozenset) -> dict:
    """phenocode -> weight for a disease (only endpoints in `have`)."""
    w = G._onto_w(disease_id)
    M = endpoint_map()
    M = M[M.phenocode.isin(have) & M.disease_id.isin(w.keys())]
    out = {}
    for r in M.itertuples():
        out[r.phenocode] = max(out.get(r.phenocode, 0.0), w[r.disease_id])
    return out


def _channel(B, kind, perm=False):
    out = np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    if B.name not in OK_BENCH:
        return out
    have = frozenset(credible_sets().trait)
    for k, t in enumerate(B.tasks):
        m = task_endpoints(t.disease_id, have)
        if m:
            out[k] = np.max(np.stack([w * _endpoint_vec(e, kind, B.n_genes) for e, w in m.items()]), 0)
    if perm:
        out = out[:, np.random.default_rng(0).permutation(B.n_genes)]
    return out


def fg_hit(B, s): return _channel(B, "near")
def fg_hit_cs2g(B, s): return _channel(B, "cs2g")
def fg_hit_perm(B, s): return _channel(B, "near", perm=True)
def fg_hit_cs2g_perm(B, s): return _channel(B, "cs2g", perm=True)


CHANNELS = {f.__name__: f for f in (fg_hit, fg_hit_cs2g, fg_hit_perm, fg_hit_cs2g_perm)}
