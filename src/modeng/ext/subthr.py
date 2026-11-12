"""r4 (F2-6 / SG-1 lite): sub-threshold GWAS loci -> gene assignment.

Input: `data/subthr/leads.parquet` (scripts/subthr_extract.py): independent leads (greedy +-500 kb clumping by p over all SNPs with
p <= 1e-5, MHC removed, build-37 positions from modet's 1000G EUR panel) of modet's 108 harmonised GWAS summary statistics dated
<= 2021-04-23 (GWAS Catalog DATE). Sub-threshold lead = 5e-8 < p <= 1e-5 and > 500 kb from every stronger lead of the same study.

Disease -> study: gwas_d's ontology (self/descendants w=1, specific ancestors w=0.7) + semantic TF-IDF proxies (w=cosine), studies with
publication DATE <= the benchmark cutoff (past 2019-04-30 -> 88 studies, temporal 2021-04-30 -> 108). NaN = no study matched.

Channels (float32 [n_tasks, n_genes]; 0 for genes of a covered disease without a hit; per gene max over leads and matched studies):
  subthr_near   w * (-log10 p) for the gene with the nearest TSS (NCBI37.3 gene.loc; gnomAD v2.1.1 start as fallback) to the lead.
  subthr_cs2g   w * (-log10 p) * cS2G(lead SNP, gene) (Gazal 2022, 1000G EUR SNP-to-gene scores, hg19, built from pre-2021 resources).
  subthr_win    w * (-log10 p) * exp(-d / 50 kb), d = |lead - TSS| <= 500 kb.
  *_perm        same-width negative control: gene axis permuted (fixed seed).
  subthr_gws_*  diagnostic: the same with genome-wide significant leads (p <= 5e-8) instead.
Leakage: temporal-clean on temporal (summary statistics, gene positions and cS2G all <= 2021-04; ontology mapping as in gwas_d).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .. import data as D
from . import gwas_d as G

LEADS = Path("/home/ph/code/modeng/data/subthr/leads.parquet")
CS2G = Path("/mnt/msadb/modet/downloads/cs2g/x/cS2G_1000GEUR")
GLOC = Path("/home/ph/code/modet/data/raw/magma/NCBI37.3/NCBI37.3.gene.loc")
HGNC = Path("/home/ph/code/modet")
WIN, SCALE = 500_000, 50_000


@lru_cache(None)
def leads() -> pd.DataFrame:
    L = pd.read_parquet(LEADS)
    L["lp"] = -np.log10(L.P.clip(lower=1e-300)).clip(upper=50)
    L["chr"] = L.chr.astype(str); L["pos"] = L.pos.astype(float)
    return L


@lru_cache(None)
def tss():
    """chrom (str) and TSS (hg19) per universe gene."""
    import sys
    sys.path.insert(0, str(HGNC / "src"))
    from modet_bench.common import entrez_to_ensembl
    e2g = entrez_to_ensembl()
    u = D.universe(); idx = {g: i for i, g in enumerate(u)}
    chrom = np.array([""] * len(u), dtype=object); pos = np.full(len(u), np.nan)
    gl = pd.read_csv(GLOC, sep="\t", header=None, names=["entrez", "chr", "start", "stop", "strand", "sym"], dtype={"entrez": str, "chr": str})
    for r in gl.itertuples(index=False):
        g = e2g.get(r.entrez)
        if g in idx and chrom[idx[g]] == "":
            chrom[idx[g]] = r.chr; pos[idx[g]] = r.start if r.strand == "+" else r.stop
    c2, p2 = __import__("modeng.ext.kernels_b", fromlist=["_gene_loc"])._gene_loc()
    miss = (chrom == "") & (c2 != "")
    chrom[miss] = c2[miss]; pos[miss] = p2[miss]
    return chrom.astype(str), pos


@lru_cache(None)
def cs2g() -> pd.DataFrame:
    """cS2G rows (SNP, gene index, score) for all lead SNPs."""
    snps = set(leads().SNP)
    sym = D.symbols(); u = D.universe(); idx = {g: i for i, g in enumerate(u)}
    s2i = {}
    for g, s in sym.items():
        if isinstance(s, str) and g in idx:
            s2i.setdefault(s, idx[g])
    out = []
    for f in sorted(CS2G.glob("cS2G.*.SGscore.gz")):
        for d in pd.read_csv(f, sep="\t", usecols=["SNP", "GENE", "cS2G"], chunksize=2_000_000):
            d = d[d.SNP.isin(snps)]
            if len(d):
                out.append(d)
    d = pd.concat(out)
    d["gi"] = d.GENE.map(s2i)
    return d.dropna(subset=["gi"]).astype({"gi": int})


@lru_cache(None)
def _study_vec(gcst: str, kind: str, gws: bool, n_genes: int) -> np.ndarray:
    """Per-study gene vector: max over (sub-threshold | gws) leads of the assigned evidence."""
    L = leads(); L = L[(L.gcst == gcst) & (L.gws == gws)]
    v = np.zeros(n_genes, np.float32)
    if L.empty:
        return v
    chrom, pos = tss()
    if kind == "cs2g":
        C = cs2g(); C = C[C.SNP.isin(set(L.SNP))].merge(L[["SNP", "lp"]], on="SNP")
        np.maximum.at(v, C.gi.to_numpy(), (C.cS2G * C.lp).to_numpy(np.float32))
        return v
    for c, part in L.groupby("chr"):
        g = np.flatnonzero((chrom == c) & ~np.isnan(pos))
        if len(g) == 0:
            continue
        d = np.abs(part.pos.to_numpy()[:, None] - pos[g][None, :])        # leads x genes
        lp = part.lp.to_numpy()[:, None]
        if kind == "near":
            j = d.argmin(1); ok = d[np.arange(len(j)), j] <= WIN
            np.maximum.at(v, g[j[ok]], lp[ok, 0].astype(np.float32))
        else:
            s = np.where(d <= WIN, lp * np.exp(-d / SCALE), 0).max(0)
            v[g] = np.maximum(v[g], s.astype(np.float32))
    return v


def _channel(B, kind, gws=False, perm=False):
    have = set(leads().gcst)
    S = G.studies()
    out = np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    for k, m in enumerate(G._matches(B)):
        m = {i: w for i, w in m.items() if S.gcst[i] in have}
        if not m:
            continue
        out[k] = np.max(np.stack([w * _study_vec(S.gcst[i], kind, gws, B.n_genes) for i, w in m.items()]), 0)
    if perm:
        p = np.random.default_rng(0).permutation(B.n_genes)
        out = out[:, p]
    return out


def subthr_near(B, s): return _channel(B, "near")
def subthr_cs2g(B, s): return _channel(B, "cs2g")
def subthr_win(B, s): return _channel(B, "win")
def subthr_near_perm(B, s): return _channel(B, "near", perm=True)
def subthr_cs2g_perm(B, s): return _channel(B, "cs2g", perm=True)
def subthr_win_perm(B, s): return _channel(B, "win", perm=True)
def subthr_gws_near(B, s): return _channel(B, "near", gws=True)
def subthr_gws_cs2g(B, s): return _channel(B, "cs2g", gws=True)


CHANNELS = {f.__name__: f for f in (subthr_near, subthr_cs2g, subthr_win, subthr_near_perm, subthr_cs2g_perm, subthr_win_perm,
                                    subthr_gws_near, subthr_gws_cs2g)}
