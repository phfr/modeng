"""r6: gene-level LoF constraint (gnomAD v2.1.1, `gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz`, released 2019, on disk) as a static
gene channel. gnomad_loeuf = -oe_lof_upper (higher = more constrained); per gene the row with the most
expected LoFs; NaN for genes without a row. Same row for every task. *_perm: gene axis permuted (same-width control).
Leakage: temporal-clean on temporal (data <= 2020); on `past` (cutoff 2019-04) the v2.1.1 metrics (2019-03 data release,
constraint paper 2020) are borderline: report as transductive there."""
from __future__ import annotations

from functools import lru_cache
import numpy as np
import pandas as pd
from .. import data as D

F = "/mnt/msadb/modet/downloads/gnomad_v2.1.1/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz"


@lru_cache(None)
def _vec() -> np.ndarray:
    z = pd.read_csv(F, sep="\t", compression="gzip", usecols=["gene_id", "oe_lof_upper", "exp_lof"])
    z = z.dropna(subset=["oe_lof_upper"]).sort_values("exp_lof", ascending=False).drop_duplicates("gene_id")
    u = D.universe()
    v = np.full(len(u), np.nan, np.float32)
    j = u.get_indexer(z.gene_id); ok = j >= 0
    v[j[ok]] = -z.oe_lof_upper.to_numpy(np.float32)[ok]
    return v


def gnomad_loeuf(B, s):
    return np.tile(_vec(), (len(B.tasks), 1))


def gnomad_loeuf_perm(B, s):
    return np.tile(_vec()[np.random.default_rng(0).permutation(B.n_genes)], (len(B.tasks), 1))


CHANNELS = {"gnomad_loeuf": gnomad_loeuf, "gnomad_loeuf_perm": gnomad_loeuf_perm}
