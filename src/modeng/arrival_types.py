"""Arrival typing for `temporal` and `temporal_wide` targets (R4 synthesis item 4 / plan P3): label-process vs new data.

Every (task, target) of the two 21.04 -> 25.09 benchmarks gets ONE exclusive type, first matching rule wins:
  onto      ontology / propagation event: the target already sits in the 21.04 context row of the disease itself or of an
            ontology descendant (any of 21.04 / 25.09 / 26.09), i.e. exactly the pairs masked by `attach_families`
            ('self+desc'). It "arrives" only because modet propagates 21.04 evidence with the 25.09 ontology / the family
            was rewired.                                                                                  -> labelproc
  catlag    GWAS-Catalog lag: a GWAS Catalog association (p <= 1e-6, MAPPED_GENE = target) ADDED TO THE CATALOG on or
            before 2021-04-30 is mapped to the disease or an ontology descendant (`ext.gwas_d._assoc`; ancestors not
            used). The association was public before T0; OT attached it later.                      -> labelproc
  old       modet arrival class B: the primary source group's first evidence year is <= 2020 (old record newly
            attached: ingestion / curation lag, re-scoring of old evidence).                         -> labelproc
  new       modet arrival class A: first evidence year 2022..2025, and the label persists in 26.09 on evidence dated
            <= 2025 (A implies persistent in modet's operationalisation).                            -> newdata
  nonpersist  class X (no 26.09 row reaches the threshold on evidence <= 2025: re-scored / vanished label)
  band2021    class U (2021 evidence year: uncertain, never merged)
  untyped   no modet arrival row (temporal: modet's arrival table covers 4,981 of 6,324 pairs) or class N
Arrival classes: temporal = modet `arrival_devall_t0_string115_phys_notm_400.parquet` (read only); temporal_wide =
`tasks.wide_arrival` (modet's build_arrival run on the wide targets; cached `data/cache/arrival_wide.parquet`).
Descriptive flag `prior21`: any OT 21.04 association (any datasource, any score > 0; `associationByDatasourceDirect`) for
the pair in the disease family (disease + descendants). Weak genetic evidence is already excluded as `unknown`, so
prior21 is mostly literature / animal / expression evidence.

Alt target sets (`add_typed_alt`, called by `tasks.load` for temporal / temporal_wide; additive, old metrics unchanged):
  newdata   = targets typed `new`;  labelproc = targets typed onto | catlag | old.
  Each is ranked against non-targets only (the other targets are excluded, as for `sa`); kept when >= MIN_ALT positives.
The typing uses post-T0 information (evidence years, persistence in 26.09) by construction: allowed for labels, never an
input. Leakage class of every metric is that of the benchmark (temporal-clean).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D

MIN_ALT = 3
LABELPROC = ("onto", "catlag", "old")
NEWDATA = ("new",)
T0_DATE = "2021-04-30"
MODET_ARR = D.GT / "arrival_devall_t0_string115_phys_notm_400.parquet"


def _arrival(bench: str) -> pd.DataFrame:
    if bench == "temporal":
        a = pd.read_parquet(MODET_ARR)
    elif bench == "temporal_wide":
        a = pd.read_parquet(D._cache("arrival_wide.parquet"))
    else:
        raise ValueError(bench)
    return a[["disease_id", "gene", "arrival", "fy_primary", "evidence_class", "persistent"]]


def _catlag_pairs() -> dict:
    """trait id -> set of gene indices with a catalog association (p <= 1e-6) added <= T0."""
    from .ext import gwas_d as G
    a = G._assoc()
    a = a[a.added <= T0_DATE]
    return a.groupby("trait")["gene"].apply(lambda s: set(s.tolist())).to_dict()


def _prior21():
    """(CSR disease x gene, any 21.04 association score > 0; disease index)."""
    import scipy.sparse as sp
    from .ext import backlog_e as BE
    df = BE._assoc("2104")
    df = df[df.score > 0]
    ids = pd.Index(sorted(df.diseaseId.unique()))
    r = ids.get_indexer(df.diseaseId)
    M = sp.csr_matrix((np.ones(len(r), np.float32), (r, df.g.to_numpy())), shape=(len(ids), len(D.universe())))
    M.sum_duplicates()
    return M, ids


def type_table(B) -> pd.DataFrame:
    """One row per (task, target): disease_id, split, gene index, ENSG, type, arrival class, first year, prior21. Cached."""
    f = D._cache(f"arrival_types_{B.name}.parquet")
    if f.exists():
        return pd.read_parquet(f)
    U = D.universe()
    arr = _arrival(B.name).set_index(["disease_id", "gene"])
    cat = _catlag_pairs()
    P, pid = _prior21()
    rows = []
    for t in B.tasks:
        fam = {t.disease_id} | set(D.descendants(t.disease_id))
        onto = set(np.asarray(t.mask_c).tolist()) if t.mask_c is not None else set()
        cl = set()
        for x in fam:
            cl |= cat.get(x, set())
        pr = pid.get_indexer(sorted(fam)); pr = pr[pr >= 0]
        p21 = set(np.unique(P[pr].indices).tolist()) if len(pr) else set()
        for g in t.targets:
            ensg = U[g]
            try:
                r = arr.loc[(t.disease_id, ensg)]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                ac, fy, ec = r["arrival"], r["fy_primary"], r["evidence_class"]
            except KeyError:
                ac, fy, ec = None, np.nan, None
            if g in onto:
                ty = "onto"
            elif g in cl:
                ty = "catlag"
            elif ac == "B":
                ty = "old"
            elif ac == "A":
                ty = "new"
            elif ac == "X":
                ty = "nonpersist"
            elif ac == "U":
                ty = "band2021"
            else:
                ty = "untyped"
            rows.append((t.disease_id, t.split, int(g), ensg, ty, ac, fy, ec, g in p21))
    df = pd.DataFrame(rows, columns=["disease_id", "split", "g", "gene", "type", "arrival", "fy_primary", "evidence_class",
                                     "prior21"])
    df.to_parquet(f, index=False)
    return df


def add_typed_alt(B) -> None:
    """Attach alt target sets `newdata` and `labelproc` (see module docstring) to every task of B in place."""
    if B.name not in ("temporal", "temporal_wide"):
        return
    tt = type_table(B)
    by = {k: v for k, v in tt.groupby("disease_id")}
    for t in B.tasks:
        d = by.get(t.disease_id)
        if d is None:
            continue
        for name, types in (("newdata", NEWDATA), ("labelproc", LABELPROC)):
            pos = np.sort(d.g[d.type.isin(types)].to_numpy(np.int64))
            if len(pos) >= MIN_ALT:
                t.alt[name] = (pos, np.union1d(t.exclude, np.setdiff1d(t.targets, pos)))
