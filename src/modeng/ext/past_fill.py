"""Past-fill wrappers (r3_prospective): give the modeng-v1 channels that are NaN on the `past` benchmark (seed release 19.04)
a value, so a stacker trained on past episodes (19.04 -> 21.04, all <= T0 = 21.04) can learn their weights.

Channels `<name>_pf`:
  * on every benchmark other than `past`: IDENTICAL to `<name>` (read through modeng.channels.compute, i.e. the cached original),
    so evaluation on temporal dev/confirm uses exactly the v1 inputs (OT evidence 21.04, HPO 2021-01: temporal-clean).
  * on `past`: the nearest admissible vintage.
      - GWAS-winner priors (pri_l2g_n_pf, pri_l2g_max_pf, hz_gwas_pf and the seed mix of hz_mend_pf): OT 21.04 ot_genetics_portal
        evidence restricted to publicationYear <= 2018 (a date filter exists for this source; 2019 studies are dropped because
        the seed release is 2019-04). L2G scores themselves are from the 21.04 L2G model (residual post-19.04 vintage).
      - Mendelian prior (pri_mend_n_pf): OT 21.04 eva(>=0.5)/curated evidence, no reliable date column -> post-19.04 vintage.
      - HPO channels (pri_hpo_n_pf, hz_mend_pf, ph_mean_pf, ph_top3_pf): HPO 2021-01-15 (no older HPO vintage on disk).
    Leakage class on past = "training-side transductive": these inputs post-date the past seed release (19.04) and may already
    contain the 19.04 -> 21.04 arrivals that are the past training targets (family exclusion removes only the task's own
    disease family). This can only bias the weights the stacker learns; it cannot leak temporal test labels, because every
    input is dated <= 21.04 = the temporal seed release.
Not wrapped (already defined on past with 19.04-dated inputs): dA_pt / pt_ridge / pt_ridge_pos (PubTator PMID <= 2019-03),
gwas_catlag (GWAS Catalog date added <= 2019-04-30), cs_donor, dc_ridge(_pos).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import data as D
from . import priors_h as PH

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/cache/past_fill"
GWAS_YEAR_MAX = 2018
_M: dict = {}


class _As:
    """Proxy of a Benchmark with another name (selects priors_h's 21.04 / HPO-2021 inputs)."""

    def __init__(self, B, name):
        self._B, self.name = B, name

    def __getattr__(self, k):
        return getattr(self._B, k)


def _gwas_mats():
    """(gwas_win, gwas_score, dids): OT 21.04 ot_genetics_portal evidence with publicationYear <= GWAS_YEAR_MAX."""
    if "g" in _M:
        return _M["g"]
    f = CACHE / f"ev_2104_otg_y{GWAS_YEAR_MAX}.parquet"
    if not f.exists():
        import duckdb
        CACHE.mkdir(parents=True, exist_ok=True)
        p = PH.EVDIR / "evidence_2104" / "sourceId=ot_genetics_portal" / "*.parquet"
        duckdb.connect().execute(
            f"COPY (SELECT diseaseId, targetId, max(score) s FROM read_parquet('{p}') WHERE publicationYear <= {GWAS_YEAR_MAX} "
            f"GROUP BY ALL) TO '{f}'")
    ev = pd.read_parquet(f)
    U = D.universe()
    ev = ev.assign(g=U.get_indexer(ev.targetId))
    ev = ev[ev.g >= 0]
    dids = pd.Index(sorted(ev.diseaseId.unique()))
    r, g, s = dids.get_indexer(ev.diseaseId), ev.g.to_numpy(), ev.s.to_numpy(np.float32)
    score = sp.csr_matrix((s, (r, g)), shape=(len(dids), len(U)))          # (d, g) unique after GROUP BY
    w = s >= 0.5
    win = sp.csr_matrix((np.ones(w.sum(), np.float32), (r[w], g[w])), shape=(len(dids), len(U)))
    _M["g"] = (win, score, dids)
    return _M["g"]


def _count_excl(B, M, dids):
    col = np.asarray(M.sum(0)).ravel().astype(np.float32)
    out = np.empty((len(B.tasks), B.n_genes), np.float32)
    for i, t in enumerate(B.tasks):
        F = PH._family_rows(t, dids)
        out[i] = col - (np.asarray(M[F].sum(0)).ravel() if len(F) else 0)
    return np.log1p(out)


def _max_excl(B, M, dids):
    gmax = M.max(0).toarray().ravel().astype(np.float32)
    C = M.tocsc()
    out = np.empty((len(B.tasks), B.n_genes), np.float32)
    for i, t in enumerate(B.tasks):
        F = PH._family_rows(t, dids)
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
    W, _, dids = _gwas_mats()
    mix = np.zeros(len(B.tasks), np.float32)
    for i, t in enumerate(B.tasks):
        F = PH._family_rows(t, dids)
        if len(F) and len(t.seeds):
            mix[i] = (np.asarray(W[F].sum(0)).ravel() > 0)[t.seeds].mean()
    return mix


# --------------------------------------------------------------------------------------------- past implementations
def _p_l2g_n(B):
    W, _, dids = _gwas_mats()
    return _count_excl(B, W, dids)


def _p_l2g_max(B):
    _, S, dids = _gwas_mats()
    return _max_excl(B, S, dids)


def _p_hz_gwas(B):
    return _p_l2g_n(B) * _mix_g(B)[:, None]


def _p_hz_mend(B):
    v = np.nan_to_num(PH._hpo_n(_As(B, "temporal")), nan=0.0)
    return (1 - _mix_g(B))[:, None] * v[None, :]


PAST = {
    "pri_l2g_n": _p_l2g_n,
    "pri_l2g_max": _p_l2g_max,
    "hz_gwas": _p_hz_gwas,
    "hz_mend": _p_hz_mend,
    "pri_mend_n": lambda B: PH.pri_mend_n(_As(B, "temporal"), None),
    "pri_hpo_n": lambda B: PH.pri_hpo_n(_As(B, "temporal"), None),
    "ph_mean": lambda B: PH.ph_mean(_As(B, "temporal"), None),
    "ph_top3": lambda B: PH.ph_top3(_As(B, "temporal"), None),
}


def _make(name):
    def fn(B, split_tag):
        if B.name == "past":
            return np.asarray(PAST[name](B), np.float32)
        from .. import channels as CH
        return CH.compute(B, name, split_tag)        # identical to the v1 channel off `past`
    return fn


CHANNELS = {n + "_pf": _make(n) for n in PAST}
