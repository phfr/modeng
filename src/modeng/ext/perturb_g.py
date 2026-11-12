"""Perturbation biology channels (implementer G, round 1: CI-01 / SC-1 / SC-3-lite routing).

Atlases (row = CRISPRi perturbation target, column = measured readout gene; subex `derived/ps_<name>.npy`, top 5 PCs across
perturbations projected out, rows standardised):
  k562   Replogle 2022 K562 genome-wide (9,852 x 8,248)           readout ids: raw/perturbseq/K562_gwps_normalized_bulk_01.h5ad var
  rpe1   Replogle 2022 RPE1 essential (2,390 x 8,749)              readout ids: raw/perturbseq/rpe1_normalized_bulk_01.h5ad var
  hepg2  Nadig 2025 HepG2 essential (2,391 x 9,624)                readout ids: ps_hepg2_nadig_bulk.npz genes
  jurkat Nadig 2025 Jurkat essential (2,391 x 8,882)               readout ids: ps_jurkat_nadig_bulk.npz genes
  cd4s   Zhu/Dann 2025 primary CD4 T, Stim8hr (12,315 x 8,871)     readout ids: ps_cd4_2025_bulk.npz genes
  cd4r   Zhu/Dann 2025 primary CD4 T, Rest (12,264 x 8,690)        readout ids: ps_cd4_2025_rest_bulk.npz genes

Per task and atlas (seeds only; no context labels are used):
  upstream (regulator) z of perturbed gene g: U_g = mean over seed readouts of X[g, seed] (signed, coherent shift) and
      A_g = mean over seed readouts of |X[g, seed]| (magnitude); null mean = the same statistic for random readout sets
      stratified on readout responsiveness deciles (column sd), computed analytically from per-row decile means; null sd from
      48 random readout sets per size on a log grid (interpolated in log n; captures readout correlation).
      Requires >= 3 seeds measured as readouts.
  responder z of readout gene g: D_g = mean over seed perturbations of X[seed, g]; null mean = column mean, null sd from
      48 random perturbation-row sets per size (interpolated). Requires >= 3 seeds perturbed.
  coherence of the seeds in the atlas = mean over perturbations of z_up^2 - 1 (excess variance of the seed program over
      random readout sets): how much the atlas "moves" the seed program at all. Used for routing (SC-3 / SC-9 lite).
Channels ([n_tasks, n_genes], NaN = gene not in any usable atlas):
  pg_up      max over atlases of |z_up|
  pg_upmag   max over atlases of z of the magnitude statistic
  pg_dn      max over atlases of |z_dn|
  pg_upr     |z_up| in the routed atlas (largest coherence for this task)
  pg_dnr     |z_dn| in the routed atlas (largest responder coherence)
  pg_ups     task-specific upstream: pg_up standardised per gene across the split's tasks (removes "regulates everything",
             e.g. essential / stress genes)
  pg_dns     same for pg_dn
  pg_rsm     readout co-response (SC-1 lite, the seed module as readout): mean cosine of gene g's response profile across all
             perturbations to the seed readouts' profiles (max over atlases); pg_rsx = mean of the top-3 cosines
  pg_coh     task-level max coherence (constant over genes; lets the stacker gate the channel)
Negative control (same code, row ids and readout ids permuted within each atlas, so coverage is unchanged and only the
gene-content link is broken): `pgneg_*`.

Leakage: every atlas is dated after OT 21.04 (Replogle 2022; Nadig GEO 2024-05; CD4 bioRxiv 2025-12), so on `temporal`
these channels are TRANSDUCTIVE. They contain no disease labels. On `forward` (seeds 25.09): K562/RPE1/HepG2/Jurkat are
dated before 25.09 (fair); the CD4 atlas (2025-12) is after the forward seed date (transductive by the date rule, though
label-free). On `holdout`: holdout-current. A no-CD4 variant (`pgnc_*`) is provided for a strictly date-fair forward run.
"""
from __future__ import annotations

import numpy as np

from .. import data as D

DER = "/mnt/msadb/subex/derived"
RAW = "/mnt/msadb/subex/raw/perturbseq"
ATLASES = {
    "k562": ("ps_k562_gwps", "K562_gwps_normalized_bulk_01.h5ad"),
    "rpe1": ("ps_rpe1_ess", "rpe1_normalized_bulk_01.h5ad"),
    "hepg2": ("ps_hepg2_nadig", None),
    "jurkat": ("ps_jurkat_nadig", None),
    "cd4s": ("ps_cd4_2025", None),
    "cd4r": ("ps_cd4_2025_rest", None),
}
NO_CD4 = ("k562", "rpe1", "hepg2", "jurkat")
GRID = np.array([3, 5, 8, 13, 20, 32, 50, 80, 128, 200, 320, 512, 1000])
NRAND = 48
MIN_SEEDS = 3
_ATL: dict = {}
_MEMO: dict = {}


def _readout_ids(name, h5):
    if h5 is None:
        z = np.load(f"{DER}/{name}_bulk.npz")
        return np.asarray(z["genes"]).astype(str)
    import h5py
    with h5py.File(f"{RAW}/{h5}", "r") as f:
        v = f["var"]["gene_id"]
        if hasattr(v, "keys"):   # categorical
            cats = v["categories"][:].astype(str)
            return cats[v["codes"][:]]
        vals = v[:]
        if "__categories" in f["var"] and "gene_id" in f["var"]["__categories"]:
            return f["var"]["__categories"]["gene_id"][:].astype(str)[vals]
        return vals.astype(str)


def _first_unique(idx):
    """keep the first occurrence of each mapped universe index; -1 = drop."""
    out = idx.copy()
    seen = set()
    for i, g in enumerate(idx):
        if g < 0 or g in seen:
            out[i] = -1
        else:
            seen.add(g)
    return out


def load_atlas(key, permute=False):
    ck = (key, permute)
    if ck in _ATL:
        return _ATL[ck]
    name, h5 = ATLASES[key]
    X = np.load(f"{DER}/{name}.npy").astype(np.float32)
    rows = open(f"{DER}/{name}_genes.txt").read().split("\n")
    cols = _readout_ids(name, h5)
    assert len(rows) == X.shape[0] and len(cols) == X.shape[1], (key, X.shape, len(rows), len(cols))
    U = D.universe()
    ri = _first_unique(U.get_indexer([r.split(".")[0] for r in rows]))
    ci = _first_unique(U.get_indexer([c.split(".")[0] for c in cols]))
    kr, kc = ri >= 0, ci >= 0
    X = np.ascontiguousarray(X[np.ix_(kr, kc)])
    ri, ci = ri[kr], ci[kc]
    if permute:
        rng = np.random.default_rng(12345)
        ri, ci = rng.permutation(ri), rng.permutation(ci)
    P, R = X.shape
    A = np.abs(X)
    # readout strata: deciles of responsiveness (column sd across perturbations)
    csd = X.std(0)
    strata = np.minimum((csd.argsort().argsort() * 10) // R, 9)
    Sm = np.zeros((R, 10), np.float32); Sm[np.arange(R), strata] = 1
    cnt = Sm.sum(0)
    mX, mA = (X @ Sm) / cnt, (A @ Sm) / cnt                         # [P, 10] per-row decile means
    rng = np.random.default_rng(0)
    # upstream null sd: random readout sets
    sdX, sdA = np.zeros((P, len(GRID)), np.float32), np.zeros((P, len(GRID)), np.float32)
    for j, n in enumerate(GRID):
        n = min(n, R)
        M = np.zeros((R, NRAND), np.float32)
        for r in range(NRAND):
            M[rng.choice(R, n, replace=False), r] = 1.0 / n
        sdX[:, j] = (X @ M).std(1); sdA[:, j] = (A @ M).std(1)
    # responder null: random perturbation sets
    cmu = X.mean(0)
    sdD = np.zeros((len(GRID), R), np.float32)
    for j, n in enumerate(GRID):
        n = min(n, P)
        M = np.zeros((NRAND, P), np.float32)
        for r in range(NRAND):
            M[r, rng.choice(P, n, replace=False)] = 1.0 / n
        sdD[j] = (M @ X).std(0)
    # readout co-response view: each readout's column across perturbations, centred and unit-norm (cosine)
    Xc = X - X.mean(0, keepdims=True)
    Cn = np.ascontiguousarray((Xc / (np.linalg.norm(Xc, axis=0, keepdims=True) + 1e-6)).T)   # [R, P]
    a = dict(X=X, A=A, ri=ri, ci=ci, strata=strata, mX=mX, mA=mA, sdX=sdX, sdA=sdA, cmu=cmu, sdD=sdD, Cn=Cn)
    return a                      # not kept in _ATL: six atlases x (X, |X|, cosine view) are ~8 GB; compute_all memoises results


def _interp_sd(sd, n):
    """sd [*, len(GRID)] (grid on axis -1) evaluated at sizes n [T] -> [*, T] by linear interpolation in log n."""
    lg = np.log(GRID.astype(float))
    x = np.clip(np.log(np.maximum(n, 1).astype(float)), lg[0], lg[-1])
    j = np.clip(np.searchsorted(lg, x) - 1, 0, len(lg) - 2)
    w = ((x - lg[j]) / (lg[j + 1] - lg[j])).astype(np.float32)
    return sd[..., j] * (1 - w) + sd[..., j + 1] * w


def atlas_scores(a, seed_lists):
    """per-task z scores on the atlas' own axes. Returns z_up [T,P], z_mag [T,P], z_dn [T,R], coh_up [T], coh_dn [T]."""
    X, A, ri, ci = a["X"], a["A"], a["ri"], a["ci"]
    P, R = X.shape
    G = len(D.universe())
    pos_c = np.full(G, -1); pos_c[ci] = np.arange(R)
    pos_r = np.full(G, -1); pos_r[ri] = np.arange(P)
    T = len(seed_lists)
    Sc = np.zeros((T, R), np.float32); Sr = np.zeros((T, P), np.float32)
    for t, s in enumerate(seed_lists):
        c = pos_c[s]; c = c[c >= 0]; Sc[t, c] = 1
        r = pos_r[s]; r = r[r >= 0]; Sr[t, r] = 1
    nc, nr = Sc.sum(1), Sr.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        # upstream
        U = (Sc @ X.T) / np.maximum(nc, 1)[:, None]                  # [T, P]
        Am = (Sc @ A.T) / np.maximum(nc, 1)[:, None]
        Kd = np.zeros((T, 10), np.float32)
        np.add.at(Kd, (np.repeat(np.arange(T), Sc.sum(1).astype(int)), a["strata"][np.nonzero(Sc)[1]]), 1)
        Kd /= np.maximum(nc, 1)[:, None]
        muX, muA = Kd @ a["mX"].T, Kd @ a["mA"].T                    # [T, P]
        sX, sA = _interp_sd(a["sdX"], nc).T, _interp_sd(a["sdA"], nc).T
        z_up = (U - muX) / np.maximum(sX, 1e-6)
        z_mag = (Am - muA) / np.maximum(sA, 1e-6)
        # responder
        Dm = (Sr @ X) / np.maximum(nr, 1)[:, None]                    # [T, R]
        sD = _interp_sd(a["sdD"].T, nr).T                           # [T, R]
        z_dn = (Dm - a["cmu"][None]) / np.maximum(sD, 1e-6)
    bad_u, bad_d = nc < MIN_SEEDS, nr < MIN_SEEDS
    z_up[bad_u] = np.nan; z_mag[bad_u] = np.nan; z_dn[bad_d] = np.nan
    coh_up = np.nanmean(z_up ** 2, 1) - 1 if P else np.full(T, np.nan)
    coh_dn = np.nanmean(z_dn ** 2, 1) - 1
    coh_up[bad_u] = np.nan; coh_dn[bad_d] = np.nan
    return z_up, z_mag, z_dn, coh_up, coh_dn


def coresponse(a, seed_lists, topk=3):
    """readout co-response: cosine of readout g's response profile (across all perturbations) to the seed readouts' profiles.
    Returns mean cosine [T,R] and mean of the top-k cosines [T,R] (seeds themselves get NaN); needs >= MIN_SEEDS seed readouts."""
    Cn, ci = a["Cn"], a["ci"]
    R = Cn.shape[0]
    pos_c = np.full(len(D.universe()), -1); pos_c[ci] = np.arange(R)
    T = len(seed_lists)
    mean, mx = np.full((T, R), np.nan, np.float32), np.full((T, R), np.nan, np.float32)
    for t, s in enumerate(seed_lists):
        c = pos_c[s]; c = c[c >= 0]
        if len(c) < MIN_SEEDS:
            continue
        C = Cn @ Cn[c].T                                             # [R, n_s]
        mean[t] = C.mean(1)
        k = min(topk, C.shape[1])
        mx[t] = np.sort(C, 1)[:, -k:].mean(1)
        mean[t, c] = np.nan; mx[t, c] = np.nan
    return mean, mx


def _nanmax_into(acc, v):
    m = np.isnan(acc)
    acc[m] = v[m]
    np.fmax(acc, v, out=acc)


def _spec(M):
    """standardise each gene (column) across tasks."""
    with np.errstate(invalid="ignore"):
        mu, sd = np.nanmean(M, 0, keepdims=True), np.nanstd(M, 0, keepdims=True)
        return (M - mu) / np.maximum(sd, 1e-3)


def compute_all(B, split_tag, permute=False, atlases=tuple(ATLASES)):
    key = (B.name, split_tag, permute, atlases, len(B.tasks))
    if key in _MEMO:
        return _MEMO[key]
    seeds = [np.asarray(t.seeds) for t in B.tasks]
    T, G = len(seeds), B.n_genes
    nan = lambda: np.full((T, G), np.nan, np.float32)
    up, mag, dn, upr, dnr = nan(), nan(), nan(), nan(), nan()
    best_u, best_d = np.full(T, -np.inf), np.full(T, -np.inf)
    coh = np.full(T, np.nan, np.float32)
    rsm, rsx = nan(), nan()
    for k in atlases:
        a = load_atlas(k, permute)
        z_up, z_mag, z_dn, cu, cd = atlas_scores(a, seeds)
        Mu, Mm, Md = nan(), nan(), nan()
        Mu[:, a["ri"]] = np.abs(z_up); Mm[:, a["ri"]] = z_mag; Md[:, a["ci"]] = np.abs(z_dn)
        _nanmax_into(up, Mu); _nanmax_into(mag, Mm); _nanmax_into(dn, Md)
        su = np.where(np.isnan(cu), -np.inf, cu); sd_ = np.where(np.isnan(cd), -np.inf, cd)
        bu, bd = su > best_u, sd_ > best_d
        upr[bu] = Mu[bu]; best_u[bu] = su[bu]
        dnr[bd] = Md[bd]; best_d[bd] = sd_[bd]
        coh = np.fmax(coh, cu.astype(np.float32))
        cm, cx = coresponse(a, seeds)
        Mc, Mx = nan(), nan()
        Mc[:, a["ci"]] = cm; Mx[:, a["ci"]] = cx
        _nanmax_into(rsm, Mc); _nanmax_into(rsx, Mx)
        print(f"[perturb_g] {B.name}/{split_tag} {k}{' PERM' if permute else ''}: P={a['X'].shape[0]} R={a['X'].shape[1]} "
              f"tasks with up {np.isfinite(cu).sum()}/{T}, dn {np.isfinite(cd).sum()}/{T}, median coh {np.nanmedian(cu):.3f}", flush=True)
        del a, Mc, Mx, Mu, Mm, Md
    out = dict(up=up, upmag=mag, dn=dn, upr=upr, dnr=dnr, ups=_spec(up), dns=_spec(dn),
               rsm=rsm, rsx=rsx, coh=np.repeat(coh[:, None], G, 1).astype(np.float32))
    _MEMO[key] = out
    return out


def _ch(stat, permute=False, atlases=tuple(ATLASES)):
    return lambda B, split_tag: compute_all(B, split_tag, permute, atlases)[stat]


STATS = ("up", "upmag", "dn", "upr", "dnr", "ups", "dns", "coh", "rsm", "rsx")
CHANNELS = {}
for _s in STATS:
    CHANNELS[f"pg_{_s}"] = _ch(_s)
    CHANNELS[f"pgneg_{_s}"] = _ch(_s, permute=True)
    CHANNELS[f"pgnc_{_s}"] = _ch(_s, atlases=NO_CD4)
