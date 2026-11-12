"""Implementer B (round 1): gene-kernel lifts of transfer. Driver for tests, stacks (with score matrices saved) and the
zero-transfer-support stratum analysis.

  python scripts/r1_kernels_b.py exact                        # lift exactness vs brute-force masked transfer
  python scripts/r1_kernels_b.py stack temporal past base,lift_paralog tag
  python scripts/r1_kernels_b.py stack forward cv base,... tag
  python scripts/r1_kernels_b.py zero temporal methodA methodB [...]   # zero-support stratum (needs saved S)
"""
from __future__ import annotations

import sys, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp

from modeng import tasks as T, bench as BM, channels as CH, stack as ST, compare as CMP, metrics as MT
from modeng.ext import kernels_b as KB

ROOT = Path(__file__).resolve().parents[1]
SDIR = ROOT / "data/cache/stackS"

GROUPS = {
    "KHOM": KB.LIFT_HOMOLOGY,
    "KT0": KB.LIFT_T0,
    "KVIEW": KB.LIFT_VIEWS,
    "KCS": ["coldstart_profile"],
    "KCS0": ["coldstart_profile_t0"],
    "KPERM": KB.PERM,
}


def expand(chs):
    out = []
    for c in chs.split(","):
        out += GROUPS.get(c, [c])
    return ",".join(out)


_orig_record = ST._record


def _record(S, B, bench, split, method, t0, write):
    SDIR.mkdir(parents=True, exist_ok=True)
    np.save(SDIR / f"{bench}_{split}_{method}.npy", S.astype(np.float32))
    return _orig_record(S, B, bench, split, method, t0, write)


ST._record = _record


def brute_transfer(B, t, k=50):
    """transfer_jaccard computed from scratch on B.context_for(t) (no correction trick)."""
    L = B.context_for(t).tocsr()
    sizes = L.getnnz(1)
    s = np.zeros(B.n_genes, np.float32); s[t.seeds] = 1
    inter = L @ s
    jac = inter / np.maximum(len(t.seeds) + sizes - inter, 1)
    own = {d: i for i, d in enumerate(B.context_ids)}.get(t.disease_id)
    if own is not None:
        jac[own] = 0
    cand = np.flatnonzero(jac > 0)
    top = cand[np.argsort(-jac[cand])[:k]]
    return np.asarray(L[top].T @ jac[top].astype(np.float32)).ravel()


def exact():
    for bench, split in (("temporal", "dev"), ("forward", "dev")):
        B = T.load(bench).split(split)
        Tm = CH.compute(B, "transfer_jaccard", split)
        K = KB.kernel("paralog_wide")
        lifted = CH.compute(B, "lift_paralog_wide", split)
        worst_T, worst_L, tie = 0, 0, 0
        for i in range(0, len(B.tasks), max(1, len(B.tasks) // 25)):
            t = B.tasks[i]
            Tb = brute_transfer(B, t)
            worst_T = max(worst_T, np.abs(Tb - Tm[i]).max())
            Lb = np.asarray(K @ Tb).ravel()
            ok = ~np.isnan(lifted[i])
            worst_L = max(worst_L, np.abs(Lb[ok] - lifted[i][ok]).max())
            # hidden-pair columns must be exactly zero in T if no visible neighbour row holds them
            tie += int(((Tm[i][t.targets] > 0) & (Tb[t.targets] == 0)).sum())
        print(f"{bench}/{split}: max|T_cached - T_brute| {worst_T:.2e}  max|lift - K T_brute| {worst_L:.2e}  "
              f"targets >0 cached but 0 brute: {tie}", flush=True)


def zero_stratum(bench, split, method):
    B = T.load(bench).split(split)
    Tm = CH.compute(B, "transfer_jaccard", split)
    if method.startswith("ch:"):
        S = CH.compute(B, method[3:], split)
    else:
        S = np.load(SDIR / f"{bench}_{split}_{method}.npy")
    deg = np.asarray(B.degree)
    rows = []
    for i, t in enumerate(B.tasks):
        z = t.targets[Tm[i, t.targets] == 0]
        nz = t.targets[Tm[i, t.targets] > 0]
        r = dict(disease_id=t.disease_id, cluster=t.cluster or t.disease_id, n_zero=len(z), n_tg=len(t.targets))
        if len(z):
            # (a) full ranking, zero-support targets as positives, supported targets removed
            m = MT.rank_metrics(S[i], z, np.concatenate([t.exclude, nz]).astype(np.int64), deg)
            r.update(z_ap=m["ap"], z_nuap=m["nuap"], z_auroc=m["auroc"], z_r100=m["recall@100"], z_r200=m["recall@200"])
            # (b) within the zero pool (genes transfer cannot rank at all)
            pool = np.ones(B.n_genes, bool); pool[t.exclude] = False; pool[nz] = False; pool &= (Tm[i] == 0)
            idx = np.flatnonzero(pool)
            pos = np.flatnonzero(np.isin(idx, z))
            m2 = MT.rank_metrics(S[i][idx], pos, np.array([], np.int64), deg[idx])
            r.update(pool_auroc=m2["auroc"], pool_ap=m2["ap"], pool_n=len(idx))
        rows.append(r)
    return pd.DataFrame(rows)


def zero(bench, *methods):
    out = []
    for split in ("dev", "confirm"):
        dfs = {m: zero_stratum(bench, split, m) for m in methods}
        d0 = dfs[methods[0]]
        print(f"\n{bench}/{split}: tasks with >=1 zero-support target {int((d0.n_zero > 0).sum())}/{len(d0)}; "
              f"zero-support targets {int(d0.n_zero.sum())}/{int(d0.n_tg.sum())} ({d0.n_zero.sum()/d0.n_tg.sum():.1%})")
        for m, d in dfs.items():
            line = dict(bench=bench, split=split, method=m, **{c: d[c].mean() for c in ("z_ap", "z_nuap", "z_auroc", "z_r100", "z_r200", "pool_auroc", "pool_ap")})
            if m != methods[0]:
                for met in ("z_ap", "z_nuap", "z_auroc", "pool_auroc"):
                    b = CMP.paired_bootstrap(d.dropna(subset=[met]), d0.dropna(subset=[met]), met, 5000)
                    line[f"d_{met}"] = b["delta"]; line[f"ci_{met}"] = f"[{b['ci_lo']:+.4f},{b['ci_hi']:+.4f}]"
            out.append(line)
    df = pd.DataFrame(out)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "exact":
        exact()
    elif cmd == "stack":
        bench, train, chs, tag = sys.argv[2:6]
        chs = expand(chs)
        if train == "past":
            res, imp = ST.run_past(bench, chs, tag)
        else:
            res, imp = ST.run(bench, chs, tag)
        print(imp.head(30).to_string())
    elif cmd == "alone":                          # alone <bench> <split> <group,...>
        bench, split = sys.argv[2:4]
        B = T.load(bench).split(split)
        rows = []
        for n in expand(sys.argv[4]).split(","):
            df = BM.evaluate(CH.compute(B, n, split), B, jobs=4)
            z = zero_stratum(bench, split, "ch:" + n)
            rows.append(dict(channel=n, ap=df.ap.mean(), nuap=df.nuap.mean(), auroc=df.auroc.mean(), r100=df["recall@100"].mean(),
                             onnet_ap=df.onnet_ap.mean(), z_ap=z.z_ap.mean(), z_nuap=z.z_nuap.mean(), z_auroc=z.z_auroc.mean(),
                             pool_auroc=z.pool_auroc.mean()))
            print(pd.DataFrame(rows[-1:]).round(4).to_string(index=False, header=len(rows) == 1), flush=True)
        pd.DataFrame(rows).to_csv(ROOT / f"results/kernels_b_alone_{bench}_{split}.tsv", sep="\t", index=False, float_format="%.4f")
    elif cmd == "zero":
        zero(sys.argv[2], *sys.argv[3:])
    elif cmd == "channels":                       # precompute channels: channels <bench> <group,...>
        bench = sys.argv[2]
        names = expand(sys.argv[3]).split(",")
        Bf = T.load(bench)
        Bs = [(Bf.split(s), s) for s in ("dev", "confirm")] if bench != "past" else []
        if bench == "past":
            P = Bf; rng = np.random.default_rng(0)
            sel = np.sort(rng.choice(len(P.tasks), min(900, len(P.tasks)), replace=False))
            Bs = [(T.Benchmark(P.name, [P.tasks[i] for i in sel], P.context, P.context_ids, P.network, P.union_network,
                               P.leakage_class, P.degree), "train900")]
        for B, s in Bs:
            for n in names:
                t0 = time.time(); X = CH.compute(B, n, s)
                print(f"{bench}/{s} {n}: nan {np.isnan(X).mean():.3f} nz {(np.nan_to_num(X) != 0).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
