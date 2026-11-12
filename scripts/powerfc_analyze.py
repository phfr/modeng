"""r4 ML-4 analysis: power-forecast variants vs modeng-v1 (stack_v3ptr_p0.35) on dev OOF and temporal confirm (rule: scripts/powerfc_runs.sh).
Endpoints = scripts/review_analyze.py `endpoints`; paired cluster bootstrap (5000) vs v1 and vs modet headline.
Writes results/powerfc/endpoints_vs_ref.tsv.  Usage: python scripts/powerfc_analyze.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.chdir("/home/ph/code/modeng"); sys.path.insert(0, "/home/ph/code/modeng/src")
import numpy as np, pandas as pd
src = open("scripts/review_analyze.py").read(); src = src[:src.index("def boot")]
exec(src)
from modeng import bench as BM

REF = "stack_v3ptr_p0.35"
VARS = ["stack_v1_powerfc", "stack_v1_powerfc2"]
CTRL = ["stack_v1_powerfc_perm", "stack_v1_gwasown"]
MET = ["ap", "nuap", "onnet_ap", "onnet_nuap", "sa_ap", "sa_nuap", "sa_onnet_ap", "sa_onnet_nuap", "degbin_ap"]
SA4 = ["sa_ap", "sa_nuap", "sa_onnet_ap", "sa_onnet_nuap"]


def load(split, m):
    return np.load(BM.RESULTS / "temporal" / split / f"{m}.scores.npy").astype(np.float32)


def boot(a, b, cl, n=5000):
    ok = np.isfinite(a) & np.isfinite(b); d = (a - b)[ok]
    g = pd.DataFrame(dict(d=d, c=cl[ok])).groupby("c")["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy(); idx = np.random.default_rng(0).integers(0, len(g), (n, len(g)))
    bt = s[idx].sum(1) / c[idx].sum(1)
    return a[ok].mean(), b[ok].mean(), d.mean(), np.quantile(bt, .025), np.quantile(bt, .975), int(ok.sum())


res = []
chosen = None
for split in ("dev", "confirm"):
    B = Bf.split(split); cl = np.array([t.cluster for t in B.tasks])
    ref = endpoints(load(split, REF), B); hl = endpoints(BL.modet_headline(B.tasks, B), B)
    E = {m: endpoints(load(split, m), B) for m in VARS + CTRL}
    E[REF] = ref
    for name, e in E.items():
        for c in MET:
            for rn, r in (("v1", ref), ("headline", hl)):
                if c in e and c in r:
                    mv, rv, d, lo, hi, n = boot(e[c].to_numpy(float), r[c].to_numpy(float), cl)
                    res.append(dict(split=split, variant=name, vs=rn, metric=c, value=mv, ref_value=rv, delta=d, lo=lo, hi=hi, n=n))
    if split == "dev":
        best = 0.0
        for m in VARS:
            sa4 = np.mean([E[m][c].mean() - ref[c].mean() for c in SA4])
            okc = E[m]["ap"].mean() >= ref["ap"].mean() - 0.003 and E[m]["nuap"].mean() >= ref["nuap"].mean() - 0.003
            print(f"[dev rule] {m}: SA4 delta {sa4:+.4f}, ap/nuap ok {okc}", flush=True)
            if okc and sa4 > best:
                best, chosen = sa4, m
        print("chosen on dev:", chosen, flush=True)
R = pd.DataFrame(res); R.to_csv("results/powerfc/endpoints_vs_ref.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 30)
for split in ("dev", "confirm"):
    for vs in ("v1", "headline"):
        x = R[(R.split == split) & (R.vs == vs)]
        if vs == "v1":
            x = x[x.variant != REF]
        x = x.assign(cell=x.apply(lambda r: f"{r.value:.3f} {r.delta:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]", axis=1))
        print(f"\n=== {split}: vs {vs} ===")
        print(x.pivot(index="variant", columns="metric", values="cell")[[c for c in MET if c in x.metric.unique()]].to_string())
print("chosen:", chosen)
