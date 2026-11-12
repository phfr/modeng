"""ML-4 channels alone on temporal dev / confirm (covered tasks only), strict-a endpoints, vs plain T0 MAGMA z (gwas_own) and the
untrained crossing transform (gwas_xpow) and transfer on the same tasks. Paired per-task differences pfc - gwas_own (cluster bootstrap).
Writes results/powerfc/alone.tsv and alone_paired.tsv.  Usage: python scripts/powerfc_alone.py
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.chdir("/home/ph/code/modeng"); sys.path.insert(0, "/home/ph/code/modeng/src")
import numpy as np, pandas as pd
src = open("scripts/review_analyze.py").read(); src = src[:src.index("def boot")]
exec(src)
from modeng import channels as CH

CHS = ["pfc_own", "pfc_sem", "pfc_own_perm", "gwas_own", "gwas_xpow"]
MET = ["ap", "nuap", "auroc", "recall@100", "sa_ap", "sa_nuap", "sa_onnet_ap", "sa_onnet_nuap"]
rows, prs = [], []
for split in ("dev", "confirm"):
    B = Bf.split(split); cl = np.array([t.cluster for t in B.tasks])
    S = {c: CH.compute(B, c, split) for c in CHS}
    cov = ~np.all(np.isnan(S["pfc_own"]), 1)
    S["transfer"] = BL.transfer_jaccard(B.tasks, B)
    E = {}
    for c, M_ in S.items():
        X = M_.copy(); X[~cov] = np.nan
        E[c] = endpoints(X, B)
    nsa = int(E["pfc_own"]["sa_ap"].notna().sum())
    for c, e in E.items():
        rows.append(dict(split=split, channel=c, n_cov=int(cov.sum()), n_sa=nsa, **{m: e[m].mean() for m in MET if m in e}))
    for c in ("pfc_own", "pfc_sem", "gwas_xpow"):
        for m in MET:
            a, b = E[c][m].to_numpy(float), E["gwas_own"][m].to_numpy(float); ok = np.isfinite(a) & np.isfinite(b)
            d = a[ok] - b[ok]; cc = cl[ok]; uc = np.unique(cc); rng = np.random.default_rng(0)
            g = pd.DataFrame(dict(d=d, c=cc)).groupby("c")["d"].agg(["sum", "count"]); s_, n_ = g["sum"].to_numpy(), g["count"].to_numpy()
            idx = rng.integers(0, len(g), (5000, len(g))); bt = s_[idx].sum(1) / n_[idx].sum(1)
            prs.append(dict(split=split, channel=c, vs="gwas_own", metric=m, n=int(ok.sum()), delta=d.mean(),
                            lo=np.quantile(bt, .025), hi=np.quantile(bt, .975)))
R = pd.DataFrame(rows); P = pd.DataFrame(prs)
os.makedirs("results/powerfc", exist_ok=True)
R.to_csv("results/powerfc/alone.tsv", sep="\t", index=False, float_format="%.4f")
P.to_csv("results/powerfc/alone_paired.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(R.round(4).to_string(index=False)); print(P.round(4).to_string(index=False))
