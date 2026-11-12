"""r4: sub-threshold channels alone on temporal dev/confirm (covered tasks only), all targets and strict-a (`sa`) targets.

  OMP_NUM_THREADS=4 PYTHONPATH=src ~/biotools/venvs/bio-net/bin/python scripts/subthr_alone.py
-> results/subthr/alone.tsv (one row per bench/split/channel), results/subthr/hits.tsv (hit counts: targets / sa targets with score > 0).
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/ph/code/modeng/src")
from modeng import tasks as T, metrics as M, channels as C
from modeng.methods import baselines as BL

OUT = "/home/ph/code/modeng/results/subthr"; os.makedirs(OUT, exist_ok=True)
CH = ["subthr_near", "subthr_cs2g", "subthr_win", "subthr_near_perm", "subthr_cs2g_perm", "subthr_win_perm", "subthr_gws_near", "subthr_gws_cs2g",
      "gwas_catlag", "gwas_own"]
rows, hits = [], []
for split in ("dev", "confirm"):
    B = T.load("temporal").split(split); deg = np.asarray(B.degree)
    S = {c: C.compute(B, c, split) for c in CH}
    S["transfer_jaccard"] = np.asarray(BL.transfer_jaccard(B.tasks, B), np.float32)
    cov = np.isfinite(S["subthr_near"]).any(1)
    for c, X in S.items():
        R = []
        for i, t in enumerate(B.tasks):
            if not cov[i]:
                continue
            s = np.nan_to_num(X[i], nan=0.0)
            m = M.rank_metrics(s, t.targets, t.exclude, deg); r = dict(ap=m["ap"], nuap=m["nuap"], auroc=m["auroc"], r100=m["recall@100"])
            if "sa" in t.alt:
                ms = M.rank_metrics(s, *t.alt["sa"], deg); r.update(sa_ap=ms["ap"], sa_nuap=ms["nuap"], sa_auroc=ms["auroc"], sa_r100=ms["recall@100"])
            R.append(r)
            if c.startswith("subthr") or c == "gwas_catlag":
                pos = s > 0; pos[t.seeds] = False
                sa = t.alt["sa"][0] if "sa" in t.alt else np.array([], int)
                hits.append(dict(split=split, channel=c, disease=t.name, n_hit=int(pos.sum()), n_tgt=len(t.targets), tgt_hit=int(pos[t.targets].sum()),
                                 n_sa=len(sa), sa_hit=int(pos[sa].sum()) if len(sa) else 0))
        d = pd.DataFrame(R)
        rows.append(dict(split=split, channel=c, n=len(d), n_sa=int(d.sa_ap.notna().sum()) if "sa_ap" in d else 0, **d.mean().to_dict()))
    print(pd.DataFrame([r for r in rows if r["split"] == split]).round(4).to_string(index=False), flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/alone.tsv", sep="\t", index=False, float_format="%.4f")
h = pd.DataFrame(hits); h.to_csv(f"{OUT}/hits.tsv", sep="\t", index=False)
print(h.groupby(["split", "channel"])[["n_hit", "n_tgt", "tgt_hit", "n_sa", "sa_hit"]].sum().to_string())
