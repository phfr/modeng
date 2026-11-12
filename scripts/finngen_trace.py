"""r6 kill rule: share of strict-a targets traced (top 200 of the candidate ranking) by any genetic witness.
Usage: python scripts/finngen_trace.py <bench> <split>   (temporal dev | temporal_wide wide_dev; never wide_confirm)"""
import sys
import numpy as np, pandas as pd
from scipy.stats import rankdata
from modeng import tasks as T, channels as CH

bench, split = sys.argv[1], sys.argv[2]
assert split != "wide_confirm"
B = T.load(bench).split(split)
W = ["gwas_catlag", "fg_hit", "fg_hit_cs2g"]
S = {w: CH.compute(B, w, split) for w in W}
rows = []
for i, t in enumerate(B.tasks):
    if "sa" not in t.alt:
        continue
    sa = t.alt["sa"][0]
    cand = np.ones(B.n_genes, bool); cand[t.exclude] = False
    ci = np.flatnonzero(cand)
    rec = dict(disease_id=t.disease_id, name=t.name, n_sa=len(sa), n_tgt=len(t.targets))
    for w in W:
        v = S[w][i, ci].astype(float)
        cov = not np.all(np.isnan(v))
        v = np.where(np.isnan(v), -np.inf, v)
        r = rankdata(-v, method="average")
        top = set(ci[r <= 200]) if cov else set()
        rec[f"cov_{w}"] = cov
        rec[f"sa_tr_{w}"] = len(set(sa) & top)
        rec[f"tg_tr_{w}"] = len(set(t.targets) & top)
        rec[f"_sa_{w}"] = set(sa) & top
    rec["sa_tr_fg"] = len(rec["_sa_fg_hit"] | rec["_sa_fg_hit_cs2g"])
    rec["sa_tr_any"] = len(rec["_sa_gwas_catlag"] | rec["_sa_fg_hit"] | rec["_sa_fg_hit_cs2g"])
    rec["sa_tr_fg_only"] = len((rec["_sa_fg_hit"] | rec["_sa_fg_hit_cs2g"]) - rec["_sa_gwas_catlag"])
    rec["fg_only_genes"] = ";".join(map(str, sorted((rec["_sa_fg_hit"] | rec["_sa_fg_hit_cs2g"]) - rec["_sa_gwas_catlag"])))
    for w in W:
        del rec[f"_sa_{w}"]
    rows.append(rec)
df = pd.DataFrame(rows)
out = f"/home/ph/code/modeng/results/finngen/trace_{bench}_{split}.tsv"
import os; os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, sep="\t", index=False)
n = df.n_sa.sum()
print(f"{bench}/{split}: {len(df)} sa tasks, {n} strict-a targets")
for c in ["sa_tr_gwas_catlag", "sa_tr_fg_hit", "sa_tr_fg_hit_cs2g", "sa_tr_fg", "sa_tr_any", "sa_tr_fg_only"]:
    print(f"  {c:20s} {df[c].sum():5d}  {df[c].sum()/n:.3f}")
for w in W:
    print(f"  covered by {w}: {df[f'cov_{w}'].sum()} sa tasks; all-target traced {df[f'tg_tr_{w}'].sum()}/{df.n_tgt.sum()}")
