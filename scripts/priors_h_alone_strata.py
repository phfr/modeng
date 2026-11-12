"""implementer H: channels alone on temporal target strata (gw = gwas_only, ng = clinvar/curated/mixed non-GWAS arrivals)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/ph/code/modeng/scripts")
from modeng import tasks as T, bench as BM, channels as CH
from priors_h_strata import add_strata, STRATA
names = sys.argv[1].split(",")
Bf = T.load("temporal"); rows = []
for split in ("dev", "confirm"):
    B = Bf.split(split); add_strata(B)
    for n in names:
        S = CH.compute(B, n, split)
        S = np.where(np.isnan(S), np.nanmin(S) - 1, S)
        df = BM.evaluate(S, B, jobs=2)
        rows.append(dict(split=split, ch=n, nuap=df.nuap.mean(), gw_nuap=df.gw_nuap.mean(), ng_nuap=df.ng_nuap.mean(),
                         gw_auroc=df.gw_auroc.mean(), ng_auroc=df.ng_auroc.mean(), n_gw=int(df.gw_nuap.notna().sum()), n_ng=int(df.ng_nuap.notna().sum())))
        print(rows[-1], flush=True)
    for k in STRATA:
        for t in B.tasks: t.alt.pop(k, None)
pd.DataFrame(rows).to_csv("results/priors_h_alone_strata_temporal.tsv", sep="\t", index=False, float_format="%.4f")
