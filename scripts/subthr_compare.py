"""r4: paired cluster bootstrap (5,000) of stacker variants vs modeng-v1 on the per-task parquets (temporal dev / confirm).

  ~/biotools/venvs/bio-net/bin/python scripts/subthr_compare.py <split> <method> [<method> ...]  -> appends to results/subthr/compare.tsv
"""
import sys
import numpy as np, pandas as pd
R = "/home/ph/code/modeng/results/temporal"
REF = "stack_v3ptr_p0.35"
split, meths = sys.argv[1], sys.argv[2:]
ref = pd.read_parquet(f"{R}/{split}/{REF}.parquet").set_index("disease_id")
out = []
for m in meths:
    d = pd.read_parquet(f"{R}/{split}/{m}.parquet").set_index("disease_id").loc[ref.index]
    cl = ref.cluster.to_numpy(); uc = np.unique(cl); rng = np.random.default_rng(0)
    for c in ("ap", "nuap", "auroc", "recall@100", "onnet_ap", "onnet_nuap", "sa_ap", "sa_nuap"):
        a, b = d[c].to_numpy(float), ref[c].to_numpy(float); ok = np.isfinite(a) & np.isfinite(b); diff = a - b
        bs = []
        for _ in range(5000):
            idx = np.concatenate([np.flatnonzero((cl == u) & ok) for u in rng.choice(uc, len(uc))])
            bs.append(diff[idx].mean())
        out.append(dict(split=split, method=m, metric=c, n=int(ok.sum()), ref=b[ok].mean(), val=a[ok].mean(), delta=diff[ok].mean(),
                        lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
r = pd.DataFrame(out); print(r.round(4).to_string(index=False))
f = "/home/ph/code/modeng/results/subthr/compare.tsv"
import os
r.to_csv(f, sep="\t", index=False, mode="a", header=not os.path.exists(f), float_format="%.4f")
