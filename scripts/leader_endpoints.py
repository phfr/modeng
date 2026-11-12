"""Red-team endpoints (from scripts/review_analyze.py) applied to the leader stack_v3cv (saved scores) vs transfer and modet headline."""
import sys, numpy as np, pandas as pd
src = open("scripts/review_analyze.py").read(); src = src[:src.index("def boot")]
exec(src)
from modeng import bench as BM
out = []
for split in ("dev", "confirm"):
    B = Bf.split(split)
    S = {"leader": np.load(BM.RESULTS / "temporal" / split / f"{sys.argv[1] if len(sys.argv)>1 else 'stack_v3cv_s'}.scores.npy").astype(np.float32),
         "transfer": BL.transfer_jaccard(B.tasks, B), "headline": BL.modet_headline(B.tasks, B)}
    E = {k: endpoints(v, B) for k, v in S.items()}
    cl = E["leader"]["cluster"].to_numpy()
    rng = np.random.default_rng(0); uc = np.unique(cl)
    for ref in ("transfer", "headline"):
        for c in ("ap", "nuap", "recall@100", "p@20", "onnet_ap", "onnet_nuap", "sa_onnet_ap", "sa_onnet_nuap", "sa_ap", "sa_nuap", "degbin_ap", "hub_ap", "nonhub_ap"):
            a, b = E["leader"][c].to_numpy(float), E[ref][c].to_numpy(float); ok = np.isfinite(a) & np.isfinite(b)
            d = a - b; ds = []
            for _ in range(3000):
                pick = rng.choice(uc, len(uc)); idx = np.concatenate([np.flatnonzero((cl == u) & ok) for u in pick]); ds.append(d[idx].mean())
            out.append(dict(split=split, ref=ref, metric=c, leader=np.nanmean(a[ok]), ref_v=np.nanmean(b[ok]), delta=d[ok].mean(), lo=np.percentile(ds, 2.5), hi=np.percentile(ds, 97.5), n=int(ok.sum())))
r = pd.DataFrame(out); r.to_csv("results/review/leader_endpoints_" + (sys.argv[1] if len(sys.argv)>1 else "v3cv") + ".tsv", sep="\t", index=False, float_format="%.4f"); print(r.round(4).to_string(index=False))
