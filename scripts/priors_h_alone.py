import sys, time, numpy as np, pandas as pd
from modeng import tasks as T, bench as BM, channels as CH
bench = sys.argv[1]; names = sys.argv[2].split(",")
Bf = T.load(bench)
rows = []
for split in ("dev", "confirm"):
    B = Bf.split(split)
    for n in names:
        t0 = time.time()
        S = CH.compute(B, n, split)
        S2 = np.where(np.isnan(S), np.nanmin(S) - 1 if np.isfinite(S).any() else 0, S)
        df = BM.evaluate(S2, B, jobs=4)
        rows.append(dict(bench=bench, split=split, ch=n, ap=df.ap.mean(), nuap=df.nuap.mean(), auroc=df.auroc.mean(),
                         r100=df["recall@100"].mean(), onnet_nuap=df.onnet_nuap.mean(), nan_frac=float(np.isnan(S).mean()), sec=round(time.time()-t0,1)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(f"results/priors_h_alone_{bench}_h.tsv", sep="\t", index=False, float_format="%.4f")
