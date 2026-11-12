"""Channel-alone metrics of implementer-A donor channels (also fills data/cache/channels). Appends to results/donors_a/alone.tsv."""
import sys, numpy as np, pandas as pd
from modeng import tasks as T, bench as BM, channels as CH
bench, chans = sys.argv[1], sys.argv[2].split(",")
Bf = T.load(bench)
for split in ("dev", "confirm"):
    B = Bf.split(split)
    for c in chans:
        S = CH.compute(B, c, split)
        df = BM.evaluate(np.nan_to_num(S, nan=-1.0), B, jobs=4)
        row = dict(bench=bench, split=split, channel=c, n=len(df), ap=df.ap.mean(), nuap=df.nuap.mean(), auroc=df.auroc.mean(),
                   r100=df["recall@100"].mean(), onnet_nuap=df.onnet_nuap.mean(), cover=float(np.isfinite(S).mean()))
        print(row, flush=True)
        pd.DataFrame([row]).to_csv("/home/ph/code/modeng/results/donors_a/alone.tsv", sep="\t", mode="a", index=False, header=False, float_format="%.4f")
