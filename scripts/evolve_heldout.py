"""F2-3 held-out evaluation of the selected evolved kernel (run once; see docs/reports/r2_evolve.md).
evo_transfer alone vs transfer_jaccard on temporal and forward dev/confirm (all-gene nuAP/AP, onnet nuAP), paired cluster
bootstrap. Post-hoc decomposition (labelled as such, not a claim): cand_7 = transfer_jaccard x (deg+1)^-0.6."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from modeng import tasks as T, bench as BM, channels as CH
from modeng.compare import paired_bootstrap, load
from modeng.evolve.harness import SafeBench
from modeng.evolve import cand_7


def record(df, bench, split, method):
    out = BM.RESULTS / bench / split / f"{method}.parquet"; df.to_parquet(out, index=False)
    row = dict(bench=bench, split=split, method=method, n=int(df["ap"].notna().sum()), ap=df.ap.mean(), nuap=df.nuap.mean(),
               auroc=df.auroc.mean(), r100=df["recall@100"].mean(), onnet_ap=df.onnet_ap.mean(), onnet_nuap=df.onnet_nuap.mean(),
               sa_ap=df["sa_ap"].mean() if "sa_ap" in df else np.nan, seconds=0, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
               git=BM.git_sha())
    pd.DataFrame([row]).to_csv(BM.LEADERBOARD, sep="\t", mode="a", header=False, index=False, float_format="%.4f")


rows = []
for bench in ("temporal", "forward"):
    for split in ("dev", "confirm"):
        B = T.load(bench).split(split)
        S = CH.compute(B, "evo_transfer", split)
        record(BM.evaluate(S, B, jobs=6), bench, split, "evo_transfer")
        sb = SafeBench(B)
        record(BM.evaluate(np.asarray(cand_7.score(sb.tasks, sb), np.float32), B, jobs=6), bench, split, "evo_c7_jacdeg")
        base = load(bench, split, "transfer_jaccard")
        for m in ("evo_transfer", "evo_c7_jacdeg"):
            a = load(bench, split, m)
            for metric in ("nuap", "ap", "onnet_nuap", "auroc"):
                r = paired_bootstrap(a, base, metric)
                rows.append(dict(bench=bench, split=split, method=m, **r))
                print(f"{bench} {split} {m} {metric}: {r['mean_a']:.4f} vs tj {r['mean_b']:.4f} d {r['delta']:+.4f} "
                      f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] P<=0 {r['p_le0']:.4f} win {r['win_rate']:.2f}", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/evolve/heldout.tsv", sep="\t", index=False, float_format="%.5f")
