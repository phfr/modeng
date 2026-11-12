"""r3_prospective: paired cluster bootstrap (modeng.compare) of the past-trained / hybrid v1 stackers vs references.
    python scripts/r3_compare.py stackP_v1past stackP_v1past_nopri stackH_v1hyb  -> results/r3/compare.tsv"""
import sys
import pandas as pd
from modeng.compare import paired_bootstrap, load
from modeng.bench import RESULTS

REFS = ["transfer_jaccard", "stack_v3ptr_p0.35", "modet_headline", "stackP_v2"]
rows = []
for a in sys.argv[1:]:
    for split in ("dev", "confirm"):
        fa = RESULTS / "temporal" / split / f"{a}.parquet"
        if not fa.exists():
            continue
        A = load("temporal", split, a)
        for b in REFS + [x for x in sys.argv[1:] if x != a]:
            if not (RESULTS / "temporal" / split / f"{b}.parquet").exists():
                continue
            Bm = load("temporal", split, b)
            for met in ("ap", "nuap", "onnet_ap", "onnet_nuap"):
                if met not in Bm or Bm[met].notna().sum() == 0:
                    continue
                r = paired_bootstrap(A, Bm, met)
                rows.append(dict(split=split, a=a, b=b, **r))
df = pd.DataFrame(rows)
df.to_csv("/home/ph/code/modeng/results/r3/compare.tsv", sep="\t", index=False, float_format="%.4f")
print(df[["split", "a", "b", "metric", "n_tasks", "mean_a", "mean_b", "delta", "ci_lo", "ci_hi", "win_rate"]].round(4).to_string(index=False))
