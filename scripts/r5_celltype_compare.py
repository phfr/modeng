"""r5: paired cluster bootstrap of stack_v1_ct vs stack_v3ptr_p0.35 (v1) and vs the permuted control stack_v1_ctperm.
-> results/r5ct/compare.tsv"""
import pandas as pd
from modeng import compare as CMP
rows = []
for split in ("dev", "confirm"):
    for a, b in (("stack_v1_ct", "stack_v3ptr_p0.35"), ("stack_v1_ctperm", "stack_v3ptr_p0.35"), ("stack_v1_ct", "stack_v1_ctperm")):
        for m in ("ap", "nuap", "sa_nuap", "auroc", "recall@100"):
            A, Bd = CMP.load("temporal", split, a), CMP.load("temporal", split, b)
            if m not in A:
                continue
            r = CMP.paired_bootstrap(A, Bd, m, 10000)
            rows.append(dict(split=split, a=a, b=b, **r))
df = pd.DataFrame(rows)
df.to_csv("results/r5ct/compare.tsv", sep="\t", index=False, float_format="%.4f")
print(df[["split", "a", "b", "metric", "mean_a", "mean_b", "delta", "ci_lo", "ci_hi"]].round(4).to_string(index=False))
