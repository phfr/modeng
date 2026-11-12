"""Summarise implementer-A donor increment tests: paired bootstrap (cluster unit) of each run vs its reference."""
import sys, pandas as pd
from modeng import compare as C
from modeng.bench import RESULTS
PAIRS = [("temporal", "stackP_bdv_dA_a_impc_orth", "stackP_bdv"), ("temporal", "stackP_bdv_dA_a_dm_codep", "stackP_bdv"),
         ("temporal", "stackP_bdv_dA_pt", "stackP_bdv"), ("temporal", "stackP_bdv_dApre21", "stackP_bdv"),
         ("temporal", "stackP_bdv_dAps", "stackP_bdv"), ("temporal", "stackP_bdv_permA", "stackP_bdv"),
         ("temporal", "stackP_bdv_dAold", "stackP_bdv"), ("temporal", "stackP_bdv_refA", "stackP_bdv"),
         ("forward", "stack_base_dA_pt", "stack_base"), ("forward", "stack_base_dAnew", "stack_base"),
         ("forward", "stack_base_permA", "stack_base"), ("forward", "stack_bdv_dAnew", "stack_bdv_w05"),
         ("forward", "stack_bdv_w05", "stack_base")]
rows = []
for bench, a, b in PAIRS:
    for split in ("dev", "confirm"):
        fa, fb = RESULTS / bench / split / f"{a}.parquet", RESULTS / bench / split / f"{b}.parquet"
        if not (fa.exists() and fb.exists()): continue
        A, Bd = pd.read_parquet(fa), pd.read_parquet(fb)
        for m in ("nuap", "ap", "auroc", "recall@100", "onnet_nuap"):
            r = C.paired_bootstrap(A, Bd, m, 5000)
            rows.append(dict(bench=bench, split=split, run=a, ref=b, metric=m, a=r["mean_a"], b=r["mean_b"], delta=r["delta"],
                             lo=r["ci_lo"], hi=r["ci_hi"], p_le0=r["p_le0"], n=r["n_tasks"]))
R = pd.DataFrame(rows)
R.to_csv(RESULTS / "donors_a" / "increments.tsv", sep="\t", index=False, float_format="%.4f")
print(R[R.metric.isin(["nuap", "ap"])].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
