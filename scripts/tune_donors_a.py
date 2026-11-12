"""Tune the donors_a hyper scorer on temporal DEV only (channel alone, mean nuAP / AP). Writes results/donors_a/tune.tsv."""
import itertools, sys, numpy as np, pandas as pd
from modeng import tasks as T, bench as BM, channels as CH
from modeng.ext import donors_a as DA
B = T.load("temporal").split("dev")
fams = sys.argv[1:] or ["hpo2101", "msig_go", "a_impc_orth", "a_dm_codep", "msig_reactome"]
rows = []
for f in fams:
    M = DA.load_any(f)
    df = BM.evaluate(CH.donor_transfer(B.tasks, M), B, jobs=4)
    rows.append(dict(fam=f, scorer="jaccard_k50", ap=df.ap.mean(), nuap=df.nuap.mean(), auroc=df.auroc.mean()))
    print(rows[-1], flush=True)
    for K, A, I, KM in itertools.product([20, 50, 200], [0, 0.5, 1], [True, False], [1, 2]):
        df = BM.evaluate(np.nan_to_num(DA.hyper_transfer(B.tasks, M, K, A, I, KM)), B, jobs=4)
        rows.append(dict(fam=f, scorer=f"hyper_K{K}_a{A}_idf{int(I)}_k{KM}", ap=df.ap.mean(), nuap=df.nuap.mean(), auroc=df.auroc.mean()))
    print(pd.DataFrame(rows).query("fam==@f").sort_values("nuap").tail(3).to_string(), flush=True)
R = pd.DataFrame(rows); R.to_csv("/home/ph/code/modeng/results/donors_a/tune.tsv", sep="\t", index=False)
piv = R.pivot(index="scorer", columns="fam", values="nuap"); piv["mean"] = piv.mean(1)
print(piv.sort_values("mean").tail(12).to_string())
