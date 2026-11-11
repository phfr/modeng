"""Parity check vs modet: evaluate modet_headline and transfer_jaccard on modet's OWN candidate set
(panel_t0_string115_phys_notm_400 = network-covered seeds/test/unknown; ranking universe = T0 STRING nodes; nuAP degree
= T0 STRING degree) and compare with modet's reported per-disease numbers (ucov_parity_string_s0, modet-U-cov-Lb).

    PYTHONPATH=src python scripts/check_modet_parity.py
"""
import numpy as np
import pandas as pd

from modeng import data as D
from modeng import metrics as M
from modeng import tasks as T
from modeng.methods import baselines as b

NET = "t0_string115_phys_notm_400"
B = T.load("temporal")
deg = D.degree(NET)
off = np.flatnonzero(deg == 0)                                # not in the T0 STRING graph -> not ranked by modet
cov = pd.read_parquet(D.GT / f"panel_{NET}.parquet").set_index("disease_id")
rep = pd.read_parquet(D.MODET / f"results/harness/ucov_parity_string_s0/temporal_{NET}_devall.parquet")
rep = rep[rep.method == "modet-U-cov-Lb"].set_index("disease_id")

# tasks rebuilt on the covered panel (same disease ids, same fam masks)
tasks = []
for t in B.tasks:
    r = cov.loc[t.disease_id]
    seeds = D.gene_idx(r.seeds)
    tg = np.setdiff1d(D.gene_idx(r.test), seeds)
    exc = np.setdiff1d(np.union1d(np.union1d(seeds, D.gene_idx(r.unknown)), off), tg)
    tasks.append(T.Task(t.disease_id, t.name, t.split, seeds, tg, exc, t.fam_rows, t.mask_r, t.mask_c, t.cluster))
Bc = T.Benchmark("temporal", tasks, B.context, B.context_ids, B.network, B.union_network, B.leakage_class, deg)

rows = []
for name in ("modet_headline", "transfer_jaccard", "rwr_string", "popularity"):
    S = getattr(b, name)(tasks, Bc)
    for t, s in zip(tasks, S):
        m = M.rank_metrics(s, t.targets, t.exclude, deg)
        rows.append(dict(method=name, disease_id=t.disease_id, split=t.split, ap=m["ap"], nuap=m["nuap"], r100=m["recall@100"]))
df = pd.DataFrame(rows)
print("modeng recomputation on modet's covered candidate set (devall = 191 non-sealed):")
print(df.groupby("method")[["ap", "nuap", "r100"]].mean().round(4).to_string())
print(df.groupby(["method", "split"])[["ap", "nuap"]].mean().round(4).to_string())
h = df[df.method == "modet_headline"].set_index("disease_id")
j = h.join(rep[["auprc", "nu_auprc", "recall@100"]], how="inner")
print(f"\nmodet reported (U-cov-Lb, devall): AP {rep.auprc.mean():.4f} nuAP {rep.nu_auprc.mean():.4f} R@100 {rep['recall@100'].mean():.4f}")
print(f"per-disease agreement with modet's own metrics: max |dAP| {np.abs(j.ap - j.auprc).max():.2e}, "
      f"max |dnuAP| {np.abs(j.nuap - j.nu_auprc).max():.2e} (n={len(j)})")
