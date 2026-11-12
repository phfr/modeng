"""r5: cell-type channels alone on temporal dev/confirm (all targets, and the zero-transfer-support stratum).
Usage: PYTHONPATH=src python scripts/r5_celltype_alone.py  -> results/r5ct/alone.tsv, results/r5ct/zero.tsv"""
import sys
sys.path.insert(0, "/home/ph/code/modeng/scripts")
import numpy as np, pandas as pd
from modeng import tasks as T, bench as BM, channels as CH, compare as CMP
import r1_kernels_b as KB
_c = CH.compute
CH.compute = lambda B, n, s: np.nan_to_num(_c(B, n, s), nan=-1e9)   # all-NaN rows (no enriched cluster) rank last

CHS = ["transfer_jaccard", "vmean_hpa_sc", "ct_act", "ct_act_hpa", "ct_act_pops", "ct_act_perm", "ct_donor", "ct_donor_perm"]
rows, zrows = [], []
Bf = T.load("temporal")
for split in ("dev", "confirm"):
    B = Bf.split(split)
    for c in CHS:
        df = BM.evaluate(CH.compute(B, c, split), B, jobs=4)
        rows.append(dict(split=split, channel=c, **{m: df[m].mean() for m in ("ap", "nuap", "auroc", "recall@100", "onnet_ap", "sa_ap")}))
        z = KB.zero_stratum("temporal", split, "ch:" + c)
        zrows.append(dict(split=split, channel=c, n_tasks_z=int((z.n_zero > 0).sum()), **{m: z[m].mean() for m in ("z_ap", "z_nuap", "z_auroc", "z_r100", "pool_auroc", "pool_ap")}))
        print(rows[-1], zrows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/r5ct/alone.tsv", sep="\t", index=False, float_format="%.4f")
pd.DataFrame(zrows).to_csv("results/r5ct/zero.tsv", sep="\t", index=False, float_format="%.4f")
print(pd.DataFrame(rows).round(4).to_string(index=False)); print(pd.DataFrame(zrows).round(4).to_string(index=False))
