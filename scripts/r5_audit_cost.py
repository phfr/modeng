"""R5 witness audit: wall-clock compute cost of each modeng-v1 channel on temporal dev (53 tasks), recomputed without the
cache (2 BLAS threads; first call includes loading inputs, a second call shows the per-task part where a module memoises).
Output results/r5/channel_cost.tsv. Indicative only (shared machine)."""
import os, sys, time
os.environ["OMP_NUM_THREADS"] = "2"; os.environ["OPENBLAS_NUM_THREADS"] = "2"; os.environ["MKL_NUM_THREADS"] = "2"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from modeng import tasks as T, channels as CH, ext
from modeng.methods import baselines as BL

B = T.load("temporal").split("dev")
reg = ext.registry()
todo = ["transfer_jaccard", "transfer_onto", "popularity", "degree", "rwr_string", "don_msig_go", "don_hpo2101", "vmean_esmc",
        "vmean_ps_k562", "cs_donor", "dA_pt", "dc_ridge", "dc_ridge_pos", "gwas_catlag", "pri_l2g_n", "pri_l2g_max", "pri_mend_n",
        "pri_hpo_n", "hz_gwas", "hz_mend", "ph_mean", "ph_top3", "pt_ridge", "pt_ridge_pos"]
rows = []
for n in todo:
    t0 = time.time()
    if n in ("transfer_jaccard", "transfer_onto", "popularity", "degree", "rwr_string"):
        getattr(BL, n)(B.tasks, B)
    elif n.startswith("don_"):
        CH.donor_transfer(B.tasks, CH.load_family(n[4:]))
    elif n.startswith("vmean_"):
        X, m = CH.load_view(n[6:]); CH.view_sim(B.tasks, X, m)
    else:
        reg[n](B, "dev_costprobe")
    rows.append(dict(channel=n, seconds_53_tasks=round(time.time() - t0, 1)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results/r5/channel_cost.tsv"), sep="\t", index=False)
