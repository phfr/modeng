"""Implementer C (disease-side transfer): evaluate channels alone and write per-task parquets.
usage: PYTHONPATH=src python scripts/r1_disease_c.py <bench> <ch1,ch2,...> [dev,confirm]"""
import sys, time, numpy as np
from modeng import tasks as T, channels as CH, bench as BM
bench = sys.argv[1]; chans = sys.argv[2].split(","); splits = sys.argv[3].split(",") if len(sys.argv) > 3 else ["dev", "confirm"]
Bf = T.load(bench)
for sp_ in splits:
    B = Bf.split(sp_)
    for c in chans:
        t0 = time.time(); S = CH.compute(B, c, sp_)
        df = BM.evaluate(S, B, jobs=4)
        out = BM.RESULTS / bench / sp_ / f"{c}.parquet"; out.parent.mkdir(parents=True, exist_ok=True); df.to_parquet(out, index=False)
        print(f"{bench} {sp_} {c:24s} AP {df.ap.mean():.4f} nuAP {df.nuap.mean():.4f} AUROC {df.auroc.mean():.4f} "
              f"R@100 {df['recall@100'].mean():.4f} onnet AP {df.onnet_ap.mean():.4f} nuAP {df.onnet_nuap.mean():.4f} "
              f"nan_tasks {int(np.isnan(S).all(1).sum())} ({time.time()-t0:.0f}s)", flush=True)
