"""Run modeng.stack (unchanged) and also save the score matrices (for the GWAS-only stratum). Implementer D.
Usage: PYTHONPATH=src python scripts/gwas_d_stack.py <past|cv> <bench> <channels> <tag>"""
import sys
import numpy as np
from modeng import stack as ST, bench as BM

mode, bench, channels, tag = sys.argv[1:5]
OUT = BM.RESULTS / "gwas_d" / "scores"; OUT.mkdir(parents=True, exist_ok=True)
_orig = ST._record


def rec(S, B, bench_, split, method, t0, write):
    np.save(OUT / f"{bench_}_{split}_{method}.npy", S.astype(np.float32))
    return _orig(S, B, bench_, split, method, t0, write)


ST._record = rec
res, imp = (ST.run_past if mode == "past" else ST.run)(bench, channels, tag)
print(imp.head(30).to_string())
