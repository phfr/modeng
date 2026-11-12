"""r6 P1b: check that every ext witness of v1clean behaves on `temporal_wide` exactly as on `temporal` (same 21.04 seed
release, same date cuts). For 3 temporal dev tasks and 3 wide_dev tasks, each channel is computed directly (no cache) on a
Benchmark named 'temporal' and on the same tasks/context named 'temporal_wide'; they must be identical (NaN-equal).
Counterfactual: the same with an unmapped name ('unmapped') shows which channels depend on the mapping (date cut / NaN).
Usage: PYTHONPATH=src python scripts/r6_wide_port_check.py
"""
import numpy as np
from modeng import tasks as T, ext

EXT = ["cs_donor", "dA_pt", "dc_ridge", "dc_ridge_pos", "gwas_catlag", "pri_l2g_n", "pri_l2g_max", "pri_mend_n", "pri_hpo_n",
       "hz_gwas", "hz_mend", "ph_mean", "ph_top3", "pt_ridge", "pt_ridge_pos",
       "vc_mean_hpa20_sc", "vc_max_hpa20_sc", "vc_mean_hpa20_tis", "vc_max_hpa20_tis", "vc_mean_gtex8", "vc_max_gtex8"]
reg = ext.registry()


def renamed(B, name, idx):
    return T.Benchmark(name, [B.tasks[i] for i in idx], B.context, B.context_ids, B.network, B.union_network,
                       B.leakage_class, B.degree)


def same(a, b):
    return a.shape == b.shape and np.array_equal(np.nan_to_num(a, nan=-7.7), np.nan_to_num(b, nan=-7.7))


Tm = T.load("temporal").split("dev")
Wd = T.load("temporal_wide").split("wide_dev")
# pick 3 tasks per benchmark that have a gwas_catlag-relevant mix: first 3 by order (deterministic)
cases = [("temporal dev", Tm, [0, 1, 2]), ("wide_dev", Wd, [0, 1, 2])]
k = 0
print(f"{'channel':20s} {'set':13s} temporal==temporal_wide  unmapped_differs  frac_nonnan")
for ch in EXT:
    for lab, B, idx in cases:
        k += 1
        a = np.asarray(reg[ch](renamed(B, "temporal", idx), f"pc{k}a"), np.float32)
        b = np.asarray(reg[ch](renamed(B, "temporal_wide", idx), f"pc{k}b"), np.float32)
        try:
            c = np.asarray(reg[ch](renamed(B, "unmapped", idx), f"pc{k}c"), np.float32)
            dif = str(not same(a, c))
        except Exception as e:                     # unmapped name raises (KeyError) where a dict lookup is strict
            dif = f"raises {type(e).__name__}"
        print(f"{ch:20s} {lab:13s} {str(same(a, b)):22s}  {dif:16s}  {np.isfinite(a).mean():.3f}", flush=True)
        assert same(a, b), (ch, lab)
print("ALL EQUAL")
