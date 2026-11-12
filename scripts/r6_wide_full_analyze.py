"""r6 P1b analysis: summary table + paired cluster bootstraps (pre-registration docs/preregistration_wide_full.md).
Seed-averaged arms (`<m>@avg`) = per-task metrics averaged over seed 0 and seed 1. Degree-matched AP (`degbin_ap`, within
degree-decile bins + off-network bin, scripts/review_analyze.py) from saved scores where available.
Output: results/r6w/summary.tsv, results/r6w/compare.tsv, results/r6w/decision.tsv"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from modeng import tasks as T, bench as BM
from modeng.compare import paired_bootstrap
from modeng.methods import baselines as BL

src = open("scripts/review_analyze.py").read(); src = src[:src.index("def endpoints")]
exec(src)                                   # within_bin_ap (+ temporal degree deciles)

SETS = {("temporal", "dev"): T.load("temporal").split("dev"), ("temporal", "confirm"): T.load("temporal").split("confirm"),
        ("temporal_wide", "wide_dev"): T.load("temporal_wide").split("wide_dev")}
METH = ["transfer_jaccard", "stack_v1clean", "stackP_v1clean", "stackP_v2", "stackW_wclean",
        "stackW_W_v1clean", "stackW_W_v1clean_s1", "stackW_W_v2", "stackW_W_v2_s1", "stackW_W_v1perm"]
MET = ["ap", "nuap", "onnet_nuap", "sa_nuap", "degbin_ap"]


def degbin(bench, split, m, B):
    f = BM.RESULTS / bench / split / f"{m}.scores.npy"
    if m == "transfer_jaccard":
        S = BL.transfer_jaccard(B.tasks, B)
    elif f.exists():
        S = np.load(f).astype(np.float32)
    else:
        return np.full(len(B.tasks), np.nan)
    return np.array([within_bin_ap(S[i], t.targets, t.exclude) for i, t in enumerate(B.tasks)])


def load(bench, split, m, B):
    f = BM.RESULTS / bench / split / f"{m}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    assert list(d.disease_id) == [t.disease_id for t in B.tasks]
    d["degbin_ap"] = degbin(bench, split, m, B)
    for c in MET:
        if c not in d:
            d[c] = np.nan
    return d


rows, cmp_rows = [], []
for (bench, split), B in SETS.items():
    D = {m: load(bench, split, m, B) for m in METH}
    D = {k: v for k, v in D.items() if v is not None}
    for a in ("stackW_W_v1clean", "stackW_W_v2"):
        if a in D and a + "_s1" in D:
            avg = D[a].copy()
            for c in MET:
                avg[c] = (D[a][c].to_numpy(float) + D[a + "_s1"][c].to_numpy(float)) / 2
            D[a + "@avg"] = avg
    for m, d in D.items():
        rows.append(dict(bench=bench, split=split, method=m, n=len(d), **{c: d[c].mean() for c in MET}))
    pairs = [(x, y) for x in ("stackW_W_v2@avg", "stackW_W_v1clean@avg", "stackW_W_v2", "stackW_W_v1clean")
             for y in ("stack_v1clean", "transfer_jaccard")]
    pairs += [("stackW_W_v2@avg", "stackP_v2"), ("stackW_W_v1clean@avg", "stackP_v1clean"),
              ("stackW_W_v2@avg", "stackW_W_v1clean@avg"), ("stackW_W_v1clean", "stackW_W_v1perm"), ("stackW_W_v2", "stackW_W_v1perm"),
              ("stackW_W_v2_s1", "stackW_W_v2"), ("stackW_W_v1clean_s1", "stackW_W_v1clean")]
    for x, y in pairs:
        if x not in D or y not in D:
            continue
        for c in MET:
            if D[x][c].notna().sum() < 3 or D[y][c].notna().sum() < 3:
                continue
            r = paired_bootstrap(D[x], D[y], c, n=10000)
            cmp_rows.append(dict(bench=bench, split=split, a=x, b=y, **r))
S = pd.DataFrame(rows); C = pd.DataFrame(cmp_rows)
S.to_csv("results/r6w/summary.tsv", sep="\t", index=False, float_format="%.4f")
C.to_csv("results/r6w/compare.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(S.round(4).to_string(index=False))
print(C[["bench", "split", "a", "b", "metric", "n_tasks", "mean_a", "mean_b", "delta", "ci_lo", "ci_hi"]].round(4).to_string(index=False))

# pre-registered decision (temporal dev and wide_dev only)
dec = []
for arm in ("stackW_W_v2", "stackW_W_v1clean"):
    ok_all, notes = True, []
    for bench, split in (("temporal", "dev"), ("temporal_wide", "wide_dev")):
        g = C[(C.bench == bench) & (C.split == split) & (C.b == "stack_v1clean")]
        ap = g[(g.a == arm + "@avg") & (g.metric == "ap")].iloc[0]
        nu = g[(g.a == arm + "@avg") & (g.metric == "nuap")].iloc[0]
        sd = C[(C.bench == bench) & (C.split == split) & (C.a == arm + "_s1") & (C.b == arm) & (C.metric == "nuap")].iloc[0]
        tol = max(0.005, abs(sd.delta))
        c1 = ap.delta > 0 and ap.ci_lo > 0
        c2 = nu.ci_hi >= 0 and nu.delta >= -tol
        ok_all &= c1 and c2
        notes.append(f"{split}: dAP {ap.delta:+.4f} [{ap.ci_lo:+.4f},{ap.ci_hi:+.4f}] {'PASS' if c1 else 'FAIL'}; "
                     f"dnuAP {nu.delta:+.4f} [{nu.ci_lo:+.4f},{nu.ci_hi:+.4f}] tol {tol:.4f} {'PASS' if c2 else 'FAIL'}")
    dec.append(dict(arm=arm, candidate=ok_all, detail=" | ".join(notes)))
Dd = pd.DataFrame(dec); Dd.to_csv("results/r6w/decision.tsv", sep="\t", index=False)
print(Dd.to_string(index=False))
