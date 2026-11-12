"""Calibrated disease-module calls from stacker scores (E-B9-lite): isotonic map score -> P(gene is a future disease gene),
fitted on dev OOF scores only, applied to confirm. Module(d, tau) = {candidate genes with calibrated P >= tau} plus the seeds.
Reports, per tau: predicted vs realised precision on confirm, module sizes, recall of targets.
Usage: python scripts/module_cutoff.py temporal stack_v3cv_s
"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/ph/code/modeng/src")
from sklearn.isotonic import IsotonicRegression
from modeng import tasks as T, bench as BM

bench, method = sys.argv[1], sys.argv[2]
Bf = T.load(bench)


def pairs(split):
    B = Bf.split(split)
    S = np.load(BM.RESULTS / bench / split / f"{method}.scores.npy").astype(np.float32)
    xs, ys, tid = [], [], []
    for i, t in enumerate(B.tasks):
        ok = np.ones(B.n_genes, bool); ok[t.exclude] = False
        g = np.flatnonzero(ok)
        y = np.zeros(B.n_genes, bool); y[t.targets] = True
        xs.append(S[i, g]); ys.append(y[g]); tid.append(np.full(len(g), i))
    return B, S, np.concatenate(xs), np.concatenate(ys), np.concatenate(tid)


Bd, Sd, xd, yd, _ = pairs("dev")
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(xd, yd)
Bc, Sc, xc, yc, tc = pairs("confirm")
pc = iso.predict(xc)
rows = []
for tau in (0.5, 0.3, 0.2, 0.1, 0.05):
    sel = pc >= tau
    sizes = np.bincount(tc[sel], minlength=len(Bc.tasks))
    hits = np.bincount(tc[sel & yc], minlength=len(Bc.tasks))
    ntg = np.array([len(t.targets) for t in Bc.tasks])
    rows.append(dict(tau=tau, predicted_precision=round(float(pc[sel].mean()) if sel.any() else np.nan, 3),
                     realised_precision=round(float(yc[sel].mean()) if sel.any() else np.nan, 3),
                     n_called=int(sel.sum()), median_module_extra=int(np.median(sizes)), diseases_with_call=int((sizes > 0).sum()),
                     pooled_recall=round(float(hits.sum() / ntg.sum()), 3), expected_hits=round(float(pc[sel].sum()), 1), realised_hits=int(hits.sum())))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
out = BM.RESULTS / bench / "confirm" / f"{method}.module_calibration.tsv"
df.to_csv(out, sep="\t", index=False)
# calibration curve by deciles of predicted probability (confirm)
q = pd.qcut(pc[pc > 0.01], 8, duplicates="drop")
cal = pd.DataFrame({"pred": pc[pc > 0.01], "y": yc[pc > 0.01], "bin": q}).groupby("bin", observed=True).agg(pred=("pred", "mean"), real=("y", "mean"), n=("y", "size"))
print(cal.round(3).to_string())

# dev-only shrink estimate (no confirm information): cross-fit the isotonic map over 5 groups of dev diseases (by cluster)
_, _, xd2, yd2, td2 = pairs("dev")
cl = np.array([t.cluster for t in Bd.tasks]); uc = np.unique(cl); rng = np.random.default_rng(0); rng.shuffle(uc)
fold = {c: k % 5 for k, c in enumerate(uc)}; fd = np.array([fold[cl[i]] for i in td2])
pp = np.zeros_like(xd2)
for k in range(5):
    m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(xd2[fd != k], yd2[fd != k]); pp[fd == k] = m.predict(xd2[fd == k])
for tau in (0.3, 0.2, 0.1):
    sel = pp >= tau
    print(f"[dev cross-fit] tau {tau}: predicted {pp[sel].mean():.3f} realised {yd2[sel].mean():.3f} -> shrink {yd2[sel].mean()/pp[sel].mean():.2f}")
