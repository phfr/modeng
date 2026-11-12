"""ML-4 power forecaster: leave-trait-out validation on study pairs dated <= cutoff (default temporal 2021-04-30).

GroupKFold(5) over small studies; held-out pairs are scored on ALL genes (no negative subsampling). Compared rankers:
  model (LightGBM, all features), z (small-study MAGMA z), model_noz_nb (no neighbour features) as an ablation.
Reports per pair AUC / AP over all genes and over 'crossers' (genes with small-study z < 4.56, i.e. not yet significant), and
pooled means over pairs (each small study weighted equally). Writes results/powerfc/validate_<cutoff>.tsv.
Usage: python scripts/powerfc_validate.py [cutoff]
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, "/home/ph/code/modeng/src")
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold
from modeng.ext import powerfc as PF, gwas_d as G

cut = sys.argv[1] if len(sys.argv) > 1 else "2021-04-30"
out = "/home/ph/code/modeng/results/powerfc"; os.makedirs(out, exist_ok=True)
P = PF.pairs(cut); hits = PF._hits()
print(f"pairs {len(P)}, small studies {P.si.nunique()}, median ratio {np.exp(P.lratio.median()):.2f}, median hits {P.nhit.median():.0f}", flush=True)
X, y, w, g, pk = PF.training_table(cut)
print(f"training rows {len(y)}, positives {y.sum()}", flush=True)
nb_cols = [PF.FEATS.index(c) for c in ("nbmax", "nbcnt")]
rows = []
for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
    test_si = np.unique(g[te])
    mods = {"model": lgb.train(PF.PARAMS, lgb.Dataset(X[tr], y[tr], weight=w[tr]), PF.ROUNDS)}
    keep = [i for i in range(X.shape[1]) if i not in nb_cols]
    mods["model_nonb"] = lgb.train(PF.PARAMS, lgb.Dataset(X[tr][:, keep], y[tr], weight=w[tr]), PF.ROUNDS)
    for r in P[P.si.isin(test_si)].itertuples():
        F = PF.study_feats(r.si, cut).copy(); F[:, -1] = r.lratio
        ok = ~np.isnan(F[:, 0]); yy = np.zeros(len(F), int); yy[hits[r.large]] = 1
        sc = {"z": F[:, 0], "model": mods["model"].predict(F), "model_nonb": mods["model_nonb"].predict(F[:, keep])}
        cross = ok & (F[:, 0] < G.Z_THR)
        for name, s in sc.items():
            for sub, m in (("all", ok), ("cross", cross)):
                if yy[m].sum() == 0 or yy[m].sum() == m.sum():
                    continue
                rows.append(dict(fold=fold, small=r.small, large=r.large, lratio=r.lratio, ranker=name, subset=sub, npos=int(yy[m].sum()),
                                 auc=roc_auc_score(yy[m], s[m]), ap=average_precision_score(yy[m], s[m]),
                                 prev=yy[m].mean()))
R = pd.DataFrame(rows); R.to_csv(f"{out}/validate_{cut}.tsv", sep="\t", index=False, float_format="%.4f")
# equal weight per small study
S = R.groupby(["subset", "ranker", "small"])[["auc", "ap", "prev"]].mean().reset_index()
print(S.groupby(["subset", "ranker"])[["auc", "ap", "prev"]].mean().round(4).to_string())
D = S.pivot_table(index=["subset", "small"], columns="ranker", values=["auc", "ap"])
for sub in ("all", "cross"):
    d = D.loc[sub]
    for met in ("auc", "ap"):
        diff = d[(met, "model")] - d[(met, "z")]
        bs = [diff.sample(len(diff), replace=True, random_state=i).mean() for i in range(2000)]
        print(f"{sub} {met}: model - z = {diff.mean():+.4f} [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}] "
              f"(model better in {(diff > 0).sum()}/{len(diff)} small studies)")
m = PF.model(cut)
imp = pd.Series(m.feature_importance("gain"), index=PF.FEATS).sort_values(ascending=False)
print("gain importance (full model):\n", (imp / imp.sum()).round(3).to_string())
