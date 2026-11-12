"""R5 witness audit, part 2a + 3: gain importance and SHAP (LightGBM pred_contrib) of the modeng-v1 dev->confirm model
(results/r5/model_full_confirm.txt, written by r5_audit_ablate.py), on a sample of task x gene rows: per task all targets
+ 500 random non-excluded candidates (seed 0). Splits: confirm (out-of-sample) and dev (in-sample for this model).
Also writes per-task disease-type context (seed GWAS / Mendelian mix from OT 21.04 evidence in the task family; modet's
primary therapeutic area) for part 3.
Outputs results/r5/{gain,shap_feature,shap_group,shap_group_by_task,task_context}.tsv
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import numpy as np, pandas as pd, lightgbm as lgb
from modeng import stack as ST, stack_gate as SG
from r5_audit_ablate import NAMES, group_of, Bf, OUT

FEATS = NAMES + ["r_" + n for n in NAMES] + ["log_nseeds"]
FGROUP = [group_of(n) for n in NAMES] * 2 + ["log_nseeds"]
model = lgb.Booster(model_file=f"{OUT}/model_full_confirm.txt")
pd.Series(model.feature_importance("gain"), index=FEATS).sort_values(ascending=False).to_csv(f"{OUT}/gain.tsv", sep="\t", header=["gain"])

feat_rows, grp_rows, task_rows, ctx_rows = [], [], [], []
for split in ("dev", "confirm"):
    B = Bf.split(split)
    m, r, ns = ST.build_feats(B, split, NAMES)
    rng = np.random.default_rng(0)
    X_all, tid, ytrue = [], [], []
    for i, t in enumerate(B.tasks):
        cand = np.setdiff1d(np.arange(B.n_genes), np.union1d(t.exclude, t.targets))
        g = np.concatenate([t.targets, rng.choice(cand, 500, replace=False)])
        X_all.append(ST.task_X(i, g, m, r, ns)); tid += [i] * len(g); ytrue += [1] * len(t.targets) + [0] * 500
    X = np.vstack(X_all); tid = np.array(tid); ytrue = np.array(ytrue)
    C = model.predict(X, pred_contrib=True)[:, :-1]          # last column = expected value
    absC = np.abs(C)
    for j, f in enumerate(FEATS):
        feat_rows.append(dict(split=split, feature=f, group=FGROUP[j], mean_abs_shap=absC[:, j].mean(),
                              mean_abs_shap_pos=absC[ytrue == 1, j].mean(), mean_shap_pos=C[ytrue == 1, j].mean()))
    G = pd.DataFrame(C, columns=FEATS).T.groupby(np.array(FGROUP)).sum().T      # additive group SHAP per row
    for gname in G.columns:
        v = G[gname].to_numpy()
        grp_rows.append(dict(split=split, group=gname, mean_abs_shap=np.abs(v).mean(), mean_abs_shap_pos=np.abs(v[ytrue == 1]).mean(),
                             mean_shap_pos_minus_neg=v[ytrue == 1].mean() - v[ytrue == 0].mean()))
    Gt = G.assign(task=tid, y=ytrue)
    for i, sub in Gt.groupby("task"):
        d = {g: sub.loc[sub.y == 1, g].mean() - sub.loc[sub.y == 0, g].mean() for g in G.columns}
        task_rows.append(dict(split=split, disease_id=B.tasks[i].disease_id, **d))
    X_ctx, cn = SG.context(B, split, area=True)
    c = pd.DataFrame(X_ctx, columns=cn); c.insert(0, "disease_id", [t.disease_id for t in B.tasks]); c.insert(0, "split", split)
    c["name"] = [t.name for t in B.tasks]; c["n_targets"] = [len(t.targets) for t in B.tasks]
    ctx_rows.append(c)
    print(split, X.shape, flush=True)

pd.DataFrame(feat_rows).to_csv(f"{OUT}/shap_feature.tsv", sep="\t", index=False, float_format="%.5f")
gr = pd.DataFrame(grp_rows); gr.to_csv(f"{OUT}/shap_group.tsv", sep="\t", index=False, float_format="%.5f")
pd.DataFrame(task_rows).to_csv(f"{OUT}/shap_group_by_task.tsv", sep="\t", index=False, float_format="%.5f")
pd.concat(ctx_rows).to_csv(f"{OUT}/task_context.tsv", sep="\t", index=False, float_format="%.4f")
print(gr.round(4).to_string(index=False))
