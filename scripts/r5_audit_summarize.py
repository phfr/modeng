"""R5 witness audit: summaries of the group ablations (paired cluster bootstrap, ablated - full), per disease type, and
redundancy clusters at two thresholds. Reads results/r5/ (r5_audit_ablate.py, r5_audit_shap.py, r5_audit_channels.py).
Outputs results/r5/{ablation_deltas,ablation_by_type,shap_group_by_type,redundancy_pairs}.tsv
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

OUT = "results/r5"
GROUPS = ["base", "donor", "view", "cs_donor", "dA_pt", "dc_ridge", "gwas_catlag", "priors", "ph", "pt_ridge", "view+donor", "modern"]
SEEDS = ["", "_seed1", "_seed2"]   # LightGBM seed 0 (frozen recipe), 1, 2
METRICS = ["ap", "nuap", "degbin_ap"]


def boot(d, cl, n=5000, seed=0):
    ok = np.isfinite(d); d, cl = d[ok], cl[ok]
    g = pd.DataFrame(dict(d=d, c=cl)).groupby("c")["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    b = s[idx].sum(1) / c[idx].sum(1)
    return d.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975), len(d)


def fmt(m, lo, hi):
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


ctx = pd.read_csv(f"{OUT}/task_context.tsv", sep="\t")
AREA = {"ctx_area_EFO_0000618": "nervous system", "ctx_area_EFO_0000319": "cardiovascular",
        "ctx_area_OTAR_0000018": "genetic/congenital", "ctx_area_other": "other"}
ctx["area"] = ctx[list(AREA)].idxmax(axis=1).map(AREA)
g, m = ctx.ctx_mix_gwas, ctx.ctx_mix_mend
ctx["dtype"] = np.where((g == 0) & (m == 0), "no genetic seed evidence", np.where(g > m, "GWAS-dominated", "Mendelian-dominated"))
ctx[["split", "disease_id", "name", "dtype", "area", "ctx_mix_gwas", "ctx_mix_mend", "n_targets"]].to_csv(f"{OUT}/task_types.tsv", sep="\t", index=False)
print(ctx.groupby(["split", "dtype"]).size().unstack(), "\n", ctx.groupby(["split", "area"]).size().unstack())

rows, trows = [], []
for split in ("dev", "confirm"):
    Fs = {k: pd.read_parquet(f"{OUT}/ep_full{k}_{split}.parquet") for k in SEEDS}
    F = Fs[""]
    ty = ctx[ctx.split == split].set_index("disease_id")
    cl = F.cluster.to_numpy()
    # retrain noise floor: full recipe with another LightGBM seed minus seed 0
    for k in SEEDS[1:]:
        r = dict(split=split, removed=f"(none: full{k} - full)")
        for mt in METRICS:
            d = Fs[k][mt].to_numpy(float) - F[mt].to_numpy(float)
            r[mt + "_seed0"] = fmt(*boot(d, cl)[:3])
        rows.append(r)
    for gname in GROUPS:
        fs = [f"{OUT}/ep_no_{gname}{k}_{split}.parquet" for k in SEEDS]
        if not os.path.exists(fs[0]):
            continue
        r = dict(split=split, removed=gname)
        A = [pd.read_parquet(f) if os.path.exists(f) else None for f in fs]
        for mt in METRICS:
            d0 = A[0][mt].to_numpy(float) - F[mt].to_numpy(float)
            r[mt + "_seed0"] = fmt(*boot(d0, cl)[:3])
            ds = [A[i][mt].to_numpy(float) - Fs[k][mt].to_numpy(float) for i, k in enumerate(SEEDS) if A[i] is not None]
            d = np.mean(ds, 0); r["n_seeds"] = len(ds)
            r[mt + "_3seed"] = fmt(*boot(d, cl)[:3]); r[mt + "_3seed_mean"] = np.nanmean(d)
            for col in ("dtype", "area"):
                lab = ty.loc[F.disease_id, col].to_numpy()
                for lv in np.unique(lab):
                    k_ = lab == lv
                    mm, lo, hi, n = boot(d[k_], cl[k_], 2000)
                    trows.append(dict(split=split, removed=gname, metric=mt, by=col, level=lv, n=n, delta=mm, lo=lo, hi=hi))
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/ablation_deltas.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(R[["split", "removed"] + [m + "_seed0" for m in METRICS]].to_string(index=False))
print(R[["split", "removed"] + [m + "_3seed" for m in METRICS]].dropna().to_string(index=False))
TT = pd.DataFrame(trows); TT.to_csv(f"{OUT}/ablation_by_type.tsv", sep="\t", index=False, float_format="%.4f")
print(TT[(TT.split == "confirm") & (TT.by == "dtype")].pivot_table(index=["removed"], columns=["metric", "level"], values="delta").round(4).to_string())

# full-model level per type (context for the deltas)
for split in ("dev", "confirm"):
    F = pd.read_parquet(f"{OUT}/ep_full_{split}.parquet").merge(ctx[ctx.split == split][["disease_id", "dtype", "area"]], on="disease_id")
    print(split, F.groupby("dtype")[METRICS].mean().round(4).to_string())

# SHAP group contribution (mean over tasks of mean SHAP(target) - mean SHAP(non-target)) by type
S = pd.read_csv(f"{OUT}/shap_group_by_task.tsv", sep="\t").merge(ctx[["split", "disease_id", "dtype", "area"]], on=["split", "disease_id"])
cols = [c for c in S.columns if c not in ("split", "disease_id", "dtype", "area")]
sg = pd.concat([S.groupby(["split", "dtype"])[cols].mean(), S.groupby(["split", "area"])[cols].mean().rename_axis(["split", "dtype"])])
sg.to_csv(f"{OUT}/shap_group_by_type.tsv", sep="\t", float_format="%.4f"); print(sg.round(3).to_string())

# redundancy: strongest cross-group pairs and clusters at |rho| >= 0.5
Rm = pd.read_csv(f"{OUT}/redundancy_spearman.tsv", sep="\t", index_col=0)
cl = pd.read_csv(f"{OUT}/redundancy_clusters.tsv", sep="\t").set_index("channel")
pairs = [(a, b, Rm.loc[a, b]) for i, a in enumerate(Rm.index) for b in Rm.index[i + 1:] if np.isfinite(Rm.loc[a, b])]
P = pd.DataFrame(pairs, columns=["a", "b", "rho"]).assign(ga=lambda x: x.a.map(cl.group), gb=lambda x: x.b.map(cl.group))
P = P.reindex(P.rho.abs().sort_values(ascending=False).index); P.to_csv(f"{OUT}/redundancy_pairs.tsv", sep="\t", index=False, float_format="%.3f")
print(P.head(40).to_string(index=False))
D = 1 - Rm.abs().fillna(0).to_numpy(); np.fill_diagonal(D, 0); D = (D + D.T) / 2
c5 = fcluster(linkage(squareform(D, checks=False), "average"), 0.5, "distance")
for c in np.unique(c5):
    mem = Rm.index[c5 == c]
    if len(mem) > 1:
        print("rho>=0.5 cluster:", ", ".join(mem))
