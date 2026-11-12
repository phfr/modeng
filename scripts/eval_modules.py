"""Evaluate the exported modules (scripts/export_modules.py) and their leiden_anno submodules (scripts/decompose_modules.py) on
the temporal confirm targets (OT 25.09 additions; used here only, never as input).

1. module level: precision of the added genes, recall of the future targets, per method (modeng-v1 calibrated, transfer_jaccard and
   modet headline at the same sizes), paired differences with a cluster bootstrap (cluster = Task.cluster).
2. submodules: how the hits among added genes distribute over submodules with / without seeds and by the seed fraction of the submodule;
   whether targets not called by the module sit near seed submodules is NOT asked (submodules only cover module genes).
3. held-out coherence: DepMap 20Q4 co-essentiality (released 2020-11, <= T0; not used by leiden_anno, which clusters on GO / Reactome):
   mean correlation of same-submodule pairs minus different-submodule pairs, z vs 200 label permutations (subex Views loader).
Writes results/modules/r4_*.tsv. Usage: python scripts/eval_modules.py
"""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "/home/ph/code/modeng/src")
from modeng import tasks as T, data as D

ROOT = Path("/home/ph/code/modeng/data/modules")
RES = Path("/home/ph/code/modeng/results/modules"); RES.mkdir(parents=True, exist_ok=True)
METHODS = ("modeng", "transfer", "headline")
TAUS = (0.2, 0.3)
B = T.load("temporal").split("confirm")
U = D.universe()
TGT = {t.disease_id: set(U[t.targets]) for t in B.tasks}
CL = {t.disease_id: t.cluster for t in B.tasks}


def boot(d, cl, n=3000, seed=0):
    rng = np.random.default_rng(seed); uc = np.unique(cl); groups = {u: np.flatnonzero(cl == u) for u in uc}; out = []
    for _ in range(n):
        idx = np.concatenate([groups[u] for u in rng.choice(uc, len(uc))]); out.append(d[idx].sum())
    return np.percentile(out, [2.5, 97.5])


def load_sub(m, did, tau):
    f = ROOT / m / did / f"sub_tau{tau}.tsv"
    if not f.exists():
        return None, None
    z = np.load(f.with_suffix(".npz"), allow_pickle=False); meta = json.loads(str(z["__meta__"]))
    names = {s["id"]: s["name"] for s in meta["submodules"]}
    return pd.read_csv(f, sep="\t"), names


# ---------------------------------------------------------------------------------------------------------------- 1. module level
rows = []
for m in METHODS:
    for tau in TAUS:
        for t in B.tasks:
            f = ROOT / m / t.disease_id / f"module_tau{tau}.tsv"
            if not f.exists():
                continue
            mod = pd.read_csv(f, sep="\t"); add = mod[mod.role == "added"]
            hit = add.gene_id.isin(TGT[t.disease_id])
            rows.append(dict(method=m, tau=tau, disease_id=t.disease_id, cluster=t.cluster, n_seeds=int((mod.role == "seed").sum()),
                             n_added=len(add), hits=int(hit.sum()), n_targets=len(TGT[t.disease_id]),
                             n_added_off_graph=int((~add.on_graph).sum()), hits_off_graph=int((hit & ~add.on_graph).sum()),
                             pred_p_sum=float(add.calib_p.sum()) if m == "modeng" else np.nan))
M = pd.DataFrame(rows); M.to_csv(RES / "r4_module_level.tsv", sep="\t", index=False)
summ = []
for (m, tau), g in M.groupby(["method", "tau"]):
    has = g[g.n_added > 0]
    summ.append(dict(method=m, tau=tau, diseases=len(g), with_added=len(has), median_added=g.n_added.median(), added=g.n_added.sum(),
                     hits=g.hits.sum(), pooled_precision=g.hits.sum() / g.n_added.sum(), mean_precision=(has.hits / has.n_added).mean(),
                     pooled_recall=g.hits.sum() / g.n_targets.sum(), mean_recall=(g.hits / g.n_targets).mean(),
                     predicted_precision=g.pred_p_sum.sum() / g.n_added.sum() if m == "modeng" else np.nan,
                     frac_added_off_graph=g.n_added_off_graph.sum() / g.n_added.sum(), hits_off_graph=g.hits_off_graph.sum()))
S1 = pd.DataFrame(summ); S1.to_csv(RES / "r4_module_summary.tsv", sep="\t", index=False, float_format="%.4f")
print(S1.round(3).to_string(index=False))
pair = []
for tau in TAUS:
    a = M[(M.method == "modeng") & (M.tau == tau)].set_index("disease_id")
    for ref in ("transfer", "headline"):
        b = M[(M.method == ref) & (M.tau == tau)].set_index("disease_id")
        ix = a.index.intersection(b.index); d = (a.loc[ix, "hits"] - b.loc[ix, "hits"]).to_numpy(); cl = a.loc[ix, "cluster"].to_numpy()
        lo, hi = boot(d, cl)
        pair.append(dict(tau=tau, ref=ref, n=len(ix), hits_modeng=int(a.loc[ix, "hits"].sum()), hits_ref=int(b.loc[ix, "hits"].sum()),
                         delta_hits=int(d.sum()), ci_lo=lo, ci_hi=hi, wins=int((d > 0).sum()), losses=int((d < 0).sum())))
P = pd.DataFrame(pair); P.to_csv(RES / "r4_module_paired.tsv", sep="\t", index=False, float_format="%.1f"); print(P.to_string(index=False))

# ---------------------------------------------------------------------------------------------------------------- 2. submodules
srows, drows = [], []
for m in METHODS:
    for tau in TAUS:
        for t in B.tasks:
            sub, names = load_sub(m, t.disease_id, tau)
            if sub is None:
                continue
            mod = pd.read_csv(ROOT / m / t.disease_id / f"module_tau{tau}.tsv", sep="\t").set_index("gene_id")
            sub = sub[sub.submodule >= 0].copy()
            sub["role"] = mod.role.reindex(sub.gene_id).to_numpy()
            sub["hit"] = sub.gene_id.isin(TGT[t.disease_id]) & (sub.role == "added")
            k = sub.submodule.nunique()
            for sid, g in sub.groupby("submodule"):
                ns, na = int((g.role == "seed").sum()), int((g.role == "added").sum())
                srows.append(dict(method=m, tau=tau, disease_id=t.disease_id, k=k, submodule=sid, name=names.get(sid, ""), size=len(g),
                                  n_seeds=ns, n_added=na, hits=int(g.hit.sum()), seed_frac=ns / len(g)))
            drows.append(dict(method=m, tau=tau, disease_id=t.disease_id, k=k))
SM = pd.DataFrame(srows); SM.to_csv(RES / "r4_submodules.tsv", sep="\t", index=False, float_format="%.3f")
out = []
for (m, tau), g in SM.groupby(["method", "tau"]):
    ws, wo = g[g.n_seeds > 0], g[g.n_seeds == 0]
    ga = g[g.n_added > 0].copy()
    ga["sf_bin"] = pd.cut(ga.seed_frac, [-0.01, 0.0, 0.5, 0.8, 1.0], labels=["0", "(0,0.5]", "(0.5,0.8]", "(0.8,1)"])
    r = dict(method=m, tau=tau, decomposed=g.disease_id.nunique(), median_k=g.groupby("disease_id").k.first().median(),
             submodules=len(g), seedfree_submodules=len(wo), added_in_seedfree=int(wo.n_added.sum()), hits_in_seedfree=int(wo.hits.sum()),
             prec_seedfree=wo.hits.sum() / max(wo.n_added.sum(), 1), added_in_seeded=int(ws.n_added.sum()), hits_in_seeded=int(ws.hits.sum()),
             prec_seeded=ws.hits.sum() / max(ws.n_added.sum(), 1))
    for b, gb in ga.groupby("sf_bin", observed=False):
        r[f"prec_sf{b}"] = gb.hits.sum() / gb.n_added.sum() if gb.n_added.sum() else np.nan
        r[f"added_sf{b}"] = int(gb.n_added.sum())
    # within-disease concentration: share of a disease's hits in the submodule with most seeds vs that submodule's share of added genes
    top = g.sort_values(["disease_id", "n_seeds", "size"], ascending=[True, False, False]).groupby("disease_id").head(1)
    tot = g.groupby("disease_id")[["n_added", "hits"]].sum()
    tt = top.set_index("disease_id")[["n_added", "hits"]].join(tot, rsuffix="_all"); tt = tt[tt.hits_all > 0]
    r["topseed_sub_share_added"] = (tt.n_added / tt.n_added_all).mean(); r["topseed_sub_share_hits"] = (tt.hits / tt.hits_all).mean()
    r["n_diseases_with_hits"] = len(tt)
    out.append(r)
S2 = pd.DataFrame(out); S2.to_csv(RES / "r4_submodule_summary.tsv", sep="\t", index=False, float_format="%.4f")
print(S2.round(3).T.to_string())

# 2b. within-disease test (removes between-disease differences in precision): statistic = mean seed fraction of the submodules of the hits
# among added genes; null = hit labels permuted among the added (on-graph, assigned) genes of the same disease, 2000 draws
rng2 = np.random.default_rng(1); wrows = []
for (m, tau), g in SM[SM.n_added > 0].groupby(["method", "tau"]):
    hs = {did: (np.repeat(gd.seed_frac.to_numpy(), gd.n_added.to_numpy()), np.concatenate([np.r_[np.ones(k), np.zeros(n - k)] for k, n in zip(gd.hits, gd.n_added)]))
          for did, gd in g.groupby("disease_id")}
    hs = {d: v for d, v in hs.items() if 0 < v[1].sum() < len(v[1]) and len(np.unique(v[0])) > 1}
    obs = np.mean(np.concatenate([sf[y > 0] for sf, y in hs.values()]))
    null = np.array([np.mean(np.concatenate([sf[rng2.permutation(y) > 0] for sf, y in hs.values()])) for _ in range(2000)])
    wrows.append(dict(method=m, tau=tau, n_diseases=len(hs), mean_seedfrac_hits=obs, null_mean=null.mean(), z=(obs - null.mean()) / null.std(),
                      p_one_sided=(1 + (null >= obs).sum()) / 2001))
W = pd.DataFrame(wrows); W.to_csv(RES / "r4_submodule_within.tsv", sep="\t", index=False, float_format="%.4f"); print(W.round(4).to_string(index=False))

# ---------------------------------------------------------------------------------------------------------------- 3. held-out coherence
sys.path.insert(0, "/home/ph/code/subex/src")
from subex.bench.decomp_eval import Views
V = Views(); rng = np.random.default_rng(0); crow = []
for m in METHODS:
    for tau in TAUS:
        for t in B.tasks:
            sub, _ = load_sub(m, t.disease_id, tau)
            if sub is None:
                continue
            sub = sub[sub.submodule >= 0]
            C, cov = V.profile_view("depmap20q4", sub.gene_id.to_numpy(str))
            lab = sub.submodule.to_numpy()[cov]; C = C[np.ix_(cov, cov)]
            if len(lab) < 6 or len(np.unique(lab)) < 2:
                continue
            iu = np.triu_indices(len(lab), 1); v = C[iu]

            def stat(l):
                same = (l[:, None] == l[None, :])[iu]
                return v[same].mean() - v[~same].mean() if same.any() and (~same).any() else np.nan
            obs = stat(lab); null = np.array([stat(rng.permutation(lab)) for _ in range(200)])
            crow.append(dict(method=m, tau=tau, disease_id=t.disease_id, n_cov=len(lab), k=len(np.unique(lab)), diff=obs,
                             z=(obs - np.nanmean(null)) / (np.nanstd(null) + 1e-9), p=(1 + (null >= obs).sum()) / 201))
CO = pd.DataFrame(crow); CO.to_csv(RES / "r4_coherence.tsv", sep="\t", index=False, float_format="%.4f")
S3 = CO.groupby(["method", "tau"]).agg(n=("z", "size"), mean_diff=("diff", "mean"), median_z=("z", "median"), frac_p05=("p", lambda x: (x <= 0.05).mean()))
S3.to_csv(RES / "r4_coherence_summary.tsv", sep="\t", float_format="%.4f"); print(S3.round(3).to_string())
