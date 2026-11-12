"""r7 / P3 analysis (pre-registration docs/preregistration_experts.md): every method evaluated from its saved score matrix with the
newdata / labelproc alt sets; seed averages; paired cluster bootstraps; pre-registered decision; disagreement list.
Outputs: results/r7/{summary,compare,decision,disagree,disagree_examples}.tsv. Never touches wide_confirm."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
assert os.environ.get("MODENG_SCORE_WIDE_CONFIRM") != "1"
import numpy as np, pandas as pd
from modeng import tasks as T, bench as BM, channels as CH, data as D, arrival_types as AT
from modeng.compare import paired_bootstrap
from modeng.methods import baselines as BL

OUT = "results/r7"
Tm, W = T.load("temporal"), T.load("temporal_wide")
SETS = {("temporal", "dev"): Tm.split("dev"), ("temporal_wide", "wide_dev"): W.split("wide_dev"),
        ("temporal", "confirm"): Tm.split("confirm")}
TYPES = {"temporal": AT.type_table(Tm), "temporal_wide": AT.type_table(W)}
SEEDED = {"W": "stackW_W_v1clean", "EB": "stackW_E_B", "EL": "stackW_E_L", "G": "stackW_G"}
MET = ["ap", "nuap", "newdata_ap", "newdata_nuap", "labelproc_ap", "labelproc_nuap", "sa_ap", "sa_nuap"]
K, POOL = 50, 500


def scores(bench, split, m, sfx):
    return np.load(BM.RESULTS / bench / split / f"{m}{sfx}.scores.npy").astype(np.float32)


rows, cmp_rows, dec, dis_rows, ex_rows = [], [], [], [], []
for (bench, split), B in SETS.items():
    per = {}
    per["transfer_jaccard"] = BM.evaluate(BL.transfer_jaccard(B.tasks, B), B, jobs=8)
    for k, m in SEEDED.items():
        a = BM.evaluate(scores(bench, split, m, ""), B, jobs=8)
        b = BM.evaluate(scores(bench, split, m, "_s1"), B, jobs=8)
        per[k + "_s0"], per[k + "_s1"] = a, b
        avg = a.copy()
        for c in MET:
            if c in a:
                avg[c] = (a[c].to_numpy(float) + b[c].to_numpy(float)) / 2
        per[k + "@avg"] = avg
    for m, d in per.items():
        rows.append(dict(bench=bench, split=split, method=m, n=len(d),
                         **{c: d[c].mean() for c in MET if c in d}, **{f"n_{c}": int(d[c].notna().sum()) for c in MET if c in d}))
    for x in ("G@avg", "EB@avg", "EL@avg", "transfer_jaccard", "G_s1"):
        y = "W@avg" if x != "G_s1" else "G_s0"
        for c in MET:
            if c in per[x] and per[x][c].notna().sum() >= 3:
                cmp_rows.append(dict(bench=bench, split=split, a=x, b=y, **paired_bootstrap(per[x], per[y], c, n=10000)))
    for c in ("ap", "nuap", "newdata_ap", "newdata_nuap"):
        cmp_rows.append(dict(bench=bench, split=split, a="W_s1", b="W_s0", **paired_bootstrap(per["W_s1"], per["W_s0"], c, n=10000)))

    # ---------------- disagreement list
    tt = TYPES[bench]; tt = tt[tt.disease_id.isin([t.disease_id for t in B.tasks])]
    by = {k: v for k, v in tt.groupby("disease_id")}
    pop = CH.compute(B, "popularity", split)
    sym = D.symbols(); U = D.universe()
    for sfx in ("", "_s1"):
        sB, sL, sW = (scores(bench, split, SEEDED[k], sfx) for k in ("EB", "EL", "W"))
        for i, t in enumerate(B.tasks):
            d = by.get(t.disease_id)
            if d is None:
                continue
            nd = set(d.g[d.type == "new"]); lp = set(d.g[d.type.isin(AT.LABELPROC)]); allt = set(t.targets.tolist())
            if not nd:
                continue
            ok = np.ones(B.n_genes, bool); ok[t.exclude] = False
            cand = np.flatnonzero(ok)
            def pr(v):
                r = np.empty(len(cand)); r[np.argsort(v[cand], kind="stable")] = np.arange(len(cand)); return r / len(cand)
            rB, rL = pr(sB[i]), pr(sL[i])
            pool = np.argsort(-sB[i][cand], kind="stable")[:POOL]
            lst = pool[np.argsort(-(rB[pool] - rL[pool]), kind="stable")[:K]]
            wtop = np.argsort(-sW[i][cand], kind="stable")[:K]
            btop = np.argsort(-sB[i][cand], kind="stable")[:K]
            isnd = np.isin(cand, list(nd)); islp = np.isin(cand, list(lp)); ist = np.isin(cand, list(allt))
            # popularity-matched random (analytic expectation): decile of popularity among candidates
            pv = np.nan_to_num(pop[i][cand], nan=-1.0)
            dec_ = np.minimum((pd.Series(pv).rank(method="first").to_numpy() - 1) * 10 // len(cand), 9).astype(int)
            pnd = np.bincount(dec_, weights=isnd, minlength=10) / np.maximum(np.bincount(dec_, minlength=10), 1)
            ptg = np.bincount(dec_, weights=ist, minlength=10) / np.maximum(np.bincount(dec_, minlength=10), 1)
            r = dict(bench=bench, split=split, seed=sfx or "_s0", disease_id=t.disease_id, cluster=t.cluster, name=t.name,
                     n_new=len(nd), n_lp=len(lp), n_t=len(allt))
            for lab, sel in (("dis", lst), ("W", wtop), ("EB", btop)):
                r[f"{lab}_new"] = isnd[sel].sum(); r[f"{lab}_lp"] = islp[sel].sum(); r[f"{lab}_all"] = ist[sel].sum()
            r["poolrand_new"] = K * isnd[pool].mean(); r["poolrand_all"] = K * ist[pool].mean()
            r["popmatch_new"] = pnd[dec_[lst]].sum(); r["popmatch_all"] = ptg[dec_[lst]].sum()
            r["dis_medpop"] = float(np.median(pv[lst])); r["W_medpop"] = float(np.median(pv[wtop]))
            dis_rows.append(r)
            if sfx == "":
                for j in lst[isnd[lst]]:
                    g = cand[j]
                    ex_rows.append(dict(bench=bench, split=split, disease_id=t.disease_id, name=t.name, gene=sym.get(U[g], U[g]),
                                        ensg=U[g], rB=round(rB[j], 4), rL=round(rL[j], 4),
                                        W_rank=int((sW[i][cand] > sW[i][g]).sum()) + 1,
                                        fy=d.set_index("g").fy_primary.get(g), ev=d.set_index("g").evidence_class.get(g)))

S = pd.DataFrame(rows); C = pd.DataFrame(cmp_rows)
S.to_csv(f"{OUT}/summary.tsv", sep="\t", index=False, float_format="%.4f")
C.to_csv(f"{OUT}/compare.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(S[["bench", "split", "method"] + MET].round(4).to_string(index=False))
print(C[["bench", "split", "a", "b", "metric", "n_tasks", "mean_a", "mean_b", "delta", "ci_lo", "ci_hi"]].round(4).to_string(index=False))

# ---------------- pre-registered decision (temporal dev and wide_dev)
ok_all = True
for bench, split in (("temporal", "dev"), ("temporal_wide", "wide_dev")):
    c = C[(C.bench == bench) & (C.split == split) & (C.a == "G@avg") & (C.b == "W@avg")].set_index("metric")
    a = c.delta["ap"] >= -0.008 and c.delta["nuap"] >= -0.008
    b = c.delta["newdata_ap"] > 0 and c.delta["newdata_nuap"] > 0
    ok_all &= a and b
    dec.append(dict(bench=bench, split=split, dAP=c.delta["ap"], dnuAP=c.delta["nuap"], d_newdata_ap=c.delta["newdata_ap"],
                    d_newdata_nuap=c.delta["newdata_nuap"], overall_ok=a, newdata_better=b))

DS = pd.DataFrame(dis_rows)
DS.to_csv(f"{OUT}/disagree_pertask.tsv", sep="\t", index=False, float_format="%.4f")
cols = [c for c in DS.columns if c.endswith(("_new", "_lp", "_all")) and c not in ("n_new",)]
avg = DS.groupby(["bench", "split", "disease_id", "cluster", "name"])[cols + ["n_new", "dis_medpop", "W_medpop"]].mean().reset_index()
drows = []
for (bench, split), g in avg.groupby(["bench", "split"]):
    n = len(g) * K
    r = dict(bench=bench, split=split, n_tasks=len(g), n_newdata_targets=g.n_new.sum())
    for c in cols:
        r[f"prec_{c}"] = g[c].sum() / n
    # cluster bootstrap of the pooled-precision difference dis - W (newdata)
    cl = g.groupby("cluster")[["dis_new", "W_new", "popmatch_new", "poolrand_new"]].sum()
    cnt = g.groupby("cluster").size().to_numpy()
    rng = np.random.default_rng(0); idx = rng.integers(0, len(cl), (10000, len(cl)))
    for other in ("W_new", "popmatch_new", "poolrand_new"):
        diff = (cl["dis_new"].to_numpy() - cl[other].to_numpy())
        bt = diff[idx].sum(1) / (cnt[idx].sum(1) * K)
        r[f"d_vs_{other}"] = diff.sum() / n; r[f"ci_lo_{other}"] = np.quantile(bt, 0.025); r[f"ci_hi_{other}"] = np.quantile(bt, 0.975)
    r["newshare_dis"] = g.dis_new.sum() / max(g.dis_all.sum(), 1e-9); r["newshare_W"] = g.W_new.sum() / max(g.W_all.sum(), 1e-9)
    r["newshare_EB"] = g.EB_new.sum() / max(g.EB_all.sum(), 1e-9)
    r["medpop_dis"] = g.dis_medpop.median(); r["medpop_W"] = g.W_medpop.median()
    drows.append(r)
DR = pd.DataFrame(drows)
DR.to_csv(f"{OUT}/disagree.tsv", sep="\t", index=False, float_format="%.4f")
print(DR.T.to_string())
prod = all(DR.set_index("split").loc[s, "d_vs_W_new"] > 0 for s in ("dev", "wide_dev"))
dec.append(dict(bench="both", split="dev+wide_dev", overall_ok=ok_all, newdata_better=ok_all, product=prod))
pd.DataFrame(dec).to_csv(f"{OUT}/decision.tsv", sep="\t", index=False, float_format="%.4f")
print(pd.DataFrame(dec).to_string(index=False))
print("seed-wise list precision:\n", DS.groupby(["split", "seed"])[["dis_new", "W_new", "EB_new", "popmatch_new", "poolrand_new"]].sum()
      .div(DS.groupby(["split", "seed"]).size() * K, axis=0).round(4).to_string())
EX = pd.DataFrame(ex_rows)
EX.to_csv(f"{OUT}/disagree_examples.tsv", sep="\t", index=False)
top = EX[EX.split != "confirm"].groupby(["split", "disease_id", "name"]).size().sort_values(ascending=False).head(3)
print(top.to_string())
for (sp, did, nm) in top.index:
    print(EX[(EX.disease_id == did)].to_string(index=False))
