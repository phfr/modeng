"""R3 red team (modeng-v1 = stack_v3ptr_p0.35): shared helpers. Nothing is written to the leaderboard or the channel cache."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
ROOT = "/home/ph/code/modeng"; os.chdir(ROOT); sys.path.insert(0, ROOT + "/src")
import numpy as np, pandas as pd
from modeng import tasks as T, metrics as M, channels as CH, stack as ST
from modeng.methods import baselines as BL
OUT = ROOT + "/results/review2/"
CHANNELS = "base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,hz_gwas,hz_mend,ph_mean,ph_top3,pt_ridge,pt_ridge_pos"
NAMES = CH.all_channel_names(tuple(CHANNELS.split(",")))
POWER = 0.35
LGB = dict(ST.LGB); LGB["num_threads"] = int(os.environ.get("R2_THREADS", "4"))
Bf = T.load("temporal"); Bd, Bc = Bf.split("dev"), Bf.split("confirm"); deg = np.asarray(Bf.degree, np.float32)


def cached(split, name):
    return np.load(f"{ROOT}/data/cache/channels/temporal_{split}_{name}.npy")


def feats(mats, B):
    ranks = [ST.pct_rank(m) for m in mats]
    ns = np.log1p(np.array([len(t.seeds) for t in B.tasks], np.float32))
    return mats, ranks, ns


def rows(B, idx, F, neg=3000, seed=0, gene_ok=None):
    """stack.training_rows with POWER and an optional gene filter (bool [n_genes]) applied after sampling."""
    mats, ranks, ns = F
    rng = np.random.default_rng(seed); Xs, ys, ws = [], [], []
    for i in idx:
        t = B.tasks[i]
        excl = np.zeros(B.n_genes, bool); excl[t.exclude] = True; excl[t.targets] = True
        cand = np.flatnonzero(~excl)
        negs = rng.choice(cand, min(neg, len(cand)), replace=False); tg = t.targets
        if gene_ok is not None:
            negs = negs[gene_ok[negs]]; tg = tg[gene_ok[tg]]
            if len(tg) == 0:
                continue
        g = np.concatenate([tg, negs])
        Xs.append(ST.task_X(i, g, mats, ranks, ns)); ys.append(np.r_[np.ones(len(tg)), np.zeros(len(negs))])
        wp = 1.0 / np.maximum(deg[tg], 1.0) ** POWER; wp = wp / wp.mean()
        ws.append(np.r_[wp, np.ones(len(negs))])
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def fit(X, y, w, rounds=400):
    import lightgbm as lgb
    return lgb.train(LGB, lgb.Dataset(X, y, weight=w), rounds)


def predict(model, B, F):
    S = np.zeros((len(B.tasks), B.n_genes), np.float32); allg = np.arange(B.n_genes)
    for i in range(len(B.tasks)):
        S[i] = model.predict(ST.task_X(i, allg, *F), num_threads=LGB["num_threads"])
    return S


def ep(S, B):
    """per task: ap, nuap, onnet ap/nuap, strict-a onnet ap/nuap."""
    out = []; off = np.flatnonzero(deg == 0)
    for i, t in enumerate(B.tasks):
        m = M.rank_metrics(S[i], t.targets, t.exclude, deg); r = dict(disease=t.disease_id, cluster=t.cluster, ap=m["ap"], nuap=m["nuap"], auroc=m["auroc"], r100=m["recall@100"])
        if "onnet" in t.alt:
            mo = M.rank_metrics(S[i], *t.alt["onnet"], deg); r["onnet_ap"], r["onnet_nuap"] = mo["ap"], mo["nuap"]
        if "sa" in t.alt:
            kept, sexc = t.alt["sa"]; ko = np.setdiff1d(kept, off)
            if len(ko):
                ms = M.rank_metrics(S[i], ko, np.union1d(sexc, np.setdiff1d(off, ko)), deg); r["sa_onnet_ap"], r["sa_onnet_nuap"] = ms["ap"], ms["nuap"]
        out.append(r)
    return pd.DataFrame(out)


def boot(a, b, cl, n=5000, seed=0):
    """paired cluster bootstrap of mean(a - b); returns delta, lo, hi."""
    a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b); d = (a - b)[ok]; cl = np.asarray(cl)[ok]
    uc, inv = np.unique(cl, return_inverse=True); s = np.bincount(inv, d); c = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed); pick = rng.integers(0, len(uc), (n, len(uc)))
    ds = s[pick].sum(1) / c[pick].sum(1)
    return d.mean(), np.percentile(ds, 2.5), np.percentile(ds, 97.5)


def fmt(x):
    return "%+.4f [%+.4f, %+.4f]" % x
