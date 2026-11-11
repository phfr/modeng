"""Learned cross-disease stacker over feature channels (`modeng.channels`).

Training: rows = (task, gene) pairs, all targets + `neg` sampled non-target candidates per task; features = raw channel
values + within-task percentile ranks + log(n_seeds). LightGBM binary. Evaluation on a split:
  * dev: out-of-fold over GroupKFold(5) by task cluster (each task scored by a model that never saw it or its cluster)
  * confirm: model trained on all dev tasks.
Usage: python -m modeng.stack --bench temporal --channels base,donor,view --tag full
"""
from __future__ import annotations

import argparse, time
import numpy as np, pandas as pd

from . import tasks as T, bench as BM, channels as CH

LGB = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.7,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=16, seed=0)


def pct_rank(S):
    out = np.full_like(S, np.nan)
    for i in range(S.shape[0]):
        v = S[i]; ok = ~np.isnan(v)
        if ok.sum() < 2:
            continue
        r = v[ok].argsort().argsort().astype(np.float32)
        out[i, ok] = r / ok.sum()
    return out


def build_feats(B, split_tag, names):
    mats = [CH.compute(B, n, split_tag) for n in names]
    ranks = [pct_rank(m) for m in mats]
    ns = np.log1p(np.array([len(t.seeds) for t in B.tasks], np.float32))
    return mats, ranks, ns


def task_X(i, genes, mats, ranks, ns):
    cols = [m[i, genes] for m in mats] + [r[i, genes] for r in ranks] + [np.full(len(genes), ns[i], np.float32)]
    return np.stack(cols, 1)


POS_DEG_POWER = 0.5   # positives weighted deg^-POWER (nuAP-aligned, as modet's bcv); 0 = unweighted


def training_rows(B, idx, mats, ranks, ns, neg=3000, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys, ws = [], [], []
    deg = np.asarray(B.degree, np.float32)
    for i in idx:
        t = B.tasks[i]
        excl = np.zeros(B.n_genes, bool); excl[t.exclude] = True; excl[t.targets] = True
        cand = np.flatnonzero(~excl)
        negs = rng.choice(cand, min(neg, len(cand)), replace=False)
        g = np.concatenate([t.targets, negs])
        Xs.append(task_X(i, g, mats, ranks, ns)); ys.append(np.r_[np.ones(len(t.targets)), np.zeros(len(negs))])
        wp = 1.0 / np.maximum(deg[t.targets], 1.0) ** POS_DEG_POWER
        wp = wp / wp.mean() * (len(negs) / max(len(t.targets), 1)) ** 0.0
        ws.append(np.r_[wp, np.ones(len(negs))])
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def fit(X, y, w=None, rounds=400):
    import lightgbm as lgb
    return lgb.train(LGB, lgb.Dataset(X, y, weight=w), rounds)


def predict_all(model, B, idx, mats, ranks, ns, out):
    allg = np.arange(B.n_genes)
    for i in idx:
        out[i] = model.predict(task_X(i, allg, mats, ranks, ns))


def run(bench, channels, tag, rounds=400, neg=3000, folds=5, write=True):
    t0 = time.time()
    names = CH.all_channel_names(tuple(channels.split(",")))
    method = f"stack_{tag}"
    Bfull = T.load(bench)
    Bd, Bc = Bfull.split("dev"), Bfull.split("confirm")
    md, rd, nd = build_feats(Bd, "dev", names)
    print(f"[{bench}] {len(names)} channels built ({time.time()-t0:.0f}s)", flush=True)
    # dev OOF
    clusters = np.array([t.cluster or t.disease_id for t in Bd.tasks])
    uc = np.unique(clusters); rng = np.random.default_rng(0); rng.shuffle(uc)
    fold_of = {c: k % folds for k, c in enumerate(uc)}
    fid = np.array([fold_of[c] for c in clusters])
    S = np.zeros((len(Bd.tasks), Bd.n_genes), np.float32)
    for k in range(folds):
        tr, te = np.flatnonzero(fid != k), np.flatnonzero(fid == k)
        X, y, w = training_rows(Bd, tr, md, rd, nd, neg)
        m = fit(X, y, w, rounds)
        predict_all(m, Bd, te, md, rd, nd, S)
    res = {}
    res["dev"] = _record(S, Bd, bench, "dev", method, t0, write)
    # confirm: train on all dev
    X, y, w = training_rows(Bd, np.arange(len(Bd.tasks)), md, rd, nd, neg)
    m = fit(X, y, w, rounds)
    mc, rc, nc = build_feats(Bc, "confirm", names)
    Sc = np.zeros((len(Bc.tasks), Bc.n_genes), np.float32)
    predict_all(m, Bc, np.arange(len(Bc.tasks)), mc, rc, nc, Sc)
    res["confirm"] = _record(Sc, Bc, bench, "confirm", method, t0, write)
    imp = pd.Series(m.feature_importance("gain"), index=names + ["r_" + n for n in names] + ["log_nseeds"]).sort_values(ascending=False)
    return res, imp


def _record(S, B, bench, split, method, t0, write):
    df = BM.evaluate(S, B, jobs=8)
    if write:
        out = BM.RESULTS / bench / split / f"{method}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        row = dict(bench=bench, split=split, method=method, n=int(df["ap"].notna().sum()), ap=df["ap"].mean(),
                   nuap=df["nuap"].mean(), auroc=df["auroc"].mean(), r100=df["recall@100"].mean(),
                   onnet_ap=df["onnet_ap"].mean(), onnet_nuap=df["onnet_nuap"].mean(),
                   sa_ap=df["sa_ap"].mean() if "sa_ap" in df else np.nan, seconds=round(time.time() - t0, 1),
                   timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), git=BM.git_sha())
        pd.DataFrame([row]).to_csv(BM.LEADERBOARD, sep="\t", mode="a", header=False, index=False, float_format="%.4f")
    print(f"{bench:9s} {split:8s} {method:22s} AP {df.ap.mean():.4f} nuAP {df.nuap.mean():.4f} AUROC {df.auroc.mean():.4f} "
          f"R@100 {df['recall@100'].mean():.4f} | onnet AP {df.onnet_ap.mean():.4f} nuAP {df.onnet_nuap.mean():.4f}", flush=True)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="temporal"); ap.add_argument("--channels", default="base")
    ap.add_argument("--tag", default=None); ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--pos-deg-power", type=float, default=0.5)
    a = ap.parse_args()
    POS_DEG_POWER = a.pos_deg_power
    res, imp = run(a.bench, a.channels, a.tag or a.channels.replace(",", "_"), a.rounds)
    print(imp.head(25).to_string())
