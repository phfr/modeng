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
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=8, seed=0)


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


def training_rows(B, idx, mats, ranks, ns, neg=3000, seed=0, max_pos=None, drop=None):
    """drop (optional): task index -> genes removed from that task's training rows (neither positive nor negative)."""
    rng = np.random.default_rng(seed)
    Xs, ys, ws = [], [], []
    deg = np.asarray(B.degree, np.float32)
    for i in idx:
        t = B.tasks[i]
        excl = np.zeros(B.n_genes, bool); excl[t.exclude] = True; excl[t.targets] = True
        dg = drop.get(i) if drop else None
        if dg is not None and len(dg):
            excl[dg] = True
        cand = np.flatnonzero(~excl)
        negs = rng.choice(cand, min(neg, len(cand)), replace=False)
        tgt = t.targets if dg is None or not len(dg) else np.setdiff1d(t.targets, dg)
        if len(tgt) == 0:
            continue
        tg = tgt if (max_pos is None or len(tgt) <= max_pos) else rng.choice(tgt, max_pos, replace=False)
        g = np.concatenate([tg, negs])
        Xs.append(task_X(i, g, mats, ranks, ns)); ys.append(np.r_[np.ones(len(tg)), np.zeros(len(negs))])
        wp = 1.0 / np.maximum(deg[tg], 1.0) ** POS_DEG_POWER
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


def run_past(bench, channels, tag, rounds=600, neg=1500, max_pos=60, n_train=900, write=True):
    """Train on `past` episodes (19.04 -> 21.04, <= T0), evaluate on temporal dev and confirm (no CV needed)."""
    assert bench == "temporal"
    t0 = time.time()
    names = CH.all_channel_names(tuple(channels.split(",")))
    method = f"stackP_{tag}"
    P = T.load("past")
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(P.tasks), min(n_train, len(P.tasks)), replace=False))
    Pt = T.Benchmark(P.name, [P.tasks[i] for i in sel], P.context, P.context_ids, P.network, P.union_network, P.leakage_class, P.degree)
    mp_, rp_, np_ = build_feats(Pt, f"train{n_train}", names)
    X, y, w = training_rows(Pt, np.arange(len(Pt.tasks)), mp_, rp_, np_, neg, max_pos=max_pos)
    print(f"[past] {len(names)} channels, {len(y)} rows, {int(y.sum())} pos ({time.time()-t0:.0f}s)", flush=True)
    m = fit(X, y, w, rounds)
    Bfull = T.load(bench)
    res = {}
    for split in ("dev", "confirm"):
        Bs = Bfull.split(split)
        ms, rs, ns_ = build_feats(Bs, split, names)
        S = np.zeros((len(Bs.tasks), Bs.n_genes), np.float32)
        predict_all(m, Bs, np.arange(len(Bs.tasks)), ms, rs, ns_, S)
        res[split] = _record(S, Bs, bench, split, method, t0, write)
    imp = pd.Series(m.feature_importance("gain"), index=names + ["r_" + n for n in names] + ["log_nseeds"]).sort_values(ascending=False)
    return res, imp


def relative_target_guard(Wt, evals):
    """For each wide_train task: the targets of every evaluation task (list of Benchmarks) whose disease is the train
    disease's ontology relative (ancestor / descendant, any release) or the same id. Training on an ancestor's 25.09
    labels would otherwise teach gene identity (via gene-level features such as popularity / degree) of the evaluation
    targets in the same time window. Returns ({train index: genes}, n_positive_pairs_dropped)."""
    from . import data as D
    by_id = {}
    for E in evals:
        for t in E.tasks:
            by_id.setdefault(t.disease_id, []).append(t.targets)
    drop, n = {}, 0
    for i, t in enumerate(Wt.tasks):
        rel = (set(D.relatives(t.disease_id)) | {t.disease_id}) & by_id.keys()
        if rel:
            g = np.unique(np.concatenate([x for d in rel for x in by_id[d]]))
            drop[i] = g
            n += len(np.intersect1d(g, t.targets))
    return drop, n


def run_wide(channels, tag, rounds=600, neg=1500, max_pos=60, score_confirm=False, guard=True, write=True):
    """Train on temporal_wide `wide_train` (same window 21.04 -> 25.09), evaluate on temporal dev / confirm and wide_dev
    (wide_confirm only with score_confirm=True AND MODENG_SCORE_WIDE_CONFIRM=1: the frozen fold, pre-registered use only).
    guard: drop, per train task, the targets of evaluation tasks that are its ontology relatives (relative_target_guard)."""
    import os
    t0 = time.time()
    names = CH.all_channel_names(tuple(channels.split(",")))
    method = f"stackW_{tag}"
    W = T.load("temporal_wide")
    Wt = W.split("wide_train")
    Tm = T.load("temporal")
    evals = [("temporal", Tm.split("dev")), ("temporal", Tm.split("confirm")), ("temporal_wide", W.split("wide_dev"))]
    if score_confirm:
        if os.environ.get(T.WIDE_CONFIRM_FLAG) != "1":
            raise PermissionError(f"--score-confirm needs {T.WIDE_CONFIRM_FLAG}=1 (pre-registered scoring only)")
        evals.append(("temporal_wide", W.split("wide_confirm")))
    drop, ndrop = relative_target_guard(Wt, [E for _, E in evals]) if guard else ({}, 0)
    mw, rw, nw = build_feats(Wt, "wide_train", names)
    X, y, w = training_rows(Wt, np.arange(len(Wt.tasks)), mw, rw, nw, neg, max_pos=max_pos, drop=drop)
    print(f"[wide_train] {len(Wt.tasks)} tasks, {len(names)} channels, {len(y)} rows, {int(y.sum())} pos; guard dropped "
          f"{ndrop} positive pairs in {len(drop)} tasks ({time.time()-t0:.0f}s)", flush=True)
    m = fit(X, y, w, rounds)
    res = {}
    for bench, Bs in evals:
        split = Bs.tasks[0].split
        ms, rs, ns_ = build_feats(Bs, split, names)
        S = np.zeros((len(Bs.tasks), Bs.n_genes), np.float32)
        predict_all(m, Bs, np.arange(len(Bs.tasks)), ms, rs, ns_, S)
        res[(bench, split)] = _record(S, Bs, bench, split, method, t0, write)
    imp = pd.Series(m.feature_importance("gain"), index=names + ["r_" + n for n in names] + ["log_nseeds"]).sort_values(ascending=False)
    return res, imp


def _record(S, B, bench, split, method, t0, write):
    import os
    if os.environ.get("MODENG_SAVE_SCORES"):
        out = BM.RESULTS / bench / split / f"{method}.scores.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, S.astype(np.float16))
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
    ap.add_argument("--train", default="cv", help="cv (grouped CV on dev) | past (train on 19.04->21.04 episodes; temporal only) "
                    "| wide (train on temporal_wide wide_train; evaluate temporal dev/confirm + wide_dev)")
    ap.add_argument("--score-confirm", action="store_true", help="--train wide: also score the FROZEN wide_confirm fold "
                    "(needs MODENG_SCORE_WIDE_CONFIRM=1; pre-registered use only)")
    ap.add_argument("--no-guard", action="store_true", help="--train wide: keep relatives' evaluation targets in training")
    a = ap.parse_args()
    POS_DEG_POWER = a.pos_deg_power
    tag = a.tag or a.channels.replace(",", "_")
    if a.train == "wide":
        res, imp = run_wide(a.channels, tag, score_confirm=a.score_confirm, guard=not a.no_guard)
    elif a.train == "past":
        res, imp = run_past(a.bench, a.channels, tag)
    else:
        res, imp = run(a.bench, a.channels, tag, a.rounds)
    print(imp.head(25).to_string())
