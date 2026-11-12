"""r7 / P3: two-arm experts (biology vs label process) + gate, trained on temporal_wide wide_train
(pre-registration docs/preregistration_experts.md, commit 328dabf). Never touches wide_confirm.

  experts <seed>   Expert-B / Expert-L via stack.run_wide (guarded, deg^-0.35) -> stackW_E_B[_s1], stackW_E_L[_s1]
  oof              5-fold (cluster-grouped) out-of-fold expert scores on wide_train (seed 0), eval guard + within-train
                   relative guard -> results/r7/oof_{B,L}.npy
  gate             logistic gate on OOF rows, applied to full experts s0 / s1 -> stackW_G[_s1] (scores + parquet)
Outputs under results/r7/ and results/<bench>/<split>/stackW_{E_B,E_L,G}*.{parquet,scores.npy}.
"""
import os, sys, time, pickle
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ["MODENG_SAVE_SCORES"] = "1"
assert os.environ.get("MODENG_SCORE_WIDE_CONFIRM") != "1"
import numpy as np, pandas as pd
from modeng import stack as ST, tasks as T, channels as CH, data as D, bench as BM

ST.POS_DEG_POWER = 0.35
OUT = os.path.join(ROOT, "results/r7"); os.makedirs(OUT, exist_ok=True)
V1 = open("results/v1clean_channels.txt").read().strip().split(",")
L_CH = ["popularity", "gwas_catlag", "pri_l2g_n", "pri_l2g_max", "pri_mend_n", "pri_hpo_n", "hz_gwas", "hz_mend"]
B_CH = [c for c in V1 if c not in L_CH]
assert len(B_CH) == 43 and len(L_CH) == 8
ARMS = {"B": ",".join(B_CH), "L": ",".join(L_CH)}


def experts(seed):
    ST.LGB["seed"] = int(seed)
    for arm, ch in ARMS.items():
        tag = f"E_{arm}" + (f"_s{seed}" if int(seed) else "")
        res, imp = ST.run_wide(ch, tag)
        imp.to_csv(f"{OUT}/imp_{tag}.tsv", sep="\t")


def eval_sets():
    W = T.load("temporal_wide"); Tm = T.load("temporal")
    return W, W.split("wide_train"), [("temporal", Tm.split("dev")), ("temporal", Tm.split("confirm")),
                                      ("temporal_wide", W.split("wide_dev"))]


def oof(folds=5):
    t0 = time.time()
    ST.LGB["seed"] = 0
    W, Wt, evals = eval_sets()
    drop0, _ = ST.relative_target_guard(Wt, [E for _, E in evals])
    clusters = np.array([t.cluster or t.disease_id for t in Wt.tasks])
    uc = np.unique(clusters); rng = np.random.default_rng(0); rng.shuffle(uc)
    fold_of = {c: k % folds for k, c in enumerate(uc)}
    fid = np.array([fold_of[c] for c in clusters])
    np.save(f"{OUT}/oof_fold.npy", fid)
    rel = {t.disease_id: set(D.relatives(t.disease_id)) for t in Wt.tasks}
    for arm in ("B", "L"):
        names = CH.all_channel_names(tuple(ARMS[arm].split(",")))
        mw, rw, nw = ST.build_feats(Wt, "wide_train", names)
        S = np.zeros((len(Wt.tasks), Wt.n_genes), np.float32)
        for k in range(folds):
            tr, te = np.flatnonzero(fid != k), np.flatnonzero(fid == k)
            te_by = {Wt.tasks[j].disease_id: Wt.tasks[j].targets for j in te}
            drop = {}
            for i in tr:
                gs = [drop0[i]] if i in drop0 else []
                d = Wt.tasks[i].disease_id
                gs += [te_by[x] for x in (rel[d] & te_by.keys())]
                if gs:
                    drop[i] = np.unique(np.concatenate(gs))
            X, y, w = ST.training_rows(Wt, tr, mw, rw, nw, 1500, max_pos=60, drop=drop)
            m = ST.fit(X, y, w, 600)
            ST.predict_all(m, Wt, te, mw, rw, nw, S)
            print(f"[oof {arm}] fold {k} rows {len(y)} ({time.time()-t0:.0f}s)", flush=True)
        np.save(f"{OUT}/oof_{arm}.npy", S)
        del mw, rw


def _logit(p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def context(B, split_tag):
    from modeng.ext import priors_h as PH
    mix = PH._mix_g(B)
    ns = np.log1p([len(t.seeds) for t in B.tasks])
    cat = CH.compute(B, "gwas_catlag", split_tag)
    cov = (~np.isnan(cat).all(1)).astype(float)
    pop = CH.compute(B, "popularity", split_tag)
    sp = np.array([np.nanmean(np.log1p(np.maximum(pop[i, t.seeds], 0))) for i, t in enumerate(B.tasks)])
    return np.c_[mix, ns, cov, np.nan_to_num(sp)]


def gate_feats(sB, sL, rB, rL, c):
    lB, lL = _logit(sB), _logit(sL)
    return np.c_[lB, lL, rB, rL, np.repeat(c[None, :], len(lB), 0) if c.ndim == 1 else c,
                 lB[:, None] * (c if c.ndim == 2 else c[None, :]), lL[:, None] * (c if c.ndim == 2 else c[None, :])]


def gate():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    W, Wt, evals = eval_sets()
    drop0, _ = ST.relative_target_guard(Wt, [E for _, E in evals])
    SB, SL = np.load(f"{OUT}/oof_B.npy"), np.load(f"{OUT}/oof_L.npy")
    RB, RL = ST.pct_rank(SB), ST.pct_rank(SL)
    C = context(Wt, "wide_train")
    # gate training rows: same sampling rule as the experts (seed 0): <= 60 targets + 1,500 negatives per task, guard applied
    rng = np.random.default_rng(0); deg = np.asarray(Wt.degree, np.float32)
    Xs, ys, ws = [], [], []
    for i, t in enumerate(Wt.tasks):
        excl = np.zeros(Wt.n_genes, bool); excl[t.exclude] = True; excl[t.targets] = True
        dg = drop0.get(i)
        if dg is not None:
            excl[dg] = True
        negs = rng.choice(np.flatnonzero(~excl), 1500, replace=False)
        tgt = t.targets if dg is None else np.setdiff1d(t.targets, dg)
        if not len(tgt):
            continue
        tg = tgt if len(tgt) <= 60 else rng.choice(tgt, 60, replace=False)
        g = np.concatenate([tg, negs])
        Xs.append(gate_feats(SB[i, g], SL[i, g], RB[i, g], RL[i, g], C[i]))
        ys.append(np.r_[np.ones(len(tg)), np.zeros(len(negs))])
        ws.append(np.r_[1.0 / np.maximum(deg[tg], 1.0) ** 0.35, np.ones(len(negs))])
    X, y, w = np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y, sample_weight=w)
    cols = ["lB", "lL", "rB", "rL", "c_gwas", "c_nseeds", "c_catcov", "c_seedpop"] + \
           [f"lB*{c}" for c in ("gwas", "nseeds", "catcov", "seedpop")] + [f"lL*{c}" for c in ("gwas", "nseeds", "catcov", "seedpop")]
    coef = pd.Series(lr.coef_[0], index=cols)
    coef.to_csv(f"{OUT}/gate_coef.tsv", sep="\t")
    print(coef.round(3).to_string(), flush=True)
    pickle.dump((sc, lr), open(f"{OUT}/gate.pkl", "wb"))
    for bench, Bs in evals:
        split = Bs.tasks[0].split
        Cs = context(Bs, split)
        for sfx in ("", "_s1"):
            sB = np.load(BM.RESULTS / bench / split / f"stackW_E_B{sfx}.scores.npy").astype(np.float32)
            sL = np.load(BM.RESULTS / bench / split / f"stackW_E_L{sfx}.scores.npy").astype(np.float32)
            rB, rL = ST.pct_rank(sB), ST.pct_rank(sL)
            G = np.zeros_like(sB)
            for i in range(len(Bs.tasks)):
                G[i] = lr.predict_proba(sc.transform(gate_feats(sB[i], sL[i], rB[i], rL[i], Cs[i])))[:, 1]
            ST._record(G, Bs, bench, split, f"stackW_G{sfx}", t0, True)


if __name__ == "__main__":
    mode = sys.argv[1]
    {"experts": experts, "oof": oof, "gate": gate}[mode](*sys.argv[2:])
