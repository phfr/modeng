"""r3_prospective hybrid: modeng-v1 (past-fill channels) trained on past episodes (19.04 -> 21.04, 900 sampled as in
stack.run_past) + temporal dev tasks (53, 21.04 -> 25.09), evaluated on temporal confirm (138). Optional --dev-oof: dev scored
out-of-fold (past + 4/5 of dev clusters, same folds as stack.run).
    MODENG_SAVE_SCORES=1 python scripts/r3_hybrid.py [--dev-oof]
"""
import sys, time
import numpy as np
from modeng import stack as ST, tasks as T, channels as CH

ST.POS_DEG_POWER = 0.35
CHS = ("base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n_pf,pri_l2g_max_pf,pri_mend_n_pf,"
       "pri_hpo_n_pf,hz_gwas_pf,hz_mend_pf,ph_mean_pf,ph_top3_pf,pt_ridge,pt_ridge_pos")
names = CH.all_channel_names(tuple(CHS.split(",")))
method, t0 = "stackH_v1hyb", time.time()
P = T.load("past")
rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(P.tasks), min(900, len(P.tasks)), replace=False))
Pt = T.Benchmark(P.name, [P.tasks[i] for i in sel], P.context, P.context_ids, P.network, P.union_network, P.leakage_class, P.degree)
fp = ST.build_feats(Pt, "train900", names)
Xp, yp, wp = ST.training_rows(Pt, np.arange(len(Pt.tasks)), *fp, 1500, max_pos=60)
Bf = T.load("temporal")
Bd, Bc = Bf.split("dev"), Bf.split("confirm")
fd = ST.build_feats(Bd, "dev", names)


def fit_with(dev_idx):
    Xd, yd, wd = ST.training_rows(Bd, dev_idx, *fd, 3000)
    print(f"[hybrid] past rows {len(yp)} + dev rows {len(yd)} ({time.time()-t0:.0f}s)", flush=True)
    return ST.fit(np.vstack([Xp, Xd]), np.r_[yp, yd], np.r_[wp, wd], 600)


m = fit_with(np.arange(len(Bd.tasks)))
fc = ST.build_feats(Bc, "confirm", names)
Sc = np.zeros((len(Bc.tasks), Bc.n_genes), np.float32)
ST.predict_all(m, Bc, np.arange(len(Bc.tasks)), *fc, Sc)
ST._record(Sc, Bc, "temporal", "confirm", method, t0, True)
if "--dev-oof" in sys.argv:
    clusters = np.array([t.cluster or t.disease_id for t in Bd.tasks])
    uc = np.unique(clusters); r = np.random.default_rng(0); r.shuffle(uc)
    fold_of = {c: k % 5 for k, c in enumerate(uc)}
    fid = np.array([fold_of[c] for c in clusters])
    S = np.zeros((len(Bd.tasks), Bd.n_genes), np.float32)
    for k in range(5):
        mk = fit_with(np.flatnonzero(fid != k))
        ST.predict_all(mk, Bd, np.flatnonzero(fid == k), *fd, S)
    ST._record(S, Bd, "temporal", "dev", method, t0, True)
