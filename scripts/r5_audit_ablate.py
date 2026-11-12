"""R5 witness audit, part 2: frozen modeng-v1 recipe (stack_v3ptr_p0.35, cv, deg^-0.35) retrained with and without each
channel group on temporal; dev = grouped-CV OOF, confirm = model trained on all dev. Also saves the full dev->confirm model
(LightGBM text file) and per-task endpoints (ap, nuap, degbin_ap) for every arm.

Descriptive only: temporal confirm has been viewed many times overnight; no claim is decided here.
Usage (repo root irrelevant): ~/biotools/venvs/bio-net/bin/python scripts/r5_audit_ablate.py [arm ...]
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd
from modeng import tasks as T, channels as CH, metrics as M
from modeng import stack as ST

ST.POS_DEG_POWER = 0.35
OUT = os.path.join(ROOT, "results/r5"); os.makedirs(OUT, exist_ok=True)
CHANNELS = ("base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,"
            "hz_gwas,hz_mend,ph_mean,ph_top3,pt_ridge,pt_ridge_pos")
NAMES = CH.all_channel_names(tuple(CHANNELS.split(",")))
BASE = ["transfer_jaccard", "transfer_onto", "popularity", "degree", "rwr_string"]


def group_of(n):
    if n in BASE: return "base"
    if n.startswith("don_"): return "donor"
    if n.startswith("vmean_") or n.startswith("vmax_"): return "view"
    if n in ("dc_ridge", "dc_ridge_pos"): return "dc_ridge"
    if n.startswith("pri_") or n.startswith("hz_"): return "priors"
    if n.startswith("ph_"): return "ph"
    if n.startswith("pt_ridge"): return "pt_ridge"
    return n   # cs_donor, dA_pt, gwas_catlag


GROUPS = ["base", "donor", "view", "cs_donor", "dA_pt", "dc_ridge", "gwas_catlag", "priors", "ph", "pt_ridge"]
ARMS = ["full"] + ["no_" + g for g in GROUPS]
# extra arms (added after the pre-specified ones): two-group removal, transductive (post-T0 'modern') inputs removed,
# and LightGBM seed replicates of the full recipe (noise floor of a retrain; negatives sample unchanged)
MODERN = {"vmean_" + v for v in ("hpa_sc", "gtex", "hpa_tissue", "ps_k562", "ps_rpe1", "ps_cd4", "ps_hepg2", "ps_jurkat")} | \
         {"vmax_" + v for v in ("hpa_sc", "gtex", "hpa_tissue", "ps_k562", "ps_rpe1", "ps_cd4", "ps_hepg2", "ps_jurkat")} | \
         {"don_km_hpa_sc", "don_km_gtex", "don_km_ps_k562", "don_km_ps_cd4"}
EXTRA = ["no_view+donor", "no_modern", "full_seed1", "full_seed2"]


def dropped(arm, n):
    arm = arm.split("_seed")[0]          # "<arm>_seed<k>" = same arm with LightGBM seed k
    if arm.startswith("full"): return False
    if arm == "no_modern": return n in MODERN
    return group_of(n) in arm[3:].split("+")

# degree-matched AP as in scripts/review_analyze.py (deciles of positive degree + off-network bin)
Bf = T.load("temporal"); DEG = np.asarray(Bf.degree)
_q = np.quantile(DEG[DEG > 0], np.linspace(0, 1, 11)[1:-1]); DBIN = np.where(DEG > 0, 1 + np.searchsorted(_q, DEG, side="left"), 0)


def within_bin_ap(s, pos, exclude):
    mask = np.ones(len(s), bool); mask[exclude] = False
    y = np.zeros(len(s), bool); y[pos] = True
    s = np.where(np.isfinite(s), s, -1e30) + np.random.default_rng(0).uniform(0, 1e-12, len(s))
    num = den = 0.0
    for b in np.unique(DBIN[pos]):
        m = mask & (DBIN == b); yy = y[m]; n = int(yy.sum())
        if n == 0 or n == m.sum(): continue
        o = np.argsort(-s[m]); r = np.flatnonzero(yy[o]); ap = ((np.arange(n) + 1) / (r + 1)).mean()
        num += n * ap; den += n
    return num / den if den else np.nan


def endpoints(S, B):
    rows = []
    for i, t in enumerate(B.tasks):
        m = M.rank_metrics(S[i], t.targets, t.exclude, DEG)
        rows.append(dict(disease_id=t.disease_id, cluster=t.cluster, ap=m["ap"], nuap=m["nuap"],
                         degbin_ap=within_bin_ap(S[i], t.targets, t.exclude)))
    return pd.DataFrame(rows)


def main(arms):
    t0 = time.time()
    Bd, Bc = Bf.split("dev"), Bf.split("confirm")
    md, rd, nd = ST.build_feats(Bd, "dev", NAMES)
    mc, rc, nc = ST.build_feats(Bc, "confirm", NAMES)
    print(f"features loaded {time.time()-t0:.0f}s, {len(NAMES)} channels", flush=True)
    clusters = np.array([t.cluster or t.disease_id for t in Bd.tasks])
    uc = np.unique(clusters); rng = np.random.default_rng(0); rng.shuffle(uc)
    fid = np.array([{c: k % 5 for k, c in enumerate(uc)}[c] for c in clusters])
    for arm in arms:
        if os.path.exists(f"{OUT}/ep_{arm}_confirm.parquet"):
            print("skip", arm); continue
        keep = [j for j, n in enumerate(NAMES) if not dropped(arm, n)]
        ST.LGB["seed"] = int(arm.split("seed")[1]) if "_seed" in arm else 0
        sel = lambda L: [L[j] for j in keep]
        a, r = sel(md), sel(rd)
        S = np.zeros((len(Bd.tasks), Bd.n_genes), np.float32)
        for k in range(5):
            tr, te = np.flatnonzero(fid != k), np.flatnonzero(fid == k)
            X, y, w = ST.training_rows(Bd, tr, a, r, nd, 3000)
            ST.predict_all(ST.fit(X, y, w, 400), Bd, te, a, r, nd, S)
            del X
        X, y, w = ST.training_rows(Bd, np.arange(len(Bd.tasks)), a, r, nd, 3000)
        model = ST.fit(X, y, w, 400); del X
        Sc = np.zeros((len(Bc.tasks), Bc.n_genes), np.float32)
        ST.predict_all(model, Bc, np.arange(len(Bc.tasks)), sel(mc), sel(rc), nc, Sc)
        if arm == "full":
            model.save_model(f"{OUT}/model_full_confirm.txt")
            np.save(f"{OUT}/S_full_dev.npy", S.astype(np.float16)); np.save(f"{OUT}/S_full_confirm.npy", Sc.astype(np.float16))
        for split, SS, B in (("dev", S, Bd), ("confirm", Sc, Bc)):
            e = endpoints(SS, B); e.to_parquet(f"{OUT}/ep_{arm}_{split}.parquet", index=False)
            print(f"{arm:16s} {split:8s} AP {e.ap.mean():.4f} nuAP {e.nuap.mean():.4f} degbinAP {e.degbin_ap.mean():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ARMS)
