"""P4: the ONE pre-registered scoring of the frozen wide_confirm fold, cv-recipe arms A (modeng-v1) and B (v1clean).
Pre-registration: docs/preregistration_wide.md, Amendment 1 (committed before this script touched wide_confirm).

    MODENG_SCORE_WIDE_CONFIRM=1 python scripts/p4_score.py wide_confirm     # the one confirmatory run (writes)
    python scripts/p4_score.py wide_dev                                     # dry run (no writes; checks the code path)

Each arm = its cv recipe (stack.run: LightGBM params stack.LGB, seed 0, 400 rounds, 3000 negatives, positives weighted
deg^-0.35) trained on ALL 53 temporal dev diseases, exactly the model stack.run uses for temporal confirm; applied once to
<split>. The same model is also applied to temporal confirm (write=False) to check it reproduces the leaderboard row of the
recipe (stack_v3ptr_p0.35 confirm AP 0.1532; stack_v1clean confirm AP 0.1497). transfer_jaccard is recorded on <split> too.
Scores are saved (MODENG_SAVE_SCORES=1) for the secondary endpoints (P@20, degree-matched AP).
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
import numpy as np
from modeng import stack as ST, tasks as T, channels as CH
from modeng.methods import baselines as BL

SPLIT = sys.argv[1]
assert SPLIT in ("wide_confirm", "wide_dev")
WRITE = SPLIT == "wide_confirm"
if WRITE:
    os.environ["MODENG_SAVE_SCORES"] = "1"
    assert os.environ.get(T.WIDE_CONFIRM_FLAG) == "1"
else:
    os.environ.pop("MODENG_SAVE_SCORES", None)
    assert os.environ.get(T.WIDE_CONFIRM_FLAG) != "1"

ST.POS_DEG_POWER = 0.35
ST.LGB["seed"] = 0
V1 = ("base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,hz_gwas,"
      "hz_mend,ph_mean,ph_top3,pt_ridge,pt_ridge_pos")                       # docs/preregistration_modeng_v1.md
V1CLEAN = open("results/v1clean_channels.txt").read().strip()               # Amendment 1 of that pre-registration
ARMS = {"v1": (V1, 0.1532), "v1clean": (V1CLEAN, 0.1497)}

Bt = T.load("temporal")
Bd, Bc = Bt.split("dev"), Bt.split("confirm")
Bs = T.load("temporal_wide").split(SPLIT)
if SPLIT == "wide_confirm":
    assert sorted(t.disease_id for t in Bs.tasks) == [l.strip() for l in open("data/wide_confirm_ids.txt") if l.strip()]
print(f"{SPLIT}: {len(Bs.tasks)} tasks, {sum(len(t.targets) for t in Bs.tasks)} targets", flush=True)

t0 = time.time()
ST._record(BL.transfer_jaccard(Bs.tasks, Bs), Bs, "temporal_wide", SPLIT, "transfer_jaccard", t0, WRITE)

for arm, (chans, ref_conf) in ARMS.items():
    t0 = time.time()
    names = CH.all_channel_names(tuple(chans.split(",")))
    md, rd, nd = ST.build_feats(Bd, "dev", names)
    X, y, w = ST.training_rows(Bd, np.arange(len(Bd.tasks)), md, rd, nd, 3000)
    print(f"[{arm}] {len(names)} channels, {len(y)} rows, {int(y.sum())} pos ({time.time()-t0:.0f}s)", flush=True)
    m = ST.fit(X, y, w, 400)
    # reproduction check on temporal confirm (no writes)
    mc, rc, nc = ST.build_feats(Bc, "confirm", names)
    Sc = np.zeros((len(Bc.tasks), Bc.n_genes), np.float32)
    ST.predict_all(m, Bc, np.arange(len(Bc.tasks)), mc, rc, nc, Sc)
    save = os.environ.pop("MODENG_SAVE_SCORES", None)
    dc = ST._record(Sc, Bc, "temporal", "confirm", f"stack_{arm}", t0, False)
    if save:
        os.environ["MODENG_SAVE_SCORES"] = save
    print(f"[{arm}] temporal confirm AP {dc.ap.mean():.4f} (recipe row {ref_conf}; reproduced: {abs(dc.ap.mean() - ref_conf) < 5e-4})", flush=True)
    ms, rs, ns = ST.build_feats(Bs, SPLIT, names)
    S = np.zeros((len(Bs.tasks), Bs.n_genes), np.float32)
    ST.predict_all(m, Bs, np.arange(len(Bs.tasks)), ms, rs, ns, S)
    ST._record(S, Bs, "temporal_wide", SPLIT, f"stack_{arm}", t0, WRITE)
