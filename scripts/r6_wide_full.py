"""r6 P1b: full witness recipe trained at scale on temporal_wide wide_train (pre-registration: docs/preregistration_wide_full.md).

  wide <arm> <seed>   stack.run_wide (600 rounds, 1500 negatives, <= 60 positives, relative-target guard), deg^-0.35,
                      LightGBM seed <seed>; arms v1clean (51), v2 (23), v1perm (v2 + the 28 dropped channels gene-permuted)
                      -> method stackW_W_<arm>[_s<seed>] on temporal dev / confirm and wide_dev
  cv2wide <arm>       cv recipe (stack.run: 400 rounds, 3000 negatives, deg^-0.35, seed 0) trained on all 53 temporal dev
                      tasks, scored on wide_dev -> temporal_wide/wide_dev/stack_<arm>
  past <arm>          past-trained twin (stack.run_past, deg^-0.35) on temporal dev / confirm and wide_dev -> stackP_<arm>
Never touches wide_confirm. Scores saved (MODENG_SAVE_SCORES=1) for the degree-matched endpoint.
"""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("OMP_NUM_THREADS", "8"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ["MODENG_SAVE_SCORES"] = "1"
assert os.environ.get("MODENG_SCORE_WIDE_CONFIRM") != "1"
import numpy as np
from modeng import stack as ST, tasks as T, channels as CH

ST.POS_DEG_POWER = 0.35
V1 = open("results/v1clean_channels.txt").read().strip()
V2 = open("results/v2_channels.txt").read().strip()
DROP = [c for c in V1.split(",") if c not in V2.split(",")]
ARMS = {"v1clean": V1, "v2": V2, "v1perm": V2 + "," + ",".join("perm_" + c for c in DROP)}


def wide(arm, seed):
    ST.LGB["seed"] = int(seed)
    tag = f"W_{arm}" + (f"_s{seed}" if int(seed) else "")
    res, imp = ST.run_wide(ARMS[arm], tag)
    imp.to_csv(f"results/r6w/imp_{tag}.tsv", sep="\t")


def cv2wide(arm):
    t0 = time.time()
    names = CH.all_channel_names(tuple(ARMS[arm].split(",")))
    Bd = T.load("temporal").split("dev")
    md, rd, nd = ST.build_feats(Bd, "dev", names)
    X, y, w = ST.training_rows(Bd, np.arange(len(Bd.tasks)), md, rd, nd, 3000)
    m = ST.fit(X, y, w, 400)
    Bs = T.load("temporal_wide").split("wide_dev")
    ms, rs, ns = ST.build_feats(Bs, "wide_dev", names)
    S = np.zeros((len(Bs.tasks), Bs.n_genes), np.float32)
    ST.predict_all(m, Bs, np.arange(len(Bs.tasks)), ms, rs, ns, S)
    ST._record(S, Bs, "temporal_wide", "wide_dev", f"stack_{arm}", t0, True)


def past(arm):
    t0 = time.time()
    ST.run_past("temporal", ARMS[arm], arm)
    names = CH.all_channel_names(tuple(ARMS[arm].split(",")))
    P = T.load("past"); rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(P.tasks), min(900, len(P.tasks)), replace=False))
    Pt = T.Benchmark(P.name, [P.tasks[i] for i in sel], P.context, P.context_ids, P.network, P.union_network, P.leakage_class, P.degree)
    m_, r_, n_ = ST.build_feats(Pt, "train900", names)
    X, y, w = ST.training_rows(Pt, np.arange(len(Pt.tasks)), m_, r_, n_, 1500, max_pos=60)
    m = ST.fit(X, y, w, 600)
    Bs = T.load("temporal_wide").split("wide_dev")
    ms, rs, ns = ST.build_feats(Bs, "wide_dev", names)
    S = np.zeros((len(Bs.tasks), Bs.n_genes), np.float32)
    ST.predict_all(m, Bs, np.arange(len(Bs.tasks)), ms, rs, ns, S)
    ST._record(S, Bs, "temporal_wide", "wide_dev", f"stackP_{arm}", t0, True)


def cvscores(arm):
    """Re-run the cv recipe on temporal (write=False: parquet / leaderboard untouched) only to save its scores for the
    degree-matched endpoint; prints AP to check it reproduces the leaderboard row."""
    ST.run("temporal", ARMS[arm], arm, write=False)


if __name__ == "__main__":
    mode = sys.argv[1]
    {"wide": wide, "cv2wide": cv2wide, "past": past, "cvscores": cvscores}[mode](*sys.argv[2:])
