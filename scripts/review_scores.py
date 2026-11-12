"""R1 red-team: retrain stackP_bdv_cs_pt (and ablations) from cached channels, save score matrices (no leaderboard writes)."""
import os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
sys.path.insert(0, "src")
from modeng import tasks as T, channels as CH, stack as ST, bench as BM
ST.LGB = dict(ST.LGB, num_threads=4)
OUT = "results/review"
names = CH.all_channel_names(("base", "donor", "view", "cs_donor", "dA_pt"))
MODERN = [n for n in names if any(k in n for k in ("hpa_sc", "gtex", "hpa_tissue", "_ps_"))]
print("modern channels dropped in 'clean':", MODERN, flush=True)
t0 = time.time()
P = T.load("past"); rng = np.random.default_rng(0)
sel = np.sort(rng.choice(len(P.tasks), 900, replace=False))
Pt = T.Benchmark(P.name, [P.tasks[i] for i in sel], P.context, P.context_ids, P.network, P.union_network, P.leakage_class, P.degree)
mp_, rp_, np_ = ST.build_feats(Pt, "train900", names)
Bf = T.load("temporal")
feats = {s: ST.build_feats(Bf.split(s), s, names) for s in ("dev", "confirm")}
print("features", time.time() - t0, flush=True)
pt19 = {s: (np.load(f"{OUT}/pt19_{s}.npy") if os.path.exists(f"{OUT}/pt19_{s}.npy") else None) for s in ("dev", "confirm")}

def sub(mats, ranks, keep):
    return [mats[j] for j in keep], [ranks[j] for j in keep]

def go(tag, keep, power=0.5, swap_pt=False):
    ST.POS_DEG_POWER = power
    m_, r_ = sub(mp_, rp_, keep)
    X, y, w = ST.training_rows(Pt, np.arange(len(Pt.tasks)), m_, r_, np_, 1500, max_pos=60)
    m = ST.fit(X, y, w, 600)
    for s in ("dev", "confirm"):
        Bs = Bf.split(s); ms, rs, ns = feats[s]
        ms, rs = list(ms), list(rs)
        if swap_pt:
            j = names.index("dA_pt"); ms[j] = pt19[s]; rs[j] = ST.pct_rank(pt19[s])
        ms, rs = sub(ms, rs, keep)
        S = np.zeros((len(Bs.tasks), Bs.n_genes), np.float32)
        ST.predict_all(m, Bs, np.arange(len(Bs.tasks)), ms, rs, ns, S)
        np.save(f"{OUT}/S_{tag}_{s}.npy", S)
        df = BM.evaluate(S, Bs, jobs=1)
        print(f"{tag:14s} {s:8s} AP {df.ap.mean():.4f} nuAP {df.nuap.mean():.4f} R@100 {df['recall@100'].mean():.4f} onnet AP {df.onnet_ap.mean():.4f} nuAP {df.onnet_nuap.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)

allk = list(range(len(names)))
which = sys.argv[1].split(",")
if "full" in which: go("full", allk)
if "clean" in which: go("clean", [j for j in allk if names[j] not in MODERN])
if "pt19" in which and pt19["dev"] is not None: go("pt19lag", allk, swap_pt=True)
if "p025" in which: go("p025", allk, power=0.25)
if "p0" in which: go("p0", allk, power=0.0)
