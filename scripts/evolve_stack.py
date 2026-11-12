"""Memory-lean replica of modeng.stack.run_past (same rows, seeds, params; LightGBM num_threads=6) for F2-3:
frees the past feature matrices before predicting, saves the model. Usage: evolve_stack.py <channels> <tag>"""
import gc, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from modeng import stack as S, tasks as T, channels as CH

channels, tag = sys.argv[1], sys.argv[2]
S.LGB["num_threads"] = 6
t0 = time.time()
names = CH.all_channel_names(tuple(channels.split(",")))
P = T.load("past")
sel = np.sort(np.random.default_rng(0).choice(len(P.tasks), min(900, len(P.tasks)), replace=False))
Pt = T.Benchmark(P.name, [P.tasks[i] for i in sel], P.context, P.context_ids, P.network, P.union_network, P.leakage_class, P.degree)
mp_, rp_, np_ = S.build_feats(Pt, "train900", names)
X, y, w = S.training_rows(Pt, np.arange(len(Pt.tasks)), mp_, rp_, np_, 1500, max_pos=60)
del mp_, rp_; gc.collect()
print(f"[past] {len(names)} channels, {len(y)} rows ({time.time()-t0:.0f}s)", flush=True)
m = S.fit(X, y, w, 600); del X, y, w; gc.collect()
m.save_model(str(ROOT / f"results/evolve/model_stackP_{tag}.txt"))
B = T.load("temporal")
for split in ("dev", "confirm"):
    Bs = B.split(split)
    ms, rs, ns_ = S.build_feats(Bs, split, names)
    Sc = np.zeros((len(Bs.tasks), Bs.n_genes), np.float32)
    S.predict_all(m, Bs, np.arange(len(Bs.tasks)), ms, rs, ns_, Sc)
    del ms, rs; gc.collect()
    S._record(Sc, Bs, "temporal", split, f"stackP_{tag}", t0, True)
