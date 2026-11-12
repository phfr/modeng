"""Past-trained twin (19.04->21.04 episodes, as stack.run_past) scored on temporal_wide wide_dev."""
import sys, time, numpy as np
from modeng import stack as ST, tasks as T, channels as CH
chans, tag = sys.argv[1], sys.argv[2]
t0 = time.time()
names = CH.all_channel_names(tuple(chans.split(",")))
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
ST._record(S, Bs, "temporal_wide", "wide_dev", f"stackP_{tag}", t0, True)
