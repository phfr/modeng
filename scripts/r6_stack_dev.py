"""r6: dev-only OOF stack run (same recipe as modeng.stack.run, cv on temporal dev, deg^-POWER), confirm NOT scored.
Usage: python scripts/r6_stack_dev.py <tag> <channels> [power=0.35]
Writes results/temporal/dev/stack_<tag>.parquet (+ .scores.npy); no leaderboard row."""
import sys, time
import numpy as np
from modeng import stack as S, tasks as T, channels as CH, bench as BM

tag, chans = sys.argv[1], sys.argv[2]
S.POS_DEG_POWER = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
names = CH.all_channel_names(tuple(chans.split(",")))
Bd = T.load("temporal").split("dev")
t0 = time.time()
md, rd, nd = S.build_feats(Bd, "dev", names)
clusters = np.array([t.cluster or t.disease_id for t in Bd.tasks])
uc = np.unique(clusters); rng = np.random.default_rng(0); rng.shuffle(uc)
fold_of = {c: k % 5 for k, c in enumerate(uc)}
fid = np.array([fold_of[c] for c in clusters])
Sc = np.zeros((len(Bd.tasks), Bd.n_genes), np.float32)
for k in range(5):
    tr, te = np.flatnonzero(fid != k), np.flatnonzero(fid == k)
    X, y, w = S.training_rows(Bd, tr, md, rd, nd, 3000)
    S.predict_all(S.fit(X, y, w, 400), Bd, te, md, rd, nd, Sc)
df = BM.evaluate(Sc, Bd, jobs=4)
out = BM.RESULTS / "temporal" / "dev" / f"stack_{tag}.parquet"
df.to_parquet(out, index=False); np.save(str(out).replace(".parquet", ".scores.npy"), Sc.astype(np.float16))
print(f"stack_{tag}: AP {df.ap.mean():.4f} nuAP {df.nuap.mean():.4f} sa_AP {df.sa_ap.mean():.4f} sa_nuAP {df.sa_nuap.mean():.4f} ({time.time()-t0:.0f}s)")
