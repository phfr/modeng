"""R1 red-team: brute-force recompute of cached channels on a few tasks with B.context_for; task/masking audit."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, scipy.sparse as sp
sys.path.insert(0, "src")
from modeng import tasks as T, channels as CH, data as D
from modeng.ext import lead_cs, donors_a as A
C = "data/cache/channels/"
B = T.load("temporal").split("confirm")
rng = np.random.default_rng(1)
nm = np.array([len(t.mask_r) for t in B.tasks])
idx = sorted(set(np.argsort(-nm)[:4].tolist()) | set(rng.choice(len(B.tasks), 6, replace=False).tolist()))
print("tasks", idx, "masked pairs", nm[idx])
def mini(Bx, ii):
    return T.Benchmark(Bx.name, [Bx.tasks[i] for i in ii], Bx.context, Bx.context_ids, Bx.network, Bx.union_network, Bx.leakage_class, Bx.degree)
def cmp(name, new, old):
    a, b = np.nan_to_num(new, nan=-9), np.nan_to_num(old, nan=-9)
    d = np.abs(a - b).max(1)
    rk = [np.corrcoef(a[i], b[i])[0, 1] for i in range(len(a))]
    print(f"{name:18s} max|diff| per task {np.round(d,6)}  min corr {min(rk):.6f}")
# transfer brute force
cache = np.load(C + "temporal_confirm_transfer_jaccard.npy")
out = np.zeros((len(idx), B.n_genes), np.float32)
for k, i in enumerate(idx):
    t = B.tasks[i]; L = B.context_for(t)
    sz = L.getnnz(1); inter = np.asarray(L[:, t.seeds].sum(1)).ravel()
    jac = inter / np.maximum(len(t.seeds) + sz - inter, 1)
    own = B.own_row(t)
    if own >= 0: jac[own] = 0
    cand = np.flatnonzero(jac > 0); top = cand[np.argsort(-jac[cand])[:50]]
    out[k] = np.round(np.asarray(L[top].T @ jac[top].astype(np.float32)).ravel(), 5)
cmp("transfer_jaccard", out, cache[idx])
Bm = mini(B, idx)
cmp("cs_donor", lead_cs._cs(Bm, "x"), np.load(C + "temporal_confirm_cs_donor.npy")[idx])
cmp("dA_pt(a_pt21)", A.hyper_transfer(Bm.tasks, A.load_any("a_pt21"), **A.PARAMS), np.load(C + "temporal_confirm_dA_pt.npy")[idx])
cmp("dA_pt(a_pt25)?", A.hyper_transfer(Bm.tasks, A.load_any("a_pt25"), **A.PARAMS), np.load(C + "temporal_confirm_dA_pt.npy")[idx])
cmp("don_hpo2101", CH.donor_transfer(Bm.tasks, CH.load_family("hpo2101")), np.load(C + "temporal_confirm_don_hpo2101.npy")[idx])
X, m = CH.load_view("esmc"); mean, mx = CH.view_sim(Bm.tasks, X, m)
cmp("vmax_esmc", mx, np.load(C + "temporal_confirm_vmax_esmc.npy")[idx])
# past: PubTator cut used for training
P = T.load("past"); sel = np.sort(np.random.default_rng(0).choice(len(P.tasks), 900, replace=False))
pi = [0, 100, 500, 899]
Pm = mini(P, [sel[j] for j in pi])
old = np.load(C + "past_train900_dA_pt.npy", mmap_mode="r")[pi]
cmp("past dA_pt(a_pt19)", A.hyper_transfer(Pm.tasks, A.load_any("a_pt19"), **A.PARAMS), old)
cmp("past dA_pt(a_pt21)?", A.hyper_transfer(Pm.tasks, A.load_any("a_pt21"), **A.PARAMS), old)
cmp("past cs_donor", lead_cs._cs(Pm, "x"), np.load(C + "past_train900_cs_donor.npy", mmap_mode="r")[pi])
# ---- task audit (all 191 temporal tasks)
Bf = T.load("temporal"); L = Bf.context.tocsr(); ids = Bf.context_ids; pos = {d: i for i, d in enumerate(ids)}
tot = own = masked = anc_leak = neardup = anywhere = 0
sizes = L.getnnz(1)
for t in Bf.tasks:
    assert len(np.intersect1d(t.seeds, t.targets)) == 0 and len(np.intersect1d(t.exclude, t.targets)) == 0
    tot += len(t.targets)
    o = Bf.own_row(t)
    if o >= 0: own += int(L[o][:, t.targets].nnz)
    mt = np.unique(t.mask_c); masked += len(mt)
    Lt = Bf.context_for(t)
    if len(mt):
        anc_leak += int((np.asarray(Lt[:, mt].sum(0)).ravel() > 0).sum())
    inter = np.asarray(Lt[:, t.seeds].sum(1)).ravel()
    jac = inter / np.maximum(len(t.seeds) + Lt.getnnz(1) - inter, 1)
    if o >= 0: jac[o] = 0
    nd = np.flatnonzero(jac >= 0.5)
    if len(nd): neardup += int((np.asarray(Lt[nd][:, t.targets].sum(0)).ravel() > 0).sum())
    anywhere += int((np.asarray(Lt[:, t.targets].sum(0)).ravel() > 0).sum())
print(f"targets {tot}; in own 21.04 row {own}; with >=1 masked pair {masked}; of those still in another (unmasked) context row {anc_leak}; "
      f"targets in a near-duplicate row (J>=0.5 with seeds) {neardup}; targets in any context row {anywhere}")
# sealed check
ex = D.excluded_task_ids()
print("sealed/excluded ids among temporal tasks:", sum(t.disease_id in ex for t in Bf.tasks), "; in context rows:", int(ids.isin(list(ex)).sum()),
      "; among past tasks:", sum(t.disease_id in ex for t in P.tasks), "; past context rows:", int(P.context_ids.isin(list(ex)).sum()))
tid = {t.disease_id for t in Bf.tasks}
print("past training tasks that are temporal eval diseases:", sum(P.tasks[i].disease_id in tid for i in sel), "of", len(sel))
