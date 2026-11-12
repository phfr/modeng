"""R3 red team: (a) per-channel discrimination of confirm targets (near-deterministic channel?), (b) same-window training:
how much of confirm is 'genes that also arrive for dev diseases', relation dev <-> confirm, gain split by that.
Outputs results/review2/channel_auroc.tsv, window.log"""
import sys
sys.path.insert(0, "/home/ph/code/modeng/scripts")
from review2_common import *
from modeng import data as D
S1 = np.load(ROOT + "/results/temporal/confirm/stack_v3ptr_p0.35.scores.npy").astype(np.float32)
TR = BL.transfer_jaccard(Bc.tasks, Bc)

# ---------------------------------------------------------------- (a) per channel
rows_ = []
for n in NAMES + ["gwas_catlag12", "ph_mean_perm", "pri_l2g_n_perm"]:
    X = cached("confirm", n); au, ap, cov, p1 = [], [], [], []
    for i, t in enumerate(Bc.tasks):
        if not np.isfinite(X[i]).any():
            continue
        m = M.rank_metrics(X[i], t.targets, t.exclude, deg); au.append(m["auroc"]); ap.append(m["ap"])
        v = X[i][t.targets]; cov.append(np.mean(np.isfinite(v) & (v != 0)))
        mask = np.ones(Bc.n_genes, bool); mask[t.exclude] = False; s = np.where(np.isfinite(X[i]), X[i], -1e30); c = np.flatnonzero(mask); top = c[np.argsort(-s[c], kind="stable")[:20]]
        p1.append(np.isin(top, t.targets).mean())
    rows_.append(dict(channel=n, n_tasks=len(au), auroc=np.mean(au), auroc_max=np.max(au), n_tasks_auroc_gt95=int((np.array(au) > 0.95).sum()), ap=np.mean(ap), p20=np.mean(p1), target_nonzero=np.mean(cov)))
C = pd.DataFrame(rows_).sort_values("auroc", ascending=False); C.to_csv(OUT + "channel_auroc.tsv", sep="\t", index=False, float_format="%.4f")
print("== (a) per-channel mean per-task AUROC / AP on confirm targets (top 15 by AUROC, then top 8 by AP)"); print(C.head(15).round(4).to_string(index=False)); print(C.sort_values("ap", ascending=False).head(8).round(4).to_string(index=False))
e1 = ep(S1, Bc); print(f"modeng-v1 saved scores: AP {e1.ap.mean():.4f} nuAP {e1.nuap.mean():.4f} AUROC {e1.auroc.mean():.4f}; tasks with AUROC > 0.95: {(e1.auroc > 0.95).sum()}, AP > 0.8: {(e1.ap > 0.8).sum()}, max AP {e1.ap.max():.3f}")

# ---------------------------------------------------------------- (b) same-window structure
cnt = np.zeros(Bd.n_genes, np.float32)
for t in Bd.tasks:
    cnt[t.targets] += 1
devg = cnt > 0
tp = np.concatenate([t.targets for t in Bc.tasks])
print(f"\n== (b) genes that are a target of >= 1 dev task: {int(devg.sum())} of {Bd.n_genes} ({devg.mean():.3f}); confirm target pairs whose gene is a dev target: {devg[tp].mean():.3f}")
allneg = []
for t in Bc.tasks:
    ex = np.zeros(Bc.n_genes, bool); ex[t.exclude] = True; ex[t.targets] = True; allneg.append(devg[~ex].mean())
print(f"   confirm non-target candidates that are dev-target genes: {np.mean(allneg):.3f}")
cd, cc = {t.cluster for t in Bd.tasks}, {t.cluster for t in Bc.tasks}; did = {t.disease_id: t for t in Bd.tasks}
print(f"   bootstrap clusters dev {len(cd)}, confirm {len(cc)}, shared {len(cd & cc)}; confirm tasks in a shared cluster: {sum(t.cluster in cd for t in Bc.tasks)}")
rel, same = [], []
for t in Bc.tasks:
    r = [did[x] for x in (set(D.relatives(t.disease_id)) & set(did))]
    rel.append(len(r) > 0)
    same.append(np.mean(np.isin(t.targets, np.concatenate([x.targets for x in r]))) if r else 0.0)
rel = np.array(rel); same = np.array(same)
print(f"   confirm tasks with an ontology relative (ancestor/descendant) among dev tasks: {rel.sum()} of {len(rel)}; their targets that are also targets of that dev relative: {same[rel].mean() if rel.any() else float('nan'):.3f}")
# best Jaccard of target sets with any dev task
bj = []
for t in Bc.tasks:
    bj.append(max(len(np.intersect1d(t.targets, x.targets)) / len(np.union1d(t.targets, x.targets)) for x in Bd.tasks))
bj = np.array(bj); print(f"   max target-set Jaccard with any dev task: median {np.median(bj):.3f}, > 0.2: {(bj > 0.2).sum()} tasks, > 0.5: {(bj > 0.5).sum()} tasks")
# scorer = dev arrival count (a pure 'which genes arrive in this window' scorer, no disease information)
SC = np.tile(cnt, (len(Bc.tasks), 1)); ec = ep(SC, Bc); et = ep(TR, Bc)
print(f"   dev-arrival-count scorer on confirm: AP {ec.ap.mean():.4f} nuAP {ec.nuap.mean():.4f} AUROC {ec.auroc.mean():.4f} | transfer AP {et.ap.mean():.4f} nuAP {et.nuap.mean():.4f} AUROC {et.auroc.mean():.4f} | popularity AUROC see channel table")
cl = e1.cluster.to_numpy()
for nm, sel in (("all", np.ones(len(rel), bool)), ("no dev relative & target Jaccard <= 0.2", ~rel & (bj <= 0.2)), ("dev relative or target Jaccard > 0.2", rel | (bj > 0.2))):
    print(f"   v1 - transfer, tasks [{nm}] n={sel.sum()}: AP {fmt(boot(e1.ap[sel], et.ap[sel], cl[sel]))} nuAP {fmt(boot(e1.nuap[sel], et.nuap[sel], cl[sel]))}")
# split targets: gene is a dev target vs not (other class removed from the ranking)
def sub_ep(S, keep_dev):
    out = []
    for i, t in enumerate(Bc.tasks):
        k = t.targets[devg[t.targets] == keep_dev]
        if len(k) < 3:
            out.append((np.nan, np.nan)); continue
        m = M.rank_metrics(S[i], k, np.union1d(t.exclude, np.setdiff1d(t.targets, k)), deg); out.append((m["ap"], m["nuap"]))
    return np.array(out)
for kd, nm in ((True, "targets whose gene is a dev target"), (False, "targets whose gene is NOT a dev target")):
    a, b = sub_ep(S1, kd), sub_ep(TR, kd)
    print(f"   {nm}: tasks {np.isfinite(a[:, 0]).sum()}; v1 AP {np.nanmean(a[:, 0]):.4f} vs transfer {np.nanmean(b[:, 0]):.4f}: {fmt(boot(a[:, 0], b[:, 0], cl))}; nuAP {np.nanmean(a[:, 1]):.4f} vs {np.nanmean(b[:, 1]):.4f}: {fmt(boot(a[:, 1], b[:, 1], cl))}")
# false-positive side: among the top-100 non-target candidates of v1 vs transfer, share that are dev-target genes
fp1, fpt = [], []
for i, t in enumerate(Bc.tasks):
    ex = np.zeros(Bc.n_genes, bool); ex[t.exclude] = True; ex[t.targets] = True; c = np.flatnonzero(~ex)
    fp1.append(devg[c[np.argsort(-S1[i][c])[:100]]].mean()); fpt.append(devg[c[np.argsort(-TR[i][c], kind="stable")[:100]]].mean())
print(f"   top-100 false positives that are dev-target genes: v1 {np.mean(fp1):.3f}, transfer {np.mean(fpt):.3f} (base rate {devg.mean():.3f})")
np.save(OUT + "devcount.npy", cnt)
