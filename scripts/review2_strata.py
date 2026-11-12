"""R3 red team: modeng-v1 endpoints (saved scores) after removing targets that already had Open Targets 21.04 evidence for the
task family (own disease + ontology descendants), and by cluster overlap with the training (dev) diseases. Log: results/review2/strata.log"""
import sys
sys.path.insert(0, "/home/ph/code/modeng/scripts")
from review2_common import *
from modeng import data as D
U = D.universe()
S1 = np.load(ROOT + "/results/temporal/confirm/stack_v3ptr_p0.35.scores.npy").astype(np.float32)
SS = {"v1": S1, "transfer": BL.transfer_jaccard(Bc.tasks, Bc), "headline": BL.modet_headline(Bc.tasks, Bc)}
for extra in sys.argv[1:]:
    SS[extra] = np.load(OUT + f"S_{extra}.npy").astype(np.float32)
ev = pd.read_parquet(ROOT + "/data/cache/priors_h/ev_2104.parquet"); ev["g"] = U.get_indexer(ev.targetId); ev = ev[ev.g >= 0]
MEND = ("eva", "gene2phenotype", "genomics_england", "clingen", "uniprot_literature", "uniprot_variants", "orphanet")
strong = ev[(ev.src.isin(MEND) & ~((ev.src == "eva") & (ev.s < 0.5))) | ((ev.src == "ot_genetics_portal") & (ev.s >= 0.5))]
by_any = ev.groupby("diseaseId").g.apply(lambda x: np.unique(x.to_numpy())).to_dict(); by_str = strong.groupby("diseaseId").g.apply(lambda x: np.unique(x.to_numpy())).to_dict()
off = np.flatnonzero(deg == 0); cd = {t.cluster for t in Bd.tasks}
fam_any, fam_str, anc_str = [], [], []
for t in Bc.tasks:
    fam = {t.disease_id} | set(D.descendants(t.disease_id)); anc = set(D.relatives(t.disease_id)) - fam
    cat = lambda m, ids: np.unique(np.concatenate([m[d] for d in ids if d in m] + [np.array([], int)]))
    fam_any.append(cat(by_any, fam)); fam_str.append(cat(by_str, fam)); anc_str.append(cat(by_str, fam | anc))

def sub(S, drop, onnet=False, min_pos=3):
    out = []
    for i, t in enumerate(Bc.tasks):
        k = np.setdiff1d(t.targets, drop[i]) if drop is not None else t.targets
        exc = np.union1d(t.exclude, np.setdiff1d(t.targets, k))
        if onnet:
            k = np.setdiff1d(k, off); exc = np.union1d(exc, np.setdiff1d(off, k))
        if len(k) < min_pos:
            out.append((np.nan, np.nan)); continue
        m = M.rank_metrics(S[i], k, exc, deg); out.append((m["ap"], m["nuap"]))
    return np.array(out)
cl = np.array([t.cluster for t in Bc.tasks]); res = []
tot = sum(len(t.targets) for t in Bc.tasks)
for nm, drop in (("all targets", None), ("targets without strong (Mendelian / L2G>=0.5) 21.04 family evidence", fam_str),
                 ("targets without strong 21.04 evidence in family or any ancestor", anc_str), ("targets without ANY 21.04 family evidence (any source, incl. text mining)", fam_any)):
    kept = tot if drop is None else sum(len(np.setdiff1d(t.targets, drop[i])) for i, t in enumerate(Bc.tasks))
    print(f"\n== {nm}: {kept} of {tot} target pairs kept")
    for onnet in (False, True):
        E = {k: sub(v, drop, onnet) for k, v in SS.items()}
        for k in SS:
            if k == "v1" or (k == "headline" and not onnet):
                continue
            for j, met in enumerate(("ap", "nuap")):
                d = boot(E["v1"][:, j], E[k][:, j], cl)
                print(f"   {'onnet' if onnet else 'all-gene':8s} {met:4s} v1 {np.nanmean(E['v1'][:, j]):.4f} vs {k:9s} {np.nanmean(E[k][:, j]):.4f}: {fmt(d)}  (n tasks {int(np.isfinite(E['v1'][:, j] - E[k][:, j]).sum())})")
                res.append(dict(stratum=nm, cand="onnet" if onnet else "all", metric=met, ref=k, v1=np.nanmean(E["v1"][:, j]), ref_v=np.nanmean(E[k][:, j]), delta=d[0], lo=d[1], hi=d[2]))
pd.DataFrame(res).to_csv(OUT + "strata.tsv", sep="\t", index=False, float_format="%.4f")
shared = np.array([t.cluster in cd for t in Bc.tasks]); E = {k: sub(v, None) for k, v in SS.items()}
print(f"\n== cluster overlap with training diseases: confirm tasks in a cluster that also holds dev (training) diseases {shared.sum()}, in confirm-only clusters {(~shared).sum()}")
for nm, sel in (("shared cluster", shared), ("confirm-only cluster", ~shared)):
    for j, met in enumerate(("ap", "nuap")):
        print(f"   [{nm}] {met}: v1 {np.nanmean(E['v1'][sel, j]):.4f} vs transfer {np.nanmean(E['transfer'][sel, j]):.4f}: {fmt(boot(E['v1'][sel, j], E['transfer'][sel, j], cl[sel]))}")
