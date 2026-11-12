"""R5 KGE: paired cluster bootstrap of stack_v1_kge vs stack_v1_kgeperm (control) and vs stack_v3ptr_p0.35 (modeng-v1) on
temporal dev / confirm: ap, nuap, sa_nuap (per-task parquet, modeng.compare) and degbin_ap (degree-matched AP, review endpoints
from saved scores); plus a secondary 'clean' target set without the KG-leaky pairs (descendant / synonym OT node 21.04 edges)."""
import sys, pickle, copy, numpy as np, pandas as pd
sys.argv = sys.argv[:1]
src = open("/home/ph/code/modeng/scripts/review_analyze.py").read(); src = src[:src.index("def boot")]
exec(src)                                          # defines Bf (temporal), endpoints, M, BL, ...
from modeng import bench as BM, compare as C, data as D

LEAK = sys.argv[1] if len(sys.argv) > 1 else None
A, CTRL, REF = "stack_v1_kge", "stack_v1_kgeperm", "stack_v3ptr_p0.35"
leak = pd.read_pickle("/home/ph/code/modeng/results/review/r5_kge_leak.pkl").set_index("d")["leak"].to_dict()
U = D.universe(); pos = {g: i for i, g in enumerate(U)}
rows = []
for split in ("dev", "confirm"):
    B = Bf.split(split)
    P = {m: C.load("temporal", split, m) for m in (A, CTRL, REF)}
    S = {m: np.load(BM.RESULTS / "temporal" / split / f"{m}.scores.npy").astype(np.float32) for m in (A, CTRL, REF)}
    E = {m: endpoints(S[m], B).assign(disease_id=[t.disease_id for t in B.tasks], cluster=[t.cluster for t in B.tasks]) for m in S}
    # clean target set: leaky pairs moved from targets to exclude
    Bc = copy.deepcopy(B)
    for t in Bc.tasks:
        lk = np.array(sorted(pos[g] for g in leak.get(t.disease_id, []) if g in pos), dtype=np.int64)
        if len(lk):
            t.targets = np.setdiff1d(t.targets, lk); t.exclude = np.union1d(t.exclude, lk)
    Ec = {m: BM.evaluate(S[m], Bc, jobs=4).assign(cluster=[t.cluster for t in Bc.tasks]) for m in S}
    for a, b in ((A, CTRL), (A, REF), (CTRL, REF)):
        for met in ("ap", "nuap", "sa_nuap"):
            rows.append(dict(split=split, a=a, b=b, **C.paired_bootstrap(P[a], P[b], met)))
        rows.append(dict(split=split, a=a, b=b, **C.paired_bootstrap(E[a], E[b], "degbin_ap")))
        for met in ("ap", "nuap"):
            r = C.paired_bootstrap(Ec[a], Ec[b], met); r["metric"] = "clean_" + met
            rows.append(dict(split=split, a=a, b=b, **r))
r = pd.DataFrame(rows)[["split", "a", "b", "metric", "mean_a", "mean_b", "delta", "ci_lo", "ci_hi", "p_le0", "win_rate", "n_tasks"]]
r.to_csv("/home/ph/code/modeng/results/review/r5_kge_compare.tsv", sep="\t", index=False, float_format="%.4f")
print(r.round(4).to_string(index=False))
