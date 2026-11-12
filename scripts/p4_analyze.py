"""P4 analysis (pre-registered: docs/preregistration_wide.md, Amendment 1). Reads saved score matrices of the arms on <split>,
computes per-task endpoints (scripts/review_analyze.py `endpoints`: AP, nuAP, P@20, onnet, strict-a, degree-matched AP) and
paired cluster bootstraps (modeng.compare.paired_bootstrap, 10,000, clusters = ontology components, seed 0).

    MODENG_SCORE_WIDE_CONFIRM=1 python scripts/p4_analyze.py wide_confirm
Outputs results/p4/{per_task,summary,primary,secondary,between,strata}_<split>.tsv
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd
from modeng import tasks as T, bench as BM
from modeng.compare import paired_bootstrap
from modeng.methods import baselines as BL

src = open("scripts/review_analyze.py").read(); src = src[:src.index("def boot")]
exec(src)                                    # endpoints(), within_bin_ap() with temporal degree deciles (same network)

SPLIT = sys.argv[1]
B = T.load("temporal_wide").split(SPLIT)
ARMS = {"A_v1": "stack_v1", "B_v1clean": "stack_v1clean", "C_W_v1clean": "stackW_P4C_v1clean",
        "D_W_v1clean_nolab": "stackW_P4D_v1clean_nolab"}
REF = "transfer_jaccard"
D = {}
for arm, m in ARMS.items():
    f = BM.RESULTS / "temporal_wide" / SPLIT / f"{m}.scores.npy"
    if not f.exists():
        print(f"MISSING {arm}: {f} (reported as failure)"); continue
    D[arm] = endpoints(np.load(f).astype(np.float32), B)
D[REF] = endpoints(BL.transfer_jaccard(B.tasks, B), B)
for k, d in D.items():
    d.insert(0, "method", k)
pd.concat(D.values()).to_csv(f"results/p4/per_task_{SPLIT}.tsv", sep="\t", index=False, float_format="%.5f")

MET = ["ap", "nuap", "degbin_ap", "p@20", "onnet_ap", "onnet_nuap", "auroc", "recall@100", "sa_ap", "sa_nuap"]
summ = pd.DataFrame([dict(method=k, n=len(d), **{c: d[c].mean() if c in d else np.nan for c in MET},
                          n_sa=int(d["sa_ap"].notna().sum()) if "sa_ap" in d else 0) for k, d in D.items()])
summ.to_csv(f"results/p4/summary_{SPLIT}.tsv", sep="\t", index=False, float_format="%.4f")


def cmp(a, b, metrics, min_n=3):
    out = []
    for c in metrics:
        if c not in D[a] or c not in D[b] or D[a][c].notna().sum() < min_n:
            continue
        out.append(dict(a=a, b=b, **paired_bootstrap(D[a], D[b], c, n=10000, seed=0)))
    return out


prim = pd.DataFrame([r for arm in ARMS if arm in D for r in cmp(arm, REF, ["ap", "nuap"])])
prim["ci_excl0"] = (prim.ci_lo > 0) | (prim.ci_hi < 0)
claim = prim.groupby("a").apply(lambda g: bool(((g.ci_lo > 0).all()) and len(g) == 2)).rename("claim").reset_index()
prim = prim.merge(claim, on="a")
prim.to_csv(f"results/p4/primary_{SPLIT}.tsv", sep="\t", index=False, float_format="%.4f")
sec = pd.DataFrame([r for arm in ARMS if arm in D for r in cmp(arm, REF, ["degbin_ap", "p@20", "onnet_ap", "onnet_nuap"])])
sec.to_csv(f"results/p4/secondary_{SPLIT}.tsv", sep="\t", index=False, float_format="%.4f")
pairs = [("C_W_v1clean", "B_v1clean"), ("B_v1clean", "A_v1"), ("D_W_v1clean_nolab", "C_W_v1clean"), ("C_W_v1clean", "A_v1")]
btw = pd.DataFrame([r for a, b in pairs if a in D and b in D for r in cmp(a, b, ["ap", "nuap"])])
btw.to_csv(f"results/p4/between_{SPLIT}.tsv", sep="\t", index=False, float_format="%.4f")

nt = D[REF]["n_targets"].to_numpy()
strata = pd.cut(nt, [0, 5, 10, 10**6], labels=["3-5", "6-10", ">10"])
st = []
for k, d in D.items():
    for s in ["3-5", "6-10", ">10"]:
        sel = np.asarray(strata == s)
        st.append(dict(method=k, stratum=s, n=int(sel.sum()), ap=d["ap"][sel].mean(), nuap=d["nuap"][sel].mean()))
pd.DataFrame(st).to_csv(f"results/p4/strata_{SPLIT}.tsv", sep="\t", index=False, float_format="%.4f")

pd.set_option("display.width", 250)
cols = ["a", "b", "metric", "n_tasks", "n_clusters", "mean_a", "mean_b", "delta", "ci_lo", "ci_hi", "win_rate"]
print(summ.round(4).to_string(index=False))
print("\nPRIMARY\n" + prim[cols + ["claim"]].round(4).to_string(index=False))
print("\nSECONDARY\n" + sec[cols].round(4).to_string(index=False))
print("\nBETWEEN (descriptive)\n" + btw[cols].round(4).to_string(index=False))
print("\nSTRATA\n" + pd.DataFrame(st).round(4).to_string(index=False))
