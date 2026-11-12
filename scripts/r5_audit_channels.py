"""R5 witness audit, parts 1 + 4: per-channel coverage / triviality / per-gene-vs-per-task / alone metrics (temporal dev and
confirm, cached channel matrices of modeng-v1), and within-task Spearman redundancy between channel scores (confirm, 2,000
random candidate genes per task, pairwise-complete) with average-linkage clusters at |rho| >= 0.7.
Outputs results/r5/channels_inventory.tsv, results/r5/redundancy_spearman.tsv, results/r5/redundancy_clusters.tsv.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from modeng import tasks as T, channels as CH, metrics as M
from r5_audit_ablate import NAMES, group_of

OUT = os.path.join(ROOT, "results/r5"); os.makedirs(OUT, exist_ok=True)
Bf = T.load("temporal"); DEG = np.asarray(Bf.degree)


def inventory():
    rows = []
    for split in ("dev", "confirm"):
        B = Bf.split(split)
        for n in NAMES:
            S = CH.compute(B, n, split)
            fin = np.isfinite(S)
            cov = fin.mean(1)                                     # per task share of genes with a value
            nontriv = np.array([fin[i].sum() > 1 and np.nanstd(S[i]) > 0 and (np.nan_to_num(S[i]) != 0).sum() > 0 for i in range(len(S))])
            nz = np.array([(np.nan_to_num(S[i]) != 0).mean() for i in range(len(S))])
            # per-gene channel: rows identical across tasks (up to NaN pattern)
            A = np.nan_to_num(S, nan=-9e9)
            per_gene = bool((A == A[0:1]).all())
            aps, nuaps, aucs = [], [], []
            for i, t in enumerate(B.tasks):
                m = M.rank_metrics(S[i], t.targets, t.exclude, DEG)
                aps.append(m["ap"]); nuaps.append(m["nuap"]); aucs.append(m["auroc"])
            rows.append(dict(split=split, channel=n, group=group_of(n), gene_cov=cov.mean(), nonzero_share=nz.mean(),
                             task_nontrivial=nontriv.mean(), per_gene=per_gene, alone_ap=np.nanmean(aps),
                             alone_nuap=np.nanmean(nuaps), alone_auroc=np.nanmean(aucs)))
        print(split, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/channels_inventory.tsv", sep="\t", index=False, float_format="%.4f")
    return df


def redundancy(n_genes=2000):
    B = Bf.split("confirm")
    mats = {n: CH.compute(B, n, "confirm") for n in NAMES}
    rng = np.random.default_rng(0)
    k = len(NAMES); acc = np.zeros((k, k)); cnt = np.zeros((k, k))
    for i, t in enumerate(B.tasks):
        cand = np.setdiff1d(np.arange(B.n_genes), t.exclude)
        g = rng.choice(cand, min(n_genes, len(cand)), replace=False)
        X = pd.DataFrame({n: mats[n][i, g] for n in NAMES})
        X = X.loc[:, X.nunique() > 1]
        C = X.corr(method="spearman", min_periods=200)
        ix = [NAMES.index(c) for c in C.columns]
        v = C.to_numpy(); ok = np.isfinite(v)
        acc[np.ix_(ix, ix)] += np.where(ok, v, 0); cnt[np.ix_(ix, ix)] += ok
    R = pd.DataFrame(acc / np.maximum(cnt, 1), index=NAMES, columns=NAMES)
    R[cnt == 0] = np.nan
    R.to_csv(f"{OUT}/redundancy_spearman.tsv", sep="\t", float_format="%.3f")
    D = 1 - R.abs().fillna(0).to_numpy(); np.fill_diagonal(D, 0); D = (D + D.T) / 2
    Z = linkage(squareform(D, checks=False), "average")
    cl = fcluster(Z, 0.3, "distance")
    out = pd.DataFrame(dict(channel=NAMES, group=[group_of(n) for n in NAMES], cluster=cl))
    out.to_csv(f"{OUT}/redundancy_clusters.tsv", sep="\t", index=False)
    for c, sub in out.groupby("cluster"):
        if len(sub) > 1:
            print(c, ", ".join(sub.channel))
    return R


if __name__ == "__main__":
    print(inventory().query("split=='confirm'").round(3).to_string(index=False))
    redundancy()
