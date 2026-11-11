"""Quick test: universal donor transfer (UDT) = transfer over non-disease gene-set families (virtual donors).
Usage: python scripts/exp_udt.py temporal dev
Per family: donor weight w_j = Jaccard(seeds, set_j) (top-k=50); score(g) = sum_j w_j * set_j[g].
Reports channel alone and rank-mixed with transfer_jaccard (alpha grid) on the given bench/split.
"""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
sys.path.insert(0, "/home/ph/code/modeng/src")
from modeng import tasks as T, bench as BM
from modeng.methods import baselines as BL

DON = Path("/home/ph/code/modeng/data/donors")


def load_family(name):
    z = np.load(DON / f"{name}.npz")
    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))


def donor_transfer(tasks, M, k=50, n_genes=None):
    S = BL._indicator([t.seeds for t in tasks], M.shape[1])
    inter = (S @ M.T).toarray()
    sizes = M.getnnz(1)
    ns = np.array([len(t.seeds) for t in tasks])[:, None]
    jac = inter / np.maximum(ns + sizes[None] - inter, 1)
    out = np.zeros((len(tasks), M.shape[1]), np.float32)
    for i in range(len(tasks)):
        top = np.argsort(-jac[i])[:k]; w = jac[i, top]
        out[i] = np.asarray(M[top].T @ w.astype(np.float32)).ravel()
    return out


def rank01(S):
    r = S.argsort(1).argsort(1).astype(np.float32)
    return r / S.shape[1]


def summ(df):
    return f"AP {df.ap.mean():.4f} nuAP {df.nuap.mean():.4f} R@100 {df['recall@100'].mean():.4f} AUROC {df.auroc.mean():.4f}"


if __name__ == "__main__":
    bench, split = sys.argv[1], sys.argv[2]
    fams = sys.argv[3].split(",") if len(sys.argv) > 3 else [l.split("\t")[0] for l in open(DON / "MANIFEST.tsv")]
    B = T.load(bench).split(split)
    TR = BL.transfer_jaccard(B.tasks, B)
    print("transfer_jaccard", summ(BM.evaluate(TR, B, 8)), flush=True)
    Rtr = rank01(TR + 1e-9 * np.random.default_rng(0).random(TR.shape))
    allU = None
    for f in fams:
        M = load_family(f)
        U = donor_transfer(B.tasks, M)
        allU = U / (U.max(1, keepdims=True) + 1e-9) if allU is None else allU + U / (U.max(1, keepdims=True) + 1e-9)
        line = f"{f:16s} alone {summ(BM.evaluate(U, B, 8))}"
        mix = BM.evaluate(TR / (TR.max(1, keepdims=True) + 1e-9) + 0.2 * U / (U.max(1, keepdims=True) + 1e-9), B, 8)
        print(line, "| tr+0.2*udt", summ(mix), flush=True)
    print("ALL families alone", summ(BM.evaluate(allU, B, 8)))
    for a in (0.05, 0.1, 0.2, 0.4):
        mix = TR / (TR.max(1, keepdims=True) + 1e-9) + a * allU / len(fams)
        print(f"tr + {a}*ALL", summ(BM.evaluate(mix, B, 8)), flush=True)
