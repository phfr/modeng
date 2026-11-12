"""Evaluate settx predictions alone on temporal dev / confirm (+ zero-transfer-support stratum).

    python -m modeng.settx.evaluate --variant all [--write]
Zero-support stratum: per task, genes with transfer_jaccard == 0 that are rankable (not excluded); AUROC / AP of the score
among them (targets in the stratum vs the stratum's non-targets), averaged over tasks with >= 1 stratum target.
"""
from __future__ import annotations

import argparse, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from .. import tasks as T, bench as BM, channels as CH
from . import data as SD


def load_pred(variant, bench, split):
    z = np.load(SD.CACHE / f"pred_{variant}_{bench}_{split}.npz" if bench == "temporal" else SD.CACHE / f"pred_{variant}_past.npz")
    return z["ids"], z["S"]


def zero_stratum(S, B, tj):
    rows = []
    for i, t in enumerate(B.tasks):
        ok = np.ones(B.n_genes, bool); ok[t.exclude] = False
        z = ok & (tj[i] <= 0)
        y = np.zeros(B.n_genes, bool); y[t.targets] = True
        yz = y[z]
        if yz.sum() == 0 or yz.all():
            rows.append((np.nan, np.nan, int(yz.sum()), len(t.targets))); continue
        rows.append((roc_auc_score(yz, S[i, z]), average_precision_score(yz, S[i, z]), int(yz.sum()), len(t.targets)))
    return pd.DataFrame(rows, columns=["z_auroc", "z_ap", "z_npos", "npos"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", nargs="+", required=True)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    Bt = T.load("temporal")
    for split in ("dev", "confirm"):
        Bs = Bt.split(split)
        tj = CH.compute(Bs, "transfer_jaccard", split)
        ref = {"transfer_jaccard": tj, "popularity": CH.compute(Bs, "popularity", split)}
        for v in a.variant:
            ids, S = load_pred(v, "temporal", split)
            assert list(ids) == [t.disease_id for t in Bs.tasks]
            ref[f"settx_{v}"] = S
        for name, S in ref.items():
            df = BM.evaluate(S.astype(np.float32), Bs, jobs=4)
            zs = zero_stratum(S, Bs, tj)
            print(f"temporal {split:8s} {name:22s} AP {df.ap.mean():.4f} nuAP {df.nuap.mean():.4f} AUROC {df.auroc.mean():.4f} "
                  f"R@100 {df['recall@100'].mean():.4f} onnet AP {df.onnet_ap.mean():.4f} nuAP {df.onnet_nuap.mean():.4f} "
                  f"sa_AP {df.sa_ap.mean():.4f} | zero-support AUROC {zs.z_auroc.mean():.4f} AP {zs.z_ap.mean():.4f} "
                  f"(frac targets {zs.z_npos.sum()/zs.npos.sum():.2f})", flush=True)
            if a.write and name.startswith("settx_"):
                out = BM.RESULTS / "temporal" / split / f"{name}.parquet"
                pd.concat([df, zs], axis=1).to_parquet(out, index=False)


if __name__ == "__main__":
    main()
