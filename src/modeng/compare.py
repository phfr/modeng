"""Paired bootstrap of method A vs B on one metric (resampling unit = task cluster; modet fold clusters on `temporal`,
the task itself elsewhere).

    python -m modeng.compare --bench temporal --split dev --a transfer_jaccard --b rwr_string [--metric ap] [--n 10000]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .bench import RESULTS


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, metric: str = "ap", n: int = 10000, seed: int = 0) -> dict:
    df = a[["disease_id", "cluster", metric]].merge(b[["disease_id", metric]], on="disease_id", suffixes=("_a", "_b"))
    df = df.dropna(subset=[f"{metric}_a", f"{metric}_b"])
    df["d"] = df[f"{metric}_a"] - df[f"{metric}_b"]
    g = df.groupby("cluster")["d"].agg(["sum", "count"])
    sums, cnts = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    boot = sums[idx].sum(1) / cnts[idx].sum(1)
    delta = df["d"].mean()
    return dict(metric=metric, n_tasks=len(df), n_clusters=len(g), mean_a=df[f"{metric}_a"].mean(),
                mean_b=df[f"{metric}_b"].mean(), delta=delta, ci_lo=np.quantile(boot, 0.025), ci_hi=np.quantile(boot, 0.975),
                p_le0=float((boot <= 0).mean()), win_rate=float((df["d"] > 0).mean()))


def load(bench: str, split: str, method: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS / bench / split / f"{method}.parquet")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True); ap.add_argument("--split", default="dev")
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--metric", default="ap"); ap.add_argument("--n", type=int, default=10000)
    x = ap.parse_args(argv)
    r = paired_bootstrap(load(x.bench, x.split, x.a), load(x.bench, x.split, x.b), x.metric, x.n)
    print(f"{x.bench}/{x.split} {x.metric}: {x.a} {r['mean_a']:.4f} vs {x.b} {r['mean_b']:.4f}  delta {r['delta']:+.4f} "
          f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  P(delta<=0) {r['p_le0']:.4f}  win {r['win_rate']:.2f}  "
          f"(n={r['n_tasks']}, clusters={r['n_clusters']})")


if __name__ == "__main__":
    main()
