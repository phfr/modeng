"""Run a method on a benchmark split, write per-task metrics and a leaderboard row.

    python -m modeng.bench --method rwr_string --bench temporal --split dev [--jobs 8]
    python -m modeng.bench --method all --bench all --split all          # every registered method x bench x {dev,confirm}

Outputs: results/<bench>/<split>/<method>.parquet (one row per task) and a row appended to results/leaderboard.tsv
(bench, split, method, n, ap, nuap, auroc, recall@100, onnet_ap / onnet_nuap (network genes only), sa_ap (temporal strict-a), seconds, timestamp, git sha).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from . import tasks as T
from .methods import BENCHES as METHOD_BENCHES
from .methods import METHODS, get

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
LEADERBOARD = RESULTS / "leaderboard.tsv"

_G: dict = {}


def _metrics_chunk(idx):
    S, tasks, deg = _G["S"], _G["tasks"], _G["deg"]
    return [M.task_metrics(S[i], tasks[i], deg) for i in idx]


def evaluate(S: np.ndarray, B: T.Benchmark, jobs: int = 1) -> pd.DataFrame:
    """Per-task metric table for a score matrix [n_tasks, n_genes]."""
    assert S.shape == (len(B.tasks), B.n_genes), f"score shape {S.shape} != {(len(B.tasks), B.n_genes)}"
    deg = B.degree
    if jobs > 1 and len(B.tasks) > 50:
        _G.update(S=S, tasks=B.tasks, deg=deg)
        chunks = np.array_split(np.arange(len(B.tasks)), jobs * 4)
        with mp.get_context("fork").Pool(jobs) as pool:
            rows = [r for part in pool.map(_metrics_chunk, chunks) for r in part]
        _G.clear()
    else:
        rows = [M.task_metrics(S[i], t, deg) for i, t in enumerate(B.tasks)]
    df = pd.DataFrame(rows)
    df.insert(0, "disease_id", [t.disease_id for t in B.tasks])
    df.insert(1, "name", [t.name for t in B.tasks])
    df.insert(2, "split", [t.split for t in B.tasks])
    df.insert(3, "cluster", [t.cluster for t in B.tasks])
    df.insert(4, "n_seeds", [len(t.seeds) for t in B.tasks])
    return df


def git_sha() -> str:
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "src"], capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "?"


def run(method: str, bench: str, split: str = "dev", jobs: int = 1, write: bool = True, verbose: bool = True) -> pd.DataFrame:
    B = T.load(bench).split(split)
    t0 = time.time()
    S = np.asarray(get(method)(B.tasks, B), dtype=np.float32)
    t_score = time.time() - t0
    df = evaluate(S, B, jobs)
    secs = time.time() - t0
    if write:
        out = RESULTS / bench / split / f"{method}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        row = dict(bench=bench, split=split, method=method, n=int(df["ap"].notna().sum()),
                   ap=df["ap"].mean(), nuap=df["nuap"].mean(), auroc=df["auroc"].mean(), r100=df["recall@100"].mean(),
                   onnet_ap=df["onnet_ap"].mean(), onnet_nuap=df["onnet_nuap"].mean(),
                   sa_ap=df["sa_ap"].mean() if "sa_ap" in df else np.nan, seconds=round(secs, 1),
                   timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), git=git_sha())
        new = not LEADERBOARD.exists()
        pd.DataFrame([row]).to_csv(LEADERBOARD, sep="\t", mode="a", header=new, index=False, float_format="%.4f")
    if verbose:
        sa = f" | onnet AP {df['onnet_ap'].mean():.4f} nuAP {df['onnet_nuap'].mean():.4f}"
        sa += f" saAP {df['sa_ap'].mean():.4f}" if "sa_ap" in df else ""
        print(f"{bench:9s} {split:8s} {method:18s} n={df['ap'].notna().sum():4d}  AP {df['ap'].mean():.4f}  "
              f"nuAP {df['nuap'].mean():.4f}  AUROC {df['auroc'].mean():.4f}  R@100 {df['recall@100'].mean():.4f}{sa}  "
              f"({t_score:.1f}s score, {secs:.1f}s total)", flush=True)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, help="registered name, comma list, or 'all'")
    ap.add_argument("--bench", default="temporal", help="temporal|forward|holdout, comma list, or 'all'")
    ap.add_argument("--split", default="dev", help="dev|confirm|all, or 'both' = dev then confirm")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    methods = sorted(METHODS) if a.method == "all" else a.method.split(",")
    benches = list(T.BENCHES) if a.bench == "all" else a.bench.split(",")
    splits = ["dev", "confirm"] if a.split == "both" else [a.split]
    for b in benches:
        for m in methods:
            if m in METHOD_BENCHES and b not in METHOD_BENCHES[m]:
                continue
            for s in splits:
                run(m, b, s, a.jobs, write=not a.no_write)


if __name__ == "__main__":
    main()
