"""FROZEN evaluator for F2-3 (evolved transfer kernel). Do not edit after the first evaluation (see docs/reports/r2_evolve.md).

    python scripts/evolve_eval.py <n> [--parent <p,q>] [--complement]

Imports modeng.evolve.cand_<n> (must expose `score(tasks, B) -> [n_tasks, n_genes]`, called with the target-free
harness.SafeBench), scores it on a FIXED random subset of 600 `past` episodes (rng = np.random.default_rng(0),
choice(n_past, 600, replace=False), sorted) and appends one row to results/evolve/log.tsv.
fitness = mean nuAP (tie-break mean AP). `--complement` scores the other past episodes instead (pre-registered: used only for
cand_0 = transfer_jaccard and the final winner, never for selection). Lint (logged): no disease/gene id strings, no integer
literals > 1000, no access to private harness attributes (`_B`). Failures are logged with status != ok.
"""
from __future__ import annotations

import argparse, ast, importlib, os, re, sys, time, traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modeng import tasks as T, bench as BM          # noqa: E402
from modeng.evolve.harness import SafeBench          # noqa: E402

LOG = ROOT / "results/evolve/log.tsv"
N_FIT, SEED = 600, 0
ID_RE = re.compile(r"(EFO_|MONDO_|Orphanet_|HP_|DOID|ENSG|OTAR_|GO_)")


def subset(complement=False):
    P = T.load("past")
    sel = np.sort(np.random.default_rng(SEED).choice(len(P.tasks), N_FIT, replace=False))
    if complement:
        sel = np.setdiff1d(np.arange(len(P.tasks)), sel)
    return T.Benchmark(P.name, [P.tasks[i] for i in sel], P.context, P.context_ids, P.network, P.union_network,
                       P.leakage_class, P.degree)


def lint(path):
    tree = ast.parse(Path(path).read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and ID_RE.search(node.value):
                bad.append(f"id-string:{node.value[:20]}")
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool) and abs(node.value) > 1000:
                bad.append(f"big-literal:{node.value}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_B"):
            bad.append("private-access")
    return ";".join(bad) or "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n"); ap.add_argument("--parent", default="")
    ap.add_argument("--complement", action="store_true")
    a = ap.parse_args()
    path = ROOT / f"src/modeng/evolve/cand_{a.n}.py"
    B = subset(a.complement)
    row = dict(cand=a.n, parent=a.parent, mode="complement" if a.complement else "fit600", n=len(B.tasks),
               nuap=np.nan, ap=np.nan, auroc=np.nan, r100=np.nan, seconds=np.nan, lint="", status="ok", desc="")
    t0 = time.time()
    try:
        row["lint"] = lint(path)
        mod = importlib.import_module(f"modeng.evolve.cand_{a.n}")
        row["desc"] = " ".join((getattr(mod, "DESC", None) or (mod.__doc__ or "").strip().split("\n")[0]).split())[:300]
        sb = SafeBench(B)
        S = np.asarray(mod.score(sb.tasks, sb), np.float32)
        assert S.shape == (len(B.tasks), B.n_genes), S.shape
        df = BM.evaluate(S, B, jobs=6)
        row.update(nuap=df.nuap.mean(), ap=df.ap.mean(), auroc=df.auroc.mean(), r100=df["recall@100"].mean())
    except Exception as e:  # failures are logged, never deleted
        row["status"] = f"fail:{type(e).__name__}:{str(e)[:120]}"
        traceback.print_exc()
    row["seconds"] = round(time.time() - t0, 1)
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    new = not LOG.exists()
    pd.DataFrame([row]).to_csv(LOG, sep="\t", mode="a", header=new, index=False, float_format="%.5f")
    print(f"cand_{a.n} [{row['mode']}] nuAP {row['nuap']:.5f} AP {row['ap']:.5f} AUROC {row['auroc']:.4f} "
          f"R@100 {row['r100']:.4f} lint={row['lint']} {row['status']} ({row['seconds']}s)  {row['desc'][:80]}")


if __name__ == "__main__":
    main()
