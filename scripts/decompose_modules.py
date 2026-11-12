"""Run subex's `leiden_anno` decomposition (S-62/S-63, the confirmed best) on every exported module (scripts/export_modules.py).

Same code path as `subex decompose --method leiden_anno --overlap none --graph t0_string115_phys_notm_400` (subex/cli.py, decompose branch:
graph -> load_module -> load_library -> decompose_stage -> save_submodules + submodule_table), but the graph and the T0 GO / Reactome
library are loaded once per worker instead of once per module (the CLI spends ~45 s per call on loading). subex is only imported, never
edited. Check: `--check` runs the real CLI on one module and compares the labels.
Writes data/modules/<method>/<disease>/sub_tau{T}.npz / .tsv. 6 worker processes, 1 thread each.
Usage: python scripts/decompose_modules.py [--check]
"""
import os, sys, subprocess, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = Path("/home/ph/code/modeng/data/modules")
GRAPH = "t0_string115_phys_notm_400"
SUBEX = "/home/ph/biotools/venvs/bio-net/bin/subex"
_G = _lib = None


def _init():
    global _G, _lib
    import torch; torch.set_num_threads(1)
    from subex.cli import graph
    from subex.annot import load_library
    _G = graph(GRAPH); _lib = load_library(_G)


def run_one(mod: Path):
    from subex.io import load_module, load_submodules, save_submodules
    from subex.stages import decompose_stage
    from subex.cli import submodule_table
    out = mod.with_name(mod.name.replace("module_", "sub_"))
    if out.exists():
        return str(mod), "skip", 0.0
    t0 = time.time()
    try:
        mr = load_module(mod, _G)
        if len(mr.genes) < 3:
            return str(mod), "too_small", 0.0
        er, ov = decompose_stage(_G, mr, _lib, overlap=None, module_kind="connected", method="leiden_anno")
        save_submodules(out, er, _G, provenance=dict(module=str(mod), overlap="none", method="leiden_anno", tool="modeng decompose_modules.py (subex decompose path)"), overlap=ov)
        load_submodules(out, _G)
        submodule_table(_G, er, ov).to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
        return str(mod), "ok", time.time() - t0
    except Exception as e:  # recorded, not hidden
        return str(mod), f"failed: {type(e).__name__}: {e}", time.time() - t0


def check():
    mod = sorted(ROOT.glob("modeng/*/module_tau0.3.npz"))[0]
    cli_out = Path("/tmp/claude-1000/-home-ph-code-modetor/e1bb738e-6e86-40a7-aee2-526b20823cc9/scratchpad/r4_cli_check.npz")
    subprocess.run([SUBEX, "decompose", "--module", str(mod), "--graph", GRAPH, "--method", "leiden_anno", "--overlap", "none",
                    "--out", str(cli_out)], check=True, capture_output=True)
    a = np.load(cli_out, allow_pickle=False); b = np.load(mod.with_name("sub_tau0.3.npz"), allow_pickle=False)
    same = (a["module"] == b["module"]).all() and (a["labels"] == b["labels"]).all()
    print(f"CLI check on {mod}: identical = {same}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check(); sys.exit()
    order = [("modeng", "0.2"), ("modeng", "0.3"), ("transfer", "0.2"), ("headline", "0.2"), ("transfer", "0.3"), ("headline", "0.3")]
    mods = [m for meth, tau in order for m in sorted(ROOT.glob(f"{meth}/*/module_tau{tau}.npz"))]
    if "--main-only" in sys.argv:          # modeng both taus + comparators at tau 0.2
        mods = [m for m in mods if not (m.parts[-3] != "modeng" and m.name.endswith("0.3.npz"))]
    with Pool(6, initializer=_init) as P:
        res = []
        for k, r in enumerate(P.imap_unordered(run_one, mods, chunksize=2)):
            res.append(r)
            if k % 50 == 0:
                print(k, r, flush=True)
    df = pd.DataFrame(res, columns=["module", "status", "seconds"])
    df.to_csv(ROOT / "decompose_status.tsv", sep="\t", index=False)
    print(df.status.str.slice(0, 40).value_counts())
