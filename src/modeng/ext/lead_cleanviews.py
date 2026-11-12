"""Temporal-clean replacements for the post-dated expression views (lead, 2026-11-12; see docs/invalidations.md 'Vintage flag').
hpa20_sc   : HPA v20.1 rna_single_cell_type (NX, log1p, genes x cell types)      [HPA v20.1 = 2020-11, pre-T0]
hpa20_tis  : HPA v20.1 rna_tissue_consensus (NX, log1p)                           [pre-T0]
gtex8      : GTEx v8 median TPM (modet features gtex_v8.parquet, log2(TPM+1))     [2017, pre-T0]
Channels: vc_mean_<v>, vc_max_<v> = mean / top-3 cosine of each gene to the seeds (same as channels.view_sim), z-scored
gene profiles, cached to data/views_clean/<v>.npz. Used to build the 'clean' arm of modeng-v1 (modern expression and
Perturb-seq views removed)."""
import io, zipfile
import numpy as np
import pandas as pd
from pathlib import Path

from modeng import data as D
from modeng.channels import view_sim

OUT = Path("/home/ph/code/modeng/data/views_clean")
HPA = Path("/mnt/msadb/modet/downloads/hpa_v20.1")


def _matrix(name):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{name}.npz"
    if f.exists():
        z = np.load(f)
        return z["X"], z["mask"]
    u = D.universe()
    if name in ("hpa20_sc", "hpa20_tis"):
        fn = {"hpa20_sc": "rna_single_cell_type.tsv.zip", "hpa20_tis": "rna_tissue_consensus.tsv.zip"}[name]
        with zipfile.ZipFile(HPA / fn) as zf:
            d = pd.read_csv(io.BytesIO(zf.read(zf.namelist()[0])), sep="\t")
        col = d.columns[2]
        M = d.pivot_table(index="Gene", columns=col, values="NX", aggfunc="mean")
        M = np.log1p(M)
    else:
        M = pd.read_parquet("/home/ph/code/modet/data/processed/features/gtex_v8.parquet")
    M = M.reindex(u)
    mask = M.notna().all(1).to_numpy()
    X = M.fillna(0).to_numpy(np.float32)
    mu, sd = X[mask].mean(0), X[mask].std(0) + 1e-6
    X = (X - mu) / sd
    X[~mask] = 0
    np.savez_compressed(f, X=X, mask=mask)
    return X, mask


_CACHE = {}


def _make(view, which):
    def fn(B, split_tag):
        key = (B.name, split_tag, len(B.tasks), view)
        if key not in _CACHE:
            X, m = _matrix(view)
            _CACHE[key] = view_sim(B.tasks, X, m)
        mean, mx = _CACHE[key]
        return mean if which == "mean" else mx
    return fn


CHANNELS = {}
for _v in ("hpa20_sc", "hpa20_tis", "gtex8"):
    CHANNELS[f"vc_mean_{_v}"] = _make(_v, "mean")
    CHANNELS[f"vc_max_{_v}"] = _make(_v, "max")
