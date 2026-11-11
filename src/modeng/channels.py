"""Feature channels for the learned stacker: each channel is a [n_tasks, n_genes] float32 score matrix computed from
the task's seeds and the (masked) context only. Channels are cached per (benchmark, split, channel) in data/cache/channels.

Families:
  base:   transfer_jaccard, transfer_onto, popularity, degree, rwr_string
  donor:  universal donor transfer per virtual-donor family (data/donors/*.npz)
  view:   seed similarity in each gene-embedding view (data/views/*.npz): mean and max cosine to the seeds
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from pathlib import Path

from . import data as D
from .methods import baselines as BL

ROOT = Path(__file__).resolve().parents[2]
DON = ROOT / "data/donors"
VIEWS = ROOT / "data/views"
CACHE = ROOT / "data/cache/channels"


def donor_families():
    return [l.split("\t")[0] for l in open(DON / "MANIFEST.tsv")]


def view_names():
    return [l.split("\t")[0] for l in open(VIEWS / "MANIFEST.tsv")]


def load_family(name):
    z = np.load(DON / f"{name}.npz")
    assert str(z["order"]) == "sorted_ensg"
    return sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))


def load_view(name):
    z = np.load(VIEWS / f"{name}.npz")
    assert str(z["order"]) == "sorted_ensg"
    return z["X"], z["mask"]


def donor_transfer(tasks, M, k=50):
    S = BL._indicator([t.seeds for t in tasks], M.shape[1])
    inter = (S @ M.T).toarray()
    sizes = M.getnnz(1)
    ns = np.array([len(t.seeds) for t in tasks])[:, None]
    jac = inter / np.maximum(ns + sizes[None] - inter, 1)
    out = np.zeros((len(tasks), M.shape[1]), np.float32)
    for i in range(len(tasks)):
        top = np.argsort(-jac[i])[:k]
        w = jac[i, top].astype(np.float32)
        if w.max() > 0:
            out[i] = np.asarray(M[top].T @ w).ravel()
    return out


def view_sim(tasks, X, mask, topk=3):
    """mean cosine to seeds and mean of the top-k cosines (robust max). Genes without the view get NaN."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-6)
    mean = np.full((len(tasks), X.shape[0]), np.nan, np.float32)
    mx = np.full_like(mean, np.nan)
    for i, t in enumerate(tasks):
        s = t.seeds[mask[t.seeds]]
        if len(s) == 0:
            continue
        C = Xn @ Xn[s].T                     # [n_genes, n_seeds]
        mean[i] = C.mean(1)
        k = min(topk, C.shape[1])
        mx[i] = np.sort(C, axis=1)[:, -k:].mean(1)
    mean[:, ~mask] = np.nan
    mx[:, ~mask] = np.nan
    return mean, mx


def compute(B, name: str, split_tag: str) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{B.name}_{split_tag}_{name}.npy"
    if f.exists():
        return np.load(f)
    tasks = B.tasks
    if name in ("transfer_jaccard", "transfer_onto", "popularity", "degree", "rwr_string"):
        S = getattr(BL, name)(tasks, B)
    elif name.startswith("don_"):
        S = donor_transfer(tasks, load_family(name[4:]))
    elif name.startswith("vmean_") or name.startswith("vmax_"):
        v = name.split("_", 1)[1]
        X, m = load_view(v)
        mean, mx = view_sim(tasks, X, m)
        np.save(CACHE / f"{B.name}_{split_tag}_vmean_{v}.npy", mean)
        np.save(CACHE / f"{B.name}_{split_tag}_vmax_{v}.npy", mx)
        S = mean if name.startswith("vmean_") else mx
    else:
        from . import ext
        reg = ext.registry()
        if name not in reg:
            raise KeyError(name)
        S = reg[name](B, split_tag)
    S = np.asarray(S, np.float32)
    np.save(f, S)
    return S


def all_channel_names(include=("base", "donor", "view")):
    """include: groups 'base','donor','view' and/or explicit channel names (e.g. ext channels)."""
    out = [n for n in include if n not in ("base", "donor", "view")]
    if "base" in include:
        out += ["transfer_jaccard", "transfer_onto", "popularity", "degree", "rwr_string"]
    if "donor" in include:
        out += ["don_" + f for f in donor_families()]
    if "view" in include:
        for v in view_names():
            out += ["vmean_" + v, "vmax_" + v]
    return out
