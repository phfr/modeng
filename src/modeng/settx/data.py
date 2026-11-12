"""settx inputs: gene view matrix and per-episode context features, cached to data/cache/settx.

Gene inputs: the 15 views in data/views (per-view standardised, missing rows = 0, plus a missing-view mask).
  view sets: "clean" = seq / pre21 / annot views (temporal-clean), "all" = + modern expression / Perturb-seq views
  (transductive on temporal), "seq" = sequence-only views (esmc, prott5, esm2_15b, phylo_npp): the only views dated
  before 19.04 in content, i.e. also clean for the `past` training episodes (go_t0 / reactome_t0 / depmap20q4 postdate 19.04).
Episode context features (per (episode, gene)): transfer_jaccard on the episode's masked context, popularity on the
masked context (hidden pairs subtracted), network degree. Computed with the harness baselines so they match the
stacker's channels exactly.
Episodes: `past` (19.04 -> 21.04 arrivals, context 19.04, all <= T0) for training; temporal dev / confirm for inference
(context 21.04, harness masking).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from .. import tasks as T, channels as CH
from ..methods import baselines as BL

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data/cache/settx"
CLEAN_KINDS = ("seq", "pre21", "annot")


def view_table():
    rows = [l.rstrip("\n").split("\t") for l in open(CH.VIEWS / "MANIFEST.tsv")]
    return [(r[0], r[1], int(r[2])) for r in rows]


def gene_inputs(viewset="all", permute_seed=None):
    """-> list of per-view X (float32 [n_genes, d]) and mask [n_genes, n_views]. permute_seed: negative control,
    rows of every view permuted with one shared gene permutation (breaks gene identity <-> features)."""
    Xs, Ms, names = [], [], []
    for name, kind, d in view_table():
        if viewset == "clean" and kind not in CLEAN_KINDS:
            continue
        if viewset == "seq" and kind != "seq":
            continue
        X, m = CH.load_view(name)
        X = X.astype(np.float32).copy()
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-6
        X = (X - mu) / sd
        X[~m] = 0
        X = np.clip(X, -6, 6)
        Xs.append(X); Ms.append(m); names.append(name)
    M = np.stack(Ms, 1).astype(np.float32)
    if permute_seed is not None:
        p = np.random.default_rng(permute_seed).permutation(M.shape[0])
        Xs = [X[p] for X in Xs]; M = M[p]
    return Xs, M, names


def episode_features(bench="past", split="all"):
    """Cached dict: tj [n, G] transfer_jaccard, pop [n, G] popularity (float32), ids."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"feat_{bench}_{split}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {k: z[k] for k in z.files}
    B = T.load(bench)
    if split != "all":
        B = B.split(split)
    if bench == "temporal":
        tj = CH.compute(B, "transfer_jaccard", split)
        pop = CH.compute(B, "popularity", split)
    else:
        tj = BL.transfer_jaccard(B.tasks, B)
        pop = BL.popularity(B.tasks, B)
    out = dict(tj=tj.astype(np.float32), pop=pop.astype(np.float32),
               ids=np.array([t.disease_id for t in B.tasks]))
    np.savez(f, **out)
    return out


def aux_episodes(sources=("ctx1904", "hpo2101", "msig_go", "msig_reactome", "msig_wiki", "impc13", "complexportal"),
                 n_max=2000, frac=0.3, min_size=10, max_size=500, seed=0):
    """Self-supervised set-completion episodes (all inputs <= T0): hide `frac` of a set, rank the rest.
    ctx1904 = rows of the OT 19.04 context (past benchmark's context), harness family masking (descendants and
    near-duplicates fully masked, ancestors re-propagated); donor families: own row masked. Context features (transfer,
    popularity) are computed on the episode's own family exactly as for benchmark tasks. Cached."""
    import pickle
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"aux_{'-'.join(sources)}_{n_max}_{seed}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    rng = np.random.default_rng(seed)
    P = T.load("past")
    tasks_all, tj_all, pop_all, src_all = [], [], [], []
    for src in sources:
        if src == "ctx1904":
            L, ids = P.context, P.context_ids
        else:
            L = CH.load_family(src).tocsr(); ids = pd.Index([f"{src}:{i}" for i in range(L.shape[0])])
        n = L.getnnz(1)
        rows = np.flatnonzero((n >= min_size) & (n <= max_size))
        if len(rows) > n_max:
            rows = np.sort(rng.choice(rows, n_max, replace=False))
        tasks = []
        for i in rows:
            g = L[i].indices.astype(np.int64)
            hid = np.sort(rng.choice(g, max(1, int(round(frac * len(g)))), replace=False))
            seeds = np.setdiff1d(g, hid)
            tasks.append(T.Task(ids[i], ids[i], "aux", seeds, hid, seeds, cluster=src))
        if src == "ctx1904":
            T.attach_families(tasks, L, ids, "family")
        else:
            for t, i in zip(tasks, rows):
                t.fam_rows = np.array([i]); t.mask_r = np.full(len(t.targets), i, np.int64); t.mask_c = t.targets.copy()
        Bx = T.Benchmark(src, tasks, L.astype(np.float32).tocsr(), ids, P.network, P.union_network, "aux", P.degree)
        tj_all.append(BL.transfer_jaccard(tasks, Bx)); pop_all.append(BL.popularity(tasks, Bx))
        tasks_all += tasks; src_all += [src] * len(tasks)
        print(f"aux {src}: {len(tasks)} episodes", flush=True)
    out = dict(tasks=tasks_all, tj=np.concatenate(tj_all).astype(np.float32), pop=np.concatenate(pop_all).astype(np.float32),
               src=np.array(src_all))
    with open(f, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return out
