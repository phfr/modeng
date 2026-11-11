"""Benchmarks: lists of `Task`s plus a shared `context` label matrix with per-task leakage masks.

Three benchmarks (leakage class in brackets, see CLAUDE.md):
  temporal  [temporal-clean]  modet's all-gene STRING panel (`panel_allgene_t0_string115_phys_notm_400`): seeds = OT 21.04
            genes, targets = 21.04 -> 25.09 arrivals, unknown (weak evidence) excluded. Non-sealed diseases only (191);
            split dev = modet's old dev diseases (53), confirm = the rest (138). Context = OT 21.04 sets.
            Secondary target set `sa` = modet strict-a endpoint (arrival class A, persistent; other test genes removed
            from the ranking; >= 5 kept positives). Networks: T0 STRING 11.5 (degree / RWR).
  forward   [temporal-clean w.r.t. labels; inputs up to 25.09]  OT 25.09 -> 26.09: seeds = 25.09 genes (5..1000),
            targets = 26.09 genes not in 25.09 (>= 3). Context = OT 25.09 sets. Network: STRING v12 phys no-TM >= 0.4.
  holdout   [holdout-current]  OT 26.09 diseases with 15..1000 genes, <= 600 sampled (seed 0); per disease a random 20 %
            of its genes are hidden (targets), the rest are seeds. Context = OT 26.09 sets.
dev / confirm for forward / holdout: sha1(disease_id) parity (deterministic, ~50/50).

Secondary target sets (`Task.alt`, metrics prefixed): `onnet` (all benchmarks; only network genes ranked, see
`add_onnet`) and `sa` (temporal: modet strict-a).

Sealed policy: ids in `data.excluded_task_ids()` (modet sealed diseases + their ontology relatives that are not non-sealed
modet panel diseases) are NEVER tasks. In the temporal context they are also removed as rows (so nothing about the sealed
fold enters a temporal-clean run); in forward / holdout they stay as context rows (other diseases' knowledge).

Leakage masking (`Task.mask_r`, `Task.mask_c` = context (row, gene) pairs zeroed in `Benchmark.context_for(task)`):
  holdout ("family" scope): OT propagates genes to ancestors and the hidden gene of d may come from a descendant.
      (1) `fam_rows` = the disease, its ontology descendants (21.04 + 25.09 + 26.09 union) and near-duplicate rows
      (Jaccard(context row, seeds+targets) >= 0.5, modet's family rule): all their target pairs are zeroed.
      (2) ancestors are RE-PROPAGATED, not blanket-masked: ancestor a keeps target g iff some descendant of a outside
      fam_rows still has g (sibling evidence); otherwise (a, g) is zeroed. Blanket masking would leave a tell (gene in a
      sibling's row but missing from the common ancestor = hidden), which a learned method could exploit. Approximation:
      a's own direct evidence for g is not recoverable from propagated sets, so it is dropped (conservative).
  temporal / forward ("self+desc" scope): the context is a PAST snapshot, so it cannot contain future evidence; masking
      the whole family would delete legitimate past knowledge (e.g. 71 % of forward targets are already in some ancestor's
      25.09 set via a sibling). Masked: the disease's own row and its descendants' rows (any release), i.e. genes that only
      "arrive" because of ontology rewiring / a different propagation ontology. (temporal: 287 of 6,324 targets are in
      the disease's own 21.04 row, because modet propagates T0 evidence with the 25.09 ontology.)
  forward seeds are re-propagated with the 26.09 ontology: seeds = 25.09 genes of d and of its 26.09 descendants, so a
      gene that only arrives through a new descendant link is a seed, not a target (12 % of raw targets).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import data as D

BENCHES = ("temporal", "forward", "holdout")
LEAKAGE_CLASS = {"temporal": "temporal-clean", "forward": "temporal-clean (labels 25.09->26.09; inputs <= 25.09)",
                 "holdout": "holdout-current"}
NETWORK = {"temporal": "t0_string115_phys_notm_400", "forward": "string_phys_notm_400", "holdout": "string_phys_notm_400"}
UNION = {"temporal": "t0_string115_phys_notm_400+t0_biogrid4197_phys+huri",
         "forward": "union_string_biogrid_huri", "holdout": "union_string_biogrid_huri"}
NEARDUP_JACCARD = 0.5


@dataclass
class Task:
    disease_id: str
    name: str
    split: str
    seeds: np.ndarray            # universe indices
    targets: np.ndarray          # held-out positives
    exclude: np.ndarray          # genes removed from the ranking (seeds, unknown); never contains targets
    fam_rows: np.ndarray = None  # context rows whose (row, targets) pairs are all masked
    mask_r: np.ndarray = None    # masked context pairs (row, gene), only pairs present in the context
    mask_c: np.ndarray = None
    cluster: str = ""            # resampling unit for paired bootstrap
    alt: dict = field(default_factory=dict)   # secondary target sets: name -> (targets, exclude)


@dataclass
class Benchmark:
    name: str
    tasks: list
    context: sp.csr_matrix       # disease x gene (float32 0/1), UNMASKED; use context_for(task)
    context_ids: pd.Index
    network: str
    union_network: str
    leakage_class: str
    degree: np.ndarray = None    # metric degree (nuAP weights) = degree in `network`

    @property
    def n_genes(self) -> int:
        return self.context.shape[1]

    def split(self, which: str) -> "Benchmark":
        tasks = self.tasks if which == "all" else [t for t in self.tasks if t.split == which]
        return Benchmark(self.name, tasks, self.context, self.context_ids, self.network, self.union_network,
                         self.leakage_class, self.degree)

    def own_row(self, task: Task) -> int:
        """Context row of the task's own disease, -1 if absent."""
        i = self.context_ids.get_indexer([task.disease_id])[0]
        return int(i)

    def mask_matrix(self, task: Task) -> sp.csr_matrix:
        """0/1 matrix with ones at the masked (row, gene) pairs (all of them are 1 in the context)."""
        return sp.csr_matrix((np.ones(len(task.mask_r), np.float32), (task.mask_r, task.mask_c)), shape=self.context.shape)

    def context_for(self, task: Task) -> sp.csr_matrix:
        """The context label matrix as this task may see it (hidden pairs zeroed / re-propagated)."""
        L = self.context
        if task.mask_r is None or len(task.mask_r) == 0:
            return L
        out = (L - self.mask_matrix(task)).tocsr()
        out.eliminate_zeros()
        return out

    def hidden_counts(self, task: Task) -> np.ndarray:
        """Per gene: number of context rows in which it is masked for this task (for column-sum corrections)."""
        return np.bincount(task.mask_c, minlength=self.n_genes).astype(np.float32)

    def row_masked_counts(self, task: Task) -> np.ndarray:
        """Per context row: number of masked genes (for row-size corrections)."""
        return np.bincount(task.mask_r, minlength=self.context.shape[0]).astype(np.float32)

    def summary(self) -> dict:
        rows = []
        for s in ("dev", "confirm"):
            ts = [t for t in self.tasks if t.split == s]
            rows.append(dict(bench=self.name, split=s, n_tasks=len(ts),
                             median_seeds=float(np.median([len(t.seeds) for t in ts])) if ts else np.nan,
                             median_targets=float(np.median([len(t.targets) for t in ts])) if ts else np.nan,
                             median_fam_rows=float(np.median([len(t.fam_rows) for t in ts])) if ts else np.nan,
                             n_context_rows=self.context.shape[0], network=self.network,
                             leakage=self.leakage_class))
        return rows


def hash_split(disease_id: str) -> str:
    return "dev" if int(hashlib.sha1(disease_id.encode()).hexdigest(), 16) % 2 == 0 else "confirm"


def _indicator(sets: list, n: int) -> sp.csr_matrix:
    r = np.concatenate([np.full(len(s), i) for i, s in enumerate(sets)]) if sets else np.array([], int)
    c = np.concatenate(sets) if sets else np.array([], int)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(sets), n))


def _pairs(L: sp.csr_matrix, rows: np.ndarray, genes: np.ndarray):
    """(row, gene) pairs among rows x genes that are 1 in L."""
    if len(rows) == 0 or len(genes) == 0:
        return np.array([], np.int64), np.array([], np.int64)
    sub = L[rows][:, genes].tocoo()
    return rows[sub.row].astype(np.int64), genes[sub.col].astype(np.int64)


def attach_families(tasks: list, L: sp.csr_matrix, ids: pd.Index, scope: str = "family") -> None:
    """Fill `fam_rows` and the masked pairs `mask_r` / `mask_c` (see module docstring).
    scope "self+desc": own row + ontology descendants (any release), all target pairs masked.
    scope "family": own + descendants + near-duplicates fully masked, ancestors re-propagated from the other descendants."""
    pos = {d: i for i, d in enumerate(ids)}
    L = L.tocsr()
    jac = None
    if scope == "family":
        X = _indicator([np.union1d(t.seeds, t.targets) for t in tasks], L.shape[1])
        inter = (X @ L.T).toarray()
        sz_t = np.asarray(X.sum(1)).ravel()[:, None]
        sz_l = np.asarray(L.sum(1)).ravel()[None, :]
        jac = inter / np.maximum(sz_t + sz_l - inter, 1)
    else:
        assert scope == "self+desc", scope
    for k, t in enumerate(tasks):
        fam = {pos[d] for d in D.descendants(t.disease_id) if d in pos}
        if t.disease_id in pos:
            fam.add(pos[t.disease_id])
        if jac is not None:
            fam |= set(np.flatnonzero(jac[k] >= NEARDUP_JACCARD).tolist())
        t.fam_rows = np.array(sorted(fam), dtype=np.int64)
        r, c = _pairs(L, t.fam_rows, t.targets)
        rs, cs = [r], [c]
        if scope == "family":
            # bottom-up (fewest descendants first) so an intermediate ancestor's masking is seen by the ones above it
            anc = sorted({pos[a] for a in D.relatives(t.disease_id) if a in pos} - fam,
                         key=lambda a: len(D.descendants(ids[a])))
            masked: dict[int, set] = {}
            for a in anc:
                tg = t.targets[np.asarray(L[a][:, t.targets].todense()).ravel() > 0]
                if len(tg) == 0:
                    continue
                sup = sorted({pos[x] for x in D.descendants(ids[a]) if x in pos} - fam)
                plain = np.array([x for x in sup if x not in masked], dtype=np.int64)
                has = (np.asarray(L[plain][:, tg].sum(0)).ravel() > 0) if len(plain) else np.zeros(len(tg), bool)
                for x in sup:
                    if x in masked:
                        row = np.asarray(L[x][:, tg].todense()).ravel() > 0
                        has |= row & ~np.isin(tg, list(masked[x]))
                drop = tg[~has]
                masked[a] = set(drop.tolist())
                rs.append(np.full(len(drop), a, np.int64)); cs.append(drop.astype(np.int64))
        t.mask_r, t.mask_c = np.concatenate(rs), np.concatenate(cs)


def _cluster_map() -> dict:
    f = D.modet_folds()
    return {d: f"c{c}" for d, c in f["cluster"].items()}


# ------------------------------------------------------------------------------------------------------- temporal
def build_temporal() -> Benchmark:
    panel = pd.read_parquet(D.GT / "panel_allgene_t0_string115_phys_notm_400.parquet")
    excl = D.excluded_task_ids()
    panel = panel[~panel.disease_id.isin(excl)].reset_index(drop=True)
    L, ids = D.label_matrix("2104")
    keep = ~ids.isin(excl)                    # sealed fold (and relatives) never enter the temporal context
    L, ids = L[np.flatnonzero(keep)].tocsr(), ids[keep]
    arr = pd.read_parquet(D.GT / "arrival_devall_t0_string115_phys_notm_400.parquet")
    sa = arr[(arr["arrival"] == "A") & (arr["persistent"].astype(bool) | arr["margin_ok"].astype(bool))]
    sa_keep = sa.groupby("disease_id")["gene"].apply(set).to_dict()
    clus = _cluster_map()
    tasks = []
    for r in panel.itertuples():
        seeds = D.gene_idx(r.seeds)
        targets = np.setdiff1d(D.gene_idx(r.test), seeds)
        exclude = np.setdiff1d(np.union1d(seeds, D.gene_idx(r.unknown)), targets)
        t = Task(r.disease_id, r.name, "dev" if r.split == "dev" else "confirm", seeds, targets, exclude,
                 cluster=clus.get(r.disease_id, r.disease_id))
        kept = D.gene_idx(sa_keep.get(r.disease_id, set()))
        kept = np.intersect1d(kept, targets)
        if len(kept) >= 5:
            t.alt["sa"] = (kept, np.union1d(exclude, np.setdiff1d(targets, kept)))
        tasks.append(t)
    attach_families(tasks, L, ids, "self+desc")
    return Benchmark("temporal", tasks, L, ids, NETWORK["temporal"], UNION["temporal"], LEAKAGE_CLASS["temporal"])


# -------------------------------------------------------------------------------------------------------- forward
def build_forward(min_seeds=5, max_seeds=1000, min_new=3) -> Benchmark:
    L0, ids0 = D.label_matrix("2509")
    L1, ids1 = D.label_matrix("2609")
    excl = D.excluded_task_ids()
    i1 = {d: i for i, d in enumerate(ids1)}
    names = D.disease_names("2609")
    tasks = []
    i0 = {d: i for i, d in enumerate(ids0)}
    for i, d in enumerate(ids0):
        if d in excl or d not in i1:
            continue
        rows = [i] + [i0[x] for x in D.descendants(d, ("2609",)) if x in i0]
        seeds = np.unique(L0[rows].indices).astype(np.int64)
        new = np.setdiff1d(L1[i1[d]].indices, seeds).astype(np.int64)
        if not (min_seeds <= len(seeds) <= max_seeds and len(new) >= min_new):
            continue
        tasks.append(Task(d, str(names.get(d, d)), hash_split(d), np.sort(seeds), new, np.sort(seeds), cluster=d))
    attach_families(tasks, L0, ids0, "self+desc")
    return Benchmark("forward", tasks, L0, ids0, NETWORK["forward"], UNION["forward"], LEAKAGE_CLASS["forward"])


# -------------------------------------------------------------------------------------------------------- holdout
def build_holdout(min_genes=15, max_genes=1000, n_max=600, frac=0.2, seed=0) -> Benchmark:
    L, ids = D.label_matrix("2609")
    excl = D.excluded_task_ids()
    names = D.disease_names("2609")
    n = L.getnnz(1)
    cand = [i for i in range(len(ids)) if min_genes <= n[i] <= max_genes and ids[i] not in excl]
    rng = np.random.default_rng(seed)
    if len(cand) > n_max:
        cand = sorted(rng.choice(cand, n_max, replace=False).tolist())
    tasks = []
    for i in cand:
        g = np.sort(L[i].indices.astype(np.int64))
        r = np.random.default_rng([seed, int(hashlib.sha1(ids[i].encode()).hexdigest()[:8], 16)])
        k = max(1, int(round(frac * len(g))))
        hid = np.sort(r.choice(g, k, replace=False))
        seeds = np.setdiff1d(g, hid)
        tasks.append(Task(ids[i], str(names.get(ids[i], ids[i])), hash_split(ids[i]), seeds, hid, seeds, cluster=ids[i]))
    attach_families(tasks, L, ids)
    return Benchmark("holdout", tasks, L, ids, NETWORK["holdout"], UNION["holdout"], LEAKAGE_CLASS["holdout"])


# ----------------------------------------------------------------------------------------------------------- past
def build_past(min_seeds=3, max_seeds=1000, min_new=3) -> Benchmark:
    """TRAINING episodes for the temporal benchmark (all information <= T0): seeds = OT 19.04 sets (propagated over the
    21.04 descendants), targets = genes new in 21.04, context = 19.04 matrix. Sealed ids and relatives excluded.
    Every task has split 'train'. Never an evaluation benchmark."""
    L0, ids0 = D.label_matrix("1904")
    L1, ids1 = D.label_matrix("2104")
    excl = D.excluded_task_ids()
    keep = ~ids0.isin(excl)
    L0, ids0 = L0[np.flatnonzero(keep)].tocsr(), ids0[keep]
    i1 = {d: i for i, d in enumerate(ids1)}
    i0 = {d: i for i, d in enumerate(ids0)}
    names = D.disease_names("2104") if True else {}
    clus = _cluster_map()
    tasks = []
    for i, d in enumerate(ids0):
        if d not in i1:
            continue
        rows = [i] + [i0[x] for x in D.descendants(d, ("2104",)) if x in i0]
        seeds = np.unique(L0[rows].indices).astype(np.int64)
        new = np.setdiff1d(L1[i1[d]].indices, seeds).astype(np.int64)
        if not (min_seeds <= len(seeds) <= max_seeds and len(new) >= min_new):
            continue
        tasks.append(Task(d, str(names.get(d, d)), "train", np.sort(seeds), new, np.sort(seeds), cluster=clus.get(d, d)))
    attach_families(tasks, L0, ids0, "self+desc")
    return Benchmark("past", tasks, L0, ids0, NETWORK["temporal"], UNION["temporal"], "training only (<= T0)")


BUILDERS = {"past": build_past, "temporal": build_temporal, "forward": build_forward, "holdout": build_holdout}


@lru_cache(None)
def load(bench: str) -> Benchmark:
    """Build (cached to data/cache/bench_<name>.pkl) a benchmark with its metric degree attached."""
    import pickle
    f = D._cache(f"bench_{bench}.pkl")
    if f.exists():
        with open(f, "rb") as fh:
            b = pickle.load(fh)
    else:
        b = BUILDERS[bench]()
        with open(f, "wb") as fh:
            pickle.dump(b, fh)
    b.degree = D.degree(b.network)
    add_onnet(b)
    return b


def add_onnet(b: Benchmark) -> None:
    """Secondary target set `onnet` (modet's candidate set): only genes in the benchmark network (degree > 0) are ranked
    and counted; off-network targets are dropped. On temporal this reproduces modet's covered panel exactly for
    network-only scorers (scripts/check_modet_parity.py). The main metrics stay on the all-gene universe."""
    off = np.flatnonzero(b.degree == 0)
    for t in b.tasks:
        tg = np.setdiff1d(t.targets, off)
        if len(tg):
            t.alt["onnet"] = (tg, np.union1d(t.exclude, np.setdiff1d(off, tg)))
