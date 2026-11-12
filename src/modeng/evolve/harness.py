"""Target-free view of a benchmark for evolved candidates.

Candidates get `SafeBench` / `SafeTask` objects: seeds, disease id and the exactly masked context (`B.context_for`) only.
Targets, exclude lists, alt target sets and the mask pairs (which name target genes) are not attributes of what candidates
see. Allowed inputs: seeds, masked context, ontology (release-matched to the benchmark's seed date), network degree.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from .. import data as D

# ontology releases a benchmark may use (seed date): past/temporal = 21.04, forward = <= 25.09, holdout = all (current)
ONTO = {"past": ("2104",), "temporal": ("2104",), "forward": ("2104", "2509"), "holdout": ("2104", "2509", "2609")}


@dataclass(frozen=True)
class SafeTask:
    idx: int
    disease_id: str
    seeds: np.ndarray


class SafeBench:
    def __init__(self, B):
        self._B = B
        self.name = B.name
        self.n_genes = B.n_genes
        self.context_ids = B.context_ids
        self.n_rows = B.context.shape[0]
        self.degree = np.asarray(B.degree, np.float32)          # network degree (metric network)
        self.tasks = [SafeTask(i, t.disease_id, np.asarray(t.seeds)) for i, t in enumerate(B.tasks)]
        self._pos = {d: i for i, d in enumerate(B.context_ids)}

    def L(self, t: SafeTask) -> sp.csr_matrix:
        """Context label matrix (diseases x genes, 0/1 float32) exactly as this task may see it."""
        return self._B.context_for(self._B.tasks[t.idx])

    def own_row(self, t: SafeTask) -> int:
        return self._pos.get(t.disease_id, -1)

    def relatives(self, t: SafeTask) -> np.ndarray:
        """Context rows of ontology ancestors/descendants (release-matched), own row excluded."""
        rel = _rel(ONTO[self.name]).get(t.disease_id, frozenset())
        return np.array(sorted(self._pos[d] for d in rel if d in self._pos and d != t.disease_id), dtype=np.int64)


@lru_cache(None)
def _rel(releases):
    return D.relatives_map(releases)


def indicator(sets, n):
    r = np.concatenate([np.full(len(s), i) for i, s in enumerate(sets)])
    c = np.concatenate(sets)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(sets), n))
