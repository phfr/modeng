"""Method registry: name -> callable `score(tasks, B) -> np.ndarray [n_tasks, n_genes]`.

Add a method by writing the function anywhere under `modeng.methods` and registering it here (or with `@register`).
`benches` restricts a method to some benchmarks (others are skipped by `run_all`)."""
from __future__ import annotations

from . import baselines as _b

METHODS: dict = {}
BENCHES: dict = {}


def register(name: str, benches=None):
    def deco(fn):
        METHODS[name] = fn
        if benches:
            BENCHES[name] = tuple(benches)
        return fn
    return deco


for _name in ("random", "degree", "popularity", "rwr_string", "rwr_union", "transfer_jaccard", "transfer_onto"):
    register(_name)(getattr(_b, _name))
register("modet_headline", benches=("temporal",))(_b.modet_headline)


def get(name: str):
    if name not in METHODS:
        raise KeyError(f"unknown method {name!r}; registered: {sorted(METHODS)}")
    return METHODS[name]
