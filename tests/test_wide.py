"""temporal_wide: construction rules, frozen-fold guard, leakage masks, split independence, stack guard.
Run: PYTHONPATH=src python -m pytest -q tests/test_wide.py"""
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from modeng import data as D
from modeng import tasks as T

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def W():
    return T.load("temporal_wide")


@pytest.fixture(scope="module")
def temporal():
    return T.load("temporal")


def test_frozen_fold_guard(W, monkeypatch):
    monkeypatch.delenv(T.WIDE_CONFIRM_FLAG, raising=False)
    with pytest.raises(PermissionError):
        W.split("wide_confirm")
    assert not any(t.split == "wide_confirm" for t in W.split("all").tasks)
    monkeypatch.setenv(T.WIDE_CONFIRM_FLAG, "1")
    assert len(W.split("wide_confirm").tasks) > 0


def test_frozen_ids_match(W, monkeypatch):
    """data/wide_confirm_ids.txt (hash in docs/preregistration_wide.sha256) == the rebuilt wide_confirm ids."""
    f = ROOT / "data" / "wide_confirm_ids.txt"
    if not f.exists():
        pytest.skip("not frozen yet")
    frozen = [l.strip() for l in open(f) if l.strip() and not l.startswith("#")]
    monkeypatch.setenv(T.WIDE_CONFIRM_FLAG, "1")
    assert sorted(t.disease_id for t in W.split("wide_confirm").tasks) == sorted(frozen)
    sha = (ROOT / "docs" / "preregistration_wide.sha256").read_text()
    assert hashlib.sha256(f.read_bytes()).hexdigest() in sha


def test_no_overlap_with_temporal_or_sealed(W, temporal):
    ids = {t.disease_id for t in W.tasks}
    assert len(ids) == len(W.tasks)
    assert not ids & {t.disease_id for t in temporal.tasks}
    assert not ids & set(D.excluded_task_ids())
    assert not W.context_ids.isin(list(D.sealed_ids())).any()
    assert W.context.shape == temporal.context.shape and (W.context != temporal.context).nnz == 0


def test_task_rules(W):
    for t in W.tasks:
        assert 5 <= len(t.seeds) <= 1000 and len(t.targets) >= 3
        assert len(np.intersect1d(t.seeds, t.targets)) == 0
        assert len(np.intersect1d(t.exclude, t.targets)) == 0
        assert len(np.setdiff1d(t.seeds, t.exclude)) == 0
        assert t.split in ("wide_train", "wide_dev", "wide_confirm")
        if "sa" in t.alt:
            kept, exc = t.alt["sa"]
            assert len(kept) >= 5 and len(np.setdiff1d(kept, t.targets)) == 0 and len(np.intersect1d(kept, exc)) == 0


def test_masking_no_leak(W):
    """Hidden targets are never in the task's context view for the disease or its ontology descendants."""
    pos = {d: i for i, d in enumerate(W.context_ids)}
    rng = np.random.default_rng(0)
    for k in rng.choice(len(W.tasks), 60, replace=False):
        t = W.tasks[k]
        C = W.context_for(t)
        assert C[t.fam_rows][:, t.targets].nnz == 0
        own = W.own_row(t)
        if own >= 0:
            assert C[own][:, t.targets].nnz == 0
        desc = [pos[d] for d in D.descendants(t.disease_id) if d in pos]
        if desc:
            assert C[desc][:, t.targets].nnz == 0
        assert (W.context - C).nnz == len(t.mask_r)


def test_eval_splits_unrelated(W, temporal, monkeypatch):
    """wide_dev / wide_confirm diseases are not ontology relatives of any temporal task, of each other across splits,
    or of non-hub wide_train diseases; hashed components never mix splits."""
    monkeypatch.setenv(T.WIDE_CONFIRM_FLAG, "1")
    tids = {t.disease_id for t in temporal.tasks}
    split = {t.disease_id: t.split for t in W.tasks}
    clus = {t.disease_id: t.cluster for t in W.tasks}
    for t in W.tasks:
        if t.split == "wide_train":
            continue
        rel = set(D.relatives(t.disease_id))
        assert not rel & tids, t.disease_id
        for r in rel & split.keys():
            # a relative in another split can only be a peeled hub (always wide_train)
            assert split[r] == t.split or clus[r].startswith("hub:"), (t.disease_id, r)
    by = {}
    for t in W.tasks:
        if t.cluster.startswith("comp:") and t.split != "wide_train":
            by.setdefault(t.cluster, set()).add(t.split)
    assert all(len(v) == 1 for v in by.values())


def test_relative_target_guard(W, temporal):
    from modeng.stack import relative_target_guard
    Wt = W.split("wide_train")
    evals = [temporal.split("dev"), temporal.split("confirm"), W.split("wide_dev")]
    drop, n = relative_target_guard(Wt, evals)
    assert n > 0
    by = {t.disease_id: t.targets for E in evals for t in E.tasks}
    for i, t in enumerate(Wt.tasks):
        for d in set(D.relatives(t.disease_id)) & by.keys():
            assert len(np.setdiff1d(by[d], drop[i])) == 0
