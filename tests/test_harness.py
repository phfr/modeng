"""Fast harness tests: masking leaks, metric sanity, shapes. Run: PYTHONPATH=src python -m pytest -q tests"""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from modeng import data as D
from modeng import metrics as M
from modeng import tasks as T
from modeng.methods import METHODS


@pytest.fixture(scope="module", params=T.BENCHES)
def bench(request):
    return T.load(request.param)


def test_no_sealed_tasks(bench):
    excl = D.excluded_task_ids()
    assert not any(t.disease_id in excl for t in bench.tasks)
    assert not any(t.disease_id in D.sealed_ids() for t in bench.tasks)
    if bench.name == "temporal":       # sealed fold never enters the temporal context
        assert not bench.context_ids.isin(list(D.sealed_ids())).any()


def test_task_sets_disjoint(bench):
    for t in bench.tasks:
        assert len(np.intersect1d(t.seeds, t.targets)) == 0
        assert len(np.intersect1d(t.exclude, t.targets)) == 0
        assert len(np.setdiff1d(t.seeds, t.exclude)) == 0
        assert len(t.targets) > 0


def test_masking_no_leak(bench):
    """A hidden target is never 1 in the task's context view for the disease, its descendants (and near-duplicates)."""
    rng = np.random.default_rng(0)
    pos = {d: i for i, d in enumerate(bench.context_ids)}
    for k in rng.choice(len(bench.tasks), min(40, len(bench.tasks)), replace=False):
        t = bench.tasks[k]
        C = bench.context_for(t)
        assert C[t.fam_rows][:, t.targets].nnz == 0
        own = bench.own_row(t)
        if own >= 0:
            assert C[own][:, t.targets].nnz == 0
        desc = [pos[d] for d in D.descendants(t.disease_id) if d in pos]
        if desc:
            assert C[desc][:, t.targets].nnz == 0
        # the view differs from the context only on the masked pairs
        assert (bench.context - C).nnz == len(t.mask_r)


def test_holdout_no_mask_tell():
    """holdout: a masked ancestor pair (a, g) never has g in any descendant of a in the view (otherwise the gap
    'in a sibling but missing from the common ancestor' would reveal the hidden gene)."""
    B = T.load("holdout")
    pos = {d: i for i, d in enumerate(B.context_ids)}
    fam_checked = 0
    for t in B.tasks[:80]:
        C = B.context_for(t)
        fam = set(t.fam_rows.tolist())
        for a, g in zip(t.mask_r, t.mask_c):
            if a in fam:
                continue
            desc = [pos[x] for x in D.descendants(B.context_ids[a]) if x in pos]
            assert not desc or C[desc][:, [g]].nnz == 0
            fam_checked += 1
    assert fam_checked > 0


def test_holdout_targets_in_original(bench):
    if bench.name != "holdout":
        pytest.skip()
    t = bench.tasks[0]
    assert bench.context[bench.own_row(t)][:, t.targets].nnz == len(t.targets)   # they really were known


def test_metrics_perfect_and_sklearn():
    n = 1000
    rng = np.random.default_rng(1)
    pos = rng.choice(n, 20, replace=False)
    exc = np.setdiff1d(rng.choice(n, 50, replace=False), pos)
    s = np.zeros(n); s[pos] = 1 + rng.random(20)
    m = M.rank_metrics(s, pos, exc, np.ones(n))
    assert m["ap"] == pytest.approx(1.0) and m["auroc"] == pytest.approx(1.0) and m["nuap"] == pytest.approx(1.0)
    assert m["recall@50"] == 1.0 and m["first_hit"] == 1 and m["ndcg@100"] == pytest.approx(1.0)
    s = rng.random(n)
    deg = rng.integers(0, 50, n)
    m = M.rank_metrics(s, pos, exc, deg)
    keep = np.ones(n, bool); keep[exc] = False
    y = np.zeros(n, bool); y[pos] = True
    assert m["ap"] == pytest.approx(average_precision_score(y[keep], s[keep]), abs=1e-9)
    assert m["auroc"] == pytest.approx(roc_auc_score(y[keep], s[keep]), abs=1e-9)
    w = np.ones(keep.sum()); d = np.maximum(deg[keep], 1).astype(float); wp = 1 / d[y[keep]]; w[y[keep]] = wp / wp.mean()
    assert m["nuap"] == pytest.approx(average_precision_score(y[keep], s[keep], sample_weight=w), abs=1e-9)


def test_constant_scores_not_good():
    n = 2000
    pos = np.arange(10)
    m = M.rank_metrics(np.zeros(n), pos, np.array([], int), np.ones(n))
    assert m["ap"] < 0.05


@pytest.mark.parametrize("name", ["random", "degree", "popularity", "transfer_jaccard"])
def test_method_shapes(name):
    B = T.load("temporal").split("dev")
    S = METHODS[name](B.tasks[:5], B)
    assert S.shape == (5, B.n_genes)
