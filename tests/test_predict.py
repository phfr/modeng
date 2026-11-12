"""modeng-v2 inference (`modeng.predict`): two ad-hoc diseases, CLI smoke test, sealed refusal, and the reproducibility check
(predict on a temporal dev disease with the benchmark seeds + mask == benchmark-path score of the saved model, within 1e-6).
Needs the saved model (scripts/train_v2.py). Run: PYTHONPATH=src python -m pytest -q tests/test_predict.py   (~2-4 min)"""
import json

import numpy as np
import pandas as pd
import pytest

from modeng import data as D, tasks as T
from modeng import predict as P

DISEASES = ("EFO_0000270", "EFO_0000685")      # asthma, rheumatoid arthritis (not sealed-related)


@pytest.fixture(scope="module")
def model():
    cfg, _ = P.load_config()
    if not (P.default_model_dir(cfg) / "model.txt").exists():
        pytest.skip("modeng-v2 model not trained (scripts/train_v2.py)")
    return P.Model()


@pytest.mark.parametrize("d", DISEASES)
def test_predict_adhoc(model, d):
    r = P.predict(model, d)
    t, tab, s = r["task"], r["table"], r["scores"]
    L, ids = D.label_matrix("2104")
    assert np.array_equal(t.seeds, np.sort(L[ids.get_loc(d)].indices))          # seeds = own 21.04 set
    assert len(tab) == len(s) - len(t.seeds) and not np.isin(D.gene_idx(tab.gene_id), t.seeds).any()
    assert (np.diff(tab.score.to_numpy()) <= 0).all()
    assert tab.calib_p.between(0, model.shrink + 1e-9).all()
    assert tab.calib_p.is_monotonic_decreasing
    # own row reduced to the seeds = nothing to mask (seeds are the release's own set; descendants are subsets)
    assert len(t.mask_r) == 0
    # contributions: groups + bias reproduce the raw margin
    X = P.features(r["bench"], model.channels)
    raw = model.booster.predict(X[:200], raw_score=True)
    full = model.booster.predict(X[:200], pred_contrib=True)
    assert np.allclose(r["contrib"].to_numpy()[:200].sum(1) + full[:, -1], raw, atol=1e-6)
    # a reasonable ranking: top 100 candidates hold more 25.09 arrivals than random
    L1, ids1 = D.label_matrix("2509")
    new = np.setdiff1d(L1[ids1.get_loc(P.resolve_id(d, "2509"))].indices, t.seeds)
    hits = np.isin(D.gene_idx(tab.gene_id.head(100)), new).sum()
    assert hits > 100 * len(new) / len(tab) * 3


def test_cli_with_module(model, tmp_path):
    out, mod = tmp_path / "ranked.tsv", tmp_path / "module.tsv"
    seeds = tmp_path / "seeds.txt"
    seeds.write_text("# asthma core\nIL13\nIL4R\nTSLP\nIL33\nORMDL3\nGSDMB\nENSG00000113302\n")    # last = IL12B
    P.main(["--disease", "EFO:0000270", "--seeds", str(seeds), "--out", str(out), "--module", str(mod), "--tau", "0.2", "--no-subex"])
    tab = pd.read_csv(out, sep="\t")
    side = json.loads((tmp_path / "ranked.tsv.json").read_text())
    assert side["n_seeds"] == 7 and side["disease_id"] == "EFO_0000270"
    assert side["n_masked_pairs"] > 0                       # own asthma row has genes beyond these 7 seeds: masked
    assert {"gene_id", "symbol", "score", "calib_p", "top_witnesses"} <= set(tab.columns)
    m = pd.read_csv(mod, sep="\t")
    assert (m.role == "seed").sum() == 7 and (m.loc[m.role == "added", "calib_p"] >= 0.2).all()


def test_resolve_id():
    assert P.resolve_id("EFO:0000270", "2609") == "MONDO_0004979"
    assert P.resolve_id("MONDO_0004979", "2104") == "EFO_0000270"


def test_sealed_refused(model):
    d = sorted(D.sealed_ids())[0]
    with pytest.raises(PermissionError):
        P.predict(model, d, seeds=np.arange(10))


def test_reproduces_benchmark_score(model):
    """Ad-hoc predict on a temporal dev disease (benchmark seeds + the benchmark's masked genes) == benchmark-path score."""
    Bd = T.load("temporal").split("dev")
    ref = np.load(model.dir / "scores_temporal_dev.npy")
    L, ids = P.context("2104")
    assert (L != Bd.context).nnz == 0 and ids.equals(Bd.context_ids)
    for i in (0, 7):
        t = Bd.tasks[i]
        r = P.predict(model, t.disease_id, seeds=t.seeds, mask_genes=t.targets, contrib=False)
        assert np.array_equal(r["task"].mask_r, t.mask_r) and np.array_equal(r["task"].mask_c, t.mask_c)
        assert np.abs(r["scores"] - ref[i]).max() < 1e-6, (t.disease_id, np.abs(r["scores"] - ref[i]).max())
