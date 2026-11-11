"""Ranking metrics (one score vector over the universe, held-out positives, excluded genes).

Conventions follow modet (`modet/src/modet_bench/metrics.py`):
  * excluded genes (seeds, unknown) are removed from the ranking entirely;
  * ties are broken by a tiny seeded jitter (rng(0), uniform 1e-12) so constant scores do not look good;
  * nuAP (`node_uniform_auprc`): each positive weighted 1 / max(degree, 1), normalised to mean 1 over the positives,
    negatives weight 1; degree = unweighted degree in the benchmark network, off-network genes degree 0 -> weight 1
    (modet `universe.nu_weight`). Our implementation sorts once and computes all metrics from the same order; it equals
    sklearn's average_precision_score (tested).
Non-finite scores are treated as the lowest score.
"""
from __future__ import annotations

import numpy as np

KS = (50, 100, 200)
METRICS = ("ap", "nuap", "auroc", "recall@50", "recall@100", "recall@200", "ndcg@100", "first_hit", "n_pos", "n_cand")


def _weighted_ap(yo: np.ndarray, wo: np.ndarray) -> float:
    """AP of an ordered 0/1 vector with sample weights (sklearn step-wise definition, no ties)."""
    tp = np.cumsum(wo * yo)
    fp = np.cumsum(wo * (1 - yo))
    prec = tp / np.maximum(tp + fp, 1e-300)
    return float((prec * wo * yo).sum() / (wo * yo).sum())


def rank_metrics(scores: np.ndarray, pos: np.ndarray, exclude: np.ndarray, degree: np.ndarray | None = None) -> dict:
    n = len(scores)
    mask = np.ones(n, bool)
    mask[exclude] = False
    y = np.zeros(n, bool)
    y[pos] = True
    y &= mask
    s = np.asarray(scores, float)[mask]
    y = y[mask]
    npos = int(y.sum())
    out = dict.fromkeys(METRICS, np.nan)
    out["n_pos"], out["n_cand"] = npos, int(mask.sum())
    if npos == 0 or npos == len(y) or not np.isfinite(s).any():
        return out
    s = np.where(np.isfinite(s), s, np.nanmin(s[np.isfinite(s)]) - 1.0)
    rng = np.random.default_rng(0)
    s = s + rng.uniform(0, 1e-12, len(s))
    order = np.argsort(-s, kind="stable")
    yo = y[order].astype(float)
    hits = np.cumsum(yo)
    ranks = np.flatnonzero(yo)                      # 0-based ranks of positives
    out["ap"] = float((hits[ranks] / (ranks + 1)).mean())
    if degree is not None:
        d = np.maximum(np.asarray(degree)[mask][order], 1).astype(float)
        w = np.ones(len(s))
        wp = 1.0 / d[yo > 0]
        w[yo > 0] = wp / wp.mean()
        out["nuap"] = _weighted_ap(yo, w)
    nneg = len(s) - npos
    # AUROC = P(score_pos > score_neg): negatives ranked below each positive
    neg_above = ranks - np.arange(npos)
    out["auroc"] = float(1.0 - neg_above.sum() / (npos * nneg))
    for k in KS:
        out[f"recall@{k}"] = float(hits[min(k, len(hits)) - 1] / npos)
    k = 100
    top = yo[:k]
    dcg = (top / np.log2(np.arange(2, len(top) + 2))).sum()
    idcg = (1 / np.log2(np.arange(2, min(npos, k) + 2))).sum()
    out["ndcg@100"] = float(dcg / idcg)
    out["first_hit"] = int(ranks[0] + 1)
    return out


def task_metrics(scores: np.ndarray, task, degree: np.ndarray) -> dict:
    """Main metrics on task.targets plus `<alt>_<metric>` for each secondary target set."""
    out = rank_metrics(scores, task.targets, task.exclude, degree)
    for name, (pos, exc) in task.alt.items():
        m = rank_metrics(scores, pos, exc, degree)
        out.update({f"{name}_{k}": v for k, v in m.items() if k in ("ap", "nuap", "auroc", "recall@100", "n_pos")})
    return out
