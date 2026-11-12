"""settx channels (implementer F, idea ML-5): identity-blind episodic set-completion transformer (src/modeng/settx).

Loads saved predictions (data/cache/settx/pred_<variant>_*.npz written by `python -m modeng.settx.train`):
  B.name == 'past'     -> cross-fitted OOF scores (3 folds over leak-aware Louvain groups, settx.groups); any split_tag
                          (rows looked up by disease_id, so the stacker's train900 subset works)
  B.name == 'temporal' -> model trained on all past episodes (19.04 -> 21.04, <= T0), scored on dev / confirm
  other benchmarks     -> NaN (not trained)
Channels / leakage class on temporal:
  settx       views = seq / pre21 / annot only (esmc, prott5, esm2_15b, depmap20q4, phylo_npp, go_t0, reactome_t0): temporal-clean
  settx_all   + modern views (HPA, GTEx, Perturb-seq): transductive on temporal
  settx_perm  negative control: settx_all with gene rows of every view permuted (one shared permutation)
  settx_auxonly  clean views, trained only on self-supervised episodes (19.04 context rows + pre21 donor sets), no arrivals
  settx_auxperm  negative control: settx_auxonly with gene rows of every view permuted
Context features inside the model (transfer_jaccard, popularity on the masked context, degree) are the harness's own.
"""
from __future__ import annotations

import numpy as np

from ..settx import data as SD

VARIANT = {"settx": "clean", "settx_all": "all", "settx_perm": "perm", "settx_auxonly": "auxonly", "settx_auxperm": "auxperm"}


def _load(variant, B, split_tag):
    if B.name == "past":
        f = SD.CACHE / f"pred_{variant}_past.npz"
    elif B.name == "temporal" and split_tag in ("dev", "confirm"):
        f = SD.CACHE / f"pred_{variant}_temporal_{split_tag}.npz"
    else:
        return np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    z = np.load(f)
    pos = {d: i for i, d in enumerate(z["ids"])}
    idx = np.array([pos[t.disease_id] for t in B.tasks])
    return z["S"][idx].astype(np.float32)


CHANNELS = {name: (lambda B, s, v=v: _load(v, B, s)) for name, v in VARIANT.items()}
