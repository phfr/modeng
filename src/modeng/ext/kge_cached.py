"""F1 cheapest decisive test (docs/research/05_model_families.md): modet's cached RotatE scores on PrimeKG-T0 as channels.

Source (READ ONLY, modet): /home/ph/code/modet/results/cache/kge/
  scores_kge_rotate_primekg_t0_t0_string115_phys_notm_400.npz       -> kge_rotate
  scores_kge_rotate_primekg_t0_perm_t0_string115_phys_notm_400.npz  -> kge_rotate_perm (negative control)
  each: nodes [12,743 STRING-T0 genes, ENSG], disease_ids [260 modet panel diseases], scores [genes x diseases] float32.
How modet built them (src/modet_bench/external/kge.py, benchmarks/run_frontier_kge.py):
  KG = PrimeKG v2 (Dataverse 2022-05-02, sources ~May 2021) with its DisGeNET `disease_protein` relation REMOVED, plus OT 21.04
  genetic edges `OT:<id> --disease_protein_t0--> gene` (panel diseases: exactly their benchmark seeds = T0 set n network; other
  OT diseases: their 21.04 T0 sets), `same_as` (OT 21.04 dbXRefs, 25.09 fallback) to PrimeKG MONDO nodes, `ot_parent` (T0
  ontology). 32 relations, 4.33 M triples. Score = RotatE(OT:d, disease_protein_t0, gene); RotatE dim 128, lr 0.01, margin 9,
  32 negatives, adversarial T 1.0; hyper-parameters AND checkpoint picked on the 53 modet dev diseases' nuAP (= modeng temporal
  dev targets: dev numbers are optimistic; confirm is not used for selection).
  perm = the same KG with every relation's tails shuffled (degree-preserving per relation; Brière DL2 null), retrained.
Leakage audit (docs/reports/r5_kge.md): the task disease's own OT node has 0 edges to its 25.09 targets (only seeds); no
  DisGeNET; but 586 / 6,324 temporal targets are 21.04 edges of a descendant OT node (322; modeng masks these pairs in the
  context) or of a synonym OT node sharing the MONDO node (279). These are 21.04-dated facts, not future labels; the report
  also evaluates with those pairs removed from the targets.
Genes absent from the KG (modet floor value) and genes outside STRING-T0 -> NaN. Benchmarks other than temporal -> all NaN.
Leakage class on temporal: temporal-clean by date except the PrimeKG vintage (~May 2021, one month after T0) and the dev-tuned
hyper-parameters.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .. import data as D

KGE = "/home/ph/code/modet/results/cache/kge/scores_kge_rotate_{}_t0_string115_phys_notm_400.npz"


@lru_cache(None)
def _load(variant: str):
    z = np.load(KGE.format(variant), allow_pickle=True)
    S = z["scores"].astype(np.float32)                    # [genes, diseases]
    S[S <= S.min() + 1e-6] = np.nan                      # modet's to_matrix floor = gene absent from the KG
    return list(z["nodes"]), {d: j for j, d in enumerate(z["disease_ids"])}, S


def _channel(B, variant: str) -> np.ndarray:
    out = np.full((len(B.tasks), B.n_genes), np.nan, np.float32)
    if B.name != "temporal":
        return out
    nodes, dpos, S = _load(variant)
    U = D.universe()
    pos = {g: i for i, g in enumerate(U)}
    gi = np.array([pos.get(g, -1) for g in nodes]); ok = gi >= 0
    for i, t in enumerate(B.tasks):
        j = dpos.get(t.disease_id)
        if j is not None:
            out[i, gi[ok]] = S[ok, j]
    return out


def kge_rotate(B, split_tag):
    return _channel(B, "primekg_t0")


def kge_rotate_perm(B, split_tag):
    return _channel(B, "primekg_t0_perm")


CHANNELS = {"kge_rotate": kge_rotate, "kge_rotate_perm": kge_rotate_perm}
