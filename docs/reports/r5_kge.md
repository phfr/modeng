# R5: cached RotatE (PrimeKG-T0) as a channel: F1 cheapest decisive test (2026-11-12)

## Pre-registered decision rule (written and committed before any run)
- Runs (temporal, cv on dev, dev-trained model on confirm, `--pos-deg-power 0.35`):
  - `v1_kge` = v1 channel list + `kge_rotate`;
  - `v1_kgeperm` = v1 channel list + `kge_rotate_perm` (same-width control: RotatE retrained on the degree-permuted PrimeKG-T0);
  - reference `stack_v3ptr_p0.35` (modeng-v1, saved scores).
- Endpoints: AP, nuAP, sa_nuAP (paired cluster bootstrap, `modeng.compare`, 10,000) and degree-matched AP (`scripts/leader_endpoints.py`).
- **Keep `kge_rotate` only if** `v1_kge` beats the control `v1_kgeperm` on BOTH AP and nuAP on BOTH temporal dev AND temporal
  confirm (mean delta > 0 in all four; CIs reported). Otherwise discard. A gain over the reference that the permuted control
  also shows counts as no gain. Caveat fixed in advance: modet picked the RotatE hyper-parameters and checkpoint on the
  53 dev diseases, so dev is optimistic; confirm is decisive for the sign.
- Secondary (reported, not deciding): the same comparisons with the 586 KG-leaky target pairs (see leakage audit) removed.
- Alone: `kge_rotate`, `kge_rotate_perm` vs `transfer_jaccard` on dev and confirm.

## What was built
- `src/modeng/ext/kge_cached.py`: channels `kge_rotate` (RotatE on PrimeKG-T0) and `kge_rotate_perm` (RotatE retrained on the
  degree-permuted PrimeKG-T0, modet's DL2 null; same width and coverage). Source: modet `results/cache/kge/scores_kge_rotate_primekg_t0{,_perm}_t0_string115_phys_notm_400.npz`
  (12,743 STRING-T0 genes x 260 panel diseases; all 191 temporal tasks covered). Genes mapped to sorted ENSG universe; genes absent
  from the KG (modet floor, 3) or off STRING-T0 (6,512) -> NaN; benchmarks other than temporal (incl. `past`) -> NaN.
- `scripts/r5_kge_compare.py` -> `results/review/r5_kge_compare.tsv` (paired cluster bootstrap 10,000; degbin_ap = review endpoints, as in `scripts/leader_endpoints.py`).
- Parity: `kge_rotate` alone on temporal dev onnet AP 0.1171 / nuAP 0.1091 = modet's frontier_kge_summary row exactly.

## How modet built the KG (read from `modet/src/modet_bench/external/kge.py`)
PrimeKG v2 (Dataverse 2022-05-02, sources ~May 2021) with its DisGeNET `disease_protein` relation removed; OT 21.04 genetic
edges `OT:<id> -disease_protein_t0-> gene` (panel diseases: exactly their seeds = T0 set n STRING-T0; other OT diseases: 21.04
T0 sets); `same_as` OT->MONDO; `ot_parent` (T0 ontology). 32 relations, 4.33 M triples. Hyper-parameters and checkpoint chosen on
modet dev nuAP (= the 53 modeng temporal dev diseases): dev optimistic.

## Leakage audit (triples file `triples_primekg_t0_t0_string115_phys_notm_400.parquet`)
- No DisGeNET relation (0 `disease_protein` triples). Task disease's own OT node -> its 25.09 targets: **0 edges** (only seeds:
  7,287 of 8,852 all-gene seeds, the rest off-network). No 25.09 data in the KG.
- But 586 / 6,324 targets (95 tasks; dev 187 / 1,886, confirm 399 / 4,438) are **21.04** `disease_protein_t0` edges of a
  descendant OT node (322 pairs; modeng masks these in the context, 314 of them) or of a synonym OT node sharing the MONDO node
  (279; most are visible in modeng's own context too). Not future labels, but a route modeng's masking closes and the KG does not.
  Secondary "clean" rows remove these pairs from the targets for every method.
- `past`: channel is NaN (KG is 21.04-dated; no 19.04 version).

## Results
Alone (all-gene universe; onnet in brackets):

| split | channel | AP | nuAP | AUROC | R@100 | sa_nuAP |
|---|---|---|---|---|---|---|
| dev | kge_rotate | 0.0985 (0.1171) | 0.0424 (0.1091) | 0.763 | 0.246 | 0.046 |
| dev | kge_rotate_perm | 0.0099 | 0.0088 | 0.715 | 0.043 | 0.006 |
| dev | transfer_jaccard | 0.1229 (0.1343) | 0.0961 (0.1328) | 0.682 | 0.257 | 0.040 |
| confirm | kge_rotate | 0.0887 (0.1045) | 0.0454 (0.0983) | 0.755 | 0.243 | 0.056 |
| confirm | kge_rotate_perm | 0.0075 | 0.0035 | 0.713 | 0.038 | 0.005 |
| confirm | transfer_jaccard | 0.1189 (0.1277) | 0.1042 (0.1276) | 0.705 | 0.283 | 0.052 |

Increment (`--pos-deg-power 0.35`, cv on dev, dev-trained on confirm), delta [95 % cluster CI]:

| split | comparison | AP | nuAP | sa_nuAP | degbin AP | clean AP | clean nuAP |
|---|---|---|---|---|---|---|---|
| dev | v1_kge vs v1_kgeperm | +0.0033 [-0.0021, +0.0093] | +0.0065 [+0.0010, +0.0131] | +0.0008 | +0.0022 | +0.0026 | +0.0068 [+0.0011, +0.0136] |
| dev | v1_kge vs modeng-v1 | +0.0000 [-0.0043, +0.0047] | +0.0076 [+0.0004, +0.0151] | +0.0000 | +0.0013 | +0.0005 | +0.0054 |
| confirm | v1_kge vs v1_kgeperm | +0.0022 [-0.0026, +0.0070] | +0.0007 [-0.0069, +0.0080] | -0.0110 [-0.0236, -0.0009] | -0.0017 | +0.0011 | +0.0008 |
| confirm | v1_kge vs modeng-v1 | -0.0028 [-0.0085, +0.0027] | -0.0042 [-0.0130, +0.0048] | -0.0127 [-0.0281, -0.0022] | -0.0006 | -0.0024 | -0.0034 |
| confirm | v1_kgeperm vs modeng-v1 | -0.0050 [-0.0093, -0.0014] | -0.0049 [-0.0107, +0.0001] | -0.0018 | +0.0011 | -0.0035 | -0.0042 |

Absolute: modeng-v1 dev AP 0.1756 / nuAP 0.1502, confirm 0.1532 / 0.1770; v1_kge dev 0.1757 / 0.1578, confirm 0.1504 / 0.1728;
v1_kgeperm dev 0.1724 / 0.1513, confirm 0.1482 / 0.1721. Full table: `results/review/r5_kge_compare.tsv`.

## Verdict
- **Pre-registered rule (vs control): formally met** (all four mean deltas > 0), but only dev nuAP has a CI above 0; confirm
  deltas are +0.002 / +0.001 with CIs spanning 0, and confirm sa_nuAP is significantly worse.
- **Against modeng-v1 there is no gain on confirm** (AP -0.003, nuAP -0.004, sa_nuAP -0.013); the "gain over control" on confirm
  is the control's own cost (an extra noise channel costs v1 about -0.005). Dev nuAP +0.008 is on the diseases modet tuned RotatE on.
- F1 kill rule (05_model_families.md): dev AP +0.000 < +0.005 over modeng-v1 -> the shallow-KGE half of F1 is **killed**.
  **Do not add `kge_rotate` to modeng-v1.** KG disease layer (shallow RotatE) = transfer-level; O-13 stays open only for a
  path-based model with per-query masking (which must also mask the 586 descendant/synonym pairs).
- Runtime: about 25 min wall (2 stack runs, 8 LightGBM threads each, sequential).
