# r7 / P3: arrival typing + two-arm experts (biology vs label process) with a gate, 2026-11-12
Pre-registration (committed before any scoring): `docs/preregistration_experts.md`, commit `328dabf`.
Leakage class: temporal-clean (21.04 inputs; `--train wide` = same-window training with the relative-target guard, 12,202 positive
pairs dropped). **wide_confirm was not scored** (flag never set; both drivers assert it). Temporal confirm is spent: descriptive.

## 1. Arrival typing (`src/modeng/arrival_types.py`; alt sets `newdata`, `labelproc` added by `tasks.load`, additive)
Exclusive type per (task, target), first rule wins: `onto` (target already in the 21.04 context row of the disease or a descendant,
i.e. the `self+desc` masked pairs: ontology / propagation event) > `catlag` (GWAS Catalog association p <= 1e-6 added <= 2021-04-30,
mapped to the disease or a descendant) > `old` (modet arrival B, first evidence <= 2020: ingestion / curation lag) > `new`
(modet arrival A: first evidence 2022-2025, persistent in 26.09) > `nonpersist` (X) > `band2021` (U) > `untyped`.
newdata = `new`; labelproc = `onto|catlag|old`; ranked against non-targets only; kept at >= 3 positives.
Sources: modet `arrival_devall_t0_string115_phys_notm_400.parquet` (temporal), `data/cache/arrival_wide.parquet` (modet's
`build_arrival` on wide targets), modet GWAS Catalog copy, OT 21.04 `associationByDatasourceDirect` (descriptive flag `prior21`).

| set | targets | new | old | catlag | onto | nonpersist | 2021 | untyped | newdata / labelproc tasks |
|---|---|---|---|---|---|---|---|---|---|
| temporal dev | 1,886 | 501 | 465 | 129 | 104 | 247 | 77 | 363 | 33 / 46 of 53 |
| temporal confirm | 4,438 | 1,306 | 993 | 120 | 222 | 713 | 199 | 885 | 94 / 117 of 138 |
| wide_train | 23,281 | 7,461 | 7,354 | 1,178 | 1,540 | 4,379 | 1,369 | 0 | 366 / 568 of 712 |
| wide_dev | 446 | 151 | 138 | 2 | 16 | 127 | 12 | 0 | 19 / 33 of 45 |

About a third of the targets are new data, 35-43 % label process, 13-28 % non-persistent. 54 % of `new` wide pairs and 37 % of
temporal ones already had some 21.04 OT evidence (mostly literature) in the family (`prior21`). Cache: `data/cache/arrival_types_*.parquet`.

## 2. Experts and gate (`scripts/r7_experts.py`; recipe of `stack.run_wide`, seeds 0 and 1)
- **Expert-B** = 43 biology / knowledge channels of v1clean (transfer, rwr, degree, donors, views, cs_donor, dA_pt, dc_ridge(_pos),
  ph_*, pt_ridge(_pos)); **Expert-L** = 8 label-process channels (popularity, gwas_catlag, pri_*, hz_*). B + L = W_v1clean's 51.
- **Gate G** = logistic regression on logit / rank of both experts + 4 task-context features (seed GWAS share, log n_seeds,
  catalog coverage, seed popularity) and their interactions with the expert logits; trained on 5-fold out-of-fold expert scores on
  wide_train (folds by cluster, evaluation guard + within-train relative guard). Coefficients (`results/r7/gate_coef.tsv`):
  logit B 1.06, logit L 0.38; **lB x GWAS share -0.78, lL x GWAS share +0.60**: the gate learned to shift weight from biology to the
  label-process expert in GWAS-dominated diseases (r5's "two models inside v1").

Seed-averaged (W_v1clean evaluated from saved scores: s0 dev AP 0.1909 = r6):

| set | method | AP | nuAP | newdata AP / nuAP | labelproc AP / nuAP | sa AP |
|---|---|---|---|---|---|---|
| temporal dev | transfer_jaccard | 0.122 | 0.096 | 0.026 / 0.032 | 0.172 / 0.189 | 0.031 |
| | **W_v1clean** | **0.189** | **0.180** | 0.056 / 0.061 | **0.256 / 0.307** | 0.063 |
| | Expert-B | 0.160 | 0.155 | 0.044 / 0.048 | 0.207 / 0.249 | 0.037 |
| | Expert-L | 0.063 | 0.061 | 0.034 / 0.038 | 0.098 / 0.140 | 0.048 |
| | **G** | 0.182 | 0.172 | **0.062 / 0.064** | 0.231 / 0.273 | 0.061 |
| wide_dev | transfer_jaccard | 0.100 | 0.129 | 0.011 / 0.011 | 0.141 / 0.182 | 0.026 |
| | **W_v1clean** | 0.128 | 0.184 | **0.038 / 0.048** | 0.171 / 0.249 | 0.057 |
| | Expert-B | 0.129 | 0.196 | 0.023 / 0.027 | 0.170 / 0.248 | 0.031 |
| | Expert-L | 0.031 | 0.032 | 0.017 / 0.009 | 0.037 / 0.090 | 0.025 |
| | **G** | **0.140** | **0.199** | 0.026 / 0.027 | **0.180 / 0.267** | 0.049 |
| temporal confirm (descr.) | W_v1clean / Expert-B / G | 0.172 / 0.157 / 0.171 | 0.205 / 0.186 / 0.198 | 0.054 / 0.044 / 0.058 | 0.222 / 0.199 / 0.220 | 0.073 / 0.055 / 0.076 |

Paired cluster bootstrap vs W_v1clean (10,000; full: `results/r7/compare.tsv`):

| delta vs W_v1clean | temporal dev | wide_dev | confirm (descr.) |
|---|---|---|---|
| G: AP / nuAP | -0.008 [-0.022, +0.003] / **-0.008** [-0.020, +0.001] | +0.012 [-0.006, +0.030] / +0.016 [-0.008, +0.043] | -0.001 / -0.007 [-0.015, -0.000] |
| G: newdata AP / nuAP | +0.006 [+0.000, +0.013] / +0.003 [-0.006, +0.013] | -0.012 [-0.042, +0.005] / -0.022 [-0.058, +0.003] | +0.004 [-0.001, +0.010] / +0.000 |
| G: labelproc AP / nuAP | -0.025 [-0.057, -0.002] / -0.035 [-0.067, -0.009] | +0.009 / +0.019 (n.s.) | -0.003 / -0.013 |
| Expert-B: AP / newdata AP | -0.029 [-0.051, -0.014] / -0.011 [-0.031, +0.002] | +0.001 / -0.015 [-0.030, -0.004] | -0.015 / -0.010 (CI < 0) |
| seed noise W s1 - s0: AP / newdata AP | -0.003 / +0.005 | +0.008 / +0.001 | 0.000 / -0.001 |

**Decision (pre-registered): G does not replace W_v1clean.** Temporal dev: newdata better (+0.006 / +0.003) but nuAP -0.0081 just
beyond the -0.008 tolerance (AP -0.0077 inside). wide_dev: overall better (+0.012 / +0.016) but newdata worse (-0.012 / -0.022).
The two dev sets point in opposite directions on every endpoint and all CIs include 0 except labelproc on dev; nothing here is
distinguishable from noise at 45-53 tasks (19-33 newdata tasks).

## 3. Product: the disagreement list (pre-registered: Expert-B top 500, top 50 by rank_B - rank_L)
Pooled precision over tasks with >= 1 newdata target, averaged over seeds (`results/r7/disagree.tsv`, per task `disagree_pertask.tsv`):

| set (tasks) | list newdata | W top-50 newdata | Expert-B top-50 | pool-random | popularity-matched random | list all targets / W | median popularity list / W |
|---|---|---|---|---|---|---|---|
| temporal dev (49) | **0.0006** | 0.0290 | 0.0218 | 0.0056 | 0.0005 | 0.011 / 0.180 | 5 / 40 |
| wide_dev (32) | **0.0003** | 0.0184 | 0.0131 | 0.0028 | 0.0002 | 0.004 / 0.047 | 7 / 51 |
| temporal confirm (122, descr.) | 0.0020 | 0.0320 | 0.0252 | 0.0074 | 0.0005 | 0.009 / 0.140 | 6 / 43 |

List minus W top-50 (newdata): dev -0.028 [-0.051, -0.013], wide_dev -0.018 [-0.031, -0.007]. **Not a product (rule failed on
both sets, CIs < 0).** The list is worse than random genes from the same Expert-B pool and equal to popularity-matched random
on both dev sets: "high biology, low label-process" selects the least-studied genes (median popularity 5-7 vs 40-51), and the
biology expert has no signal there. Its newdata share of hits is not higher than W's (dev 6 % vs 16 %; wide_dev 8 % vs 39 %).
The examples are all there is, one newdata hit each (seed 0, outcome-selected, illustration only):
- sensorineural hearing loss (EFO_1001176, temporal dev): **HGF** (ClinVar, 2025), rank_B 0.977 / rank_L 0.746, W rank 518;
- primary viral infectious disease (MONDO_0100329, dev): **CCR1** (GWAS, 2023), 0.989 / 0.579, W rank 653;
- functional neutrophil defect (MONDO_0015978, wide_dev): **ITGA4** (GWAS, 2024), 0.979 / 0.699, W rank 222.

## 4. Reading
- **The label-process channels also carry the new-data endpoint.** Removing them (Expert-B) costs as much on newdata (dev -0.011,
  wide_dev -0.015, confirm -0.010 AP) as overall; Expert-L alone beats transfer on newdata on dev (0.034 vs 0.026) and wide_dev
  (0.017 vs 0.011). Popularity / L2G priors are partly study-bias priors that also predict which genes the next real studies hit.
  So "label-process" channels are not removable for a biology claim without a measurable cost even on new-data arrivals.
- **Separation of biology from label process cannot be done by channel partition** at the gene level: what the biology expert
  ranks high and the label-process expert low is the low-popularity tail, where nothing currently predicts arrivals (R4-B's wall:
  untraced targets are ranked at popularity level). The disagreement product as defined is dead; the R4-A / 05 kill criterion
  (Bio-favoured tercile >= 1.5x) would also fail (list < pool-random).
- The gate learns a sensible, interpretable mixing (GWAS-heavy diseases -> label expert) but does not beat a joint stacker that
  sees all channels; consistent with 05/R4-B: the perfect-gate bound is small and experts use the same witnesses.
- The typing itself is the durable output: every future table can report newdata / labelproc next to AP / nuAP. On it W_v1clean's
  newdata AP is 0.04-0.06 vs 0.17-0.26 on labelproc: the benchmark's AP is dominated by label-process arrivals.

## Files
`src/modeng/arrival_types.py` (new), `src/modeng/tasks.py` (load hook, additive), `scripts/r7_experts.py`, `scripts/r7_analyze.py`,
`docs/preregistration_experts.md` (+ `.sha256`), `results/r7/` (summary, compare, decision, disagree*, gate_coef, imp_*, logs;
OOF matrices `oof_*.npy` not committed), per-task parquets / scores `stackW_{E_B,E_L,G}[_s1]` in `results/temporal/{dev,confirm}/`,
`results/temporal_wide/wide_dev/`. Tests: `tests/test_wide.py` + `tests/test_harness.py` 24 pass, 2 skip. Runtime ~25 min, 8 threads, peak ~10 GB.
