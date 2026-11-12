# r4: ML-4 learned GWAS power forecaster for strict-a (2026-11-12, ~09:00-10:20)

**Goal.** modeng-v1 (`stack_v3ptr_p0.35`) only ties modet's headline on strict-a (genetic arrivals first seen 2022+, `sa` targets).
ML-4 (`docs/ideation/r1_ml.md`): learn which sub-threshold gene signals cross significance as studies grow, from study pairs
dated <= T0, and use the forecast crossing probability as a channel.

## Verdict
**Discard (for strict-a).** The forecaster is clearly better than plain z at predicting later hits within pre-T0 study pairs
(leave-trait-out AUC 0.82 vs 0.65), and alone it beats plain T0 MAGMA z on strict-a for covered diseases, but as a stacker input it does
not move strict-a: the dev-chosen variant (`stack_v1_powerfc2`) gives dev sa_AP +0.009 [+0.001,+0.017] that does not replicate on
confirm (sa_AP -0.001 [-0.008,+0.004], sa_nuAP -0.004 [-0.016,+0.004]). vs modet's headline, strict-a stays a tie on modet's candidate
set (sa_onnet_nuAP 0.107 vs 0.109, -0.002 [-0.017,+0.015]) and below on all genes (sa_nuAP -0.023 [-0.039,-0.005]).
Plain `gwas_own` as the increment does the same (confirm sa deltas -0.002 to -0.005). Coverage is the ceiling: only 12/28 dev and
34/71 confirm strict-a diseases have an own T0 MAGMA study.

## What was built (commit 9b96abe = code + pre-registration before any stack run; report commit follows)
- `src/modeng/ext/powerfc.py`: channels `pfc_own`, `pfc_sem`, `pfc_own_perm` (docstring has full definitions).
  - Training pairs (cutoff = 2021-04-30 on temporal, 2019-04-30 on past): small = one of the 108 MAGMA T0 studies of `gwas_d.studies()`;
    large = any GWAS Catalog study dated <= cutoff sharing a MAPPED_TRAIT_URI id, effective N (cases, else total) >= 1.2x, >= 1 hit.
    **146 pairs, 20 small studies** (median N ratio 5.4; SCZ PheCode 48 and bipolar 32 pairs dominate); past: 72 pairs, 14 small studies.
  - Label = gene is a MAPPED_GENE of a p <= 5e-8 association of the large study (catalog-hit label, close to OT's GWAS route; MAGMA
    p < 2.5e-6 in the large study would need summary statistics + MAGMA runs of every large study: not done, time).
  - Features: small-study z, z percentile, max z / # z>3 among 10 position-neighbours each side, nsnps, gene length, local gene density,
    gnomAD LOEUF/pLI, pleiotropy (# other available studies with z > 4.56, trait-sharing studies excluded), log N_eff, # sig genes,
    lambda, log N ratio. LightGBM, negatives 1:10 subsampled (weighted).
  - Channel: per task, forecast at N ratio x2.5 for each matched own study (gwas_own ontology matching; `pfc_sem` adds TF-IDF proxies),
    max of w * P. NaN without a study. On past: model trained on pairs <= 2019-04 and applied to studies <= 2019-04 (NaN otherwise).
- `scripts/powerfc_validate.py` (pair-level validation), `scripts/powerfc_alone.py` (channels alone), `scripts/powerfc_runs.sh`
  (4 stack runs; pre-registered rule in header), `scripts/powerfc_analyze.py` (dev rule, paired cluster bootstrap 5000).
- Outputs: `results/powerfc/{validate_2021-04-30,alone,alone_paired,endpoints_vs_ref}.tsv`, run logs; scores
  `results/temporal/<split>/stack_v1_{powerfc,powerfc2,powerfc_perm,gwasown}.scores.npy`; `results/review/leader_endpoints_stack_v1_powerfc2.tsv`.
  Rows were appended to `results/leaderboard.tsv` (not committed here; the file had other uncommitted edits).

## 1. Forecaster validation (GroupKFold 5 over small studies, all genes of held-out pairs; mean over small studies)
| ranker | AUC all | AP all | AUC crossers (z < 4.56) | AP crossers |
|---|---|---|---|---|
| small-study z | 0.647 | 0.063 | 0.607 | 0.018 |
| model | **0.817** | 0.040 | **0.797** | 0.026 |
| model without neighbour features | 0.814 | 0.037 | 0.797 | 0.026 |
Model - z: AUC all +0.170 [+0.100,+0.252] (18/20 studies), AP all -0.023 [-0.066,+0.013]; crossers AUC +0.190 [+0.114,+0.277], AP +0.008
[-0.006,+0.022] (16/19). Gain importance: log N ratio 0.22, gene length 0.20, z 0.12, z pct 0.07, pLI 0.06. The AUC gain is mostly a
gene prior (long, constrained genes are the ones catalog hits get mapped to); the top of the ranking (AP) is not better than z.

## 2. Channels alone (temporal, covered tasks only; dev n=16 (sa 12), confirm n=46 (sa 34))
| split | channel | AP | nuAP | AUROC | R@100 | sa_AP | sa_nuAP |
|---|---|---|---|---|---|---|---|
| dev | pfc_own | 0.023 | 0.026 | 0.731 | 0.063 | 0.013 | 0.019 |
| dev | gwas_own (plain z) | 0.021 | 0.025 | 0.589 | 0.058 | 0.009 | 0.013 |
| dev | pfc_own_perm | 0.003 | 0.003 | 0.487 | 0.004 | 0.002 | 0.003 |
| confirm | pfc_own | 0.029 | 0.029 | 0.712 | 0.071 | 0.024 | 0.021 |
| confirm | pfc_sem | 0.031 | 0.031 | 0.715 | 0.077 | 0.026 | 0.021 |
| confirm | gwas_own | 0.025 | 0.025 | 0.598 | 0.051 | 0.019 | 0.016 |
| confirm | gwas_xpow (untrained) | 0.030 | 0.028 | 0.597 | 0.058 | 0.024 | 0.017 |
| confirm | transfer (same tasks) | 0.101 | 0.082 | 0.675 | 0.232 | 0.056 | 0.044 |
Paired pfc_own - gwas_own: dev sa_AP +0.004 [+0.001,+0.008], sa_nuAP +0.007 [+0.002,+0.012]; confirm sa_AP +0.005 [+0.003,+0.009],
sa_nuAP +0.005 [-0.002,+0.015], R@100 +0.020 [+0.006,+0.033], AUROC +0.11. So the forecast adds to plain z alone, but both are
far below transfer (AP 0.03 vs 0.10).

## 3. Increment over modeng-v1 (cv on dev, power 0.35; delta vs v1, paired cluster bootstrap)
Pre-registered (commit 9b96abe): variants V1 = v1+pfc_own, V2 = v1+pfc_own+pfc_sem; controls permuted (same width) and v1+gwas_own;
dev rule = highest dev mean SA4 delta with dev AP, nuAP >= v1 - 0.003. Dev SA4: V1 +0.003, **V2 +0.009 -> chosen**.

| variant | split | AP | nuAP | sa_AP | sa_nuAP | sa_onnet_AP | sa_onnet_nuAP |
|---|---|---|---|---|---|---|---|
| V1 pfc_own | dev | -0.002 | +0.003 | +0.001 | +0.006 | +0.001 | +0.005 |
| **V2 pfc_own+pfc_sem** | dev | +0.000 | +0.006 [-0.002,+0.014] | **+0.009 [+0.001,+0.017]** | +0.009 [-0.000,+0.017] | +0.009 [+0.001,+0.017] | +0.008 |
| perm control | dev | -0.006 | -0.004 | -0.000 | -0.001 | +0.002 | -0.002 |
| gwas_own | dev | -0.003 | +0.002 | +0.008 | +0.008 | +0.013 [+0.001,+0.027] | +0.009 |
| V1 pfc_own | confirm | +0.001 | -0.001 | -0.000 | -0.003 | -0.001 | -0.003 |
| **V2 (chosen)** | confirm | -0.002 [-0.005,+0.002] | -0.002 [-0.007,+0.004] | **-0.001 [-0.008,+0.004]** | **-0.004 [-0.016,+0.004]** | -0.001 [-0.007,+0.003] | -0.003 [-0.013,+0.005] |
| perm control | confirm | -0.001 | -0.001 | -0.002 | -0.004 | -0.003 | -0.004 |
| gwas_own | confirm | +0.002 | +0.001 | -0.002 | -0.004 | -0.003 | -0.005 |
Absolute confirm V2: AP 0.151, nuAP 0.175, onnet nuAP 0.209 (+0.006 [-0.001,+0.013] vs v1), sa_onnet_nuAP 0.107.
vs modet headline (confirm, `leader_endpoints_stack_v1_powerfc2.tsv`): sa_onnet_nuAP 0.107 vs 0.109 (-0.002 [-0.017,+0.015]),
sa_onnet_AP +0.021 [+0.005,+0.036], sa_nuAP -0.023 [-0.039,-0.005] = unchanged from v1 (tie / below).
**Variant count:** 4 stack runs, all 4 scored on confirm once (stack.py prints confirm; the analysis was run once after all finished);
2 eligible variants for the strict-a claim.

## Leakage
temporal-clean by publication date: pairs, labels (catalog associations of studies dated <= 2021-04-30) and inputs (T0 MAGMA) are pre-T0.
Residual routes as for gwas_catlag: 2026 catalog trait/gene mapping of old studies; ontology releases for task-study matching.
Past channel uses only <= 2019-04 pairs and studies.

## Why it fails, and what might work
- Dev strict-a gains (28 tasks, 12 covered) do not replicate; the same happened in r3. Dev cannot select strict-a variants.
- 20 training traits (two psychiatric traits dominate), catalog-hit labels, and the model mostly learned a gene prior that the stacker
  already has (base/priors). The trees cannot use a channel that is NaN for half the strict-a diseases.
- Ideas: (a) MAGMA labels from downloaded pre-T0 large-study summary statistics (MAGMA + 1000G ref are on disk; ~1 GB per study);
  (b) equal weight per trait; (c) coverage is the real limit: forecasting needs a T0 study per disease (FinnGen R4/R5, Neale UKB v2
  2018, ~2,000 traits, would raise coverage) - this is the only version with a chance on strict-a.
Runtime: ~10 min validation/alone, ~40 min for 4 stack runs (8 threads, load 30-40).
