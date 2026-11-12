# Pre-registration: full witness recipe trained at scale on `temporal_wide` (r6 P1b), 2026-11-12
Written and committed **before** any model in this file is scored. wide_confirm is not touched (no `MODENG_SCORE_WIDE_CONFIRM`).

## Arms (all: `stack.run_wide`, wide_train 712 tasks, relative-target guard on, 600 rounds, 1,500 negatives, <= 60 positives
per task, positives weighted deg^-0.35, script `scripts/r6_wide_full.py`)
- `stackW_W_v1clean`: the 51 channels of `results/v1clean_channels.txt`.
- `stackW_W_v2` (primary candidate): v1clean minus 28 channels = all 20 view similarities (14 `vmean_/vmax_` + 6 `vc_*`),
  `cs_donor`, `dA_pt`, the 6 priors `pri_*` / `hz_*` -> 23 channels (`results/v2_channels.txt`: base, 11 donor families,
  `dc_ridge(_pos)`, `gwas_catlag`, `ph_mean`, `ph_top3`, `pt_ridge(_pos)`).
  Note: the audit's "30 view similarities" were v1's; v1clean holds 20 (its 6 `vc_*` are the clean replacements of the
  HPA / GTEx views); all 20 are dropped.
- Seed noise: both arms again with LightGBM seed 1 (`_s1`; negative sample unchanged).
- Same-width permuted control `stackW_W_v1perm`: v2 + the 28 dropped channels with their gene axis permuted per task
  (`ext/wide_perm.py`). v1clean vs v1perm = information in the dropped channels; v2 vs v1perm = cost of width / noise.
- Comparators: cv-trained v1clean (`stack_v1clean`: temporal dev = grouped-CV OOF (existing row), wide_dev = model trained on
  all 53 temporal dev tasks, 400 rounds, 3,000 negatives, deg^-0.35, seed 0); past-trained twins `stackP_v1clean`,
  `stackP_v2` (19.04 -> 21.04; priors / ph are NaN on `past`); `transfer_jaccard`.

## Endpoints
AP, nuAP, onnet nuAP, sa nuAP (where defined), degree-matched AP; temporal dev (53), wide_dev (45); temporal confirm (138,
spent: descriptive only). Paired cluster bootstrap (`modeng.compare`, 10,000; modet fold clusters on temporal, task on wide).

## Decision rule (fixed now)
A wide-trained arm X in {W_v2 (primary), W_v1clean} is **the new candidate** iff, with per-task metrics averaged over the
two LightGBM seeds, versus cv-trained `stack_v1clean`:
1. AP: delta > 0 with the 95 % CI lower bound > 0 on temporal dev **and** on wide_dev;
2. nuAP: no loss beyond noise on both sets = the 95 % CI upper bound >= 0 **and** delta >= -max(0.005, |seed0 - seed1| nuAP
   difference of X on that set).
If both arms pass, W_v2 is the candidate (fewer witnesses). If an arm has positive AP point estimates on both sets but a CI
including 0 on one, it is reported as "not decided (underpowered)", not as the candidate. Temporal confirm, the permuted
control, the past twins and transfer_jaccard are descriptive only and cannot change the decision.
