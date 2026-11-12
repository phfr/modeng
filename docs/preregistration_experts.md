# Pre-registration: arrival typing + two-arm experts with a gate (R4 plan P3), 2026-11-12
Written and committed BEFORE any expert, gate or disagreement list is scored. Only label counts of the typing (no model output)
were looked at before this file. **wide_confirm is never scored** (MODENG_SCORE_WIDE_CONFIRM is never set; the driver asserts it).

## 1. Arrival typing (labels only; `src/modeng/arrival_types.py`)
Exclusive type per (task, target), first rule wins: `onto` (target already in the 21.04 context row of the disease or a descendant =
the `self+desc` masked pairs) > `catlag` (GWAS Catalog association p <= 1e-6 added <= 2021-04-30 mapped to the disease or a
descendant) > `old` (modet arrival B, first year <= 2020) > `new` (modet arrival A, first year 2022..2025, persistent in 26.09)
> `nonpersist` (X) > `band2021` (U) > `untyped`. Alt target sets: **newdata** = `new`; **labelproc** = `onto|catlag|old`; each ranked
against non-targets only, kept when >= 3 positives. Counts (looked at before this file): temporal dev 501 new / 698 labelproc pairs
(33 / 46 tasks), wide_dev 151 / 156 (19 / 33 tasks), wide_train 7,461 / 10,072.

## 2. Models (all trained on temporal_wide wide_train, `stack.run_wide` recipe: 600 rounds, 1,500 negatives, <= 60 positives,
deg^-0.35, relative-target guard; LightGBM seeds 0 and 1)
- **Expert-B** (biology / knowledge, 43 channels): transfer_jaccard, transfer_onto, rwr_string, degree, the 11 don_*, the 14 vmean/vmax,
  the 6 vc_*, cs_donor, dA_pt, dc_ridge, dc_ridge_pos, ph_mean, ph_top3, pt_ridge, pt_ridge_pos.
- **Expert-L** (label process, 8 channels): popularity, gwas_catlag, pri_l2g_n, pri_l2g_max, pri_mend_n, pri_hpo_n, hz_gwas, hz_mend.
  (B + L = exactly the 51 channels of W_v1clean.)
- **Gate G**: logistic regression (L2, C = 1, standardised inputs, positives weighted deg^-0.35) on features
  [logit sB, logit sL, within-task percentile rB, rL, context c, logit sB x c, logit sL x c], context c = (seed GWAS share `_mix_g`,
  log1p n_seeds, gwas_catlag coverage flag, mean log1p popularity of the seeds). Training rows = the wide_train training rows
  (targets <= 60 + 1,500 negatives, seed 0) with **out-of-fold** expert scores: 5 folds grouped by wide_train cluster; each fold's
  experts are trained with the evaluation guard plus a within-train guard (targets of held-out relatives dropped). Expert seed 0 for OOF.
  At evaluation G is applied to the full-trained experts of seed s (G_s0, G_s1).
- Comparator **W_v1clean** (r6, seeds 0 / 1, existing scores). All methods are evaluated from their saved score matrices with the
  same code (alt sets newdata / labelproc added).

## 3. Endpoints and decision rule (evaluation sets: temporal dev 53, wide_dev 45; temporal confirm descriptive only, spent)
Seed-averaged per-task metrics (`@avg` = mean of s0 and s1). Paired cluster bootstrap (10,000) CIs reported for every delta.
- **G replaces W_v1clean** only if on BOTH temporal dev and wide_dev: (a) delta AP >= -0.008 and delta nuAP >= -0.008 (point estimates;
  0.008 = seed noise from r6) AND (b) delta newdata_ap > 0 and delta newdata_nuap > 0 (point estimates). Otherwise W_v1clean stays.
- Secondary (no decision): Expert-B and Expert-L alone vs W_v1clean on all / newdata / labelproc; labelproc deltas; sa.

## 4. Product: the disagreement list
Per task (non-excluded genes only): r = within-task percentile of Expert-B and Expert-L scores; pool = Expert-B top 500;
**list = top K = 50 of the pool by D = rB - rL**. Same-size comparators: W_v1clean top 50, Expert-B top 50, pool-random (50 random
genes of the pool, 100 draws), popularity-matched random (per list gene a random non-excluded gene of the same popularity decile,
100 draws). Metric: pooled newdata precision = newdata hits / list genes over the tasks with >= 1 newdata target; also precision on all
targets and labelproc, and the newdata share of hits. Lists built per seed (experts s0 / s1, W s0 / s1), precision averaged over seeds.
- **The disagreement list is a product** if its newdata precision exceeds W_v1clean top-50's newdata precision on BOTH dev sets
  (point estimate; CI of the difference reported, cluster bootstrap over tasks).
- 3 example diseases (illustration, outcome-selected and labelled so): the tasks with most newdata hits in the list.
