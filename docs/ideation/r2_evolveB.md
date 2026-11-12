# Round 2, evolution ideator B: the learner and the training signal (2026-11-12)

Independent of the other round-2 ideator. Scope: no new data channel; everything here changes HOW the stacker is trained, on WHICH
episodes, with WHICH labels / weights / constraints, and how its scores become a module.

## Measured facts this builds on (all from `docs/insights.md`, `docs/reports/r1_*.md`, `results/stackP_power_sweep.txt`)
- **Current leader** (temporal confirm, all-gene, past-trained): `stackP_bdv_cs_pt` AP 0.111 / nuAP 0.175 (dev 0.122 / 0.150);
  `+dc_ridge`, `+gwas_catlag`, `+ph_*` are measured Pareto increments not yet all in one run. "Leader" below = that stack with the
  kept channels; every test is an A/B with identical channels, only the learner changes.
- **The AP/nuAP dial** (`stackP_bdv`, confirm): positive weight deg^-p with p = 0 / 0.25 / 0.5 gives AP 0.106 / 0.105 / 0.091 and
  nuAP 0.095 / 0.131 / 0.151. p = 0.25 costs almost no AP; between 0.25 and 0.5 the model pays 0.014 AP for 0.020 nuAP.
  These three points are the frontier any learner change must dominate.
- **The learner** (`src/modeng/stack.py`): LightGBM `binary`, rows pooled over tasks (all targets capped at 60 + 1,500 uniform
  negatives), features = raw channel + within-task percentile rank + `log_nseeds` (the ONLY task-level feature), no constraints,
  no query structure, 900 of 1,786 past episodes.
- **Strata separate cleanly** (`r1_priors.md`): typed transfer beats full transfer within its own arrival class (GWAS stratum nuAP
  +0.02); `pri_l2g_n` ranks GWAS arrivals at AUROC 0.83 but costs nuAP on the Mendelian stratum; `gwas_catlag` gains sit only in the
  ~50 % covered diseases (uncovered: delta 0 or slightly negative); `gwas_seedw` is used as a negative cue.
- **Wide feature sets hurt**: 18 noise channels cost -0.010 nuAP; permuted controls cost -0.003 each.
- **Vintage mismatch in training**: on past episodes (19.04 -> 21.04) the donor families `hpo2101` (2021-01), `msigdb v7.4`,
  `complexportal 2021-03`, `km_depmap20q4` already contain part of the arrivals they are asked to predict (71 % of curated arrivals are
  HPO genes); `lead_cs.py` notes this "only biases the learned weight". `don_hpo2101` is the #2 feature by gain. At test time
  (21.04 -> 25.09) the same file has no such foresight. This is a train/test shift in the meaning of a feature, not only in its distribution.
- **Window mismatch**: training window 2.0 years, test window 4.4 years; 19.04 -> 21.04 arrivals are dominated by pipeline changes.
- CV inside `past` must use `settx.groups.leak_folds` (cluster CV leaks).

Rules used for every idea: increment tests through the learned stacker only (no rank averaging); same-width permuted control when
features are added; a claim needs AP and nuAP together (or the frontier test below); dev decides, confirm is touched once.

**Frontier test (used where an idea moves along the dial instead of off it).** Run the idea at its own dial settings, plot
(AP, nuAP) on dev, compare with the reference sweep p in {0, 0.25, 0.5, 0.75}. "Dominates" = at matched AP (linear interpolation of the
reference frontier) nuAP is higher with paired cluster-bootstrap CI excluding 0, or the reverse.

---

## MUTATE

### E-B1: Query-wise LambdaRank with degree-scaled gains
- **Mechanism.** Replace `objective=binary` on pooled rows by LightGBM `lambdarank` with one query per task; label gain of a positive
  = deg^-p (bucketed to 4-5 integer levels), `lambdarank_truncation_level` ~ 300, same negatives. Pairwise swaps are weighted by their
  effect on a per-task ranking metric, so a disease with 60 targets no longer outweighs one with 3, and the degree preference enters
  through the metric gain instead of through sample weights that shrink the effective number of hub positives.
- **Builds on.** `stack.py` (binary + weights), the power sweep, and the task-averaged evaluation (the binary loss optimises a
  pooled, target-count-weighted objective; AP/nuAP are macro averages over diseases).
- **Leakage class.** Unchanged (temporal-clean; past-trained).
- **Test (<= 1 h).** In a copy of `run_past`: `lgb.Dataset(..., group=rows_per_task)`, objective `lambdarank` and `rank_xendcg`
  as second arm, p in {0, 0.25, 0.5}; reuse cached channel matrices (only the fit changes, ~5 min per fit). Frontier test on dev.
- **Expected effect vs leader.** AP +0.004 to +0.010, nuAP -0.003 to +0.008 at matched p; the gain should come from small-target
  tasks (median task), check by tertile of `len(targets)`.
- **Kill rule.** Frontier not dominated on dev (CI includes 0 at both matched points), or the gain is only in tasks with > 30
  targets. Known risk: lambdarank overfits the head of the list with 1,500 sampled negatives; if R@100 rises while AP falls, stop.

### E-B2: Shape-constrained, coverage-aware stacker (monotone + missing-as-NaN + task-context gates)
- **Mechanism.** (a) Monotone-increasing constraints on every evidence channel and its rank feature (transfer, cs_donor, dc_ridge,
  dA_pt, donors, catlag, ph_*), so a tree can never use a sparse channel as a negative cue or fit noise in both directions;
  (b) channels with no coverage for a task (no own GWAS, no HPO profile for the seeds, no donor with overlap) are coded NaN for the
  whole task instead of 0, plus one coverage flag per channel group; (c) 8-10 task-context features (share of seeds that are
  GWAS-type vs Mendelian-type by the seed-date evidence, max donor Jaccard, seed mean degree, ontology depth, covered-by-catalog)
  so splits can gate channel groups by disease type. Today `log_nseeds` is the only way the model can tell diseases apart.
- **Builds on.** H's strata result (channel priors help GWAS, hurt Mendelian), D's covered/uncovered split, the noise-channel cost,
  `gwas_seedw` being used as a negative cue; register idea CI-05 (per-disease gating, never tested) gets its learner-side version.
- **Leakage class.** Temporal-clean if the seed evidence mix is read from the seed-date release (19.04 for past, 21.04 for temporal:
  `ot2104/associationByDatasourceIndirect` exists; the 19.04 equivalent must be checked, else fall back to "seed is in HPO / seed has
  L2G for any trait").
- **Test (<= 1 h).** Three arms on the leader's channels: +monotone only (`monotone_constraints`, `monotone_constraints_method=advanced`);
  +NaN coding; +context features, with a control arm of the same number of task-level features permuted across tasks. Also rerun the
  18-noise-channel test under monotone constraints: the cost should shrink.
- **Expected effect vs leader.** AP +0.003 to +0.008, nuAP +0.003 to +0.010; GWAS-stratum AP up without the Mendelian-stratum nuAP
  loss H measured; noise-channel cost from -0.010 to > -0.004.
- **Kill rule.** Context arm not above its permuted-task control by >= +0.003 on both metrics (dev), and monotone arm does not
  reduce the noise-channel cost by half. If monotone alone costs > 0.003 AP, some channel is legitimately non-monotone: drop (a).

---

## CROSSOVER

### E-B3: Arrival-class heads x disease-type mixing (H's typed strata x the stacker)
- **Mechanism.** Train two heads on past episodes with typed labels, "arrives through GWAS evidence" and "arrives through
  ClinVar / curated / other evidence", each seeing all channels but only its own positives (the other class's positives are dropped,
  not used as negatives). Final score = pi_d * P_gwas + (1 - pi_d) * P_mend (optionally 1 - (1 - a)(1 - b)), where pi_d is the
  expected GWAS share of the disease's next arrivals, predicted from task-context features (seed evidence mix, catalog coverage) by a
  small model fitted on past. One pooled binary label forces the dominant mechanism to own the top of every list (r1_recsys A10).
- **Builds on.** `priors_h.py` typed transfer and strata tables, `gwas_d.py` covered / uncovered split, E-B2's context features.
- **Leakage class.** Temporal-clean, provided arrival classes of PAST targets come from 21.04 evidence (<= T0). The existing
  `arrival_devall_...parquet` classes describe 25.09 arrivals and may only be used for evaluation strata.
- **Test (<= 1 h).** Derive the class of each past target from `ot2104/associationByDatasourceIndirect` (genetic_association sources
  ot_genetics_portal only -> GWAS; else non-GWAS), fit two heads + a ridge/GBM for pi_d (target: observed GWAS share per past task),
  score temporal dev. Controls: pi_d = constant global share; classes shuffled within task.
- **Expected effect vs leader.** AP +0.005 to +0.012, nuAP +0.003 to +0.010; per stratum: GWAS-only AP up >= +0.01, Mendelian
  stratum not down.
- **Kill rule.** Not better than the constant-pi control by +0.003 on both metrics on dev, or pi_d has rank correlation < 0.3 with
  the realised GWAS share on dev (then the gate is not learnable and E-B2's soft gating is enough).

### E-B4: Two episode families in one model (past arrivals x T0 hide-and-recover episodes, with an episode-type feature)
- **Mechanism.** Past episodes have the right mechanism (real arrivals) but an old, short window and old-vintage features; add
  hide-and-recover episodes built at T0 itself (mask 20 % of a disease's 21.04 genes with the holdout benchmark's ancestor-safe
  masking), which have the test-time feature distribution and 3-5x more diseases but the wrong mechanism (hidden genes are
  better-studied than future arrivals). Train one stacker on both with an `episode_type` feature and a per-family weight; predict with
  `episode_type = arrival`. Trees share channel-combination structure and keep mechanism-specific splits separate.
- **Builds on.** `tasks.build_past` + `tasks.build_holdout` (re-pointed at the 21.04 matrix), the measured gap between holdout-style
  and arrival-style tasks, settx's finding that self-supervised set completion alone reaches nuAP 0.124-0.130.
- **Leakage class.** Temporal-clean (all labels <= 21.04). Must reuse the fixed holdout masking (the first version leaked through
  ancestors) and quantised subtract-corrections. On `forward` the same recipe is stronger and cleaner than today's CV on 26.09 labels:
  train on past + temporal arrival episodes (both <= 25.09) and test on forward with no forward label in training.
- **Test (<= 1 h).** 600 T0-holdout episodes on non-sealed, non-dev/confirm-cluster diseases (dev/confirm diseases may appear only as
  context, as now); channels from cache where the split tag allows, else the cheap ones (base, donors, cs_donor); arms: past only,
  past + holdout with type feature, past + holdout without it, holdout only. Weight grid for the holdout family {0.25, 0.5, 1}.
- **Expected effect vs leader.** AP +0.003 to +0.010 (more hub-like positives teach the head), nuAP -0.003 to +0.006; the type
  feature should make it at least neutral on nuAP. Forward variant: expect within 0.01 nuAP of the CV stacker, which would show the
  label-process part can be learned from earlier windows.
- **Kill rule.** "With type feature" not better than "past only" on both metrics on dev; or "holdout only" already equals it (then
  arrival episodes add nothing and the past-training result itself needs re-examination).

---

## REPAIR

### E-B5: Vintage parity: give each training episode features of its own date
- **Mechanism.** Rebuild every time-stamped donor / view family at the past seed date (HPO annotations <= 2019-04, MSigDB v6.2 /
  GO 2019, Complex Portal and DepMap 19Q1 where available) and use those for past episodes only, as `dA_pt` already does with its
  PMID cut. Where no old vintage exists, mark the family "transductive on past" and test dropping it from training-sensitive use.
  The stacker then learns the weight a feature deserves when it has NO foresight, which is the situation at test time.
- **Builds on.** `lead_cs.py` comment on `hpo2101`; `don_hpo2101` = #2 gain feature; the dev < confirm instability of several
  increments; `dA_pt`'s clean design as the template.
- **Leakage class.** Makes the training side strictly cleaner; test side unchanged (temporal-clean).
- **Test (<= 1 h).** HPO only first (largest suspect): fetch the 2019-03/04 `genes_to_phenotype` / phenotype.hpoa archive (HPO
  GitHub releases or Wayback, as was done for 2021-01), build `hpo1903.npz` with the existing donor script, route past episodes to it
  (split-tag switch as in `donors_a.py`), refit. Diagnostic that needs no download: share of past targets inside `hpo2101` genes vs
  share of temporal targets (if past >> temporal, the bias is confirmed).
- **Expected effect vs leader.** AP +0.003 to +0.010, nuAP +0.003 to +0.012, mostly Mendelian stratum; the stacker shifts gain from
  `don_hpo*` to transfer / cs_donor / dA_pt. Sign is not guaranteed: if foresight made the model over-trust HPO and HPO is in fact
  the best Mendelian channel at test time too, the effect is ~0.
- **Kill rule.** |delta| < 0.002 on both metrics on dev AND the diagnostic shows equal enrichment in both windows: then the
  "only biases the weight" note was right and the mismatch can stay.

### E-B6: Take degree out of the relevance model, put it back as one dial (repair of the AP-for-nuAP trade)
- **Mechanism.** Stage 1 learns evidence combination from ALL positives at full weight: unweighted labels, negatives sampled
  degree-matched to the positives of the task, and no degree / popularity features, so degree carries no signal and hub positives
  still teach which channels to trust. Stage 2 adds a one-parameter prior: score = logit f(x) + lambda * g(deg, popularity), g fitted
  by isotonic regression on past residuals; lambda in [-1, 1] traces the frontier. deg^-0.5 sample weights instead throw away most of
  the information in hub positives (effective sample size) to buy the degree tilt.
- **Builds on.** Power sweep; the evolved-kernel lesson (75-85 % of a nuAP gain can be degree alone); `stack.py::training_rows`.
- **Leakage class.** Unchanged.
- **Test (<= 1 h).** Two fits (stage 1 with and without rank features of degree-correlated channels such as RWR), lambda grid of 9
  values applied post hoc to cached predictions (no refit); frontier test against p in {0, 0.25, 0.5, 0.75}. Mandatory baseline:
  the same lambda dial applied to the CURRENT p = 0 model (if that already matches the weighted models, weighting was never a model
  gain). Add a degree-matched endpoint (AP within degree deciles, averaged) to the report for all arms.
- **Expected effect vs leader.** At nuAP = 0.175 AP +0.006 to +0.015 (recovering most of the 0.014 lost between p = 0.25 and 0.5);
  at AP = leader, nuAP +0.005 to +0.015. If the frontier is merely reproduced, the result is still a simpler, honest dial.
- **Kill rule.** Frontier not dominated at any lambda on dev; or degree-decile AP is lower than the leader's (then the weighted
  model was learning real non-hub structure, not a tilt).

---

## FRESH (p_typical < 0.10)

### E-B7: Seed self-audit meta-features: each disease tells the stacker which channels work for it
- **Mechanism.** For every task and channel compute how well the channel recovers the task's OWN seeds: for seed-independent
  channels (catlag, priors, HPO phenotype, MAGMA z, popularity, views) directly as AUROC / top-1 % recall of seeds vs non-seeds; for
  seed-dependent channels (transfer, cs_donor, ridge, donors) by 3-fold seed hold-out. These per-task reliabilities become task-level
  features next to the channel values, so the stacker learns interactions "trust channel c where its self-audit is high" and, as with
  `gwas_seedw`, "distrust it where the seeds already exhaust it"; the sign is learned on past episodes, not assumed.
- **p_typical** 0.07 (per-query feature reliability exists in learning-to-rank; using the seed set as an internal validation set to
  gate evidence channels per disease is not in the register; F2-2 uses seed hold-out as a referee for LLM hypothesis programs, not as
  stacker features).
- **Builds on.** E-B2 context features (this is their data-driven version), D's negative `gwas_seedw` result (seed enrichment is not
  monotone good), H's strata.
- **Leakage class.** Temporal-clean (seeds only).
- **Test (<= 1 h).** Seed-independent channels + transfer / donor families only (lasso and ridge hold-outs are too slow for the first
  pass): ~20 reliabilities per task. Control: reliabilities permuted across tasks (same width).
- **Expected effect vs leader.** AP +0.004 to +0.012, nuAP +0.004 to +0.012; larger for tasks with >= 10 seeds, none for 3-5 seeds
  (audit too noisy; pass `log_nseeds` so the model can discount it).
- **Kill rule.** Not above the permuted-task control by +0.003 on both metrics on dev, or gain confined to tasks with > 50 seeds.

### E-B8: Label-free shift correction of the training episodes (density-ratio weights from T0 features)
- **Mechanism.** Fit a domain classifier past-row vs temporal-row on the stacker's feature vectors (no labels; temporal features are
  T0 inputs), twice: at task level (context features, channel coverage, seed count) and at (task, gene) level. Weight past training
  rows by the clipped density ratio so the 19.04 episodes that look like 21.04 diseases count more; in addition use the classifier as a
  diagnostic: features with high domain importance (candidates: `don_hpo2101`, popularity, catlag coverage) are the shifted ones and are
  re-tested under E-B5 or dropped.
- **p_typical** 0.08 (covariate-shift weighting is textbook; applying it across label-release windows of a disease-gene benchmark,
  with the classifier as a vintage-leak detector, is not in the register; CI-07 "invariant predictors" is a different mechanism).
- **Builds on.** Past -> temporal shift; E-B5 (this finds the mismatches E-B5 repairs); E-B9 (same weights feed weighted conformal).
- **Leakage class.** Temporal-clean in labels; transductive in the ML sense (uses unlabeled test-period inputs). Fit the density
  ratio on temporal DEV rows only, so confirm stays untouched; report under that name.
- **Test (<= 1 h).** Domain classifier AUC per feature group (10 min, the diagnostic alone is worth it), then refit the leader with
  task-level weights clipped to [0.2, 5]; arms: task-level, row-level, both.
- **Expected effect vs leader.** AP +0.002 to +0.006, nuAP +0.002 to +0.008; if domain AUC < 0.6 there is no shift to correct and the
  expected effect is 0 (a useful negative: the remaining gap is concept shift, not covariate shift).
- **Kill rule.** Domain AUC < 0.6, or weighted refit within +-0.002 of the leader on both metrics on dev, or effective sample size
  after weighting < 30 % (variance cost exceeds the bias gain).

---

## Calibrated module cutoff

### E-B9: Two-layer module by conformal risk control over diseases (core = precision-controlled, shell = recall-controlled)
- **Mechanism.** Calibrate the stacker on past episodes out of leak-fold (isotonic per seed-count x disease-type bin) to a
  probability of arrival, then choose thresholds by conformal risk control with the disease as the exchangeable unit: the CORE is the
  largest prefix whose false-discovery proportion has expected value <= alpha across calibration diseases; the SHELL is the smallest
  prefix whose missed-arrival proportion is <= beta. Thresholds are set on a per-disease normalised score (calibrated probability
  divided by the disease's predicted arrival count from a count model on seed count, growth 19.04 -> 21.04 and context features),
  Mondrian by disease type, and reweighted with E-B8's task-level density ratios (weighted conformal) for the past -> temporal shift.
- **Builds on.** `settx.groups.leak_folds`, E-B2 / E-B3 disease types, E-B8 weights, backlog report's near-calibrated conversion curve.
- **Leakage class.** Temporal-clean (calibration on past only). Labels are positive-unlabelled: a "false discovery" is "not confirmed
  within the window", so realised precision is a LOWER bound on true precision and the core guarantee is conservative; the shell
  guarantee refers to arrivals within a window of the calibration length (2 years) and is expected to under-cover on the 4.4-year
  test window: report coverage by 23.xx if an intermediate release is available, else state the caveat.
- **Test (<= 1 h).** No refit: leak-fold OOF scores on past (10 folds, 900 episodes), thresholds for alpha in {0.5, 0.7, 0.8} and
  beta in {0.5, 0.2}, then on temporal dev the realised mean FDP / miss rate, module size distribution, share of diseases with an empty
  core. Baselines: fixed top-k (k = 50, 100), global probability threshold without per-disease normalisation, unweighted conformal.
- **Expected effect vs leader.** AP and nuAP unchanged by construction (ranking untouched; the per-bin isotonic step is monotone
  within a task). New endpoints: realised risk within alpha +- 0.05; set-level F1 above top-100 by >= 10 % relative; core sizes
  typically 5-40 genes at alpha = 0.7, empty for a sizeable share of 3-5-seed diseases (an honest "no module yet").
- **Kill rule.** Realised FDP on temporal dev outside alpha +- 0.07 (bootstrap CI over clusters) for the weighted, Mondrian version,
  or F1 not above the best fixed top-k. If only the unweighted version fails, the shift (not the method) is the finding.

---

## Order by expected value per hour (own prior)
1. **E-B6** (degree decoupling): attacks the one measured trade directly, needs two fits and cached predictions, and its mandatory
   baseline settles whether sample weighting ever was a model gain.
2. **E-B5** (vintage parity): a concrete train/test inconsistency in the #2 feature; free diagnostic first.
3. **E-B2** then **E-B3** (gating by disease type): the strata evidence is already measured; E-B2 is the cheap soft version, E-B3 the
   explicit one, and E-B2's features are reused by E-B3, E-B7 and E-B9.
4. **E-B7** (seed self-audit), **E-B1** (lambdarank), **E-B4** (two episode families), **E-B8** (shift weights; run its 10-minute
   diagnostic early because it tells which of the others matter).
5. **E-B9** runs on whatever stacker wins; it needs no new model.

Interactions to keep the tests readable: run E-B6 and E-B1 on the same channels and report one joint frontier plot; E-B2 / E-B3 / E-B7
each against their own permuted-task control; do not stack accepted learner changes without re-running the permuted `cs_donor_perm` /
`noise1` controls, because a more flexible gate can also gate noise.

To verify before implementation (not checked here): availability of 19.04 per-datasource association scores for seed evidence mix
(E-B2, E-B3); an archived HPO annotation file from spring 2019 (E-B5); whether `tasks.build_holdout` can be pointed at the 21.04
matrix without touching sealed ids (E-B4).
