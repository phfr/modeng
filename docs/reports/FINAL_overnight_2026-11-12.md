# modeng overnight sprint, 2026-11-11 23:56 -> 2026-11-12 08:00: final report

> **Corrected summary (after the final Fable review `docs/reviews/R2_fable_final_report.md`; all 16 corrections applied below):**
> 1. On the temporal benchmark (138 confirm diseases) the final stacker `modeng-v1` ranks future Open Targets disease genes better than raw
>    transfer (AP 0.153 vs 0.119, nuAP 0.177 vs 0.104) and better than modet's headline on modet's candidate set for all arrivals
>    (AP +0.063 [+0.049, +0.082], nuAP +0.031 [+0.012, +0.054]), but it ties modet on modet's primary strict-a endpoint (nuAP +0.001 [-0.016, +0.025]).
> 2. These are development numbers, not a confirmation: 59 stacker variants were scored on confirm, the degree power was fixed after confirm
>    results by power were known, the stacker is trained on arrivals of the same time window (the past-trained version has AP equal to transfer),
>    and the red team reviewed only the earlier past-trained model.
> 3. The forward replication (AP 0.151 vs 0.085, nuAP 0.214 vs 0.083, 797 diseases) has no valid CIs and leans on channels that forecast the
>    Open Targets label pipeline (GWAS-Catalog lag, L2G priors), so the next step is one pre-registered run on untouched diseases with the recipe frozen.
> Also: strict-a on all genes is lower than modet's headline (stack_v3cv nuAP -0.027 [-0.050, -0.005]); module calibration (4b) belongs to stack_v3cv.

## 1. Result in five sentences
1. A cross-disease LightGBM stacker over **many weak, mostly new channels** (`stack_v3cv_ptr`, cv-trained on the 53 dev diseases) beats raw
   disease-to-disease transfer and modet's headline model on the temporal benchmark (seeds Open Targets 21.04, targets new by 25.09, 138 confirm
   diseases never used for model fitting, but the stacker is trained on dev arrivals of the same 21.04 -> 25.09 window) and replicates on the forward benchmark (25.09 -> 26.09, 797 confirm diseases).
2. vs modet's headline `modet-U-cov-Lb` on modet's own candidate set (per-disease identical scoring to modet): **nuAP 0.219 vs 0.173
   (+0.046 [+0.028, +0.067]), AP 0.150 vs 0.096 (+0.054 [+0.041, +0.070])**; dev 53 diseases +0.039 [+0.004, +0.077] / +0.060 [+0.037, +0.088]. Final model modeng-v1 (section 1c): nuAP +0.031 [+0.012, +0.054], AP +0.063 [+0.049, +0.082].
3. vs raw transfer on all 19,255 genes (temporal confirm): **nuAP 0.184 vs 0.104 (+0.080 [+0.062, +0.103]), AP 0.143 vs 0.119 (+0.025 [+0.002, +0.048])**,
   recall@100 +0.031 [-0.004, +0.061]. Forward confirm (v3 recipe without pt_ridge): nuAP 0.222 vs 0.083, AP 0.140 vs 0.085.
4. The new channels that carry the gain (Pareto gains on AP and nuAP against same-width permuted controls; priors + HPO only +0.006 nuAP [-0.001, +0.011], PubTator ridge +0.008 [+0.005, +0.011]): an EASE-style
   disease ridge regression, a joint non-negative lasso of the seeds over donor diseases (an idea from round-2 ideation), PubTator3
   literature donor sets cut by PMID date, HPO-2021 phenotype matching + channel priors (L2G-winner / Mendelian gene), virtual donor families
   (HPO, GO, Reactome, complexes, embedding modules) and gene-embedding similarities; GWAS-Catalog ingestion lag helps the GWAS-only stratum.
5. Caveats: 59 stacker variants (78 methods) were scored on temporal confirm during the night (no winner's-curse correction), several channels use
   pre-T0 resources whose annotation vintage is post-T0 (PubTator 2026 mapping of old PMIDs; GWAS-Catalog 2026 trait mapping; modern
   expression/perturbation views = transductive), and the Fable red-team review (`docs/reviews/R1_fable_redteam.md`) covers the past-trained round-1 stacker only; the cv-trained leader and modeng-v1 are not red-teamed.

## 1b. What may be claimed (red-team endpoints of `docs/reviews/R1_fable_redteam.md` recomputed for `stack_v3cv`, the leader without PubTator ridge; `results/review/leader_endpoints.tsv`)
- Claimable: on the temporal benchmark (138 confirm diseases), the stacker ranks future Open Targets disease genes better than modet's headline
  on its own candidate set for **all arrivals**: AP +0.050 [+0.037, +0.067], nuAP +0.039 [+0.023, +0.060], degree-matched AP +0.077 [+0.061, +0.094],
  precision@20 +0.067; better for hub and non-hub genes alike. The red team found no leakage, masking or selection artefact in the past-trained round-1 stacker; the v3 channels (priors, HPO, PubTator ridge) and cv training were not reviewed.
- **Not claimable: a gain on modet's primary endpoint (strict-a: genetic discoveries first seen 2022+): tie** on modet's candidate set (nuAP 0.108 vs 0.109, AP +0.010 n.s.); on all genes strict-a nuAP is lower (-0.027 [-0.050, -0.005]).
  The new channels mostly help with arrivals that already have older evidence somewhere (literature, phenotypes, ClinVar-type knowledge).
- vs raw transfer: better on degree-matched AP (+0.056) and precision@20 (+0.033) but worse on hub genes (hub AP -0.026): the deg^-0.5 training
  weight trades hubs for non-hubs; power 0.25 gave higher AP (0.127 vs 0.111) and recall@100 but lower nuAP (0.151 vs 0.175) in the red team's run on confirm; any power chosen after that needs fresh data.
- 59 stacker variants (78 methods) were scored on temporal confirm overnight; dev and confirm increments agree, but a fresh fold is the real test.

## 1c. Final model `modeng-v1` = `stack_v3ptr_p0.35` (leader recipe, positive weight deg^-0.35 chosen on dev mean of AP and nuAP; grid and rule were set after the red team showed confirm results by power, so the confirm numbers below are not a clean confirmation)
Temporal confirm, 138 diseases (cluster bootstrap): vs raw transfer AP 0.153 vs 0.119 (+0.034 [+0.011, +0.059]), nuAP 0.177 vs 0.104 (+0.073 [+0.053, +0.099]),
precision@20 +0.050, degree-matched AP +0.061, hub-gene AP -0.006 (n.s.). vs modet headline on its own candidate set: AP +0.063 [+0.049, +0.082],
nuAP +0.031 [+0.012, +0.054], degree-matched AP +0.082, hub and non-hub AP both better; strict-a: AP +0.023 [+0.006, +0.040], nuAP tie (+0.001 [-0.016, +0.025]). vs the leader `stack_v3cv_ptr`: AP +0.010 [+0.005, +0.014], nuAP -0.007 [-0.013, -0.002].
Forward replication (25.09 -> 26.09, inputs <= 25.09, 797 confirm diseases, v3 recipe without pt_ridge, same power; no CIs because every forward task is its own cluster; gwas_catlag is the 2nd most important feature): AP 0.151 vs transfer 0.085, nuAP 0.214 vs 0.083 (dev 0.166 / 0.236).
Endpoints: `results/review/leader_endpoints_stack_v3ptr_p0.35.tsv`. Command: see `docs/NEXT.md` (channels list) with `--pos-deg-power 0.35`.

## 2. What was done (process)
- Ideation process researched first (`docs/research/00_ideation_process.md`): independent generation, diversity prompting, novelty check,
  critic panel, closed loop with a frozen evaluator; LLM judges/tournaments treated as weak (not used for selection).
- Round 1: 8 independent Fable ideators (73 ideas), 3 Fable critics (statistics/leakage, biology/curation, engineering), literature novelty check.
- Wave 1: 8 implementers (+ lead) tested ~30 ideas as channels against a learned stacker with permuted controls; one evolutionary
  program-search loop (F2-3). Round 2: 2 fresh-island ideators (16 ideas, one -> cs_donor, a positive result), 2 evolution ideators from measured results.
- Usage limit stopped the sprint 02:50-04:11; resumed from `docs/NEXT.md`.

## 3. Measured table (temporal confirm, 138 diseases; full rows in `results/leaderboard.tsv`, facts in `docs/insights.md`)
| model | AP | nuAP | onnet nuAP |
|---|---|---|---|
| transfer_jaccard | 0.119 | 0.104 | 0.128 |
| modet headline (onnet) | - | - | 0.173 (onnet AP 0.096) |
| stackP_base (5 base channels, past-trained) | 0.081 | 0.134 | 0.161 |
| stackP_bdv (+donor families, +view similarities) | 0.091 | 0.151 | 0.171 |
| stackP_v2 (+cs_donor, dA_pt, dc_ridge, gwas_catlag; past-trained) | 0.119 | 0.182 | 0.205 |
| stack_v3cv (v2 channels + priors + HPO phenotypes; cv) | 0.140 | 0.176 | 0.212 |
| stack_v3cv_ptr (+ PubTator ridge; cv) = nuAP leader | 0.143 | 0.184 | 0.219 (onnet AP 0.150) |
| **stack_v3ptr_p0.35 = modeng-v1 (final)** | **0.153** | **0.177** | **0.203** (onnet AP 0.160) |

## 4. Forward replication (25.09 -> 26.09)
Same channel recipe (v3, without pt_ridge and without backlog channels), cv on 842 dev diseases, inputs dated <= 25.09: confirm (797) nuAP 0.222
vs stack_base 0.137 (+0.085) and transfer 0.083; AP 0.140 vs 0.077 / 0.085. Caveats: forward CIs treat tasks as independent (too narrow);
gwas_catlag (2nd most important feature of the forward model) and L2G-winner priors forecast the label pipeline; no forward run without them exists, so the forward gain is not evidence for new biology.

## 4b. Module calls
Isotonic score -> probability map fitted on dev OOF (stack_v3cv), applied to confirm: at tau 0.2 a median of 11 extra genes per disease
(136/138 diseases) at realised precision 0.21 (predicted 0.28), 10 % of all future targets recovered; tau 0.5: precision 0.50 (pred 0.70).
Calibration is optimistic by ~1.4x; the factor 0.7 is estimated on confirm and the map belongs to `stack_v3cv`, not modeng-v1 (refit needed). `scripts/module_cutoff.py`.

## 5. Negative results (kept in `docs/invalidations.md` / `docs/ideas.md`)
Evolved transfer kernel (nuAP via degree only; AP halves), seed-set transformer (memorises arrivals; self-supervised twin = transfer level),
gene-kernel / homology lifts (AUROC only), Perturb-seq regulators/responders and perturbation/DepMap/IMPC donor sets (chance for OT labels),
trending / directed transfer, learned disease encoder, own/proxy-trait MAGMA z, channel-typed transfer, joint lasso over gene-set families.

## 6. Label-process forecasting (separate product, not biology)
On forward, sub-threshold evidence already present at the seed release (backlog channels) more than doubles the stacker (nuAP 0.137 -> 0.307,
AP 0.077 -> 0.196): 47 % of next-year arrivals already had evidence for the disease family. Useful to forecast Open Targets, not new biology.

## 7. Leaks and bugs caught (all fixed)
Float-residue tie leak in masked transfer (holdout transfer_onto 0.117 -> 0.036); universe-order bug in views/donors; holdout ancestor tell;
past-cluster CV leak (Louvain leak folds); evaluation-universe artefact that made transfer look better than modet's headline.

## 8. Next
Fresh confirmation on untouched diseases (e.g. sealed-like new fold or the next OT release), archived PubTator/GWAS-Catalog vintages,
module output (calibrated cutoff + subex `leiden_anno` decomposition), round-2 evolution ideas (`docs/ideation/r2_evolve*.md`).

## 9. Morning follow-up (07:20-08:30)
- **R3 red team of modeng-v1** (`docs/reviews/R3_fable_redteam_v1.md`): holds with caveats; no leak, no memorisation; out of window the gain is carried by
  label-process channels; removing confirm targets with pre-existing 21.04 family evidence keeps the AP gain over transfer (+0.025) but makes the nuAP gain
  over modet's headline a tie.
- **Prospective version** (trained on 19.04->21.04 only, `docs/reports/r3_prospective.md`): vs transfer nuAP +0.082 [+0.057, +0.110], AP +0.018 (n.s.);
  vs modet headline onnet AP +0.046, nuAP +0.032 (CIs > 0); strict-a tie. Half of the AP margin of v1 comes from same-window training.
- **Forward without label-process channels** (in-window): AP 0.103 vs 0.085 (+0.018), nuAP 0.156 vs 0.083.
- **Strict-a attack** (gating, class-A head; `docs/reports/r3_strict_a.md`): no strict-a gain; context features +0.006 AP overall.
- **Module calibration refit** for modeng-v1; dev-only shrink 0.8-0.9 (confirm ~0.72).
- **Sealed-fold recommendation: not now.** The robust claim (nuAP and onnet AP over transfer and over modet's headline, also prospectively) is on all
  arrivals; modet's sealed-fold primary is strict-a, where every variant ties. Use the next OT release as the fresh fold; make the no-label-process arm co-primary.
