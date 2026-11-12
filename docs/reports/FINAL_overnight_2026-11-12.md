# modeng overnight sprint, 2026-11-11 23:56 -> 2026-11-12 08:00: final report

## 1. Result in five sentences
1. A cross-disease LightGBM stacker over **many weak, mostly new channels** (`stack_v3cv_ptr`, cv-trained on the 53 dev diseases) beats raw
   disease-to-disease transfer and modet's headline model on the temporal benchmark (seeds Open Targets 21.04, targets new by 25.09, 138 confirm
   diseases never used for model fitting) and replicates on the forward benchmark (25.09 -> 26.09, 797 confirm diseases).
2. vs modet's headline `modet-U-cov-Lb` on modet's own candidate set (per-disease identical scoring to modet): **nuAP 0.219 vs 0.173
   (+0.046 [+0.028, +0.067]), AP 0.150 vs 0.096 (+0.054 [+0.041, +0.070])**; dev 53 diseases +0.039 [+0.004, +0.077] / +0.060 [+0.037, +0.088].
3. vs raw transfer on all 19,255 genes (temporal confirm): **nuAP 0.184 vs 0.104 (+0.080 [+0.062, +0.103]), AP 0.143 vs 0.119 (+0.025 [+0.002, +0.048])**,
   recall@100 +0.031 [-0.004, +0.061]. Forward confirm (v3 recipe without pt_ridge): nuAP 0.222 vs 0.083, AP 0.140 vs 0.085.
4. The new channels that carry the gain (each a Pareto gain on AP and nuAP with a same-width permuted control at ~0): an EASE-style
   disease ridge regression, a joint non-negative lasso of the seeds over donor diseases (an idea from round-2 ideation), PubTator3
   literature donor sets cut by PMID date, HPO-2021 phenotype matching + channel priors (L2G-winner / Mendelian gene), virtual donor families
   (HPO, GO, Reactome, complexes, embedding modules) and gene-embedding similarities; GWAS-Catalog ingestion lag helps the GWAS-only stratum.
5. Caveats: about 20 stacker variants were viewed on temporal confirm during the night (no winner's-curse correction), several channels use
   pre-T0 resources whose annotation vintage is post-T0 (PubTator 2026 mapping of old PMIDs; GWAS-Catalog 2026 trait mapping; modern
   expression/perturbation views = transductive), and a Fable red-team review (`docs/reviews/R1_fable_redteam.md`) decides what may be claimed.

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
| **stack_v3cv_ptr (+ PubTator ridge; cv) = leader** | **0.143** | **0.184** | **0.219** (onnet AP 0.150) |

## 4. Forward replication (25.09 -> 26.09)
Same channel recipe (v3, without pt_ridge and without backlog channels), cv on 842 dev diseases, inputs dated <= 25.09: confirm (797) nuAP 0.222
vs stack_base 0.137 (+0.085) and transfer 0.083; AP 0.140 vs 0.077 / 0.085. Caveats: forward CIs treat tasks as independent (too narrow);
gwas_catlag and L2G-winner priors partly forecast the label pipeline.

## 4b. Module calls
Isotonic score -> probability map fitted on dev OOF (stack_v3cv), applied to confirm: at tau 0.2 a median of 11 extra genes per disease
(136/138 diseases) at realised precision 0.21 (predicted 0.28), 10 % of all future targets recovered; tau 0.5: precision 0.50 (pred 0.70).
Calibration is optimistic by ~1.4x: multiply predictions by 0.7. `scripts/module_cutoff.py`.

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
