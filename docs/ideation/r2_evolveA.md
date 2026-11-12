# Round 2, evolution ideator A (2026-11-12)

Inputs read: `docs/reports/r1_summary.md`, `docs/insights.md`, `docs/invalidations.md`, `docs/reports/r1_*.md` (verdicts, failures, new ideas),
`docs/reports/r2_evolve.md`, `docs/ideas.md`, `docs/ideation/brief.md`, `implementer_brief.md`, code `ext/lead_cs.py`, `ext/donors_a.py`,
`ext/disease_c.py`, `stack.py`, `results/leaderboard.tsv`. No web search. Nothing here was run: all effects are priors.

**Reference leader L** = bdv + cs_donor + dA_pt + dc_ridge, temporal, `--train past`. Measured: bdv + cs + pt = AP 0.111 / nuAP 0.175 (confirm).
`stackP_v1` / `stackP_v2` in the leaderboard (AP 0.114 / 0.119, nuAP 0.181 / 0.182) look like L, but I did not verify their channel lists.
Minimum detectable effect ~0.01 nuAP on confirm, so most single ideas below are expected to sit at 0.3-1 MDE; the test for each is
dev first (pre-registered sign on BOTH AP and nuAP), confirm once.

**Rules applied to every test** (from the invalidations): `python -m modeng.stack --bench temporal --train past --channels <L>,<new>` vs `<L>`
and vs `<L>,<new>_perm` (same-width gene-permuted control, dilution costs up to -0.010 nuAP); paired cluster bootstrap with `modeng.compare`;
a gain counts only if AP and nuAP both rise; any subtract-style masking is quantised (`np.round(x, 5)`); any model fitted inside `past`
uses `settx.groups.leak_folds`; the channel must return values on `B.name == 'past'` or NaN.

A code fact used below: `dc_ridge` is already EASE in the dual (`X^T (X X^T + lambda I)^-1 X s = (X^T X + lambda I)^-1 X^T X s`; the
zero-diagonal constraint only touches seed genes, which are not ranked). So "gene-gene EASE" is not a new idea; it is the same channel.

---

## MUTATE

### E-A1  cs_donor with seed leave-out: per-task alpha and a reliability feature
- **Mechanism.** `cs_donor` uses one fixed alpha (1e-5) and reports the in-sample reconstruction. Mutation: for each task run the lasso
  on K = 5 seed folds, pick alpha on the path {1e-4 .. 1e-7} by the rank of the held-out seeds, emit the fold-averaged out-of-sample
  score `X b`, and add one task-level column: mean percentile of held-out seeds (how well this disease is explained by donors at all).
  The stacker today has only `log n_seeds` as a task-level feature, so it cannot tell a disease where transfer works from one where it does not.
- **Builds on.** F1-CSDONOR (`ext/lead_cs.py`); insight "success is decided by transfer support".
- **Data.** Context label matrix only.
- **Leakage class.** Same as the context (temporal-clean); seeds only, targets never touched; exact masking via `context_for`.
- **Test (<= 1 h).** `ext/evoA_cs.py`: channels `cs_cv` and `cs_rel` (constant per task). 5 x 4 lassos per task, 4 threads, ~25 min for
  past-900 + temporal. Stack L with `cs_donor` replaced by `cs_cv, cs_rel`; control = `cs_rel` with tasks shuffled.
- **Expected vs L.** AP +0.002 to +0.004, nuAP +0.002 to +0.005 (dev); most of it from `cs_rel` gating, not from alpha.
- **Kill.** Dev AP or nuAP delta <= 0, or the shuffled-`cs_rel` control gives >= half of the gain. Then keep fixed alpha.

### E-A2  dA_pt by disease identity instead of seed overlap (own-MeSH NPMI row)
- **Mechanism.** `dA_pt` selects MeSH donor rows by hypergeometric overlap with the seeds, so it is weak for diseases with 3-5 seeds and
  ignores the one row that names the disease. Mutation of the donor-selection component: map the task disease to its MeSH descriptor(s)
  through the 21.04 xrefs (plus ontology parents when missing) and score each gene by NPMI of gene x MeSH co-mention (count-thresholded),
  with the same PMID cut per benchmark. Seedless, so it also works where transfer has no support.
- **Builds on.** `dA_pt` (`ext/donors_a.py`, new idea 3 in `r1_donors.md`); LT-2.
- **Data.** `data/raw/pubtator3/mesh_gene_counts` (cut bins 2019-03 / 2021-03 / 2025-08), OT disease xrefs of the seed release.
- **Leakage class.** Temporal-clean by PMID date, but **label-adjacent**: this is close to an emulation of OT's Europe PMC evidence. Must
  be reported per arrival class (literature / GWAS / ClinVar-curated arrivals).
- **Test (<= 1 h).** Channel `pt_own` (NaN if no MeSH mapping; report coverage). L + `pt_own` vs L vs L + gene-permuted `pt_own`.
  Stratify the gain by arrival class and by n_seeds <= 5.
- **Expected vs L.** AP +0.004 to +0.008, nuAP +0.003 to +0.006; larger in the <= 5-seed stratum. Coverage is the main risk (MeSH mapping maybe 60-70 %).
- **Kill.** Dev gain confined to literature-borne arrivals (>= 80 % of the delta) -> keep only as a label-process baseline, not as a
  biology claim; or both deltas <= 0.

---

## CROSSOVER

### E-A3  Joint ridge over literature rows (dc_ridge solver x dA_pt matrix)
- **Mechanism.** Two results say that explaining the seed set jointly beats independent per-donor weights on the label matrix (cs_donor,
  dc_ridge), but the literature matrix is still scored donor by donor (hypergeometric top-50). Apply the ridge solver to the PubTator
  MeSH x gene matrix (rows unit-normed, optionally stacked under the label rows), score = `X_pt^T w`.
- **Why the gains should add and not overlap.** dc_ridge's gain comes from the solver on label rows; dA_pt's from a new matrix with a
  weak scorer. `cs_universal` (no gain) is not a counter-example: it added gene-set families (GO, complexes, k-means), which are not
  disease-derived; MeSH rows are disease-like rows with dense redundancy (many near-duplicate descriptors), which is exactly where
  joint regression removes double counting.
- **Builds on.** C-RIDGE (`_ridge_w`), LT-2 / dA_pt.
- **Data.** `data/donors/a_pt19|21|25.npz`. No masking needed for literature rows (no label pairs), only the PMID cut.
- **Leakage class.** As dA_pt (temporal-clean by date, label-adjacent).
- **Test (<= 1 h).** `pt_ridge`: Gram of the MeSH rows (if > 20k rows keep rows with >= 5 genes and top 15k by size), one inverse,
  lambda in {1, 10} fixed on past held-out groups. L + `pt_ridge` vs L; also L with `dA_pt` replaced by `pt_ridge`; permuted control.
- **Expected vs L.** AP +0.003 to +0.006, nuAP +0.003 to +0.007.
- **Kill.** Replacement run not better than L on both metrics on dev -> the hypergeometric scorer is already sufficient; drop.

### E-A4  Backlog transfer: donor weights applied to the relatives' catalog lag (dc_ridge / cs_donor weights x gwas_catlag matrix)
- **Mechanism.** `gwas_catlag` only helps the ~30 % of diseases with an own GWAS; that coverage limit is why confirm is not significant.
  Build the disease x gene matrix C of catalog hits not yet in the label matrix at the seed date, for every context disease, and score
  a task by `C^T w` with the non-negative ridge / cs donor weights w (`dc_ridge_pos`, `cs_donor` coefficients). A gene that a jointly
  selected donor disease has in its public backlog is predicted for the task disease.
- **Why additive.** Transfer channels see only pairs already labelled; catlag sees only the own disease's catalog rows. The cross term
  (donor's unlabelled hits) is in neither. Measured support: 86 % of GWAS-only arrivals already have L2G >= 0.5 for another trait.
  New reason vs the failed CI-08 / seed-weighted cross-trait z: thresholded catalog hits and jointly learned donor weights, not a linear
  mix of genome-wide z with overlap weights.
- **Builds on.** SG-7 (`ext/gwas_d.py`), C-RIDGE, F1-CSDONOR.
- **Data.** GWAS Catalog associations with date added <= seed date (already parsed by implementer D), context matrix.
- **Leakage class.** As gwas_catlag: clean by catalog date, residual route = 2026-10 trait/gene mapping (use an archived snapshot when available).
- **Test (<= 1 h).** Channel `catlag_tr` (zero the own-disease row and masked rows' columns as in `_ridge_w`; quantise). L + `gwas_catlag`
  + `catlag_tr` vs L + `gwas_catlag`. Report the GWAS-only stratum and the diseases without own GWAS separately.
- **Expected vs L (+catlag).** AP +0.003 to +0.006, nuAP +0.002 to +0.005; GWAS-only stratum AP 0.044 -> ~0.055; gain should appear in
  the 70 % of diseases without own GWAS (that is the pre-registered signature).
- **Kill.** No gain in the no-own-GWAS stratum on dev, or permuted-gene C gives the same delta.

---

## REPAIR

### E-A5  Repair settx_auxonly: residual (offset) training and release-matched twins
- **Named failure.** Strong alone (nuAP 0.124 / 0.130) but zero stacker increment; ~70 % of its gain is a re-learned combination of
  the context features (transfer, popularity) it is given, and its `past` predictions are cross-fitted OOF while temporal uses the final
  model (reliability shift).
- **Mechanism.** (1) Remove the context features from the input and give them as a fixed logit offset (`transfer_jaccard` + `dc_ridge`
  computed on each masked aux episode), so the network can only fit what these two do not explain. (2) Train two models, one on 19.04
  rows for `past`, one on 21.04 rows for temporal, both aux-only and both scoring tasks they never saw, so the channel has the same
  meaning in training and test. (3) Emit only the residual logit as the channel.
- **Builds on.** ML-5 aux-only twin (`src/modeng/settx`), r1_settx new ideas 1-2.
- **Data.** As settx aux episodes (masked context rows + pre-2021 donor sets, clean views).
- **Leakage class.** Temporal-clean (clean views); never sees an arrival.
- **Test (<= 1 h).** 2 x 1,500 steps via `~/biotools/bin/gpu-run --mem 8 -- ...` (background), then L + `settx_res` vs L vs L + view-permuted twin.
- **Expected vs L.** AP +0.000 to +0.003, nuAP +0.000 to +0.004; honest prior of a real gain ~25 %. The zero-support stratum AUROC
  (0.66-0.68 = popularity today) is the secondary endpoint.
- **Kill.** Real-views minus permuted-views increment < +0.003 nuAP on dev, or AP negative. Then the set transformer line is closed.

### E-A6  Repair gwas_catlag confirm: locus window with a host-gene prior instead of MAPPED_GENE
- **Named failure.** Dev significant, confirm CIs include 0 (AP +0.008, nuAP +0.003); catlag credits the catalog's mapped (mostly
  nearest) gene, while Open Targets later attaches genes by L2G; the mapping comes from a 2026 release.
- **Mechanism.** For each lagging lead SNP score every gene within +-500 kb by `exp(-dist / 50 kb)` x host-gene prior, normalised to sum
  1 per locus; prior = implementer H's `pri_l2g_n` (gene is an L2G winner for other traits at the seed date; it lifts the GWAS stratum
  AUROC but not nuAP as a global prior) plus a coding flag. Add a "locus already explained by a seed" flag (lead SNP within 500 kb of a
  seed) as a second column, since the seed-weighted result suggests such loci do not yield new genes.
- **Builds on.** SG-7, r1_gwas new ideas 2-3, r1_priors new idea 4, WC-L2GEMU (the cheap part only).
- **Data.** Catalog rows (date added <= seed), gene coordinates (GRCh38), `pri_l2g_n` at the seed release. `pri_*` are NaN on `past`
  (no 19.04 evidence): on past use distance + coding only, or run this arm with `--train cv` and say so.
- **Leakage class.** As gwas_catlag; positions instead of MAPPED_GENE removes part of the re-mapping route (rsID -> position is stable).
- **Test (<= 1 h).** Channels `catlag_locus`, `catlag_seedlocus`. L + these vs L + `gwas_catlag`; endpoints: covered-task AP / nuAP and
  GWAS-only stratum R@100, dev then confirm.
- **Expected vs L (+catlag).** AP +0.002 to +0.005, nuAP +0.002 to +0.004 overall; covered diseases +0.01; GWAS-only R@100 +0.03.
- **Kill.** GWAS-only stratum AP on dev not above the MAPPED_GENE version (0.048), or the distance-only variant equals the prior variant
  (then the prior carries nothing and the nearest gene is as good as we get).

---

## FRESH (p_typical < 0.10, not related to the result table)

### E-A7  Hubness correction in score space (CSLS / inverted softmax from cross-lingual retrieval)
- **Mechanism.** In dense retrieval some items are near-neighbours of every query; CSLS fixes this by subtracting each item's mean
  similarity to its k nearest queries. Here: for each channel score matrix [diseases x genes], subtract (or z-score by) the gene's mean
  score over its top-k context diseases, computed on context / past tasks only. This removes "scores high for every disease" without
  using network degree or popularity, which is what the AP-vs-nuAP trade-off of the deg^-0.5 weighting tries to do bluntly.
- **Builds on.** Nothing in the table; motivated by the measured AP/nuAP trade-off (power sweep) and the degree lesson of F2-3.
- **Data.** Existing channel caches for `past` (1,786 tasks) as the reference population; temporal scored against it.
- **Leakage class.** Temporal-clean (reference statistics from past-episode seeds only; no targets).
- **Test (<= 1 h).** `ext/evoA_csls.py`: `csls_<ch>` for transfer_jaccard, dc_ridge, cs_donor, dA_pt (k = 10). L + 4 columns vs L vs L + 4
  permuted columns; and L trained with `--pos-deg-power 0.25` with / without the columns (does specificity let us give AP back without losing nuAP?).
- **Expected vs L.** AP +0.002 to +0.006, nuAP +0.000 to +0.004. p(real Pareto gain) ~30 %.
- **Kill.** nuAP up and AP down (it is degree down-weighting again), or the gain vanishes on a degree-matched endpoint (`sa_ap`). p_typical ~0.05.

### E-A8  Label-noise model: forecast retraction, weight seeds and donor pairs by persistence
- **Mechanism.** Every model treats the label matrix as true, yet 32 % of GWAS-only arrivals do not persist and "recently added donor
  genes predict worse" (IR-6). Estimate for each (disease, gene) pair the probability that it survives the next release, first with a
  future-free proxy (leave-one-out reconstruction of the pair by the ridge: isolated pairs are suspect), validated on 19.04 -> 21.04
  removals; then replace the binary seed vector and donor rows by persistence weights in `dc_ridge` and `cs_donor`.
- **Builds on.** Nothing in the table (all R1 ideas model arrival, none removal). Uses the dc_ridge machinery only as a tool.
- **Data.** 19.04 / 21.04 / 25.09 label matrices (removals), context matrix.
- **Leakage class.** Proxy version temporal-clean (no future data). A removal classifier trained on 19.04 -> 21.04 is clean for
  temporal but in-sample on `past` episodes: cross-fit by leak folds or use the proxy on past.
- **Test (<= 1 h).** Step 0 (10 min, decides everything): AUROC of LOO-reconstruction for "pair of 19.04 absent in 21.04"; removal base
  rate. Step 1: `dc_ridge_w`, `cs_w` with weighted seeds / rows; L with these added vs L vs weights permuted within row.
- **Expected vs L.** AP +0.001 to +0.004, nuAP +0.001 to +0.004; the persistence score is also the missing second axis for the module (E-A9).
- **Kill.** Step 0 AUROC < 0.60 or removal rate < 3 % of pairs -> stop before any stacker run. p_typical ~0.08.

---

## MODULE

### E-A9  From ranking to a calibrated module: membership probability, FDR-controlled cutoff, donor-explained sub-modules
- **Mechanism.** (1) Calibrate the L stacker score to P(arrival within the horizon) with isotonic regression on leak-fold OOF past
  episodes, per stratum (transfer support zero / non-zero x n_seeds bin); convert to membership with a label-frequency correction
  `p_member = min(1, p_arrive / c)`, c estimated from seed hold-out recovery (Elkan-Noto: seeds are known members, the fraction of
  held-out seeds the pipeline "re-labels" at a given score gives c). (2) Module = seeds + the largest top-k whose estimated FDR
  `1 - mean(p_member of top-k)` stays <= q (q = 0.5 "candidate module", 0.2 "core"), with a conformal risk-control correction fitted on
  past so the realised FDR bound holds per disease on average; diseases where even k = 1 fails get an explicit "no confident extension".
  (3) Structure: assign each module gene to the donor(s) with the largest `cs_donor` coefficient x membership (non-negative, sparse, so
  it reads as "this part of the module is shared with disease X / MeSH term Y"), giving sub-modules; persistence (E-A8) and channel of
  evidence (transfer / literature / GWAS backlog) are reported per gene.
- **Builds on.** All L channels; cs_donor coefficients for structure; WC-CAPTURE and F1-ENDO are related but untested, this is the
  minimal version that needs no new data.
- **Data.** OOF stack scores on past (leak folds), temporal dev / confirm scores, 25.09 and 26.09 labels for evaluation.
- **Leakage class.** Temporal-clean; calibration never sees temporal targets. Caveat stated in every output: a non-arrival is not a
  non-member (positive-unlabelled), so realised FDR against 25.09 labels is an upper bound; re-score against 26.09 to show how much of
  the "false" part arrives later.
- **Test (<= 1 h).** `scripts/module_calibrate.py`: reliability curve and ECE on temporal dev / confirm; realised FDR vs nominal q;
  module size distribution; share of modules with no confident extension; fraction of "false" members that arrive by 26.09; for
  structure, enrichment of sub-modules in one MSigDB / HPO term vs size-matched random splits of the module.
- **Expected effect.** Per-disease AP / nuAP unchanged by construction (monotone within stratum; across strata a small change, |delta| < 0.002).
  New endpoints: ECE < 0.02, realised FDR within +-0.1 of nominal at q = 0.5; expected median module extension small (3-10 genes),
  many rare diseases with none: that is the honest state of the evidence.
- **Kill.** Realised FDR at q = 0.5 above 0.7 on confirm after the 26.09 re-score (calibration does not transfer across eras) -> report
  rank-only output plus per-stratum precision tables instead of a per-disease cutoff.

---

## Own ranking (expected value per hour)
1. **E-A4** backlog transfer (attacks the measured coverage limit of the strongest GWAS signal; GWAS-only is 57 % of positives).
2. **E-A3** joint ridge over literature rows (two measured mechanisms, both cheap, one inverse).
3. **E-A9** module calibration (the user's goal; no new data; produces the first module-level endpoints).
4. E-A2 own-MeSH row, 5. E-A7 CSLS, 6. E-A1 cs seed leave-out, 7. E-A6 locus prior, 8. E-A8 persistence, 9. E-A5 settx residual.

## Not verified
- Channel lists of `stackP_v1` / `stackP_v2`; number of MeSH rows in `a_pt21` (Gram size for E-A3); MeSH xref coverage of the 191
  temporal diseases (E-A2); whether implementer D kept the per-disease catalog matrix for all context diseases (E-A4) or only for tasks;
  the removal rate 19.04 -> 21.04 (E-A8); whether a 19.04 ontology is on disk for past-side mappings.
