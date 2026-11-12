# R5 witness audit of modeng-v1 (`stack_v3ptr_p0.35`), 2026-11-12

Scope: every input channel ("witness") of the frozen modeng-v1 recipe (`docs/preregistration_modeng_v1.md`): what it
computes, where its data come from and how they are dated, how much it contributes, where it matters, and which channels
repeat each other. **All numbers here are descriptive.** Temporal confirm (138 diseases) has been looked at many times
overnight; no claim is decided here, and none of the "drop / merge" suggestions below may be confirmed on temporal confirm.
They need the fresh fold of the pre-registration (section 5 there).

Scripts (all `scripts/r5_audit_*.py`, outputs in `results/r5/`):
`r5_audit_ablate.py` (frozen recipe retrained per arm, cv on 53 dev diseases, deg^-0.35, 400 rounds, 3000 negatives,
LightGBM seed 0 = recipe; arms `_seed1`, `_seed2` = same arm with LightGBM seeds 1 and 2), `r5_audit_shap.py` (gain and
`pred_contrib` SHAP of the dev->confirm model), `r5_audit_channels.py` (coverage, alone metrics, redundancy),
`r5_audit_cost.py` (timings), `r5_audit_summarize.py` (bootstrap, disease types). The seed-0 full arm reproduces the
leaderboard row of modeng-v1 exactly (dev AP 0.1756 / nuAP 0.1502, confirm 0.1532 / 0.1770).

## 1. Inventory (65 channels + log n_seeds; each channel enters as raw value and within-task percentile rank = 131 features)

T0 = seed release OT 21.04 (2021-04). Leakage classes: **clean** = temporal-clean (content and vintage <= T0);
**vintage** = pre-T0 content with post-T0 annotation / model vintage; **transductive** = post-T0 label-free data;
**label-process** = forecasts Open Targets' own evidence pipeline (dated <= T0, but it predicts how labels are made, not
biology). Coverage = share of (task, gene) cells with a value / share of cells that are non-zero, and share of temporal
tasks where the channel is non-trivial (not all-NaN, not constant); temporal confirm (dev within +-0.01 except where
noted). Alone = the channel used as the ranking by itself (confirm AP / nuAP / AUROC; transfer_jaccard 0.119 / 0.104 / 0.70
for reference). Cost = recomputation on 53 dev tasks from cached inputs (2 BLAS threads, loaded machine; `results/r5/channel_cost.tsv`).
Full per-channel table: `results/r5/channels_inventory.tsv`.

| channel (group) | computes | input, version, content date | leakage | coverage value / non-zero; tasks | per | alone AP / nuAP / AUROC | cost |
|---|---|---|---|---|---|---|---|
| `transfer_jaccard` (base) | sum of the genes of the top-50 context diseases by Jaccard with the seeds (own row, descendants masked) | OT 21.04 associations (modet groundtruth), masked context | clean | 100 % / 5 %; 100 % | task | 0.119 / 0.104 / 0.70 | < 1 s |
| `transfer_onto` (base) | same, donors weighted by ontology proximity | OT 21.04 + EFO 21.04 ontology | clean | 100 % / 2 %; 95 % | task | 0.059 / 0.055 / 0.62 | 3 s |
| `popularity` (base) | number of context diseases of the gene (hidden pairs subtracted) | OT 21.04 | clean | 100 % / 52 %; 100 % | gene (mask-adjusted) | 0.019 / 0.012 / 0.79 | < 1 s |
| `degree` (base) | STRING degree | STRING v11.5 physical, no text-mining, score >= 400 (released 2021-08, ~4 months after T0) | vintage | 100 % / 66 %; 100 % | gene | 0.004 / 0.001 / 0.61 | < 1 s |
| `rwr_string` (base) | random walk with restart 0.5 from the seeds | STRING v11.5 as above | vintage | 100 % / 66 %; 100 % | task | 0.014 / 0.006 / 0.65 | < 1 s |
| 15 donor families `don_*` (donor) | Jaccard top-50 transfer over virtual-donor gene sets (`data/donors/MANIFEST.tsv`) | MSigDB 7.4 GO / Reactome / WikiPathways (2021-03), HPO 2021-01-15, IMPC 13.0 (2020-07), Complex Portal 2021-03-10, Compara 103 paralogs (2021-02), DepMap 20Q4 k-means, CladeOScope NPP (2021-01-25) k-means = clean; ESM2-15B / ProtT5 k-means = vintage (sequence models released 2021-22); HPA / GTEx current k-means, Replogle 2022 K562, GWCD4i 2025 k-means = transductive | 9 clean, 2 vintage, 4 transductive | 100 % / 0.1-27 %; 91-100 % | task | `don_hpo2101` 0.080 / 0.068 / 0.65; MSigDB 0.020-0.024; the other 11 <= 0.010 (AUROC 0.50-0.59) | < 1 s each |
| 30 view similarities `vmean_* / vmax_*` (view) | mean and top-3 mean cosine of the gene to the seeds in 15 gene embeddings (`data/views/MANIFEST.tsv`) | ESM C 300M on Swiss-Prot 2021_02, ProtT5, ESM2-15B (vintage); DepMap 20Q4, CladeOScope 2021-01, GO 2021-02-01, Reactome v72 2020-06 (clean); HPA single-cell / tissue and GTEx (current, 'modern'), Perturb-seq K562 / RPE1 2022, CD4 2025, HepG2 / Jurkat 2025 (transductive) | 8 clean, 6 vintage, 16 transductive | 10-99 % (Perturb-seq RPE1 / HepG2 / Jurkat 10 %, K562 51 %, CD4 64 %, Reactome 55 %, GO 82 %, rest > 90 %); 84-100 % | task | AP 0.003-0.018, AUROC 0.50-0.65 (GO / Reactome / sequence best; Perturb-seq at chance) | < 1 s each |
| `cs_donor` | non-negative lasso of the seed indicator over the top-300 Jaccard context rows; score = reconstruction | OT 21.04 masked context | clean | 100 % / 1.4 %; 100 % | task | 0.076 / 0.072 / 0.61 | < 1 s |
| `dA_pt` | hypergeometric top-50 donor transfer over PubTator3 gene x MeSH-disease co-mention sets | PubTator3 (2026 annotation release) restricted to PMID <= 33,750,000 (2021-03-22) | vintage | 67 % / 12 %; 100 % | task | 0.093 / 0.095 / 0.77 | 27 s |
| `dc_ridge`, `dc_ridge_pos` (dc_ridge) | EASE / SLIM ridge regression of the seed vector on unit-norm context rows (masked rows removed exactly); `_pos` = positive weights, top-50 | OT 21.04 masked context | clean | 100 % / 52 % and 6 %; 100 % | task | 0.104 / 0.114 / 0.67 and 0.113 / 0.106 / 0.72 | 2 s each (dense Gram inverse) |
| `gwas_catlag` | max -log10 p of GWAS Catalog associations (p <= 1e-6) added <= 2021-04-30, mapped to the disease, descendants (w 1) or specific ancestors (w 0.7) | GWAS Catalog release 2026-10-22 (`MAPPED_GENE`, `MAPPED_TRAIT_URI` = 2026 mapping) | label-process + vintage | value only for covered tasks: 38 % of confirm, 45 % of dev tasks; non-zero 0.15 % of cells | task | 0.047 / 0.049 / 0.53 | 5 s |
| `pri_l2g_n`, `pri_l2g_max`, `hz_gwas` (priors) | log1p number of other diseases where the gene is an L2G >= 0.5 winner, max L2G, and `pri_l2g_n` x the task's seed GWAS share | OT 21.04 `ot_genetics_portal` evidence (task family excluded) | label-process | 100 % / 41 %, 86 %, 20 %; `hz_gwas` non-trivial in 50 % of tasks | gene (family-adjusted); `hz_gwas` task-scaled | 0.020 / 0.020 / 0.66; 0.007 / 0.008 / 0.65; 0.018 / 0.017 / 0.62 | 2 s (first call loads the evidence aggregate) |
| `pri_mend_n` (priors) | log1p number of other diseases with ClinVar >= 0.5 or curated (G2P, GEL, ClinGen, UniProt, Orphanet) evidence | OT 21.04 evidence (task family excluded) | clean (label-adjacent popularity) | 100 % / 26 %; 100 % | gene (family-adjusted) | 0.019 / 0.009 / 0.72 | < 1 s |
| `pri_hpo_n`, `hz_mend` (priors) | log1p number of HPO terms of the gene; x (1 - seed GWAS share) | HPO genes_to_phenotype 2021-01-15 | clean | 22 % (NaN = no HPO) and 100 % / 21 %; 100 % and 94 % | gene; task-scaled | 0.015 / 0.008 / 0.71; 0.015 / 0.007 / 0.70 | < 1 s |
| `ph_mean`, `ph_top3` (ph) | IDF-weighted HPO-profile cosine of the gene to the seed profile / mean of top-3 cosines to single seeds | HPO 2021-01-15 | clean | 22 % (HPO genes); 100 % | task | 0.076 / 0.069 / 0.72; 0.059 / 0.059 / 0.72 | < 1 s |
| `pt_ridge`, `pt_ridge_pos` (pt_ridge) | ridge regression of the seed vector on the top-500 PubTator literature sets (`a_pt21`) | as `dA_pt` | vintage | 100 % / 33 % and 27 %; 100 % | task | 0.075 / 0.068 / 0.70; 0.030 / 0.018 / 0.77 | 3 s |
| `log_nseeds` | log1p number of seeds | task | clean | 100 % | task | - | - |

One-time costs dominate and are outside the timings: DuckDB aggregation of `evidence_2104` (4.2 GB), PubTator3 family build
(3.1 GB raw), donor / view matrices (`scripts/build_donors*.py`, `build_views.py`; the protein-language-model embeddings were
precomputed, a rebuild would be a GPU job). One stacker retrain (cv dev + dev->confirm) takes 40-150 s at 8 threads and
~10 GB RAM depending on machine load. Per-gene channels (`degree`, `pri_hpo_n`, near-per-gene `popularity`, `pri_*`) only
act through interactions with the per-task channels; they set the gene prior.

__S2__

__S3__

## 4. Redundancy (within-task Spearman, temporal confirm, 2,000 random candidates per task; `results/r5/redundancy_spearman.tsv`, `redundancy_pairs.tsv`)

Mean within-task Spearman over 138 tasks, pairwise-complete (NaN cells dropped). Clusters = average linkage on 1 - |rho|.

- **Exact duplicates in rank (rho = 1.00):** `pri_l2g_n` ~ `hz_gwas`, `pri_hpo_n` ~ `hz_mend`. Within a task the hz_* channels
  are the pri_* channels times a task constant, so their raw value only adds the task-level GWAS share (which the stacker
  could get from one context feature); their rank twins `r_hz_*` are exact copies of `r_pri_*`.
- **Near-duplicates (|rho| >= 0.8):** `degree` ~ `rwr_string` 0.93; `ph_mean` ~ `ph_top3` 0.92; `pri_l2g_n` ~ `pri_l2g_max` 0.85;
  `pri_mend_n` ~ `hz_mend` 0.81; the mean/max pair of each Perturb-seq view (RPE1, HepG2, Jurkat 0.88), GO 0.80.
- **Clusters at |rho| >= 0.5:** (a) GWAS-prior cluster {`pri_l2g_n`, `pri_l2g_max`, `hz_gwas`, `popularity`} (popularity ~ L2G
  count 0.75: L2G winners are popular genes); (b) Mendelian / HPO prior {`pri_mend_n`, `pri_hpo_n`, `hz_mend`} (and
  ph_mean ~ hz_mend 0.57); (c) {`ph_mean`, `ph_top3`}; (d) {`dc_ridge_pos`, `transfer_jaccard`} 0.62 (the two disease-row
  transfer witnesses); (e) {`dA_pt`, `pt_ridge_pos`} 0.68 (the two literature witnesses); (f) {`degree`, `rwr_string`};
  (g) each view's mean / max pair (0.53-0.88); views of different sources barely correlate (max cross-view 0.52 GTEx ~ HPA
  tissue, ESM C ~ ProtT5 0.51).
- **Not redundant:** `cs_donor`, `dc_ridge` (unclipped), `gwas_catlag`, `transfer_onto`, the 15 donor families (max |rho| with
  anything < 0.5; `ph_mean` ~ `don_hpo2101` 0.49 is the closest, same HPO source).

## 5. Keep / merge / drop, and gaps

Suggestions for a modeng-v2 candidate recipe, to be pre-registered and tested on the fresh fold, never on temporal confirm.

__S5__
