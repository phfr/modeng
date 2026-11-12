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
| clean arm `stack_v1clean` only: `vc_mean_* / vc_max_*` (6) | view similarity as above in HPA v20.1 single-cell and tissue and GTEx v8 (replace the 16 transductive views and 4 transductive k-means donors; `src/modeng/ext/lead_cleanviews.py`, list `results/v1clean_channels.txt`, 51 channels) | HPA v20.1 (files 2020-11-16), GTEx v8 (2017-06) | clean | 88-99 %; 100 % | task | AP 0.005-0.006, AUROC 0.47-0.61 | < 1 s each |
| `log_nseeds` | log1p number of seeds | task | clean | 100 % | task | - | - |

**Clean arm.** `stack_v1clean` (all inputs clean or vintage; no transductive class) is equivalent to modeng-v1 (insights 14:00: confirm AP 0.150 vs 0.153, -0.004 [-0.009, +0.001]; nuAP -0.004 [-0.011, +0.003]; dev within noise), and this audit's `no_modern` arm (transductive views/donors removed, nothing substituted) gives the same answer (3-seed confirm dAP -0.001, dnuAP -0.000). The transductive witnesses carry nothing; the clean arm is the version for the fresh fold. Two further witness types were tested and killed in parallel this morning: cached KG embeddings (`r5_kge.md`) and cell-type-of-action (`r5_celltype.md`).

One-time costs dominate and are outside the timings: DuckDB aggregation of `evidence_2104` (4.2 GB), PubTator3 family build
(3.1 GB raw), donor / view matrices (`scripts/build_donors*.py`, `build_views.py`; the protein-language-model embeddings were
precomputed, a rebuild would be a GPU job). One stacker retrain (cv dev + dev->confirm) takes 40-150 s at 8 threads and
~10 GB RAM depending on machine load. Per-gene channels (`degree`, `pri_hpo_n`, near-per-gene `popularity`, `pri_*`) only
act through interactions with the per-task channels; they set the gene prior.

## 2. Contribution in modeng-v1

### 2a. Gain importance and SHAP (`results/r5/gain.tsv`, `shap_feature.tsv`, `shap_group.tsv`)

The model object was not saved overnight, so the dev->confirm model was retrained once with the frozen recipe (seed 0;
predictions identical to the leaderboard row) and saved as `results/r5/model_full_confirm.txt`. SHAP = LightGBM
`pred_contrib=True` (log-odds) on all targets + 500 random non-excluded candidates per task (confirm 73,438 rows,
out-of-sample; dev 28,386 rows, **in-sample** for this model, so dev SHAP measures what the model fitted, not what
generalises). Group SHAP = sum of the group's raw and rank features per row (additive), then mean |.|; "sep" = mean group SHAP
on targets minus mean on candidates (how much the group pushes targets up).

| group | gain share | mean abs SHAP dev / confirm | sep dev / confirm | top features (confirm mean abs SHAP) |
|---|---|---|---|---|
| priors (pri_*, hz_*) | 0.135 | 0.423 / 0.429 | 0.87 / 0.67 | `hz_gwas` 0.21, `r_pri_l2g_max` 0.12, `pri_l2g_max` 0.09 |
| view (30) | **0.285** | 0.412 / 0.380 | **0.90 / 0.23** | `vmax_esmc` 0.07, `vmean_hpa_tissue` 0.06, `vmax_prott5` 0.06 |
| base | 0.140 | 0.233 / 0.224 | 0.38 / 0.25 | `r_degree` 0.11, `degree` 0.09 |
| log_nseeds | 0.030 | 0.160 / 0.188 | 0.18 / 0.06 | |
| donor (15) | 0.087 | 0.166 / 0.158 | 0.36 / 0.16 | `r_don_km_ps_cd4` 0.05, `r_don_km_ps_k562` 0.05 |
| dc_ridge | 0.156 | 0.159 / 0.154 | 0.65 / **0.58** | `dc_ridge_pos` 0.07, `dc_ridge` 0.05 |
| pt_ridge | 0.037 | 0.138 / 0.133 | 0.32 / 0.27 | `r_pt_ridge` 0.07 |
| gwas_catlag | 0.070 | 0.116 / 0.093 | 0.49 / 0.16 | `r_gwas_catlag` 0.08 |
| dA_pt | 0.030 | 0.096 / 0.093 | 0.21 / 0.19 | `r_dA_pt` 0.08 |
| ph | 0.021 | 0.050 / 0.049 | 0.17 / 0.15 | `r_ph_mean` 0.02 |
| cs_donor | 0.010 | 0.019 / 0.019 | 0.08 / 0.06 | |

Reading: views and donors carry a large share of gain and |SHAP| but their target separation collapses out of sample
(view sep 0.90 -> 0.23, donor 0.36 -> 0.16), and the donor features the model leans on most are the near-chance
transductive Perturb-seq k-means families: this is fitted noise. dc_ridge, pt_ridge, dA_pt and ph keep their separation
from dev to confirm. Gain / SHAP rank the GWAS priors first, but the ablations below show that they are replaceable.

### 2b. Group ablations (`results/r5/ablation_deltas.tsv`, per task `ep_*.parquet`)

Frozen recipe retrained without each group (cv on dev, dev->confirm for confirm). Delta = ablated - full (negative = the
group helps). Paired cluster bootstrap (5,000, modet fold clusters). **Retrain noise floor:** the full recipe with LightGBM
seed 1 / 2 instead of 0 changes confirm AP by -0.0015 / -0.0056 (seed 2: CI [-0.0095, -0.0021] excludes 0) and dev AP by
-0.004 / -0.005, so single-seed deltas below ~0.005 are not interpretable. The table therefore gives the **3-seed delta**
(per task, mean over seeds k = 0, 1, 2 of ablated_k - full_k); single-seed (frozen seed 0) deltas are in the tsv; they agree in sign
with the 3-seed deltas for base, view, view+donor, dc_ridge, gwas_catlag, pt_ridge and ph, while the single-seed deltas of
donor, cs_donor, dA_pt, priors and modern flip sign between seeds on some metrics (noise-level groups). Arms `view+donor` and `modern` (the 16
transductive views + 4 transductive donor families) were added after seeing the single-group results.

| removed | dev dAP | dev dnuAP | dev d degree-matched AP | confirm dAP | confirm dnuAP | confirm d degree-matched AP |
|---|---|---|---|---|---|---|
| base | +0.006 [-0.001, +0.012] | **-0.014 [-0.022, -0.008]** | -0.006 [-0.011, +0.000] | +0.002 [-0.001, +0.005] | **-0.018 [-0.022, -0.013]** | -0.006 [-0.012, +0.000] |
| donor | +0.001 [-0.004, +0.006] | -0.001 [-0.006, +0.004] | +0.003 [-0.004, +0.009] | -0.002 [-0.005, +0.002] | -0.003 [-0.008, +0.002] | +0.001 [-0.002, +0.004] |
| view | **+0.008 [+0.002, +0.014]** | **+0.009 [+0.002, +0.016]** | +0.005 [-0.003, +0.014] | **+0.006 [+0.001, +0.012]** | **+0.008 [+0.003, +0.014]** | **+0.007 [+0.004, +0.011]** |
| view+donor | **+0.014 [+0.004, +0.024]** | **+0.016 [+0.004, +0.027]** | +0.010 [-0.001, +0.022] | +0.005 [-0.002, +0.013] | +0.008 [-0.001, +0.016] | **+0.010 [+0.005, +0.014]** |
| modern (transductive views + donors) | +0.004 [+0.000, +0.008] | +0.002 [-0.004, +0.008] | +0.003 [-0.002, +0.007] | -0.001 [-0.004, +0.003] | -0.000 [-0.005, +0.005] | +0.001 [-0.002, +0.005] |
| cs_donor | +0.000 [-0.007, +0.006] | +0.001 [-0.005, +0.005] | +0.001 [-0.005, +0.006] | +0.002 [-0.002, +0.005] | +0.001 [-0.004, +0.006] | -0.000 [-0.004, +0.003] |
| dA_pt | -0.001 [-0.004, +0.003] | +0.001 [-0.003, +0.004] | -0.003 [-0.007, +0.002] | +0.000 [-0.003, +0.004] | +0.001 [-0.003, +0.005] | -0.001 [-0.004, +0.003] |
| dc_ridge | -0.004 [-0.010, +0.003] | -0.002 [-0.007, +0.004] | **-0.006 [-0.012, -0.000]** | -0.004 [-0.007, +0.000] | **-0.006 [-0.011, -0.001]** | **-0.007 [-0.012, -0.002]** |
| gwas_catlag | **-0.025 [-0.047, -0.009]** | **-0.023 [-0.045, -0.007]** | **-0.024 [-0.046, -0.009]** | -0.004 [-0.013, +0.003] | **-0.007 [-0.016, -0.000]** | **-0.007 [-0.016, -0.001]** |
| priors | -0.001 [-0.008, +0.003] | -0.002 [-0.007, +0.003] | -0.004 [-0.011, +0.001] | +0.002 [-0.003, +0.007] | -0.001 [-0.005, +0.003] | -0.001 [-0.007, +0.004] |
| ph | -0.004 [-0.008, +0.001] | -0.003 [-0.008, +0.001] | -0.003 [-0.008, +0.002] | -0.002 [-0.006, +0.002] | -0.004 [-0.008, +0.001] | -0.002 [-0.007, +0.002] |
| pt_ridge | -0.002 [-0.005, +0.001] | -0.001 [-0.005, +0.003] | **-0.006 [-0.010, -0.002]** | **-0.004 [-0.007, -0.002]** | **-0.004 [-0.007, -0.002]** | **-0.004 [-0.006, -0.002]** |

(Full recipe, seed 0: dev AP 0.176 / nuAP 0.150 / degree-matched AP 0.256; confirm 0.153 / 0.177 / 0.252.)

Reading (descriptive):
- **Necessary witnesses** (removal hurts on dev and confirm): `gwas_catlag` (large on dev, about a third of that on confirm,
  where only 38 % of tasks are covered), `dc_ridge`, `pt_ridge`; and the base group for nuAP (the degree / popularity /
  STRING features let the deg^-0.35 weighting work; AP is unaffected).
- **Removable without loss:** `cs_donor`, `dA_pt` (given `pt_ridge`), the priors group (given `gwas_catlag`, popularity
  and the HPO channels; R3 found the same for the four label-process channels), `ph` (borderline, -0.002 to -0.004),
  the transductive views / donors (`modern`: no gain and no loss, so the transductive leakage class buys nothing).
- **Harmful:** the 30 view similarities. Removing them improves every metric on dev and on confirm (3-seed CIs > 0 for AP
  and nuAP on both), removing views + donors improves dev AP by +0.014. With 53 training diseases, 60 weak features
  (alone AUROC 0.50-0.65) are fitted noise; this matches the earlier finding that 18 noise channels cost -0.010 nuAP.
  The pre-registered recipe keeps them (frozen); a v2 without them is a candidate for the fresh fold, not a claim.

## 3. Per disease type (temporal confirm 138 / dev 53; `results/r5/task_types.tsv`, `ablation_by_type.tsv`, `shap_group_by_type.tsv`)

Type from the seeds: share of seeds that are L2G >= 0.5 GWAS winners (`ot_genetics_portal` 21.04) vs share with ClinVar /
curated evidence in the task family (OT 21.04 evidence; `stack_gate.context`, same helpers as `priors_h`). The split is
bimodal: **GWAS-dominated** (GWAS share > Mendelian share; confirm n = 39, median GWAS share 0.88), **Mendelian-dominated**
(confirm n = 96, median Mendelian share 0.80, GWAS share 0), no genetic seed evidence (n = 3 confirm / 1 dev; not
interpreted). Therapeutic area = modet's primary area (confirm: nervous system 29, genetic/congenital 17, cardiovascular 9,
other 83).

modeng-v1 itself is much weaker on GWAS-dominated diseases (confirm AP 0.092 / nuAP 0.098 / degree-matched AP 0.173) than on
Mendelian-dominated ones (0.174 / 0.210 / 0.280).

3-seed ablation deltas by type (confirm; dev in brackets where it differs in sign or size):

| removed | GWAS-dominated dAP | GWAS-dominated dnuAP | Mendelian-dominated dAP | Mendelian-dominated dnuAP |
|---|---|---|---|---|
| base | +0.006 [+0.000, +0.011] | **-0.013 [-0.020, -0.007]** | +0.000 | **-0.020 [-0.025, -0.015]** |
| gwas_catlag | -0.006 [-0.020, +0.005] (dev **-0.061 [-0.117, -0.020]**) | **-0.012 [-0.026, -0.001]** (dev -0.061) | -0.003 | -0.005 |
| donor | -0.001 | **-0.005 [-0.009, -0.000]** | -0.000 | -0.001 |
| priors | +0.002 | -0.000 | +0.002 | -0.001 |
| dc_ridge | +0.000 | -0.001 | -0.004 [-0.008, +0.000] | **-0.007 [-0.014, -0.001]** |
| pt_ridge | -0.002 | -0.004 | **-0.006 [-0.010, -0.002]** | **-0.005 [-0.009, -0.001]** |
| ph | +0.001 | -0.001 | -0.003 [-0.008, +0.002] | -0.006 [-0.011, +0.001] |
| view | +0.001 | +0.002 | **+0.010 [+0.003, +0.016]** | **+0.012 [+0.005, +0.018]** |
| view+donor | -0.002 | -0.001 | **+0.011 [+0.003, +0.019]** | **+0.013 [+0.003, +0.022]** |
| cs_donor, dA_pt, modern | within +-0.002 | | within +-0.003 | |

SHAP view (confirm, group "sep" per task averaged by type): priors dominate GWAS-dominated diseases (0.82 vs 0.32 for
Mendelian), gwas_catlag 0.18 vs 0.10; the disease-row and literature witnesses dominate Mendelian-dominated diseases
(dc_ridge 0.80 vs 0.48, pt_ridge 0.44 vs 0.18, dA_pt 0.30 vs 0.13, ph 0.28 vs 0.07, donor 0.26 vs 0.09).

Reading: **two different models live inside modeng-v1.** GWAS-dominated diseases are served by the label-process genetic
witnesses (`gwas_catlag`, with the L2G priors as a redundant backup) plus the base gene prior; nothing else matters there,
and the model is weak there (the strict-a frontier). Mendelian-dominated diseases are served by disease-row transfer
(`dc_ridge`), literature (`pt_ridge`) and HPO phenotypes (`ph`), and they are where the views hurt (+0.010 AP when removed).
By therapeutic area (confirm nuAP deltas; small groups, no CIs claimed): gwas_catlag matters most in genetic/congenital
(-0.034, n = 17), base everywhere (-0.011 to -0.019), views hurt in every area (+0.007 to +0.012).

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

**Keep** (necessary on dev and confirm, or the only witness of an evidence type):
- `dc_ridge` / `dc_ridge_pos` (best disease-row witness; Mendelian diseases).
- `pt_ridge` / `pt_ridge_pos` (literature; all six 3-seed deltas negative, all three confirm CIs exclude 0).
- `gwas_catlag` (only witness that moves GWAS-dominated diseases; label-process class: report with and without it, as
  pre-registered).
- base group (needed for nuAP; `transfer_jaccard`, `popularity`, `degree`). `ph_mean` (the only seed-specific HPO-profile witness; Mendelian diseases).

**Merge:**
- `hz_gwas` and `hz_mend` into one task-level context feature (seed GWAS share): within a task they are exact rank copies
  of `pri_l2g_n` / `pri_hpo_n` (rho = 1.00).
- `pri_l2g_n` + `pri_l2g_max` -> one L2G prior (rho 0.85); `ph_mean` + `ph_top3` -> one (rho 0.92); `degree` + `rwr_string`
  -> keep one plus RWR residual (rho 0.93).
- `dA_pt` into `pt_ridge` (same PubTator sets; `dA_pt` adds nothing once `pt_ridge` is in: 3-seed dAP +0.000).
- `cs_donor` into `dc_ridge` (same rows, both regressions of the seed vector; `cs_donor` removal +0.002 AP).
- The 15 donor families: only `don_hpo2101` and the three MSigDB families have alone AP > 0.02; collapse the rest or drop.

**Drop (candidate v2 arm for the fresh fold, pre-register first):**
- Start from the clean arm `stack_v1clean` (equivalent to v1, no transductive inputs), then apply the drops below.
- All 30 view similarities (harmful: 3-seed confirm AP +0.006, nuAP +0.008 when removed; dev +0.008 / +0.009).
- The 16 transductive views and 4 transductive donor families at minimum (no contribution at all, and they make the model
  transductive on temporal); Perturb-seq views (RPE1 / HepG2 / Jurkat cover 10 % of genes, alone AUROC 0.50).
- The 11 weak donor families (k-means of embeddings, IMPC, Complex Portal, paralogs: alone AUROC 0.50-0.59).
- Priors group as a whole is removable without loss on temporal (redundant with `gwas_catlag`, popularity and HPO), but it
  carried the forward gain (`r1_priors.md`: +0.012 / +0.009 nuAP on forward); keep `pri_l2g_n` only if the fresh fold is a forward-type window.

**Gaps (evidence types with no witness in modeng-v1):**
- **GWAS-dominated diseases without own catalog hits** (62 % of confirm tasks have no `gwas_catlag` value): no gene-level
  GWAS statistic is used (MAGMA `gwas_own` / `gwas_seedw` exist in `ext/gwas_d.py` but are not in v1; PoPS features and
  cS2G are on disk, unused); sub-threshold loci and power forecasting were tried and discarded (r4).
- **Rare-variant burden / exome evidence** (Genebass / UKB WES, OT `gene_burden`): none on disk with a T0 date; no witness.
- **Gene constraint** (gnomAD v2.1.1 pLI / LOEUF, on disk, pre-T0): not used; a natural Mendelian gene prior.
- **Drugs and clinical evidence** (ChEMBL, known drugs), **somatic / cancer** (Cancer Gene Census, IntOGen),
  **differential expression in disease** (Expression Atlas), **animal-model phenotype matching** (PhenoDigm): only reached
  through the labels; `pop_<source>` popularity channels were built in r1_priors but are not in v1.
- **Interaction networks beyond STRING physical** (BioGRID, HuRI, BioPlex, AF2 interactomes, HumanBase tissue networks,
  all on disk): only `degree` / `rwr_string` from STRING v11.5.
- **Cross-disease genetic correlation** (`neale_rg`, on disk) as donor weights: not used; donors are weighted by gene overlap only.
- **Disease-specific cell types / tissues** (seed-set cell-type enrichment from Tabula Sapiens / HPA): views use generic
  similarity only, and that form hurts.
- Annotation vintage gap: PubTator3 and GWAS-Catalog mappings are 2026 versions (no archived vintages on disk), so the
  two literature / catalog witnesses stay in the "vintage" class.
