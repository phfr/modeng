# Round 1 ideation — lens: statistical / population genetics (2026-11-12)

Sources for claims about our data: `docs/ideation/brief.md` (cited as "brief N" = fact N of the modet list), `docs/research/02_local_data_inventory.md`
("inventory"), and files I inspected read-only (named inline). `docs/research/01_sota_and_resources.md` did not exist when I wrote this.
`results/leaderboard.tsv` was empty/absent, so "transfer baseline" below means the brief's numbers (tr_max nuAP 0.122; transfer LightGBM 0.177 dev / 0.168 devall).
Web sources are listed at the end (accessed 2026-11-12). Statements marked **[memory]** are my recall and were not confirmed by a search.

Facts I measured for this note (one-line pandas, no heavy compute):
- `modet/data/processed/groundtruth/arrival_devall_t0_string115_phys_notm_400.parquet`: 4,981 (disease, gene) arrivals over 191 diseases; evidence_class
  gwas_only 2,863 (57 %), clinvar_only 827, curated_only 633, mixed_non_gwas 434, **burden_only 201 (4 %)**, gwas+other 23. `novel_gene` mean 0.104;
  by class: burden_only 0.234, gwas_only 0.098, clinvar_only 0.106, curated_only 0.090.
- `modet/data/processed/gwas_magma/t0_coverage.tsv` (STRING panel rows): 260 diseases, **only 71 have a harmonised T0 GWAS**, 100 have any T0 study.
  So own-trait GWAS covers about a quarter of the panel: any statgen idea that needs the disease's own GWAS is capped at that coverage.
- `gwas_magma_t0/*.parquet`: per-study MAGMA gene tables (columns gene, entrez, z, p, nsnps, n, gcst, chr, start, stop; ~17k genes; variants `_w0`, `_w10`, `_leads`),
  112 study prefixes. `gwas_magma/*.parquet`: 102 post-T0 per-disease tables.
- `ot_t0_sets.parquet`: 4,592 context rows; 957 have EFO ids in the EFO_0004-EFO_0019 range (many are measurement traits) and 504 are HP terms, i.e. the
  donor matrix already contains quantitative-trait rows, but only as thresholded gene sets.
- `/mnt/msadb/modet/downloads/cs2g/x/cS2G_1000GEUR/cS2G.<chr>.SGscore.gz` present (SNP-to-gene scores); `neale_rg_table/r2_results/geno_correlation_sig.r2`
  and `h2_results/ukb31063_topline_h2_4203.tsv` present (UKB genetic correlations and h2); `ot_evidence/evidence_2509/sourceId=gwas_credible_sets` (172 MB) present.

---

## Step 1 — eight assumptions of the transfer LightGBM and the baselines, and what follows if each is false

| # | Assumption | If false |
|---|---|---|
| A1 | **Evidence is binary.** A gene either is or is not associated with a donor disease (thresholded OT sets). | Sub-threshold association statistics (z between 2 and 5) carry most of the information about the *next* arrivals (brief 4: sub-threshold z AUC ~0.8). A dense, continuous gene x trait matrix would give non-zero support to genes that have "zero transfer support" today (brief 3). |
| A2 | **The gene is the unit of evidence.** Transfer is computed gene by gene. | GWAS evidence lives at loci. A donor hit assigned to the neighbour gene gives zero support to the gene L2G will pick for the target disease. Zero-support GWAS positives may have *locus-level* support. |
| A3 | **Disease similarity = overlap of known gene sets.** | With 3-5 seeds the overlap is noise. Genome-wide sharing (genetic correlation, phenotype-profile similarity) defines relatedness without needing seeds, and can relate a rare disease to a quantitative trait it shares no curated gene with. |
| A4 | **The future looks like the present, only more of it** (stationary label process). | Arrivals are a mixture of channels with different laws: GWAS arrivals follow statistical power (sample size x effect x allele frequency), burden arrivals follow coding target size and constraint, ClinVar/curated arrivals follow clinical sequencing. One score cannot rank all three; a channel-specific hazard could. |
| A5 | **Gene covariates are trait-agnostic priors** (degree, constraint, popularity). | Discoverability is channel-specific and partly irrelevant to biology: gene length, local LD, regulatory domain size and drift decide which important genes GWAS finds first (Spence et al. 2025). A model that separates "importance for the trait" from "discoverability by the next study" would extrapolate, not interpolate. |
| A6 | **Only curated databases are input.** Everything the model knows passed through the OT pipeline. | Public summary statistics and catalog rows that OT has not ingested yet, or ingested below its thresholds, are a legitimate information source at the seed date (brief 5: ~40 % of "new" positives were old evidence newly attached). |
| A7 | **Common-variant, rare-variant and Mendelian evidence are exchangeable rows** of one matrix. | They are complementary views with little overlap in top hits (Spence et al. 2025). Rare-variant burden is gene-first (no locus-to-gene step) and reaches genes with no disease record; Mendelian genes sit near GWAS loci of the matched complex trait (Freund et al. 2018), which gives a directed bridge rare disease <- quantitative trait GWAS. |
| A8 | **Gene-gene relations come from physical/functional networks** (PPI, KG), and those add nothing beyond degree (brief 2). | The relation that matters for a polygenic trait may be *regulatory and directed* (peripheral gene -> core gene, omnigenic model). Population genetics supplies such edges for free as trans-QTLs; they were never tested and are not a PPI. |

---

## Step 2 — ideas

Conventions: "dev" = dev split of the modeng benchmarks. All methods are `score(tasks, B) -> [n_tasks, n_genes]` in `src/modeng/methods/`. Where an idea covers
only a subset of diseases, the score falls back to the transfer baseline elsewhere and is rank-blended with it where it applies; I state the expected
effect on the covered subset and on the whole dev split. Effects are my estimates, not measurements. No idea uses modet's sealed-fold diseases:
`data.excluded_task_ids()` is applied by the harness, and for external trait matrices the trait columns mapped to those ids must be dropped before fitting.

### SG-1 — Locus-space transfer plus within-locus gene choice
- **Mechanism.** Do transfer in LD-block space instead of gene space: a block is "supported" if any related disease or trait has a gene in it, then a
  conditional-logit model picks the gene inside the block (distance rank, cS2G score, coding, constraint), trained on how T0 GWAS evidence resolved loci.
  This attacks A2: a zero-transfer-support GWAS positive whose neighbour is a donor gene becomes reachable. It differs from the banned "locus prior (genes
  near seeds)" (that used the target's own seeds) and from "bundle/locus units" (an evaluation endpoint): here the *donor* signal is spread over the locus.
- **p_typical:** 0.12
- **Data.** On disk: gene coordinates in the MAGMA tables (`gwas_magma_t0/*.parquet`, chr/start/stop), `cs2g/x/cS2G_1000GEUR` , OT 21.04 sets and
  `ot_evidence/evidence_2104/sourceId=ot_genetics_portal` (to learn the gene-choice model). Download: LD-block bed (Berisa & Pickrell EUR blocks, a few 100 kB,
  https://bitbucket.org/nygcresearch/ldetect-data **[memory]**). **Leakage class: temporal-clean** on `temporal` (all inputs <= 21.04). Main route: the
  gene-choice model must be fitted on context rows only, with `fam_rows x targets` masked via `B.context_for(task)`; cS2G was published 2022 **[memory]**
  and is built partly from GWAS-era annotations, so run the temporal test with distance + coding only and add cS2G as a separately labelled transductive variant.
- **Minimal test (<= 1 h, CPU).** (1) Diagnostic first, 10 min: for dev `temporal` tasks, take gwas_only positives with zero transfer support and count
  the fraction whose LD block (or +-250 kb window) contains a gene with non-zero `tr_max`. (2) `methods/locus_transfer.py`: `S_gene = tr_max` matrix ->
  `S_block = max over genes in block` -> `score = S_block[block(g)] * softmax_within_block(w . features(g))`, w fitted on dev-train context; rank-blend with
  `tr_max`. Run on dev of `temporal` and `forward`, stratify by evidence_class and by transfer support.
- **Expected effect.** If the diagnostic shows >= 25 % of zero-support GWAS positives have block support: recall@100 of that stratum from 0.4 % to 3-5 %,
  nuAP +0.008 to +0.015 on dev, AP +0.005. **Kill rule:** diagnostic fraction < 10 %, or blended nuAP gain < +0.005 on dev `temporal`.
- **Most likely failure.** Blocks hold 5-20 genes and donor support is already concentrated on the right gene, so spreading it mostly adds false positives.

### SG-2 — Phenome fingerprints: dense sub-threshold gene x trait matrix
- **Mechanism.** Replace the thresholded donor matrix by a dense matrix of gene-level association z over thousands of biobank traits; represent each gene
  by its phenome-wide profile and score a candidate by the similarity of its profile to the seed profiles (after removing same-locus genes, whose profiles
  correlate through LD). A gene never recorded for any disease still has a full profile, so this is the one transfer variant that can reach brief-3 "unknown
  to any disease" genes. Banned-adjacent ("T0 UKB MAGMA studies as donors"); the new reasons are (a) ~4,000 traits instead of ~110 studies, (b) TWAS +
  colocalisation gene assignment instead of MAGMA windows, (c) continuous profiles matched to seeds, no disease-to-study mapping required.
- **p_typical:** 0.20
- **Data.** PhenomeXcan (Pividori et al., Sci. Adv. 2020-09): S-MultiXcan z and fastENLOC RCP for 4,091 traits x ~22,500 genes; Zenodo
  DOI 10.5281/zenodo.3530669, download instructions in github.com/hakyimlab/phenomexcan; gene x trait matrices are of order 1 GB (size not verified).
  **Leakage class: temporal-clean** (published 2020-09, UKB + pre-2020 public GWAS, GTEx v8). Main route: traits that are the target disease itself or a
  relative; these are not labels, but to keep the comparison strict run a variant with trait columns whose EFO maps into `fam_rows` removed.
- **Minimal test (<= 2 h, CPU).** Download the two matrices; align to `D.universe()`; z-score columns; regress out, per gene, the mean profile of genes
  within +-500 kb (LD sharing); `score[t] = mean over seeds s (outside the candidate's locus) of cosine(profile_g, profile_s)`; also a ridge variant
  (seeds vs rest on trait columns, 5-fold over seeds). Evaluate alone, rank-blended with `tr_max`, and as one LightGBM column, on dev `temporal`;
  report the "gene unknown to any disease" stratum and zero-support stratum separately.
- **Expected effect.** Alone nuAP 0.02-0.04; blended +0.004 to +0.010 nuAP overall; in the unknown-gene pool within-pool AUC from ~0.60-0.63 to >= 0.66.
  AP roughly unchanged. **Kill rule:** unknown-pool AUC gain < 0.02 **and** blended nuAP gain < +0.004.
- **Most likely failure.** UKB power: sub-threshold TWAS profiles of non-hit genes are mostly noise, and what is not noise is LD sharing that survives the regression.

### SG-3 — Gene-level multivariate shrinkage forecast of the next GWAS
- **Mechanism.** Treat a GWAS arrival as what it is (brief 4): a gene crosses significance when N grows. Fit a multivariate adaptive shrinkage model
  (mash-style mixture of covariance patterns) on the gene x trait z matrix of related traits, then compute for every gene the posterior probability that
  its own-trait z exceeds the threshold at N' = k x N. Cross-trait borrowing gives a forecast for diseases whose own GWAS is small or missing (only 71 of
  260 panel diseases have a harmonised T0 GWAS, `t0_coverage.tsv`), which plain own-trait z cannot do.
- **p_typical:** 0.08
- **Data.** On disk: `gwas_magma_t0/*_w10.parquet` (112 studies), `neale_rg_table/r2_results/geno_correlation_sig.r2` (to pick related traits and
  initialise covariance patterns), `studies_selected_t0.tsv` (study -> disease, N, cases). Optional extension: the SG-2 matrix as additional conditions.
  **Leakage class: temporal-clean** with T0 studies. Main route: study dates; `studies_selected_t0.tsv` contains rows dated up to 2021-04-23, so filter
  on `date < 2021-04-01` and never read `gwas_magma/` (post-T0) on `temporal`. On `forward`, `gwas_magma/` studies dated <= 2025-09 are fair.
- **Minimal test (<= 2 h, CPU).** `methods/mash_forecast.py`: stack Z (genes x ~110 T0 studies), estimate null correlation from |z|<2 genes, fit 10-20
  data-driven covariance components by EM on the top 2,000 genes (numpy, minutes), posterior mean/variance per gene and study, then
  `P(|z'| > 4.5 | N' = 3N)` for the study mapped to each task (for tasks without a study: posterior of the rg-weighted mixture of the 5 most correlated
  studies). Compare against raw own-trait z (same tasks) and `tr_max`; evaluate on gwas_only targets and on all targets.
- **Expected effect.** On covered diseases, gwas_only recall@100 from ~16 % (brief 4) to 22-25 %; overall dev nuAP +0.006 to +0.012 when blended.
  Raw own-trait z should already give part of this; the claim to test is mash > raw z by >= +0.02 AUC on later GWAS positives and non-zero signal on
  uncovered diseases. **Kill rule:** blended nuAP gain < +0.005 on covered dev diseases, or mash not better than raw z.
- **Most likely failure.** MAGMA gene z is locus-smeared, so the forecast ranks the whole locus and L2G later picks a different gene (fix: combine with SG-1's gene choice).

### SG-4 — Phenotype-decomposed quantitative-trait GWAS for rare diseases
- **Mechanism.** A rare disease has no GWAS, but its clinical features do: decompose the disease into its HPO phenotypes, map each to biobank quantitative
  or common-disease traits (short stature -> height, hypercholesterolaemia -> LDL, cardiomyopathy -> cardiac traits, renal -> eGFR), and score genes by
  the frequency-weighted sub-threshold gene z of those traits. Mendelian genes are specifically enriched near GWAS loci of the matched complex trait
  (Freund et al. 2018), so this gives seed-independent evidence exactly where seeds are 3-5 and transfer overlap is noise (A3, A7).
- **p_typical:** 0.07
- **Data.** On disk: `hpo_annotations_2021-01_wayback` (disease -> HPO with frequencies, T0-dated), `gwas_magma_t0` (quantitative traits among the 112
  studies), HP rows in `ot_t0_sets.parquet` (504). Larger trait panel = SG-2 matrix. HPO -> trait mapping: label matching plus name-embedding
  nearest neighbour, then a manual check of the top 100 HPO terms. **Leakage class: temporal-clean** with the 2021-01 HPO file and T0 traits. Main route:
  an LLM or embedding model used for the mapping may know post-2021 gene-disease facts; the mapping must take only phenotype and trait *names* as input
  and must never see gene names.
- **Minimal test (<= 2 h, CPU).** `methods/hpo_gwas.py`: for each Orphanet/MONDO dev task with <= 10 seeds, `score_g = sum_h freq(h) * idf(h) * max_t in map(h) z_gt`;
  evaluate on that subset alone and blended with `tr_max`; negative control = permute the HPO -> trait map (must kill the gain).
- **Expected effect.** On few-seed rare-disease tasks nuAP +0.010 to +0.020 (blend); on the whole dev split +0.002 to +0.005, because these tasks are a
  minority and their arrivals are mostly ClinVar/curated. **Kill rule:** gain on the few-seed subset < +0.008 nuAP, or the permuted-map control retains > 50 % of the gain.
- **Most likely failure.** T0 trait coverage is too thin (112 studies) for most HPO terms, and rare-disease arrivals in OT are mainly re-attached old ClinVar records (brief 5) that no GWAS predicts.

### SG-5 — Sub-threshold rare-variant burden atlas as gene-first evidence
- **Mechanism.** Exome-wide burden tests are gene-first: no locus-to-gene step, no dependence on a prior disease record, and they rank trait-specific
  genes that GWAS ranks low (Spence et al. 2025). Use the full sub-threshold burden statistics (pLoF and missense masks) across thousands of phenotypes as
  a dense gene x phenotype matrix, read both directly (matched phenotype) and as profiles (as in SG-2). In our arrivals burden_only positives have the
  highest share of genes new to any disease (0.234 vs ~0.10, measured above), so this is the natural source for the "unknown gene" stratum.
- **p_typical:** 0.30
- **Data.** Genebass (Karczewski et al. 2022; 394,841 UKB exomes, 4,529 phenotypes, gene-based results at app.genebass.org/downloads; Hail tables,
  size not verified, expect tens of GB for gene-level results). Smaller, T0-dated alternatives **[memory]**: DDD de novo enrichment per gene (Kaplanis et al.
  2020), ASC (Satterstrom et al. 2020), Epi25 (2019) gene tables, each < 10 MB. **Leakage class: transductive on `temporal`** for Genebass (preprint
  2021-06, after T0); **temporal-clean** for the 2019-2020 consortium tables; fair on `forward` (inputs <= 2025-09). Main route: OT ingests gene-burden
  evidence, so the same test statistics become labels; on `forward` restrict to p > OT's inclusion threshold (sub-threshold only) to show the gain is forecast, not copy.
- **Minimal test (<= 2 h, CPU after download).** Export Genebass gene-level SKAT-O/burden p for pLoF and missense|LC to a genes x phenotypes parquet;
  map phenotypes to tasks (ICD/phecode -> EFO via OT ontology xrefs); `score = -log10 p` of the matched phenotype (sub-threshold only) and the SG-2-style
  profile similarity; evaluate on dev `forward`, stratified by evidence class and unknown-gene pool. T0-clean mini-test on `temporal`: DDD/ASC/Epi25
  tables for the neurodevelopmental, autism and epilepsy tasks (if in dev).
- **Expected effect.** `forward` dev: recall@100 on burden-class arrivals from near 0 to 10-15 %; overall nuAP +0.003 to +0.008 (burden arrivals are ~4 %
  of positives). AP unchanged. **Kill rule:** overall blended gain < +0.003 nuAP **and** unknown-pool AUC gain < 0.02.
- **Most likely failure.** Binary-disease burden tests in 400k exomes are under-powered: outside ~20 well-powered phenotypes the sub-threshold tail is null.

### SG-6 — Trans-QTL regulators of the seed genes (natural-perturbation edges)
- **Mechanism.** Population genetics provides directed regulatory edges: a variant that is a cis-QTL for gene A and a trans-pQTL/eQTL for protein B gives
  A -> B. Score a candidate by how many seed gene products it regulates in trans (and the reverse: targets of seed-gene cis variants), with a
  hypergeometric test against the target's trans in-degree. This is the omnigenic "peripheral regulator -> core gene" relation, not a PPI; modet's
  "network adds nothing beyond degree" result (brief 2) was never tested on directed, genetically anchored edges.
- **p_typical:** 0.06
- **Data.** UKB-PPP (Sun et al., Nature 2023): 2,923 proteins, 14,287 primary pQTL associations, summary statistics on Synapse syn51364943; the
  supplementary lead-pQTL table (a few MB) is enough for the test. T0-dated alternative: eQTLGen 2019 trans-eQTL table (https://www.eqtlgen.org,
  ~60k significant trans pairs, tens of MB **[memory]**). **Leakage class: transductive on `temporal`** with UKB-PPP (2023), **temporal-clean** with eQTLGen
  2019; fair on `forward`. Main route: the cis gene of a trans locus is assigned by the authors using prior knowledge, and eQTLGen tested trans effects only
  for GWAS-catalog variants, so "has a trans edge" is itself a proxy for "is at a GWAS locus"; control by comparing against a degree-matched (trans
  out-degree) null.
- **Minimal test (<= 1 h, CPU).** Build a sparse directed matrix T (cis gene -> trans target) from the lead tables; `score_g = -log10 hypergeom(|targets(g) ∩ seeds|)`
  plus the reverse direction; evaluate on dev `forward` (UKB-PPP) and dev `temporal` (eQTLGen), alone, blended, and against 100 out-degree-preserving rewirings of T.
- **Expected effect.** Applies only where seeds include measured plasma proteins or blood-expressed genes (immune, cardiometabolic, haematological tasks):
  there nuAP +0.005 to +0.010; whole dev +0.002 to +0.004. **Kill rule:** real T beats the rewired T by < +0.003 nuAP on the applicable subset.
- **Most likely failure.** Trans-pQTL hubs (ABO, FUT2, HLA, APOE, ASGR1 loci) dominate; after hub control too few edges remain to rank 19k genes.

### SG-7 — Ingestion-lag arbitrage: public associations not yet in the label source
- **Mechanism.** The label generator is a pipeline with a lag. At the seed date there are GWAS Catalog rows and deposited summary statistics that the
  seed-date OT release has not ingested, or holds below its score threshold; brief 5 measured the same effect for ClinVar (~40 % of "new" positives were
  old evidence newly attached). Read those sources directly at the seed date, run a cheap locus-to-gene step, and add the result as evidence. This is an
  information source, not a model, but it targets the largest hard stratum directly (82 % of strictly-new positives are GWAS-only, brief 4).
- **p_typical:** 0.10
- **Data.** GWAS Catalog associations and study tables (brief lists the GWAS Catalog as on disk; otherwise
  https://www.ebi.ac.uk/gwas/docs/file-downloads, ~300 MB, use the release archived closest to the seed date); OT 21.04 weak evidence rows in
  `ot_evidence/evidence_2104/sourceId=ot_genetics_portal` (below-threshold L2G); for `forward`, `ot_evidence/evidence_2509/sourceId=gwas_credible_sets`
  below-threshold rows plus catalog studies dated <= 2025-09. **Leakage class: temporal-clean if and only if the catalog snapshot is dated <= seed date.**
  Main route: the catalog back-fills; a current download filtered on publication date still contains curation done later (trait re-mapping, added
  studies). Use an archived release, not a date filter on today's file. Note the harness excludes "unknown (weak evidence)" genes from `temporal`
  rankings, which removes part of this signal by design; report how much.
- **Minimal test (<= 1 h, CPU).** `methods/catalog_lag.py`: archived catalog release from 2021-03 -> rows with p < 1e-6 -> EFO -> task (ontology
  descendants included) -> mapped genes, score = max -log10 p; subtract genes already in the task's seeds; blend with `tr_max`. Same for `forward` with a
  2025-09 release. Report the gain separately for gwas_only targets and for arrival classes A/B.
- **Expected effect.** gwas_only recall@100 from ~16 % to 22-28 %; nuAP +0.010 to +0.020 on `forward` dev, less on `temporal` (weak-evidence genes are
  excluded there). **Kill rule:** < +0.005 nuAP blended on `forward` dev.
- **Most likely failure.** OT's own sub-threshold rows already contain nearly all of it and the harness has removed them from the ranking, leaving almost nothing to add.

### SG-8 — Channel-specific discovery hazard: importance times discoverability
- **Mechanism.** Split the score into (i) importance of the gene for the trait (the transfer score, SG-2 profile) and (ii) the probability that the next
  study *of a given channel* detects it. GWAS discoverability depends on regulatory target size and local LD, burden discoverability on coding length times
  constraint, clinical discoverability on phenotype severity; Spence et al. 2025 show these trait-irrelevant factors drive association rankings. Each
  disease gets a channel mix from its own evidence history, and the final score is the mixture of channel hazards. Differs from the banned "tier-0
  gene-propensity prior" by being channel-specific, multiplicative with importance, and disease-specific in the mix.
- **p_typical:** 0.09
- **Data.** All on disk: MAGMA `nsnps` per gene and gene coordinates (`gwas_magma_t0`), gnomAD v2.1.1 constraint and CDS length
  (`gnomad_v2.1.1/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz`), per-source evidence at T0 (`ot_evidence/evidence_2104/sourceId=*`) for the disease channel
  mix and for fitting channel hazards on T0-internal history (19.04 -> 21.04 arrivals via `ot_t19_sets.parquet`). **Leakage class: temporal-clean.**
  Main route: fitting hazards on 21.04 -> 25.09 arrivals of dev diseases and then scoring the same diseases; fit on 19.04 -> 21.04 only.
- **Minimal test (<= 1.5 h, CPU).** `methods/channel_hazard.py`: three logistic hazards (GWAS, burden/clinical, curated) fitted on 19.04 -> 21.04
  arrivals with features {log nsnps, gene length, LOEUF, CDS length, number of T0 diseases per channel for the gene}; per task
  `score = sum_c pi_c(disease) * importance_g * h_c(g)`, pi_c from the seed evidence sources; compare with `tr_max` x single pooled prior.
- **Expected effect.** nuAP +0.003 to +0.008; the visible effect should be in the unknown-gene pool (AUC +0.03) and in calibration of how many arrivals
  per channel a disease gets. **Kill rule:** channel-specific model beats the pooled-prior control by < +0.003 nuAP.
- **Most likely failure.** The LightGBM with gene covariates already learned the pooled version, and the channel split adds variance, not signal.

### SG-9 — Genetic-correlation donor weights that do not need seeds
- **Mechanism.** Transfer weights donors by overlap with the seeds, which is unstable for few-seed diseases and circular for hub genes. Weight donors by
  genome-wide sharing instead: LDSC genetic correlation where both have GWAS, otherwise correlation of SG-2 phenome profiles averaged over the disease's
  seed loci. The donor score then flows through `rg^2`-weighted donors, giving a second, seed-independent transfer channel whose errors are not the same as tr_max's.
- **p_typical:** 0.35
- **Data.** On disk: `neale_rg_table/r2_results/geno_correlation_sig.r2`, `inputs/ukb_finngen_names.tsv` (UKB phenotype names for mapping to EFO),
  `neale_rg/ukb31063_h2_all.02Oct2019.tsv.gz`. **Leakage class: temporal-clean** (UKB round-2 results, 2018-2019). Main route: mapping UKB phenotypes to
  OT diseases by name with a post-2021 ontology can pull in merged/renamed terms; use `ontologies_2021-04`.
- **Minimal test (<= 1 h, CPU).** Map UKB phenotypes to context rows; build disease x disease weight matrix W_rg; `score = W_rg[task] @ context_for(task)`
  (row-normalised, hidden pairs masked); compare and blend with `tr_max`; report by seed-count tertile.
- **Expected effect.** +0.002 to +0.006 nuAP overall, concentrated in the lowest seed-count tertile. **Kill rule:** < +0.003 nuAP in the lowest seed tertile.
- **Most likely failure.** Only a few hundred heritable UKB phenotypes map to panel diseases, and for those the seed-overlap weights are already good.

---

## Bets

- **I would bet on SG-7 combined with SG-1.** Both follow from the measured fact that most new positives are GWAS-only power events (brief 4) and both
  have a 10-minute diagnostic that decides them before any modelling: how many arrivals were already visible in public association rows at the seed date,
  and how many zero-support positives have locus-level support. They are the least "model-like" ideas here, and SG-7 in particular should be reported as an
  information-source gain, separately from any modelling claim.
- **Most underrated: SG-3** (multivariate shrinkage forecast). It is the only idea that turns "arrival is a statistical-power event" into an explicit
  probability with a sample-size argument, which also gives the calibrated module cutoff the task asks for, and it extends own-trait sub-threshold z
  (AUC ~0.8 in 4 diseases, brief 4) to the three quarters of diseases without a T0 GWAS. Its risk is the locus-smearing of MAGMA z, which SG-1 addresses.
- Honest ranking of novelty versus payoff: SG-6 and SG-4 are the most novel but apply to subsets; SG-2 is the only one aimed squarely at genes unknown
  to any disease, and its failure would be informative (it would say biobank sub-threshold profiles carry no usable signal at gene resolution).

## Web sources (accessed 2026-11-12)
- PhenomeXcan: https://www.biorxiv.org/content/10.1101/833210.full.pdf ; published record https://knowledge.uchicago.edu/record/11012 ; data DOI https://doi.org/10.5281/zenodo.3530669
  (search snippets gave 4,091 traits and 22,515 genes in the published version, 22,255 in the preprint; file sizes not verified).
- Genebass: https://www.medrxiv.org/content/10.1101/2021.06.19.21259117v4.full.pdf ; https://labs.icahn.mssm.edu/minervalab/resources/data-ark/genebass/ (points to app.genebass.org/downloads; table size not verified).
- Spence et al., "Specificity, length and luck drive gene rankings in association studies", Nature 2025: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12823407/ ; preprint https://www.biorxiv.org/content/10.1101/2024.12.12.628073.full.pdf
- Freund et al. 2018 AJHG, Mendelian genes near GWAS regions of matched traits: https://pubmed.ncbi.nlm.nih.gov/30290150/
- UKB-PPP, Sun et al. 2023 Nature: https://www.nature.com/articles/s41586-023-06592-6.pdf ; data pointer https://metabolomics.helmholtz-munich.de/ukbbpgwas/ (Synapse syn51364943; trans table names not verified).
- Pan-UK Biobank (7,228 phenotypes, per-phenotype flat files): https://pan.ukbb.broadinstitute.org/docs/technical-overview (release date not confirmed by the search, so I did not rely on it as a T0-clean source).
- MVP 2,068-trait GWAS, Verma et al. 2024 Science: https://impact.ornl.gov/en/publications/diversity-and-scale-genetic-architecture-of-2068-traits-in-the-va/ (summary statistics are via dbGaP, access restricted; not used in any idea above).
- GWAS ATLAS gene-level downloads: search did not confirm a MAGMA gene-level bulk download; not used.
