# R1 critic: engineering / compute / what can be built and benchmarked tonight (2026-11-12, written ~00:15-01:00)

Scope: 73 ideas in `r1_{evostruct,ml,recsys,llmtext,causal,singlecell,statgen,wildcard}.md`. Checks were read-only one-liners (file exists, schema, row
counts, dates). Nothing was run on the benchmarks. Python = `~/biotools/venvs/bio-net/bin/python` (lightgbm 4.7, torch 2.11 cu128, duckdb, pyarrow, sklearn).

## 0. Facts I verified that change the ideas (read before the per-idea notes)

| # | Fact (how checked) | Consequence |
|---|---|---|
| F1 | Leaderboard: `transfer_jaccard` nuAP 0.096 / 0.104 (temporal dev / confirm) is **above** `modet_headline` 0.082 / 0.080 in this all-gene harness; forward 0.096 / 0.083, holdout 0.171 / 0.165. | The bar for a "learned baseline" is raw transfer, not the headline. If the stacker does not beat `transfer_jaccard`, that is a finding. |
| F2 | T0 MAGMA coverage: 137 usable T0 studies (`studies_selected_t0.tsv` joined to `magma_runs_t0.tsv`; 3 dated >= 2021-04-01; 81 matched via a descendant, 56 direct). Tasks with a T0 study: temporal dev **15/53**, confirm 32/138; forward dev 42/842; holdout dev 13/303. Post-T0 `gwas_magma/`: temporal dev 16, forward dev 28. | Every own-trait GWAS idea is a covered-subset experiment (15 dev diseases, MDE there about 0.03 nuAP). Only cross-trait proxies (CI-08, WC-XPOWER, LT-9) cover all tasks. |
| F3 | Temporal tasks exclude a median of **1,601 genes** per dev task from the ranking (weak-evidence "unknown"); forward/holdout exclude only the seeds (median 39). | Own-disease weak-evidence ideas (SG-7, ML-9, own-channel part of IR-7, half of LT-3/WC-BACKLOG) are removed by design on `temporal`, and are fully rankable on `forward`. |
| F4 | `evidence_2509/` holds only `gwas_credible_sets` (1,753,850 rows, 100 parts, `score` down to 0.05). No 25.09 ClinVar/G2P/Europe PMC evidence, no 25.09 association-by-datasource table on disk. | SG-7 is directly testable on `forward` (sub-threshold L2G at 25.09). Backlog / funnel ideas are temporal-only tonight unless OT 25.09 evidence is downloaded. |
| F5 | `*_leads.parquet` contain only genome-wide significant leads (80 rows, max P 4.6e-8 in `GCST001255`). | WC-L2GEMU's sub-threshold peaks need a pass over `gwas_sumstats_t0` (90 GB), not the leads files. |
| F6 | Arrival table (4,981 rows): gwas_only 2,863 (A 1,534), clinvar_only 827, curated_only 633, mixed 434, burden_only 201; `id_unmapped` = 55 (1.1 %); `novel_gene` = 518. | "Identifier migration" in the strict sense is 1 % here; the backlog ideas depend on evidence filed under another disease id for the clinvar/curated classes (29 %), which is a 20-minute diagnostic. |
| F7 | `data/donors/` already has 7 families (msig_go 9,680 sets, msig_reactome, msig_wiki, hpo2101 3,765, impc13 526 sets / 5,766 genes, complexportal, paralog30 3,014 sets / 4,694 genes); `scripts/build_donors.py` has a `clusters` branch (k-means on views) not yet run. | EV-1, EV-2, SC-2, LT-2 (HPO/GO part) need no new data, only a generic donor-transfer function. |
| F8 | HF cache: `Qwen3-Embedding-0.6B` complete (1.2 GB, `model.safetensors`); **`Qwen3-Embedding-8B` is 16 MB = no weights**. No PubMedBERT/SapBERT. `tabpfn 9.2` installed, TabICL not. `vllm` env exists. | Text ideas use the 0.6B embedder (modern, so `transductive` on temporal) or TF-IDF (clean). LT-4/LT-5 as written (8B / 27B) are out. |
| F9 | Missing on disk (find over downloads, subex raw, biotools data): LD blocks (ldetect), eQTLGen, UKB-PPP, Genebass / AZ PheWAS, PhenomeXcan, Rummagene, AlphaMissense, BioGRID ORCS, GenePT, ERC2.0, Animal QTLdb, OMIA, NIH ExPORTER, PMID-to-year table. | Each is a download with unverified size/URL; see per-idea notes. Gene coordinates exist (MAGMA parquet: `chr,start,stop`), so fixed windows replace LD blocks. |
| F10 | `ot2104/` = three association tables only (`associationByDatasourceIndirect`: 11,780,049 rows, cols `targetId, diseaseId, datasourceId, datasourceHarmonicScore, datasourceEvidenceCount, diseaseLabel`); no disease index. Synonyms/xrefs: `ontologies_2021-04/{efo_v3.29.1,mondo_v2021-04-07,hp_v2021-04-13}.obo`. | IR-7 is ready. Backlog matching takes names from `diseaseLabel` and synonyms/xrefs from the 2021-04 obo files. |
| F11 | Label-release data on disk that must never be an input on `forward`: `v2/ot2609_assoc`, `~/biotools/data/evidence/raw/opentargets/26.09/{l2g_prediction_slim,credible_set_gwas_slim}.parquet`. Post-T0 `gwas_magma/studies_selected.tsv` (26 rows) has no date column. | Forbid these paths in code review. Post-T0 MAGMA cannot be declared "<= 2025-09" without looking up each GCST date. |
| F12 | Holdout: `transfer_onto` AUROC 0.31 / 0.29 (below chance). | The family mask is visible as an anti-signal. A stacker trained on holdout can learn mask artefacts; holdout stays secondary and is never used to select channels. |
| F13 | Machine at 00:12: load average 30 on 32 cores, GPU 14 GB in use by `p17-score`, 23.8 GB free; 72 GB RAM available. | Cap every agent at 4 threads (`OMP_NUM_THREADS=4`, LightGBM `n_jobs<=8` only in the stacker); GPU jobs <= 8 GB through `gpu-run`. |

Harness API recap (from `bench.py`, `tasks.py`, `methods/`): a method is `score(tasks, B) -> float32 [n_tasks, 19255]`, registered with
`@register(name, benches=...)`; labels only via `B.context_for(task)` or the exact correction (`t.mask_r, t.mask_c`, `B.hidden_counts`,
`B.row_masked_counts`); any disease-row matrix must be indexed by `B.context_ids` (this is what removes the sealed fold on temporal).
Useful identity for many ideas: the transfer score is linear in the donor matrix, `S = w^T L_t`, so **"transfer on L @ K" equals `S_transfer @ K`**
for any gene-gene kernel K. Lifting needs no new masking code; it is one sparse product on an already masked score matrix.

Harness gaps that will bite 8 parallel agents: (1) all methods are registered in one `methods/__init__.py` (edit conflicts): switch to
auto-import of `methods/*.py` first; (2) concurrent appends to `leaderboard.tsv`; (3) no strata read-out, no blend helper, no trained-model path,
no NaN convention for partial-coverage scores; (4) `t.cluster` is the disease id on forward/holdout, so relatives are separate bootstrap/CV units.

## 1. Per-idea assessment

Format: **feasible tonight** | inputs (verified = I saw the file) | sketch against the API | runtime | bug/leak risk.
"lift" = `S_tr @ K`; "donor-transfer" = `_transfer` generalised to `vstack([context, E])`; "stacker" = section 3.

### Evolution / structure (EV)
- **EV-1 phenolog donors.** Yes. `data/donors/impc13.npz` verified (526 sets, 5,766 genes); current MGI (`~/biotools/data/evidence/raw/mgi/MGI_GenePheno.rpt`) verified but dated 2026 (not clean on any temporal bench). Sketch: donor-transfer with E = impc13, one scalar row-family weight. 15 min after the generic function exists. Risk: OT already has `phenodigm` (711k rows at 21.04) and temporal excludes weak-evidence genes (F3), so expect little; symbol-uppercase ortholog mapping is crude.
- **EV-2 paralog/fold lift.** Yes. `paralog30.npz` verified (only 4,694 genes at identity >= 30: thin); `afdb_clusters_v3` verified, not parsed. Sketch: K = I + a*P; `S_tr @ K`. 10 min. Risk: none for leakage if S is the masked score; tandem paralog arrays make this partly the banned "genes near seeds".
- **EV-3 co-evolution kernel.** Yes (NPP only). `data/views/phylo_npp.npz` verified (256-d, 18,202 genes). Sketch: cosine kNN-50 on the view -> (a) lift, (b) `rwr(tasks, A_knn)`. 15 min. ERC2.0 (7.5 GB Dryad, browser session): no. Risk: the view is a 256-d reduction, not per-clade z; the degree-vs-STRING-degree check is mandatory before reading the number.
- **EV-4 z x constraint hazard.** Yes, but 15 dev tasks (F2). Inputs verified (`gwas_magma_t0/*_w10.parquet`, covariates `gnomad_loeuf`). Sketch: stacker features `z_own`, `z_own*loeuf` (NaN where no study). 20 min inside the GWAS agent. Risk: study-date filter (3 studies dated >= 2021-04-01 must go); descendant-matched studies are not the trait itself.
- **EV-5 livestock/dog QTL donors.** No. Downloads unverified (Animal QTLdb, OMIA), megabase intervals, author-candidate circularity. Drop.
- **EV-6 convergent-evolution rates.** No tonight. Compara file verified (has `dn, ds`), but pairwise dN/dS to human is not a branch rate and the proper matrix is the ERC/TOGA download. `phylo_npp` in EV-3 covers the cheap part.
- **EV-7 interface-resolved edges.** No. Interface residues need 15.3 GB figshare models; AlphaMissense missing; `openpyxl` missing for the Burke table. The premise test (pDockQ edges vs matched random) is 30 min but coverage-limited far below the MDE.
- **EV-8 domain-level donors.** Marginal. Needs a Pfam parse of `uniprot_sprot-only2021_02.tar.gz` (verified, not opened), about 1 h. The ESM C / ProtT5 kNN lift (already in views) tests the same hypothesis in 10 min; run that first and only build Pfam if it is positive.
- **EV-9 selection-space deep sets.** No as written. Covariates verified (`gnomad_loeuf, gnomad_pli, hgnc_year, loc_loglen`); the idea reduces to stacker interactions between seed-set means of these covariates and the gene's values. Add 4 such features to the stacker (10 min); no deep-sets.

### Machine learning (ML)
- **ML-1 cold-start imputation.** Yes (ridge version). Views verified (15 npz). Sketch: ridge from concatenated views to the gene's column profile (SVD-128 of a "safe" context = context minus all task rows and their descendants), predictions made gene-fold out-of-fold; `K_hat` = imputed columns; score = `S_tr @ K_hat` or transfer on `L + lambda*L_hat`. 45 min CPU. Risk: (a) SVD on the unmasked context leaks own/descendant target pairs (287 temporal targets): use the safe context; (b) if imputation is not gene-fold cross-fitted, known genes get near-identity profiles and cold genes do not, and the stacker learns that difference.
- **ML-2 ascertainment-simulator PFN.** No. 2 h GPU plus simulator design, sim-to-real risk, nothing reusable.
- **ML-3 Rummagene set model.** Stretch. Missing: Rummagene GMT (URL and size unverified, about 1-2 GB) and a PMCID -> date table (`PMC-ids.csv.gz`, about 200 MB). If the download works: PPMI + SVD-256 on set x gene incidence = one more view, then seed-centroid and lift (45 min CPU). The transformer is out. Risk: without the date filter it is `holdout-current` only; GWAS supplement tables are the label.
- **ML-4 release-pair extrapolator.** Partly. T0/post-T0 MAGMA pairs exist for few traits (137 T0 studies, 102 post-T0 files; overlap not counted), training labels are post-T0 (transductive on temporal), covered dev tasks 15. Tonight only its diagnostic: does T0 own z rank later positives (AUC) on the 15 + 32 covered tasks. The supervised extrapolator waits.
- **ML-5 episodic set transformer.** No (2 h GPU, shared card). Cheap stand-in that tests the premise: per view, cosine of gene to the seed centroid and mean of top-3 seed similarities = 30 stacker features, trained across diseases (30 min). If these add nothing, the transformer has no case.
- **ML-6 TabICL in-context.** Control only. TabICL not installed; the "LightGBM on pooled arrival rows of the previous window" control is exactly the stacker trained on the `past` benchmark (section 2). TabPFN 9.2 can be swapped in as the learner on a 10k-row subsample later (30 min) if the control is positive.
- **ML-7 set diffusion.** No (depends on ML-5).
- **ML-8 learned disease encoder.** Yes (untrained version). Embed disease names with Qwen3-Embedding-0.6B (`gpu-run --mem 4`, 2 min), donor weight = name cosine, run transfer with those weights, stratify by seed count. 30 min. The ideator's failure mode is wrong for temporal/forward: relatives are not removed as donors there, only own+descendant target pairs are masked. Risk: modern embedder = transductive on temporal; fair on forward.
- **ML-9 evidence-trajectory hazard.** No. `publicationYear` is empty for `europepmc` (needs PMID -> year, missing), and own-pair trajectories are excluded on temporal (F3); nothing on disk for forward (F4).

### Recommender systems (IR)
- **IR-1 exposure-aware relevance.** No as a model (iALS + exposure, 2 h; exposure is near-constant within a disease). Its usable part = gene testability covariates (`nsnps`, `loc_loglen`, `loc_gdens500k`) in the stacker.
- **IR-2 locus slates** (= SG-1). Yes with windows. Gene coordinates verified in MAGMA parquet (about 17k genes; the other 2.2k need `ensembl104`). Sketch: K_locus[g, g'] = 1 if within +-250 kb; features `max` and `sum` of `S_tr` over the window, own value excluded; win propensity = stacker with distance rank and `loc_gdens500k`. 30 min. First the 10-min diagnostic (share of zero-support GWAS positives with a supported neighbour). Risk: window max copies the seed-locus prior (banned, no gain) unless seeds' own loci are reported separately.
- **IR-3 identity linking** (= LT-3 = WC-BACKLOG). Yes on temporal. Sources verified: `eva` 837,193 rows, `genomics_england` 20,687, `gene2phenotype` 2,550, all with `diseaseFromSource`, `diseaseFromSourceId`, `diseaseFromSourceMappedId`, `diseaseId`. Sketch: one projection read into a (gene, source label, source id, mapped id) table; arm A exact xref/CUI/normalised-string match to task names + 2021-04 obo synonyms; arm B char-n-gram TF-IDF cosine; score = max over the gene's rows whose `diseaseId` is not the task or a descendant. 45 min CPU. Risk: (a) matched rows may sit in family rows that are legitimately visible on temporal: then transfer already has them; (b) the hit genes may be in `exclude` (F3); (c) never use current MONDO/EFO for synonyms.
- **IR-4 powered nulls** (= WC-UPPERLIM). Diagnostic only: positive rate of (nsnps >= 50, |z| < 0.5) genes inside transfer top-200 vs the rest, on the 47 covered temporal tasks. 15 min. A veto model is not worth building for 15 dev tasks.
- **IR-5 evidence baskets.** Possible, low priority. `studyId` / `literature` verified in the schema; europepmc is 10.5 M rows. Sketch: basket x gene sparse matrix -> item-item cosine top-200 = one more K for lift and for `seeds @ K`. 1 h. Risk: it is the banned EASE result re-cut, and paper baskets carry popularity; holdout would need per-task basket masking.
- **IR-6 trending.** Yes, first thing to run. `D.label_matrix("1904")` supported (3,435 rows, verified). Sketch: align 19.04 rows to `B.context_ids`, `Lw = L21 + beta*(L21 > L19)`, run `_transfer` with donor matrix Lw (similarity still on L21), apply the same mask correction. 15 min. Risk: 19.04 rows of sealed diseases must be dropped by the `context_ids` join; diseases absent in 19.04 look entirely "new" (handle with a row flag, not as delta).
- **IR-7 multi-behaviour funnel.** Yes on temporal. `associationByDatasourceIndirect` verified (11.78 M rows, 20 data sources). Sketch: C_k = disease x gene score per source restricted to `B.context_ids`; `T_k = transfer(donor = C_k)` with Jaccard similarity from the binary context; own row and descendants zeroed for target pairs; 20 features into the stacker. 1 h. Risk: C_k has pairs that are not in L, so `mask_r/mask_c` (defined on L pairs) do not cover them: zero the task's own row and descendant rows completely in every C_k. Text-mining channel will dominate unless rank-normalised per channel.
- **IR-8 cold-start tower.** No as a neural model tonight; ML-1 ridge is the same test. Promote only if ML-1's cold-pool AUC >= 0.66.
- **IR-9 next-item sequences.** No. First-evidence dates missing for curated sources and europepmc; median 38 seeds on dev, order information is thin.
- **IR-10 burden PheWAS cross-domain.** No. Genebass is Hail tables on cloud buckets, tens of GB, needs a Hail export; AZ PheWAS has no verified bulk file.

### Language / literature (LT)
- **LT-1 masked-context cloze.** No tonight. 3 M sentences, encoder download, two masking passes, 2 h with three leak routes (residual disease strings, family sentences, head trained on dev diseases).
- **LT-2 virtual donors from MeSH/HPO/GO.** Half. HPO and GO donor rows exist (F7) and run through donor-transfer in 15 min. The PubTator3 gene x MeSH matrix (`disease2pubtator3.gz` 2.2 GB, `gene2pubtator3.gz` 758 MB, verified) is a 30-45 min DuckDB join on a loaded machine and needs a PMID cut (PMID order is only approximately chronological: cut conservatively and say so). Semantic donor selection uses the 0.6B embedder. Risk: HPO rows are disease-derived labels: fine on temporal (2021-01 file), a direct leak on holdout and partly on forward (old labels of hidden pairs): restrict hpo2101 to temporal or mask it like context.
- **LT-3 curation debt.** See IR-3 (one implementation). The weak-tier part (ClinGen Limited, PanelApp amber) is in `evidence_2104` scores; add as a feature of the same table.
- **LT-4 equal-budget dossiers.** No. Templating from six tables is 1.5 h of plumbing; the 8B embedder is not on disk (F8). The views already are the numeric version.
- **LT-5 LLM judge.** No. 26 GB GPU for 60-90 min on a shared card with 23.8 GB free, weights unverified, model cutoff unknown.
- **LT-6 mechanism cards.** No. Depends on LT-4/LT-8 text; small clusters.
- **LT-7 arrival hazard on the previous transition.** The paradigm half is the `past` benchmark + stacker (section 2/3), which will exist. The literature-momentum half needs PMID -> year: no.
- **LT-8 time-locked text two-tower.** Yes as a view. Parse UniProt 2021_02 function comments (strip DISEASE lines), embed with the 0.6B model (`gpu-run --mem 4`, about 3 min), save `data/views/uniprot_text.npz`, then seed-centroid + lift like any view. 40 min. Risk: `.dat` parsing; modern encoder = transductive on temporal.
- **LT-9 proxy traits by language.** Yes, in the GWAS agent: trait-name x disease-name cosine as an alternative weighting of the Z matrix (10 extra lines once Z exists). Trait names are in `studies_selected_t0.tsv` (verified column `trait`).

### Causal (CI)
- **CI-01 upstream-of-core** (= SC-1). Yes. Raw matrices verified (`/mnt/msadb/subex/derived/ps_{k562_gwps,cd4_2025,cd4_2025_rest,cd4_2025_stim48hr,hepg2_nadig,jurkat_nadig,rpe1_ess}.npy`, `*_genes.txt`, `*_bulk.npz`). Sketch: load, column z-score, upstream = standardised mean over seed readout columns minus matched random sets, downstream = mean of seed perturbation rows, NaN for unperturbed genes; max / mean over atlases; 4-6 stacker features. 40 min CPU, memory < 4 GB. Risk: row ids (perturbations) and column ids (readouts) come from different files: assert shapes against `_genes.txt` and `_bulk.npz`; essential-gene knock-downs score high for every disease (subtract the per-gene mean over all tasks); CD4 atlas is dated after the forward seed date (declare).
- **CI-02 trans-QTL convergence** (= SC-5, SG-6). Stretch. Missing: eQTLGen significant trans file (tens of MB, URL from memory), plus SNP -> cis gene. 1 h if the download works. Expect a near-empty score matrix; run once as a sparsity check, not as a model.
- **CI-03 cis-MR / coloc.** No. 90 GB sumstats + GTEx eQTL download + per-study SNP joins on a loaded machine for 15 covered dev tasks.
- **CI-04 directed donor graph.** Yes. Needs L19 aligned (same as IR-6). Sketch: asym(a->d) from 19.04 -> 21.04 counts, `w = jaccard * exp(theta*asym)`, forked `_transfer`. 20 min. Risk: theta tuned on dev with 53 diseases is noise; pre-register theta = 1 and report theta = 0 beside it.
- **CI-05 power-jump gate.** No as a model (15 dev tasks cannot fit a gate). A stacker with `has_study`, `log N`, `n_sig` task features learns the gate if it exists.
- **CI-06 channel detectability** (= SG-8, WC-SELFN). Reduced form yes: task features = channel mix of the seeds (share of seeds with genetics / ClinVar / literature evidence, from `evidence_2104`), gene features = architecture covariates; LightGBM learns the product. Control = same model without channel-mix features. 40 min after the evidence table exists.
- **CI-07 invariant predictors.** No. 1 h of refits for a diagnostic with expected overall change inside noise.
- **CI-08 rg-borrowed gene z.** Yes, and it covers all tasks. Sketch: Z [137 studies x genes] from `_w10`; per task ridge of the seed indicator on Z with leave-locus-out (drop genes within 1 Mb of any seed when scoring) -> proxy z; where an own study exists, also `z_own`. 40 min. Risk: locus smearing makes the proxy a "near seeds" prior (banned, no gain): the leave-locus-out variant is the claim; date filter; banned-adjacent ("T0 MAGMA studies as donors"): report correlation with a plain study-donor transfer.
- **CI-09 allelic-series concordance.** No (Genebass, see IR-10).

### Single cell / perturbation (SC)
- **SC-1 virtual Perturb-seq screen.** Yes: same implementation as CI-01; add the HPA-sc routing weight (view `hpa_sc` verified) as a second variant.
- **SC-2 experiments as donor rows.** Yes. Build E from (a) `build_donors.py clusters` (k-means on views; not yet run), (b) top-100 responders per perturbation and top regulators per readout from the raw `ps_*` matrices, (c) DepMap co-essential kNN sets. Donor-transfer with IDF row weights; rewiring control on E. 45 min. Risk: Jaccard to a 30-gene seed set is 0 or 1 gene for most rows: use overlap significance (hypergeometric) instead of Jaccard for experimental rows, or the result is noise by construction.
- **SC-3 cell-address donors.** Yes, generalised to every view: disease embedding = mean seed embedding in view v; donor embedding = mean over the donor's genes; weight = cosine; transfer with those weights. 15 views x 3 benches about 20 min. Risk: donor embeddings include hidden target genes for own/descendant rows (and family rows on holdout): subtract masked genes' contributions per task or drop `fam_rows` as donors.
- **SC-4 regulator-program-trait.** No. Signed version needs Genebass; unsigned version is PoPS on 15 covered dev tasks.
- **SC-5 trans-QTL atlas.** See CI-02.
- **SC-6 phenocopy of diseased state.** No. Census queries dominate (1.5 h), 10 pilot diseases, DE is label-adjacent.
- **SC-7 set model on experimental sets.** No (GPU transformer); SC-2 and the seed-centroid features are the cheap versions.
- **SC-8 within-cell-type programs.** No. NMF over Tabula Sapiens (54 GB) is CPU-heavy on a machine at load 30.
- **SC-9 coherence cutoff and layers.** Later tonight (20 min) on the best ranking, CPU only. It is the only cheap route to the "module with a cutoff" deliverable besides WC-CAPTURE; not a ranking test.

### Statistical genetics (SG)
- **SG-1 locus-space transfer.** See IR-2 (one implementation). Run the temporal-clean variant without cS2G.
- **SG-2 phenome fingerprints (PhenomeXcan).** Yes if the download works. Missing: Zenodo 10.5281/zenodo.3530669 gene x trait S-MultiXcan z (order 1 GB, unverified; the distributed matrices may be pandas pickles: untrusted, load with a restricted unpickler or use the tsv/h5 variant, `python -I`, own directory). Sketch: z-score, PCA-256 -> `data/views/phenomexcan.npz` (date class pre21) -> seed-centroid, lift, donor similarity; plus the leave-locus-out profile similarity. 1 h. Risk: LD neighbours share profiles (regress out the +-500 kb mean); trait columns that are the task disease or relatives (drop by name for the strict variant).
- **SG-3 mash forecast.** No tonight (EM implementation and locus smearing). The crossing-probability transform on CI-08's proxy z (WC-XPOWER) is the 15-line version.
- **SG-4 HPO-decomposed traits.** No. HPO -> trait map needs manual checking; 137 T0 studies do not span HPO.
- **SG-5 burden atlas.** No (Genebass). The T0 consortium tables cover 3-4 tasks.
- **SG-6 trans-QTL regulators.** See CI-02.
- **SG-7 ingestion lag.** Yes on forward, cheap, probably the largest single effect of the night. `evidence_2509/sourceId=gwas_credible_sets` verified (1.75 M rows, score 0.05-0.98). Sketch: per (disease, gene) max and count of L2G score at 25.09, propagated to ancestors with the 25.09 ontology, for pairs not in the seeds; score and stacker feature. 30 min. Risk: confirm that forward seeds = thresholded 25.09 association, so sub-threshold pairs are rankable (F3 says exclude = seeds); the propagation must not use the 26.09 ontology; report it as an information-source baseline, separately from modelling claims. On temporal the same signal is in `exclude`.
- **SG-8 channel hazard.** See CI-06.
- **SG-9 rg donor weights.** No tonight (UKB phenotype -> EFO name mapping is the work). CI-08's seed regression gives data-driven trait weights without a mapping.

### Wild card (WC)
- **WC-L2GEMU.** Half. Host-gene model from 21.04 L2G rows (`ot_genetics_portal` 388,324 rows, `resourceScore`, `studyId`, `variantId` verified): 30 min. Locus-crossing half needs sub-threshold peaks from 90 GB sumstats (F5): no. Tonight: gene-level `z_own` x host prior (distance rank, `loc_gdens500k`) on the 15 covered dev tasks as a go/no-go.
- **WC-XPOWER.** Yes: CI-08 plus a two-component mixture per study and `P(z' > 4.5 | N' = 3N)`; three arms (linear, crossing, crossing + transfer). 20 extra minutes.
- **WC-BACKLOG.** See IR-3.
- **WC-SELFN.** See CI-06; the per-channel flow models need the `past` benchmark and channel labels from `evidence_2104`. Reduced form tonight.
- **WC-HAWKES.** Yes with IR-6: transfer on the 19.04 context vs the 21.04 context -> level, delta, delta of donor count as stacker features. 20 min. Risk: 19.04 -> 21.04 pipeline changes; rows missing in 19.04.
- **WC-CAPTURE.** Test 1 yes (15 min after the evidence table): Chao1 / Good-Toulmin on per-gene evidence multiplicity vs realised arrival count (Spearman against `n_t0`). Test 2 (explore budget) only if an explore ranker exists.
- **WC-DCA.** No. About 10k L1 logistic fits on a loaded machine to re-derive co-occurrence (EASE is on the banned list).
- **WC-UPPERLIM.** See IR-4 (diagnostic).
- **WC-MARKET.** No. ExPORTER download plus symbol matching, 2 h, overlaps Europe PMC text mining.

Count: 73 ideas; about 24 runnable tonight in some form (many as one shared implementation), 9 as diagnostics or reduced forms, the rest not tonight.

## 2. (a) Implementation plan: 8 parallel agents, about 3 hours

### Shared components (built once, by agent 1, in this order)
1. **Plumbing (first 10 min, blocks nobody after it):** `methods/__init__.py` auto-imports every `methods/*.py`; each agent owns one file `methods/<agent>.py` and one results prefix. `bench.run(..., leaderboard=<path>)` writing to `results/leaderboard_<agent>.tsv` (no concurrent appends to the main file).
2. **Channel cache (`src/modeng/chan.py`):** `chan.get(name, bench) -> float16 [n_tasks_all, n_genes]` memmap in `data/cache/chan/<bench>/<name>.npy`, computed once over ALL tasks of the bench in task order by calling the registered method; NaN = "channel not defined for this task/gene". About 63 MB per channel on forward.
3. **Generic transfer (`methods/transfer.py`):** `transfer(tasks, B, donor=None, extra=None, sim="jaccard"|array, k=50, agg="sum"|"max", zero_rows="self+desc")` with the exact mask correction of `_transfer`, real-valued donor matrices, extra donor rows E (never masked unless flagged `label_like`), and externally supplied similarities [n_tasks, n_rows]. Returns also support diagnostics (number of donors carrying the gene, best donor similarity). Used by agents 2, 3, 4, 6.
4. **`past` benchmark (`tasks.build_past`):** seeds = 19.04, targets = 21.04 arrivals, context = 19.04, non-sealed, same Task API, `self+desc` masking. Training-only; gives arrival-objective rows that are entirely pre-T0. Used by the stacker and by IR-6 / CI-04 / WC-HAWKES tuning.
5. **Strata (`src/modeng/strata.py`):** per task: recall@100 for zero-transfer-support targets, within-pool AUROC for genes with an empty context column, seed-count bins, covered-subset flag, and (temporal) arrival class / evidence class from the arrival parquet.
6. **Stacker + increment test (`src/modeng/stack.py`, section 3).**
7. **Universal donor store:** `data/donors/*.npz` (exists) + `clusters` families + perturbation-responder rows (agent 2).
8. **Universal kernel store:** `data/kernels/<name>.npz` sparse gene x gene, row-normalised (agent 3): kNN-50 per view, paralog, +-250 kb window.

Contract for every agent: deliver channels as registered methods `ch_<name>` (raw score, NaN where undefined) plus `<name>_rb` = rank-average with
`transfer_jaccard` for the standalone leaderboard; run on temporal dev and forward dev only; 4 threads; no confirm split; write one paragraph of
leakage class per channel. The channel API is the existing method API, so nobody waits for agent 1 except for the increment numbers.

### Agents
| # | Agent | Ideas | First deliverable (T+60 min) | Second (T+120) |
|---|---|---|---|---|
| 1 | **Core / stacker** | baseline B0, ML-6 control, LT-7 paradigm, EV-9 and CI-05/CI-06 reduced forms | plumbing, `chan`, `transfer`, `past`, `strata`; B0 out-of-fold on temporal + forward dev | increment table for every delivered channel; shuffled-channel control |
| 2 | **Donor universe** | EV-1, SC-2, LT-2 (HPO/GO/MSigDB part), EV-2 as sets | donor-transfer per family (7 existing) with rewiring control | k-means view clusters + perturbation responder rows; hypergeometric similarity variant |
| 3 | **Gene kernels / cold start** | EV-2, EV-3, ML-1, IR-8 (ridge form), ML-5 (seed-centroid form), EV-8 via ESM kNN | kernel store; `S_tr @ K_v` for 15 views + paralog; seed-centroid features per view | ML-1 cross-fitted imputation; cold-pool AUC; NPP-kNN RWR |
| 4 | **Disease similarity / history** | IR-6, CI-04, WC-HAWKES, SC-3, ML-8 | IR-6 (beta = 1 pre-registered), CI-04 (theta = 1), velocity features | view-space donor similarity (15 views), name-embedding donor weights (0.6B, `gpu-run --mem 4`) |
| 5 | **GWAS z** | CI-08, WC-XPOWER, LT-9, EV-4, IR-4 / WC-UPPERLIM diagnostic, ML-4 diagnostic, IR-2 / SG-1 locus window, **SG-7 on forward** | Z matrices (T0, date-filtered), SG-7 forward channel, own-z AUC on covered tasks, locus-window diagnostic | cross-trait proxy with leave-locus-out, crossing transform, null-veto diagnostic, locus features |
| 6 | **Evidence tables** | IR-3 / LT-3 / WC-BACKLOG, IR-7, CI-06 / SG-8 / WC-SELFN channel mix, WC-CAPTURE test 1 | `data/cache/evidence2104_small.parquet`; backlog diagnostic + arm A/B; channel matrices C_k | `T_k` funnel channels; channel-mix task features; capture-recapture Spearman |
| 7 | **Perturbation** | CI-01 / SC-1, later SC-9 | virtual-screen channels (upstream, downstream, routed) on forward + temporal dev | essential-gene correction, immune/hepatic subset read-out; SC-9 cutoff on the best ranking |
| 8 | **New sources (network-bound)** | SG-2, LT-8, then CI-02 / SC-5 / SG-6, ML-3 | PhenomeXcan download -> view; UniProt-2021 text view (hand both to agent 3's kernel store) | eQTLGen trans sparsity check; Rummagene PPMI-SVD view if the download succeeded |

### Order
- **T+0:00** all agents start. Agent 1 does plumbing in 10 min. Agents 4 and 5 run their 10-15 min tests first (IR-6, CI-04, SG-7, diagnostics).
- **T+0:45** agent 1 has B0 and the increment harness; agents register first channels.
- **T+1:00 checkpoint:** standalone and rank-blend numbers for every first deliverable; kill rules applied as written by the ideators.
- **T+2:00 checkpoint:** stacker increments for all surviving channels (temporal dev, forward dev) and strata.
- **T+2:00-2:45** agent 1 fits the combined model (B0 + channels that passed, list written to a file and committed before the run); agents 2-8 finish second deliverables or stop.
- **T+2:45-3:00** one run of the pre-registered combined model and of B0 on the confirm splits; leaderboard merged; nothing is tuned after this.

## 3. (b) The strong, fair learned baseline (B0) and the increment test

**Rows.** One row per (task, gene) with gene not in `t.exclude`. Training sample per task: all targets, the top 1,000 genes by `transfer_jaccard`,
1,000 uniform random others; each task has total weight 1. Prediction on all 19,255 genes.

**B0 features (all with the exact mask; each gene-level score also as within-task rank quantile):**
transfer family: sum over top-k donors for k in {10, 50, 200}, max over donors (`tr_max`), overlap-coefficient and cosine similarity variants,
`transfer_onto`, number of top-50 donors carrying the gene, best similarity among donors carrying it; popularity (masked column count, log1p);
degree in `B.network` and `rwr_string`; gene covariates from `covariates/allgene_*.parquet` (`study_attention, hgnc_year, is_orf, go_terms,
loc_loglen, loc_gdens500k, gnomad_loeuf, gnomad_pli`; `known_t0` only on temporal; on forward/holdout replace it by "context column non-empty");
task features: number of seeds, mean seed popularity, best and 10th-best donor similarity, number of donors with similarity > 0.1.

**Learner.** LightGBM binary, fixed before any run: `num_leaves=15, learning_rate=0.05, n_estimators=300, min_data_in_leaf=50, feature_fraction=0.8,
lambda_l2=1`, three seeds averaged. One alternative (`lambdarank`, truncation 100) is compared once on dev out-of-fold and then frozen.

**Cross-disease training with disease-grouped CV.**
- Groups: temporal = modet cluster (`t.cluster`). Forward / holdout = connected components of (ontology relatives among tasks) joined with
  near-duplicates (Jaccard of seeds + targets >= 0.5); `t.cluster` is the disease id there and must not be used as the group, because a parent and
  its child share arriving genes through propagation.
- Dev: 5-fold GroupKFold, out-of-fold predictions. Confirm: model trained on all dev tasks. Confirm labels are never training rows.
- Second training regime, reported beside the first: rows from the `past` benchmark only (19.04 -> 21.04). On temporal this model has seen no
  benchmark label at all and is the cleanest claim; it also is the ML-6 / LT-7 "arrival objective" control.
- Forward: the temporal window (21.04 -> 25.09, non-sealed tasks only) may be added as training rows; holdout: not used for training or selection (F12).

**Acceptance of B0.** nuAP on dev out-of-fold must be >= `transfer_jaccard` + 0.01 on temporal (0.096) and forward (0.096). If not, report that
stacking over transfer features adds nothing and use `transfer_jaccard` + rank-blend as the reference.

**Increment test for a channel c** (one or several features):
1. Same rows, folds, seeds and hyperparameters; fit `B0 + c` and `B0`.
2. Delta = per-task nuAP (and AP, recall@100, strata) of `B0 + c` minus `B0`; paired cluster bootstrap with `compare.py` over the CV groups.
3. Controls: `B0 + shuffle(c)` (c permuted across genes within each task: gives the gain from merely adding a column, expected 0); for donor /
   kernel channels the degree-preserving rewiring of E or K; for partial-coverage channels the delta on covered tasks and on uncovered tasks
   (the latter must be 0, otherwise the NaN pattern leaks task identity).
4. Decision rule (written before running): keep c if delta nuAP >= +0.005 with a 95 % interval above 0 on forward dev (842 tasks) or on temporal
   dev, and not below -0.003 on the other; strata numbers are reported for every channel regardless of the decision.
5. The combined model contains only kept channels; its confirm run happens once.

**Leak and bug list for the stacker.** Features must be computed per task from `context_for(task)`; never fit anything (SVD, ridge, kNN on label
columns) on the unmasked context; rank-normalise per task after removing `exclude`; `known_t0`-type covariates are date-specific; NaN means
"undefined", never 0; float16 cache only for rank-like features; check that out-of-fold predictions exist for every dev task exactly once.

## 4. (c) Top 8 by expected information per hour

| Rank | Experiment | Cost | Why it is informative either way |
|---|---|---|---|
| 1 | **B0 stacker + increment harness** (with the `past`-trained variant) | 45 min | Every other number tonight is uninterpretable without it; also answers whether the arrival objective (ML-6 control, LT-7, WC-HAWKES) matters. |
| 2 | **SG-7 on forward** (25.09 sub-threshold L2G) | 30 min | Data on disk, 842 dev tasks, aimed at the GWAS-only majority. A large gain redefines the forward baseline for all other ideas; a null says OT's sub-threshold rows carry nothing. |
| 3 | **IR-6 + CI-04 + WC-HAWKES** (19.04 history) | 30 min for all three | Ten lines each on existing code; full coverage; temporal-clean. |
| 4 | **Backlog diagnostic and arm A** (IR-3 / LT-3 / WC-BACKLOG) | 20 + 25 min | Three ideators' top or "most underrated" pick; one count (share of clinvar/curated arrivals with T0 evidence under a non-family id and not in `exclude`) decides all three. |
| 5 | **Kernel lift over 15 views + seed-centroid features** (EV-2, EV-3, ML-5 form, IR-8 form; then ML-1) | 40 min | One implementation tests five ideas and the zero-support stratum directly; views are ready. |
| 6 | **IR-7 funnel channels** (donor weak evidence per data source) | 60 min | The most plausible "more of what works"; 11.8 M rows on disk; clean on temporal. |
| 7 | **CI-08 / WC-XPOWER / LT-9 cross-trait proxy z with leave-locus-out** (+ own-z and null diagnostics) | 60 min | The only GWAS idea with full task coverage; decides whether the 15-task own-trait family deserves more T0 summary statistics. |
| 8 | **CI-01 / SC-1 virtual Perturb-seq screen** (forward dev) | 40 min | Only directed, label-free signal on disk; two ideators' bet; clear kill rule on the zero-support stratum. |

Next in line: SC-2 / EV-1 donor universe (cheap, shares code with 5), SG-2 PhenomeXcan (depends on a download), SC-3 / ML-8 donor similarity.

## 5. What I did not verify
- No benchmark numbers were produced; all runtimes are estimates on a machine currently at load 30.
- Download URLs, sizes and file formats for PhenomeXcan, eQTLGen, Rummagene, Genebass (no network calls made).
- Whether forward seeds are a thresholded association score, so that 25.09 sub-threshold L2G pairs are rankable (inferred from `exclude` = seeds).
- How `metrics.task_metrics` treats NaN scores (the NaN convention above must be checked or converted to the task minimum before evaluation).
- Overlap between T0 and post-T0 MAGMA traits (ML-4), dates of post-T0 studies (F11), contents of `afdb_clusters_v3` and the Swiss-Prot `.dat`.
