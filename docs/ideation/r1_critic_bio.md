# R1 critique — disease biologist / biocurator lens (2026-11-12)

Scope: all 73 ideas in `r1_{causal,statgen,singlecell,evostruct,llmtext,ml,recsys,wildcard}.md`.
Reference baseline: `transfer_jaccard` in `results/leaderboard.tsv`: nuAP 0.096 temporal dev (0.104 confirm), 0.096 forward dev (0.083 confirm).
The ideators quoted the brief's 0.122 / 0.177; their expected gains should be read against 0.096. MDE ~0.01 nuAP (brief 7).

"Δ" below = my best realistic blended nuAP gain over `transfer_jaccard` on the idea's own benchmark (dev, all tasks);
"conf" = my probability that the measured gain reaches at least the lower end of that range. These are judgements, not measurements.
"Biology" = the idea predicts disease mechanism; "pipeline" = it predicts how Open Targets (OT) builds the label. Both can be useful; I name which.

## 0. Facts I measured for this review (one-line DuckDB / pandas, read-only)

All on `modet/data/processed/groundtruth/arrival_devall_t0_string115_phys_notm_400.parquet` (4,981 arrivals, 191 diseases) against
`/mnt/msadb/modet/downloads/ot_evidence/evidence_2104`. Limits: matches are on the **same disease id only** (descendants not expanded),
gene positions are gene midpoints from one MAGMA table, name matching is crude substring containment.

- **M1. GWAS positives are L2G picks, and the own-disease weak tail is not rankable.** modet `docs/evaluation.md` l.72: GWAS positives are *defined*
  by `gwas_credible_sets` score >= 0.5 at 25.09. Of 2,863 `gwas_only` arrivals, **0** have a T0 `ot_genetics_portal` row (L2G >= 0.05) for the same
  disease id: such genes are in `unknown` and removed from the ranking (`evaluation.md` l.37).
- **M2. GWAS arrivals are new loci, not re-assignments inside known loci.** Only **3.9 %** of `gwas_only` arrivals (n = 2,808 with coordinates) lie
  within 500 kb of a gene that had a T0 L2G row for the same disease id; 3.6 % lie within 500 kb of a T0 seed. Only 47 % of the diseases with GWAS
  arrivals have any T0 L2G row on their own id.
- **M3. GWAS arrival genes are already "L2G winners" for another trait.** **86.4 %** of `gwas_only` arrival genes had L2G >= 0.5 for some trait at T0;
  universe base rate 41.5 % (7,990 / 19,255). Any T0 L2G row: 98.8 % vs 89.5 %. The T0 L2G evidence holds 388,324 rows, 10,931 studies, 2,209
  trait ids, 17,236 genes (90,668 rows >= 0.5): a dense, label-aligned gene x trait matrix 100 times larger than the 108 MAGMA studies.
- **M4. One third of GWAS positives do not persist.** 915 / 2,863 `gwas_only` arrivals are arrival class X (`persistent = False`: gone at 26.09).
  Any model of the GWAS channel is capped by this threshold jitter.
- **M5. Burden arrivals are a source addition.** 201 `burden_only` arrivals, all class B; the 21.04 evidence folder has 20 sources and no gene-burden
  source. On `temporal`, Genebass / AZ PheWAS statistics are therefore the label itself, not a predictor.
- **M6. Curation backlog: large gene pool, small exact-match yield.** For the 1,894 non-GWAS arrivals (clinvar_only, curated_only, mixed_non_gwas):
  80-84 % of the genes have some T0 ClinVar (`eva`) row (base rate 5,586 / 19,255 = 29 %), 0 % under the same disease id, and only **7.2 %**
  under a `diseaseFromSource` label that contains or is contained in the target disease name (class A 3.7 %, class B 8.7 %). That is ~136 pairs,
  2.7 % of all positives.
- **M7. Non-GWAS arrival genes were already Mendelian disease genes with phenotype profiles.** In the HPO 2021-01 `genes_to_phenotype` file
  (4,180 universe genes, base rate 21.7 %): clinvar_only 70.7 %, curated_only 70.9 %, mixed 77 %, burden_only 67 %, gwas_only 39 %.
  `curated_only`: 588 / 633 are class B (old evidence newly attached), median first curated year 2006.
- **M8. Own-pair weak non-genetic evidence at T0 is a low-coverage signal.** T0 `europepmc` row for the same (disease id, gene): 13.4 % of arrivals
  (gwas_only 14.4 %, clinvar_only 11.7 %, curated_only 8.8 %); precision among rankable candidates 0.72 % (>= 3 PMIDs: 1.04 %) against a base
  of 0.137 %. T0 `phenodigm` (mouse model) row: 1.2 % of arrivals, precision 1.2 %.
- **M9. Compara 103 homology file:** `dn` is NULL in all 4,008,755 rows; `within_species_paralog` rows are present.
- **M10. PanelApp at T0** (`genomics_england`): 18,034 green and 2,653 amber rows, no red. `gene2phenotype` carries `allelicRequirements` (2,532 / 2,550 rows).
- Not verified: which studies (FinnGen, UKB, GWAS Catalog) support the 25.09 GWAS arrivals; the 25.09 credible-set evidence rows have no `studyId`
  (only `studyLocusId`), and the credible-set table is not on disk.

Reading of these facts: the benchmark label has two nearly disjoint channels (GWAS + Mendelian co-occur for < 1 % of new genes, modet `journey.md` l.193).
(1) **GWAS channel (57 %)**: a new locus appears for the disease and L2G picks one gene in it, usually a gene L2G already picks for other traits.
(2) **Mendelian channel (~38 %)**: a gene that OMIM / ClinVar / HPO already knew as a disease gene gets attached to this disease id.
Neither is "a gene nobody has seen". Ideas aimed at "genes unknown to any disease" chase ~10 % of positives (`novel_gene` mean 0.104).

---

## 1. Per-idea critique

### Causal lens

**CI-01 Upstream-of-core (Perturb-seq)** — Fatal: no, but most rows are noise: in the K562 genome-wide screen only about 2,000 of ~9,900
perturbations gave a strong transcriptional phenotype (Replogle 2022, from memory). Biology, not pipeline; but the biology does not match the label:
57 % of positives are positional L2G picks in new loci (M1-M3), not regulators of seed expression, and Mendelian seeds are often not expressed in
K562 / RPE1. 2-h test shows whether seed-readout enrichment is disease-specific in blood / immune tasks; it does not show anything for other tissues
and is transductive on `temporal`. Δ +0.001-0.003 overall, +0.01 on immune / blood tasks (conf 25 %). Change: restrict to CD4 atlas x immune
diseases, perturbations with confirmed knock-down and a detectable signature, and use it to label module layers (regulator / effector), not to rank.

**CI-02 Trans-QTL convergence** — Fatal: close to yes for ranking. eQTLGen phase 1 tested 10,317 trait SNPs and found trans effects for about a third,
on ~6,300 genes, in blood (from memory); a disease with 20 seed loci keeps ~7 with any trans gene, so two loci converging on one non-hub gene is rare.
Biology is right (core-gene convergence) but mislabelled for this benchmark: convergent trans targets carry no cis signal, so L2G never labels them;
they become positives only through the Mendelian / burden channel. Test shows the sparsity (count of genes with >= 2 convergent loci), not the value.
Δ +0.000-0.002 (conf 15 %). Change: use as a "core layer" annotation of the output module and validate it against Mendelian / burden arrivals only.

**CI-03 Genome-wide cis-MR + coloc** — Fatal: no. Coverage: 71 of 260 panel diseases have a harmonised T0 GWAS (`t0_coverage.tsv`, statgen file).
Biology, moderately; but less label-aligned than it looks: L2G >= 0.5 is dominated by distance and coding consequence, and eQTL-colocalising genes
differ systematically from nearest genes (Mostafavi 2023, cited in the wildcard file). Top-SNP SMR without HEIDI is LD-contaminated. Test shows
SMR vs MAGMA on GWAS-only arrivals in ~15-20 dev diseases; it cannot show the value of colocalisation. Δ +0.003-0.006 (conf 40 %), of which the
SMR increment over "nearest gene to a sub-threshold lead" is ~0. Change: emulate the labeler first (sub-threshold lead -> nearest / coding gene),
use SMR only as a tie-breaker.

**CI-04 Directed donor graph** — Fatal: no. Pipeline, mostly: the 19.04 -> 21.04 asymmetry is ontology propagation (child -> parent is hard-wired)
plus OT source and scoring changes, not causal direction. Real directed biology exists (LDL -> coronary disease) but two years of flow will not
isolate it. Test shows whether theta != 0, not why. Δ +0.002-0.005 (conf 30 %). Change: test the one-bit biological rule instead of a fitted
kernel: measurement / biomarker rows (957 EFO measurement-range rows, 504 HP rows in the context, statgen file) donate to diseases, never the reverse.

**CI-05 Power-jump gating** — Fatal: no. Pipeline (statistical power). M2 supports the premise (arrivals are new loci). Weak points: 71 / 260 coverage,
and the jump 21.04 -> 25.09 came with a pipeline change (credible-set evidence replaced `ot_genetics_portal`), which pre-T0 trends cannot forecast.
Test honestly shows predicted vs realised arrival counts. Δ +0.002-0.004 (conf 30 %); arrival-count Spearman 0.3-0.4 (conf 50 %), most of it from
"has a GWAS" and `n_t0`. Change: make the arrival count the deliverable, with `n_t0` + has-GWAS as the control, and gate on the seeds' evidence
channel mix (known at T0) instead of a fitted N^alpha curve.

**CI-06 Discovery-channel detection model** — Fatal: no. Pipeline, explicitly, and correctly so. M3 and M7 give the two concrete channel priors nobody
named: "T0 L2G winner for any trait" (86 % of GWAS arrivals vs 41.5 % base) and "T0 OMIM / HPO gene" (71 % of ClinVar / curated arrivals vs 21.7 %).
Test shows channel-specific vs pooled propensity. Δ +0.004-0.008 (conf 40 %); part is already in transfer and popularity. Change: use those two
flags (and the count of traits with L2G >= 0.5) as the channel hazards, multiplied by the disease's seed channel mix.

**CI-07 Invariant predictors** — Fatal: yes as a performance idea (ideator's own failure mode: transfer is the only strong signal and is invariant
in sign). Neither biology nor pipeline; a diagnostic. Δ 0.000 (conf 80 % that |Δ| < 0.003). Change: keep only as the reporting protocol (per-channel strata).

**CI-08 rg-borrowed gene z** — Fatal: no. Biology (pleiotropy). MAGMA z smears a locus over its genes while the label is one gene per credible set;
108 T0 studies cluster in a few trait families. Banned-adjacent. Δ +0.002-0.005 (conf 30 %). Change: borrow at lead-SNP level and through the T0 L2G
matrix (M3: 10,931 studies) instead of 108 MAGMA z vectors.

**CI-09 Allelic-series concordance** — Fatal on `temporal`: yes (M5: burden statistics are the label source). On `forward` it is legitimate but weak:
Genebass is frozen since 2022, so 26.09 burden labels can only come from new cohorts, and UKB binary-disease burden is under-powered. Biology:
the most meaningful in the set (direction of effect, dose). Test shows sub-threshold burden AUROC on mapped diseases. Δ forward +0.001-0.003
(conf 25 %). Change: use burden on quantitative proxy traits (lipids, HbA1c, eGFR, blood counts), where exomes have power, mapped to diseases, and
deliver the sign as a module annotation.

### Statistical-genetics lens

**SG-1 Locus-space transfer** — Fatal: probably. M2: own-disease known loci hold 3.9 % of GWAS arrivals; M3: 86 % of arrival genes are already
L2G winners elsewhere, so gene-level donor support mostly exists. The stratum this idea opens (gene never a winner, neighbour is a donor gene) is
at most ~14 % of GWAS arrivals. Pipeline (L2G gene choice) on top of real shared loci. The ideator's 10-min diagnostic is the right first step
and I did not run its donor version. Δ +0.002-0.005 (conf 25 %). Change: no block spreading; use the T0 L2G tail of donor traits
(scores 0.05-0.5, median 0.18) as a soft donor matrix: it already contains the locus neighbours with a principled weight.

**SG-2 Phenome fingerprints (PhenomeXcan)** — Fatal: no; temporal-clean (2020). Biology with caveats: TWAS z is co-regulation and LD as much as
causality, and S-MultiXcan genes are not L2G's nearest genes. Test shows whether profiles carry signal in the unknown-gene pool after the locus
regression. Δ +0.003-0.006 (conf 30 %); unknown-pool AUC +0.02 (conf 35 %). Change: do not match profiles to seeds; map each disease to its own
and related UKB traits through EFO and read the gene's TWAS / fastENLOC value directly. That lifts own-trait sub-threshold coverage from 71 / 260
diseases towards the 4,091-trait panel.

**SG-3 mash forecast** — Fatal: no. Pipeline (power). MAGMA significance is not the L2G label (ideator says so). Δ +0.003-0.006 (conf 30 %);
mash better than raw z: conf 25 %. Change: drop mash; compute the crossing probability at lead-SNP level and merge into WC-L2GEMU.

**SG-4 HPO-decomposed quantitative-trait GWAS for rare diseases** — Fatal for the benchmark: rare-disease arrivals are ClinVar / curated
re-attachments (M7), which no GWAS predicts; 112 T0 studies cover few HPO terms. Biology is real but the Freund 2018 enrichment is a
population-level effect, weak per gene. Δ +0.001-0.003 overall, +0.005 on few-seed tasks (conf 20 %). Change: drop the GWAS leg and match the
disease's HPO profile to gene phenotype profiles directly (M7: 71 % of ClinVar / curated arrival genes had an HPO profile at T0). See section 3, B1.

**SG-5 Sub-threshold burden atlas** — Fatal on `temporal` for Genebass (M5). The T0-dated consortium tables (DDD 2020, ASC 2020, Epi25 2019) are
clean and are real biology forecasts: sub-threshold de novo enrichment preceding G2P / ClinVar labels. Test on 3-5 neurodevelopmental tasks shows
this well; it is invisible in the aggregate. Δ forward +0.002-0.004 (conf 25 %); on the 3-5 clean tasks +0.03 (conf 50 %). Change: make the
de novo mini-test the primary experiment; it is the cleanest demonstration of biology forecasting in the whole list.

**SG-6 Trans-QTL regulators of seeds** — Duplicate of CI-02 / SC-5. Additional limit: UKB-PPP measures 2,923 proteins, so most seeds have no trans
readout. Fatal: sparsity and hubs (ABO, FUT2, HLA). Δ +0.000-0.002 (conf 15 %). Change: as CI-02.

**SG-7 Ingestion-lag arbitrage** — Fatal on `temporal`: yes for the OT sub-threshold part (M1: 0 / 2,863; those genes are in `unknown`). What is
left is GWAS evidence older than T0 that OT attached later: `gwas_only` class B = 216 pairs, 4 % of positives. Pure pipeline. On `forward` it may be
real (publication-to-ingestion lag of months). Δ temporal +0.001-0.003, forward +0.004-0.008 (conf 35 %). Change: before building anything, count on
`forward` how many 26.09 GWAS arrivals rest on studies with `publicationDate` before 2025-06 (`study_2509` has the dates; the credible-set
table is needed for the join).

**SG-8 Channel-specific hazard** — Duplicate of CI-06 / WC-SELFN / IR-1. Same verdict and same change. Δ +0.004-0.008 (conf 40 %) for the cluster, not each.

**SG-9 rg donor weights** — Fatal: no; coverage is the problem (few hundred heritable UKB phenotypes, none for rare diseases, where seed overlap is
weakest). Biology. Δ +0.001-0.003 (conf 25 %). Change: one seed-free disease similarity built from phenotype profiles (rare) and rg (common);
merge with ML-8, LT-2, SC-3.

### Single-cell / perturbation lens

**SC-1 Virtual Perturb-seq screen** — Duplicate of CI-01, with cell-type routing added (good). Same verdict. Δ +0.001-0.003 (conf 25 %).

**SC-2 Experiments as donor rows** — Fatal: likely. A 30-gene seed set and a knock-down responder set share 0-1 genes; the matching rows are the
generic ones (ideator's own failure mode). BioGRID ORCS is cancer-line fitness and drug resistance, relevant to few germline disease modules.
Biology of cell-autonomous processes only. Δ +0.001-0.004 (conf 20 %). Change: choose experimental donor rows by phenotype text / cell type
matched to the disease, not by Jaccard, and keep only screens with a disease-relevant read-out.

**SC-3 Cell-address donor selection** — Fatal: no, but the leaderboard already speaks against it: `transfer_onto` (donors by ontology position,
i.e. roughly "same organ system") reaches nuAP 0.037 against 0.096 for seed overlap on temporal dev. Same-tissue diseases are poor donors.
Biology (cell type of action) is real; a 154-cluster address from few seeds resolves only tissue class. Δ +0.001-0.003 (conf 20 %). Change: infer the
cell type from genetics (section 3, B2) and use it to annotate and filter the module, not to pick donors.

**SC-4 Signed regulator -> program -> trait** — Fatal for the leaderboard: works where a trait-matched cell type and a powered quantitative trait
exist (blood traits in K562, as in Ota 2025); disease tasks are neither. Best biology in the set. Δ ~0 (conf 80 % below 0.002). Change: run it as a
structure showcase for 5-10 haematological / immune diseases through proxy blood traits.

**SC-5 Trans-QTLs as natural perturbations** — Duplicate of CI-02 / SG-6. Δ +0.000-0.002 (conf 15 %).

**SC-6 Knock-downs that phenocopy the disease state** — Fatal: coverage (10-40 diseases) and case-control expression is consequence, treatment and
sampling. No label route (expression is not a label source here). Δ +0.000-0.001 (conf 10 %). Change: turn it round and use it as validation:
is a predicted module coherently altered in the disease-relevant cell type?

**SC-7 Set completion pretrained on experimental sets** — Fatal: likely (modet: GNN = MLP; EASE banned). Δ +0.000-0.003 (conf 15 %).
Change: run only the SVD baseline, if at all.

**SC-8 Within-cell-type state programs** — Fatal: no. Biology plausible (GWAS genes in activation / response programs) but de novo NMF on sparse
counts returns ribosomal, mitochondrial and stress programs, and covers ~25 % of genes. Δ +0.001-0.003 (conf 20 %). Change: pick the programs by
the disease's T0 GWAS enrichment (heritability-style), not by seed loading.

**SC-9 Module cutoff and layers from perturbational coherence** — Fatal for most diseases (module not expressed in the screened cell line; a third
of ranked genes have no perturbation row). One of the few ideas that addresses the module output. nuAP Δ 0; cutoff Spearman >= 0.3: conf 20 %.
Change: measure coherence with annotation (subex GO / Reactome Leiden) and phenotype similarity of members (HPO / mouse MP); use perturbation
coherence only where the cell type matches.

### Evolution / structure lens

**EV-1 Phenolog donor rows** — Fatal: no. The ideator's worry is wrong in a helpful direction: OT ingests IMPC through `phenodigm`, but animal-model
evidence does not create labels in this benchmark, so mouse rows are independent input. M8: the own-pair phenodigm row is rare (1.2 % of arrivals)
but 9 times enriched. Biology (Mendelian, developmental); nothing for the GWAS channel. Test shows the gain on ClinVar / curated arrivals.
Δ +0.003-0.006 (conf 35 %). Change: add literature-curated MGI knockout phenotypes (IMPC 13.0 covers ~7k genes) and select rows by MP-to-HPO match
to the disease's phenotype profile as well as by seed overlap.

**EV-2 Paralog / fold lift** — Fatal: no; paralog rows exist (M9). Risk the ideator missed: tandem paralog clusters (HLA, keratins, protocadherins,
olfactory receptors, UGTs, histones) turn the kernel into the banned locus prior, and in the GWAS channel the lift goes to the LD neighbour that
L2G did not pick. Biology is right for Mendelian families (channel subunits, collagens). Δ +0.003-0.006 (conf 35 %). Change: exclude paralogs
within 1 Mb and apply the lift only to donor pairs with ClinVar / curated evidence.

**EV-3 Co-evolution kernel** — Fatal: no. `phylo_npp` already exists as a 256-d view in `docs/reports/views_manifest.tsv`, so the information is
half-explored. Recovers ancient machinery (cilia, mitochondrial complexes, peroxisome); gene age correlates with Mendelian status, a confound.
Δ +0.001-0.003 (conf 20 %). Change: claim it only for ciliopathies / mitochondrial / peroxisomal diseases and report per therapeutic area.

**EV-4 Selection-aware hazard** — Fatal: no. Biology and pipeline mixed: constraint raises P(causal); L2G itself does not use constraint.
Coverage 71 / 260. Δ +0.002-0.004 for the z part, +0.001 for the interaction (conf 20 %). Change: fold constraint into the host-gene model of WC-L2GEMU.

**EV-5 Livestock / dog loci** — Fatal: yes. Intervals are megabases, author-nominated candidates are circular (ideator's own point), and genes with
known causal variants in animals are almost all orthologs of known human Mendelian genes. Δ +0.000-0.001 (conf 5 %). Change: keep only OMIA
causal-variant genes as extra phenolog rows inside EV-1.

**EV-6 Convergent-evolution trait modules** — Fatal tonight: yes. The on-disk fallback does not exist: `dn` is NULL in all 4,008,755 rows of the
Compara 103 file (M9). The real version needs TOGA / ERC2.0 and helps a handful of sensory and skin diseases. Δ +0.000-0.001 (conf 5 %). Change: drop.

**EV-7 Interface-resolved edges** — Fatal: coverage (ideator concedes below MDE). Biology is right (pathogenic missense variants concentrate on
interfaces). Δ +0.000-0.002 (conf 10 %). Change: use as structure evidence for the output module (an edge backed by an interface with pathogenic
variants is a mechanistic sub-module link), not for ranking.

**EV-8 Domain-level transfer** — Fatal: no. GWAS-defined seeds have no variant positions; promiscuous domains dominate; overlaps EV-2.
Mendelian channel only. Δ +0.002-0.004 (conf 25 %). Change: family level (not domain), restricted to diseases whose seeds are ClinVar / curated,
with T0 ClinVar missense position enrichment.

**EV-9 Mechanism signature in selection space** — Fatal: nearly (twelve constraint features are close to one-dimensional; the headline has them).
The biological point is right: haploinsufficient developmental genes and recessive enzyme genes are different classes. Δ +0.000-0.002 (conf 15 %).
Change: do not infer the mechanism from 11 seeds; read it from T0 curation fields (`allelicRequirements` in gene2phenotype, M10; inheritance in
PanelApp) and condition on that.

### Language / literature lens

**LT-1 Masked-context cloze** — Fatal: no. Not biology: it predicts what has been written about. M8: even the direct own-pair co-mention covers
13 % of arrivals at <= 1 % precision, and contexts from other diseases will mostly re-learn transfer. Δ +0.003-0.007 (conf 30 %). Change: use the
sentence to type the relation (genetic assertion "variants in X cause" vs expression / biomarker) and build a matrix of text-asserted genetic
links that are not yet in ClinVar / curated sources; that is the literature form of the curation debt.

**LT-2 Semantic transfer over virtual donors** — Fatal: no. Two different things are bundled. HPO rows: M7 shows these carry OMIM / Orphanet knowledge
that OT had not attached to the disease id; that is pipeline (curation debt) and also legitimate T0 knowledge. PubTator MeSH co-occurrence:
popularity (dense STRING with text mining hurt). Δ +0.006-0.012 (conf 45 %), nearly all from HPO rows on non-GWAS arrivals. Change: split it;
run the HPO / OMIM / Orphanet arm alone with donors matched by phenotype-profile similarity, keep PubTator as a control.

**LT-3 Curation debt** — Duplicate of IR-3 / WC-BACKLOG. Fatal: no, but the yield is measured: M6. Strings that match the target name cover 7.2 % of
non-GWAS arrivals (2.7 % of positives); the candidate pool is noisy because 80 % of these genes carry several ClinVar labels. Amber PanelApp rows
exist at T0 (M10) but same-id weak genetic evidence is in `unknown`. Pure pipeline. Δ +0.004-0.008 on the full temporal endpoint (conf 45 %),
~0 on strict-a and forward; not the +0.010-0.030 the three files expect. Change: link through concept hierarchies (MedGen CUI in
`diseaseFromSourceId`, OMIM phenotypic series, Orphanet groups) rather than strings; numbered subtypes ("..., type 11") are the bulk of the debt.

**LT-4 Equal-budget dossiers** — Fatal: no; PoPS in words, transductive on `temporal`. Δ +0.001-0.004 (conf 20 %). Change: keep only as input to LT-5.

**LT-5 Anonymised LLM judge** — Fatal: no on `forward` if cutoffs hold; re-identification will be high for annotated genes. Named arm = memory;
anonymous arm = the features. Useful as a leakage instrument. Δ anonymous +0.002-0.005 (conf 25 %). Change: use the LLM on the disease side only
(no gene names): write the phenotype profile, cell type of action, proxy traits and inheritance pattern of the disease; low leakage, and it helps
exactly where seeds are few.

**LT-6 Mechanism cards** — Fatal: no. Small ranking gain; real value for module structure. Δ +0.001-0.004 (conf 25 %). Change: build cards from
subex annotation-Leiden sub-modules of seeds plus top predictions, and require each predicted gene to attach to a card with a named evidence
channel; that is the module output format.

**LT-7 Arrival hazard with literature momentum** — Fatal: no. 19.04 -> 21.04 flows are mostly pipeline; momentum favours hubs. Δ +0.003-0.006
(conf 30 %). Change: replace momentum by relation-typed literature level (see LT-1).

**LT-8 Time-locked two-tower baseline** — Fine as the reference. UniProt disease comments at 2021_02 are already OT evidence (`uniprot_literature`,
`uniprot_variants`), so stripping them loses nothing. Δ +0.002-0.005 (conf 30 %). Change: use it only to size the memorisation gap.

**LT-9 Language-selected proxy traits** — Fatal: no. Biology (biomarker -> disease: LDL, urate, eGFR, eosinophils). Banned-adjacent; coverage of
~108 T0 traits. Δ temporal +0.003-0.006 (conf 30 %), forward +0.005-0.010 (conf 35 %). Change: read proxy traits from the T0 L2G matrix
(M3: 2,209 trait ids, continuous scores) instead of MAGMA z: far more traits, and aligned with how the label is built.

### Machine-learning lens

**ML-1 Cold-start donor imputation** — Fatal: likely no gain. Content adds <= 0.007 (brief 2); and the zero-support GWAS stratum is positional,
not content-predictable (M2, M3). Δ +0.001-0.004 (conf 20 %). Change: impute only Mendelian-channel profiles, with phenotype content
(mouse MP, HPO of paralogs), where content is known to predict disease class.

**ML-2 Ascertainment-simulator PFN** — Fatal: yes (modules are not compact in any content space, brief 2; ideator names the sim-to-real gap).
Δ +0.000-0.003 (conf 10 %). Change: an explicit two-channel generative model with the measured knobs of M3 / M7; no transformer.

**ML-3 Rummagene gene-set model** — Fatal: no. Most supplement sets are differential-expression lists: tissue and popularity, which added ~0.
Δ +0.002-0.005 (conf 25 %). Change: keep only tables with genetic content (exome, GWAS, screen hit lists; table captions are available): then it
becomes ingestion-lag evidence from supplements, including sub-threshold gene lists OT never ingests.

**ML-4 Release-pair power extrapolator** — Fatal: no. Pipeline (power). The training label (later MAGMA significance) is not the benchmark label
(later L2G pick); coverage. Δ +0.004-0.008 (conf 35 %). Change: train on "gene becomes an L2G >= 0.5 pick for the trait in a later OT release"
(21.04 -> 23.06 rows of non-task traits) with lead-SNP features; merge with WC-L2GEMU.

**ML-5 Identity-blind episodic set transformer** — Fatal: probable (GNN = MLP). HPO sets are near-labels (M7), so any gain will come from them.
Δ +0.002-0.006 (conf 20 %). Change: ablate by set family first; if HPO carries it, run the LT-2 HPO arm instead.

**ML-6 Tabular in-context learner on neighbour arrivals** — Fatal: no. The arrival objective is sensible; the 19.04 -> 21.04 context teaches that
period's pipeline. Δ +0.002-0.006 (conf 25 %). Change: run only the pooled-arrival LightGBM control, with flows split by channel.

**ML-7 Masked discrete diffusion** — Fatal: yes (20-30 positives per disease cannot identify joint structure; co-inclusion will mirror the GO /
Reactome sets used in training). Δ 0 (conf 85 %). Change: drop; size from CI-05, structure from annotation clustering.

**ML-8 Disease encoder for future overlap** — Fatal: no. `transfer_onto` (0.037 vs 0.096) shows ontology / name proximity is a poor donor selector;
the HPO-profile part is the one with a biological reason for rare diseases. Δ +0.002-0.005 overall, +0.01-0.02 for <= 5 seeds (conf 35 %).
Change: phenotype-profile similarity plus seed channel mix as the disease representation.

**ML-9 Evidence-trajectory hazard** — Fatal: mostly. M8 triggers the ideator's own coverage kill for own-pair trajectories (13 % < 25 %); the rest
goes through neighbours and is popularity. Δ +0.002-0.004 (conf 25 %). Change: fold the level features into IR-7.

### Recommender lens

**IR-1 Exposure-aware relevance** — Duplicate of the CI-06 cluster. Exposure is near-constant within a disease (ideator's own point); the usable
exposure term is gene-level and per channel (M3). Δ +0.003-0.006 (conf 30 %).

**IR-2 Locus slates** — Fatal: partly measured. M2 and M3 bound the new stratum at ~14 % of GWAS arrivals (see SG-1). Pipeline (L2G choice).
Δ +0.002-0.006 (conf 25 %), not +0.010-0.025. Change: soft L2G-weighted donor matrix instead of 0/1 rows plus block spreading; the "win propensity"
factor is the M3 prior.

**IR-3 Identity linking** — Duplicate of LT-3 / WC-BACKLOG; see M6. Δ +0.004-0.008 temporal full endpoint (conf 45 %).

**IR-4 Powered nulls as negatives** — Duplicate of WC-UPPERLIM. Fatal as a pooled veto: a flat common-variant z says nothing about the Mendelian /
burden channel (~40 % of positives), and constrained genes are depleted of common-variant signal (Mostafavi 2023). Δ -0.002 to +0.003
(conf 15 % positive). Change: apply only inside a GWAS-channel score.

**IR-5 Study-as-user baskets** — Fatal: no. The ontology-duplication point is right. But the baskets are thinner than assumed: a ClinVar RCV is one
variant in one gene; a PMID basket is text mining; a GWAS study basket is the trait row again. The new information is gene panels and per-study
GWAS rows. Δ +0.002-0.006 (conf 30 %). Change: PanelApp panels (green + amber, M10) and GWAS studies with L2G soft weights as rows; skip RCVs and PMIDs.

**IR-6 Trending items** — Fatal: no; cheapest test. Pipeline: `ot_genetics_portal` with L2G became an OT source between the two releases, so
"recent" largely means "L2G-derived", and GWAS arrivals are L2G picks (M1, M3). It may work for that reason. Δ +0.003-0.008 (conf 40 %).
Change: weight donor pairs by evidence channel explicitly (GWAS donors for GWAS-heavy diseases) rather than by a release delta.

**IR-7 Multi-behaviour funnel** — Fatal: no. On disk, temporal-clean, and it keeps what the 0/1 donor matrix throws away: donors' continuous
per-source scores. Pipeline and biology both. Own-disease weak genetic channels are in `unknown`; own-pair literature and mouse rows are usable
but thin (M8). Text mining will dominate unless held out. Δ +0.006-0.012 (conf 45 %). Change: genetic channels only for donors (continuous L2G,
eva, gene2phenotype, genomics_england, uniprot), plus own-pair literature and phenodigm; channel-to-channel weights fitted on 19.04 -> 21.04 flows.

**IR-8 Cold-start tower** — Duplicate of ML-1. Δ +0.000-0.003 (conf 15 %).

**IR-9 Next-item over discovery sequences** — Fatal: yes. `europepmc.publicationYear` is 0 % filled, curated sources have no dates, sequences are
short, and "order" is the technology era (candidate gene, GWAS, exome), which a channel feature captures. Δ +0.000-0.002 (conf 10 %). Change: drop.

**IR-10 Burden PheWAS as second log** — Duplicate of SG-5 / CI-09; forward only. Δ +0.001-0.004 (conf 25 %). Change: quantitative proxy traits.

### Wildcard lens

**WC-L2GEMU Emulate the label generator** — Fatal: no. My measurements support the design and bound it. M1: only loci with no T0 credible set for
the disease are rankable, so everything depends on sub-threshold peak detection; M2: 96 % of arrivals are indeed such new loci; M4: a third of
GWAS positives are jitter; coverage 71 / 260. The host model is fitted on the 21.04 L2G model while the label uses the 25.09 one (both
distance-dominated, acceptable). Pipeline emulation on top of real genetics. Test shows GWAS-only recall on ~15-20 covered dev diseases.
Δ +0.005-0.010 all dev, +0.02 covered (conf 45 %). Change: widen coverage with pre-T0 biobank disease sumstats (FinnGen R4, 2020; BioBank Japan
2020; Neale UKB) and evaluate also on persistent positives only.

**WC-XPOWER Crossing forecast without own GWAS** — Duplicate of CI-08 / SG-3 / LT-9. Δ +0.002-0.005 (conf 25 %). Change: T0 L2G matrix as the trait panel.

**WC-BACKLOG** — Duplicate of LT-3 / IR-3; see M6. The reporting recommendation (every result split by channel, with this arm as a baseline) is correct.

**WC-SELFN Channel-mixture arrival model** — Duplicate of CI-06 / SG-8. Δ +0.004-0.008 for the cluster (conf 40 %).

**WC-HAWKES Evidence flux** — Duplicate of IR-6 / LT-7 / ML-9 (two snapshots give one difference). Δ +0.002-0.004 (conf 25 %).

**WC-CAPTURE Capture-recapture completeness** — Fatal for the estimator: sources are not independent captures (ClinVar, G2P, PanelApp, UniProt and
Orphanet copy each other; ontology propagation duplicates rows), so it measures curation redundancy. The question (module size, completeness) is
the right one. nuAP Δ +0.000-0.002 (conf 15 %); Spearman gain over `n_t0`: conf 25 %. Change: capture occasions = independent cohorts within one
channel (loci vs sample size per trait; ClinVar submitters), merged with CI-05.

**WC-DCA Inverse Ising** — Fatal: yes. ~3k rows for ~10k spins, and the rows are ontology-propagated copies, so couplings encode the ontology tree.
EASE is banned for the same reason. Δ +0.000-0.003 (conf 10 %). Change: deduplicate to independent evidence units first, or drop.

**WC-UPPERLIM Measured nulls** — Duplicate of IR-4; same verdict. The 30-min diagnostic (positive rate of well-powered nulls in the transfer top 200)
is worth running once, per channel.

**WC-MARKET Funders and filers** — Fatal: yes. Grant-abstract co-mention is text mining with a lag in the wrong direction (M8: text co-mention is
low-coverage, <= 1 % precision). Δ +0.000-0.001 (conf 5 %). Change: use drug development records instead (ChEMBL target-indication at T0,
`/mnt/msadb/modet/downloads/ot_chembl_2104`): section 3, B3.

---

## 2. (a) Duplicate clusters

| Cluster | Members | One experiment that settles it |
|---|---|---|
| D1 Sub-threshold own-trait GWAS -> later L2G pick | CI-03, CI-05, SG-3, EV-4, ML-4, WC-L2GEMU (SC-4 in part) | WC-L2GEMU with ML-4's training target |
| D2 Cross-trait borrowing | CI-08, SG-2, SG-3, SG-9, LT-9, WC-XPOWER | proxy traits read from the T0 L2G matrix (M3) |
| D3 Locus as unit | SG-1, IR-2 | donor-locus diagnostic, then soft L2G donor matrix |
| D4 Trans-QTL edges | CI-02, SG-6, SC-5 | sparsity count on eQTLGen significant file |
| D5 Perturb-seq direction | CI-01, SC-1, SC-4 (SC-2, SC-9 reuse the matrices) | SC-1 on immune tasks with CD4 atlas |
| D6 Detection / channel / exposure factor | CI-06, CI-07, SG-8, IR-1, WC-SELFN, ML-2 | two channel priors from M3 / M7 vs pooled prior |
| D7 Curation debt / identity linking | LT-3, IR-3, WC-BACKLOG (SG-7 is the GWAS form) | concept-hierarchy linking, reported per arrival class |
| D8 Flows and momentum | CI-04, LT-7, ML-6, ML-9, IR-6, IR-9, WC-HAWKES | IR-6 (10 min), then channel-typed version |
| D9 Content imputation for cold genes | ML-1, ML-3, ML-5, IR-8, LT-4, EV-9, SC-7 | ML-1 encoder AUROC on held-out genes |
| D10 Non-disease donor rows / gene kernels | EV-1, EV-2, EV-3, EV-5, EV-6, EV-8, SC-2, LT-2, IR-5 | one `_transfer` on stacked rows, ablated by family |
| D11 Seed-free disease similarity | SG-9, SC-3, ML-8, SG-4, donor selection of LT-2 | phenotype-profile similarity vs seed overlap, by seed count |
| D12 Burden evidence | CI-09, SG-5, IR-10, SC-4 | SG-5 de novo mini-test + quantitative proxies on forward |
| D13 Measured nulls | IR-4, WC-UPPERLIM | WC-UPPERLIM diagnostic |
| D14 Module size / structure | CI-05, SC-9, ML-7, LT-6, WC-CAPTURE, EV-7 | see section 5 |

73 ideas reduce to about 14 experiments. D1, D2, D6 and D8 together hold 25 ideas and are four views of one statement: the label is a detection event.

## 3. (b) Important directions nobody proposed

- **B1. Phenotype matching for the Mendelian channel.** Score a gene by semantic similarity between the disease's phenotype profile
  (HPO annotations of the disease, T0 `phenotype.hpoa` / Orphanet) and the gene's phenotype profile (human HPO through OMIM / Orphanet at T0,
  mouse MP through MGI / IMPC), the principle behind Exomiser / Phenodigm. Needs no seeds. M7 says 71 % of ClinVar / curated arrival genes had a
  profile at T0. SG-4 and LT-2 come near it; neither does it. Related and also missing: OMIM and Orphanet gene-disease tables at T0 as explicit
  upstream sources that OT attaches late (OMIM reaches OT only through ClinVar, UniProt, G2P).
- **B2. Cell type of action inferred from genetics, then variant-to-gene in that cell type.** Use the disease's T0 summary statistics or seed loci
  to pick cell types by heritability enrichment, then score genes at sub-threshold loci by enhancer-gene links and specific expression in those
  cell types. SC-3 and SC-8 pick cell types from seed expression, which resolves only tissue; nobody used enhancer-gene maps, which are also
  features of L2G.
- **B3. Drug target data.** `ot_chembl_2104` and `drug_phase_ot_2104_2509.parquet` are on disk and unused by all eight files. Drug targets do not
  create labels here (modet: T0 genetic support of new targets ~3 %), so the nuAP gain is small; but targets of drugs for the disease and its
  relatives, and their target-family siblings, are module members by mechanism, and "does the predicted module contain later drug targets"
  is an independent validation endpoint.
- **B4. Pathway- and complex-level reasoning in the Mendelian channel.** Rank complexes / pathways first, then complete them: if several
  subunits of a complex are Mendelian genes for a phenotype, the remaining subunits are candidates (BBSome, complex I assembly, peroxins,
  Fanconi core). GO "function users" were tested and banned on the pooled endpoint; complex completion evaluated on ClinVar / curated arrivals
  only was not. CORUM and Complex Portal are on disk.
- **B5. The module as an object.** No file defines what a module is beyond "top k". Proposal in section 5.
- **B6. Persistence as the target.** M4: a third of GWAS positives vanish within a year. Train and evaluate on persistent positives; a model that
  predicts jitter is not predicting biology.
- **B7. Independent-ancestry biobanks at T0** (BioBank Japan 2020 disease GWAS) as a second look at sub-threshold loci: different LD and allele
  frequencies, pre-T0, and closer to "replication" than any cross-trait borrowing.
- **B8. Inheritance mode and direction of effect as disease-level conditioning** (`allelicRequirements` in G2P, M10): only EV-9 and CI-09 touch it,
  indirectly.

## 4. (c) Top 8 to test first, with kill rules

1. **GWAS-channel forecast, label-aligned** (WC-L2GEMU + ML-4 target + CI-05 count). Sub-threshold leads -> nearest / coding gene, crossing
   probability, coverage widened with pre-T0 biobank sumstats. Kill: GWAS-only recall@100 gain < 4 points on covered dev diseases, or < 15 dev
   diseases covered. Report also on persistent positives.
2. **Soft, channel-typed transfer** (IR-7, absorbing IR-6, IR-2, LT-9, CI-08). Donor matrix = continuous per-source scores, with the T0 L2G tail
   (10,931 studies). Kill: < +0.006 nuAP over `transfer_jaccard` on temporal dev, or the gain vanishes when the text-mining channel is removed
   from the donors.
3. **Phenotype matching + OMIM / HPO / Orphanet rows** (B1, LT-2 HPO arm, SG-4 and ML-8 as corrected). Kill: recall@100 on ClinVar / curated
   arrivals rises < 5 points, or blended gain < +0.005 nuAP.
4. **Two channel priors** (CI-06 / SG-8 / WC-SELFN reduced to M3 and M7 flags x seed channel mix). 20 minutes. Kill: < +0.003 nuAP over
   transfer x pooled popularity prior.
5. **Curation-debt linking through concept hierarchies** (LT-3 / IR-3 / WC-BACKLOG). Run as a 45-min diagnostic and keep as a reporting stratum.
   Kill: < 10 % of class-B non-GWAS arrivals in the top 100 by the debt score alone (string matching reaches 8.7 %, M6).
6. **Mendelian-channel lift** (EV-1 phenolog rows + EV-2 paralog lift without same-locus paralogs, Mendelian donor pairs only). Kill: paired gain
   < +0.004 nuAP and zero-support recall@100 < 1.5 %.
7. **Burden as biology forecast** (SG-5 de novo mini-test on `temporal`; quantitative-proxy burden on `forward`, CI-09 sign). Kill: sub-threshold
   AUROC < 0.60 for later positives on the neurodevelopmental tasks, and forward gain < +0.003 after removing near-significant own-trait pairs.
8. **Module object** (section 5: channel-layered module with LT-6 cards, size from CI-05, validated across channels and with B3 drug targets).
   Kill: the predicted module is not more enriched for held-out-channel positives and post-T0 drug targets than the transfer top-k of equal size.

Order by cost: 4, 2, 5, 3, 6 are on-disk and CPU-only; 1 needs sumstat handling; 7 and 8 need small downloads or assembly.
Not in the top 8 on purpose: all Perturb-seq and trans-QTL ranking ideas (biology the label does not reward; keep for module structure),
all content-imputation models (D9), ML-7, WC-DCA, IR-9, EV-5, EV-6, WC-MARKET.

## 5. (d) Which ideas give a biologically meaningful module, not only a better list

A ranked list scored by nuAP can be improved a long way by emulating OT (items 1, 4, 5 above). That is honest forecasting of the database, and
it should be reported as such, but it does not produce a module. Ideas by what they contribute to a module:

- **Layers with a mechanistic meaning:** CI-01 / SC-1 / SC-4 (regulators -> program -> effectors; needs a matched cell type), CI-02 / SC-5
  (core genes as convergence points of trans effects), CI-09 (direction of effect and dose). These rank poorly on this benchmark and are the
  most valuable for the module; run them on a few diseases with matched cell types as structure, not on the leaderboard.
- **Sub-modules with names:** LT-6 mechanism cards on top of the subex annotation-Leiden decomposition; B4 complex completion; EV-7 interfaces
  as edge evidence.
- **Size and completeness:** CI-05 (expected arrivals per channel), WC-CAPTURE (if rebuilt on independent cohorts), SC-9 (coherence cutoff, with
  phenotype / annotation coherence instead of K562 signatures).
- **Evidence-channel honesty:** CI-06 / WC-SELFN, because a module should say why each gene is in it.
- **Contribute nothing to a module:** WC-BACKLOG / LT-3 / IR-3, SG-7, IR-6, WC-HAWKES, LT-7, ML-6, ML-9, IR-9 (pipeline timing); ML-7 and
  WC-DCA (structure that would mirror the training sets or the ontology).

Proposed definition to build towards (my suggestion, to be decided by the user): a module is a set of genes each carrying
(i) an evidence channel (Mendelian / coding core, GWAS-proximal, regulator, effector / drug target), (ii) a sub-module (pathway, complex or
mechanism card), (iii) a cell type of action where genetics supports one, (iv) a direction where burden or drug data give one, and a size per
channel. Validation that does not depend on OT's next release: cross-channel prediction (a module built from GWAS seeds should be enriched for
later Mendelian and burden genes and the reverse), later drug targets, phenotype coherence of members' knockout phenotypes, and stability
(persistent positives only, M4).
