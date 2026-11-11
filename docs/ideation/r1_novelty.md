# Round 1 novelty check (retrieval-grounded), 2026-11-12

Scope: the 73 ideas in `r1_{causal,evostruct,llmtext,ml,recsys,singlecell,statgen,wildcard}.md` (id, title, mechanism only).
Method: 40 web searches (general web index, which covers PubMed/PMC, bioRxiv/medRxiv and arXiv), grouped by idea cluster. All sources accessed 2026-11-12.
No personal or project names were used in queries. Search snippets were not checked against full texts, so treat reported numbers as indicative.
Classes: **PRIOR-ART** = the core mechanism has been published (closest paper cited). **PARTIAL** = close work exists but differs in a stated way.
**NO-PRIOR-FOUND** = nothing close came up (the queries are given in the notes).
"Neg" marks prior art that reported a negative or limiting result. "(mem)" marks a reference cited from memory and not fetched again in this check.

Caveat for all PARTIAL / NO-PRIOR-FOUND rows: no published work was found that evaluates the idea **as a disease-module ranker on a temporal arrival benchmark**.
For almost every idea that evaluation is the main new element. Novelty of the mechanism itself is what the table rates.

## Table

| id | short title | class | closest work (year): what it showed | difference / note |
|---|---|---|---|---|
| CI-01 | Perturb-seq upstream-of-core | PARTIAL | Ota et al. Nature 2025 [S1]: regulator→program→trait graph from K562 Perturb-seq + LoF burden. Schnitzler et al. Nature 2024 [S2]: CAD loci converge on 5 EC programs, 41 genes nominated | Prior work fits one trait or locus set at a time. Here the seed set is the readout and all 19k genes are ranked genome-wide, scored on held-out arrivals |
| CI-02 | trans-QTL convergence | PARTIAL | Võsa et al. (eQTLGen) Nat Genet 2021 [S3]: trans-eQTLs found for 37 % of 10,317 trait SNPs, unlinked loci converge on core trans-genes. Downstreamer [S4]: key genes mostly outside loci, enriched for Mendelian genes | The convergence phenomenon is established and Downstreamer already ranks genes. New element: seed-loci→trans-target score benchmarked on temporal arrivals. Neg: trans effects replicate poorly and are confounded by cell composition [S3] |
| CI-03 | genome-wide cis-MR + coloc | PRIOR-ART | SMR/HEIDI (Zhu 2016, mem), TWAS, coloc. Open Targets L2G already uses colocalisation [S27] | Circularity risk: the label generator (L2G) already contains coloc, so a gain may only reproduce the label pipeline |
| CI-04 | directed donor graph | NO-PRIOR-FOUND | nearest: contrastive disease similarity (PhenoGnet 2025, CoGO) [S30] | Queries: "learned disease embedding contrastive predict future shared genes…", "temporal prediction future gene-disease associations…". No directed disease→disease gene-flow model found |
| CI-05 | power-jump hazard per disease | PARTIAL | Holland et al. Front Genet 2016 [S5]: Gaussian-mixture model of z gives replication probability and the N needed to explain h2 (SNP level) | No gene-level, per-disease gating between transfer and own GWAS found |
| CI-06 | detectability per channel | PARTIAL | Spence et al. Nature 2025 [S6]: burden ranks by gene-level trait specificity and length, GWAS by variant-level specificity; "luck" matters. Mostafavi et al. Nat Genet 2023 [S7]: selection gives GWAS/eQTL discovery differences | The phenomenon is published. Turning it into an explicit detection factor in a ranker was not found |
| CI-07 | invariant predictors across eras/channels | PARTIAL | IRM across organisms for cancer genes (O'Donoghue 2021, arXiv 2111.07348) and FairPRS (PSB 2023) [S8] | IRM over discovery eras × channels for disease-gene ranking was not found. Neg: the cross-organism IRM study was inconclusive ("further work required") |
| CI-08 | rg-borrowed gene z | PARTIAL | MTAG (Turley 2018, mem). MTAR, gene-level multi-trait rare variant test (Nat Commun 2020) [S9]: finds more lipid genes than single-trait tests | A known statistical tool. New element: used as a prospective ranker for diseases without seeds or with a small GWAS |
| CI-09 | allelic-series concordance | PRIOR-ART | Genetic priority score GPS (Duffy 2024; ML-GPS 2025) uses allelic series and direction of effect. COAST allelic-series test (AJHG 2023). Rare-to-common allelic series assigns effectors at >100 loci (Nat Genet 2025) [S10] | GPS is already on disk. Sign concordance at sub-threshold p as a ranker is only a small twist |
| EV-1 | phenolog donors | PRIOR-ART | McGary et al. PNAS 2010 [S11]: 4,390 phenologs; cross-validation predicts genes for 1/3–1/2 of ~300 diseases | Neg: a 2021 preprint [S11b] found that only mouse adds signal, more distant organisms do not. The mouse KO oracle was already proposed in modetor (banned list) |
| EV-2 | paralog / fold lift | PARTIAL | Lal et al. Genome Med 2020 [S12]: ~80 % of disease genes are in families; family de novo enrichment gives 28 new NDD candidates. Paralogue Annotation [S12b] works at variant level (precision 0.95) | Lifting the whole disease profile (L·K) inside transfer was not found |
| EV-3 | co-evolution kernel (ERC/NPP) | PRIOR-ART | Priedigkeit, Wolfe, Clark PLoS Genet 2015 [S13]: 55 % of 310 disease gene groups have an ERC signature; true gene ranked in the top 6 % on average. ERC2.0 Genome Res 2025. Phylogenetic profiling (Tabach 2013, mem) | Only the use as a degree-unbiased network on top of transfer is new. modet showed that networks add ≈0 beyond transfer |
| EV-4 | selection-aware GWAS hazard | PARTIAL | Mostafavi 2023 [S7]: constrained genes are discovered differently. Spence 2025 [S6]. Holland 2016 [S5] | A two-parameter constraint × sub-threshold z crossing hazard was not found |
| EV-5 | livestock/dog loci as donors | PARTIAL | Bouwman et al. Nat Genet 2018: cattle stature loci overlap human/dog loci. Raymond et al. PLoS Genet 2020 [S14]: 30/77 cattle stature genes are human height genes (OR 5.1) | Neg: human priors helped cattle only when no cattle prior existed, and their top SNPs explained no more variance than random genes. Only the reverse direction (animal→human) was tested |
| EV-6 | convergent-evolution trait modules | PARTIAL | RERconverge (Kowalczyk 2019), hairlessness study [S15] | Not used as donors for human disease. Neg: RERconverge can give significant p for null phenotypes [S15] |
| EV-7 | interface-resolved edges | PARTIAL | Edgetic disease mutations: 15–19 % of ClinVar disease vs 5–7 % of non-disease mutations (Nat Commun 2019). MutPred-PPI (bioRxiv 2025) on AF3 contacts [S16] | Prior work is variant-level. Edge weighting for module propagation was not found. Network propagation is banned unless the reason is new; the interface constraint is that new reason |
| EV-8 | domain-level homologous-site donors | PARTIAL | MetaDome (Hum Mutat 2019), Homologous Missense Constraint (Genome Med 2024) [S17]: domain-position pooling enriches pathogenic and NDD de novo variants | Variant interpretation only. Disease × domain-family transfer to rank genes was not found |
| EV-9 | meta-learned "selection-space" scorer | PARTIAL | Constraint features are standard in gene rankers (e.g. Mantis-ML [S25]) | No hypernetwork / deep-sets set→scorer on evolutionary features found (no dedicated query, low confidence) |
| LT-1 | masked-context cloze | PARTIAL | Temporally enabled word embeddings (nference, bioRxiv) [S18]: 13.9 % of associations predicted before first co-occurrence, median lead 4 y. DeepProphet2 (arXiv 2022) [S19] | Masking gene and disease spans and reading contexts about *other* diseases was not found |
| LT-2 | semantic virtual donors (MeSH/HPO/GO) | PARTIAL | SmuDGE (2018), PhenoGnet (2025) [S30]: phenotype/ontology-based disease–gene embeddings | Text-selected concept rows inside the transfer matrix were not found |
| LT-3 | curation debt | NO-PRIOR-FOUND | — | Queries: entity-normalisation / UKB-mapping searches [S20]. It predicts the label pipeline, not biology: may win the benchmark without telling us anything about disease. Neg-adjacent: DisGeNET reports LLMs "perform poorly in entity normalization" [S20] |
| LT-4 | equal-budget data-to-text dossiers | PARTIAL | GenePT (2023/2024) [S21]: embedded NCBI summaries match or beat Geneformer on gene tasks. Newer embedders +1–17 % (ICLR 2026) | Dossiers built from non-literature data with a fixed token budget against study bias were not found |
| LT-5 | anonymised LLM judge | PARTIAL | LLM gene-prioritisation survey (Bioinf Adv 2025) [S22]: bias toward well-studied genes, order sensitivity. NeurIPS 2026 named-vs-hidden drug-target study [S22b]: evidence changed choices no more often than shuffling | Neg: both report identity/position effects dominating evidence. A named-minus-anonymous memorisation gap as a control was not found for disease genes |
| LT-6 | mechanism cards | NO-PRIOR-FOUND | — | Covered only by the LLM/GenePT queries (low confidence) |
| LT-7 | arrival hazard + literature momentum | PARTIAL | nference temporal embeddings [S18]. Rosalind time-sliced KG predictions (Sci Rep 2020) [S23]. Stoeger 2018: attention predictable from gene properties [S24] | No hazard trained on one release transition and tested on the next was found |
| LT-8 | time-locked two-tower text baseline | PRIOR-ART | GenePT and successors [S21] | A known baseline, intended as one. The 2020-vs-modern encoder leakage contrast is a useful evaluation addition |
| LT-9 | LLM-selected proxy traits | PARTIAL | GWAS-by-proxy (Liu 2017, mem). GENPHIRE uses LLM trait embeddings (medRxiv 2025) [S20b] | Neg: DisGeNET found LLM trait/entity mapping not feasible for UKB phecodes [S20] |
| ML-1 | cold-start donor imputation (DropoutNet) | PARTIAL | DropoutNet (Volkovs 2017, mem). Disease-gene GNN with privileged info + heteroscedastic dropout (Bioinformatics 2021) [S26] | DropoutNet-style imputation of whole association profiles for empty genes was not found |
| ML-2 | ascertainment-simulator PFN | NO-PRIOR-FOUND | nearest: MapPFN (ICLR 2026) [S28], a PFN pretrained on simulated GRN knockouts | Query: "prior-fitted network … simulated … disease gene sets". No simulator of how disease gene sets get observed was found |
| ML-3 | gene-set LM on supplement tables | PRIOR-ART | Gene Set Foundation Model (bioRxiv 2025) [S29]: pretrained on >1 M Rummagene+RummaGEO sets with held-out-gene prediction; best gene-function prediction of the methods it benchmarked | Only the time-locked (pre-seed-date) corpus is new. Weights are public, so a cheap test is possible |
| ML-4 | release-pair power extrapolator | PARTIAL | Holland 2016 [S5] (SNP-level, parametric) | A model learned on paired biobank releases for gene-level crossing was not found |
| ML-5 | identity-blind episodic set completion | PARTIAL | GSFM 2025 [S29]. DeepProphet2 [S19] | Neither is identity-blind or meta-trained across set families. EASE completion is banned; the new reason is identity-blindness |
| ML-6 | tabular ICL with neighbour arrivals | NO-PRIOR-FOUND | — | Query: "TabPFN gene prioritization disease genes". Neg: TabPFN loses its edge beyond ~500 features (bioRxiv 2025). "Where tabular FMs falter on genetic data" (NeurIPS 2026) [S31] |
| ML-7 | masked discrete diffusion over sets | NO-PRIOR-FOUND | nearest: DigNet discrete diffusion for GRNs (Genome Res 2025) [S32] | Query: "discrete diffusion model generate gene sets…" |
| ML-8 | disease encoder trained on future overlap | PARTIAL | PhenoGnet / CoGO contrastive disease similarity [S30] | Training on *future* gene-overlap supervision was not found |
| ML-9 | evidence-trajectory hazard | PARTIAL | same as LT-7 [S18, S23] | Same cluster as LT-7 and IR-6. Merge them |
| IR-1 | exposure-aware relevance | PARTIAL | ExpoMF (Liang 2016, mem) in recommender systems. Stoeger 2018 [S24] | No measured-exposure factorisation for gene–disease was found (query on exposure MF / MNAR gene–disease) |
| IR-2 | locus slates + click model | PARTIAL | HyPrColoc, SigNet cross-locus Bayesian gene choice (2025), OT L2G [S27] | Transfer in locus space across traits was not found. Same cluster as SG-1 |
| IR-3 | identity linking of T0 evidence | NO-PRIOR-FOUND | — | Same cluster as LT-3, WC-BACKLOG and SG-7. Label-artefact exploitation (see LT-3) |
| IR-4 | powered nulls as negatives | NO-PRIOR-FOUND | nearest: power-based exclusion of effect sizes in negative pharmacogenetic studies [S33]. Connally eLife 2022 on informative negative colocalisation | Query: "well-powered null GWAS association as negative evidence…". Same cluster as WC-UPPERLIM |
| IR-5 | study-as-basket item-kNN | PARTIAL | DeepProphet2 (literature baskets) [S19]. Rummagene sets [S29] | Item-kNN over GWAS study / ClinVar / panel baskets was not found |
| IR-6 | trending transfer on release delta | PARTIAL | temporal link prediction with t−1,t,t+1 matrices (review arXiv 2312.01275) [S23] | Same cluster as WC-HAWKES |
| IR-7 | multi-behaviour channel funnel | PARTIAL | target prediction from Open Targets per-datatype scores (Ferrero 2017, mem). Mantis-ML [S25] | The multi-behaviour formulation with sub-threshold channels is new (not searched specifically) |
| IR-8 | cold-start item tower | PARTIAL | as ML-1 [S26] | Duplicate of ML-1 |
| IR-9 | next-item model over discovery sequences | NO-PRIOR-FOUND | nearest: DeepProphet2 (next-gene objective, but on literature, not chronological) [S19] | Query: "sequential recommendation transformer order of gene discovery…" |
| IR-10 | burden PheWAS as second domain | PARTIAL | Genebass (Cell Genomics 2022) [S25]. Mantis-ML 2.0 uses PheWAS for triage [S25] | Collective factorisation with the label matrix was not found |
| SC-1 | virtual Perturb-seq screen, seeds as readout | PARTIAL | as CI-01 [S1, S2]. Strobl & Gamazon eLife 2025 root causal genes [S34] | Same cluster as CI-01. Neg: the eLife review notes that unexpressed upstream TFs are invisible in Perturb-seq |
| SC-2 | experiments as donor rows | PARTIAL | BioGRID ORCS screens. GSFM uses experimental sets [S29, S35] | Screen and Perturb-seq sets used as transfer donors were not found |
| SC-3 | cell-address donor selection | PARTIAL | PrEDiCT (eLife 2024) [S36]: affected cell types for 328 Mendelian diseases, similar cell types across multi-tissue diseases. scDRS 2022 | Not used to pick transfer donors |
| SC-4 | signed regulator→program→trait | PRIOR-ART | Ota et al. Nature 2025 [S1] | The idea follows Ota explicitly. The new part is only the disease-generic application and benchmarking |
| SC-5 | trans-QTL natural perturbation atlas | PARTIAL | as CI-02 [S3, S4] | Same cluster as CI-02 and SG-6 |
| SC-6 | knockdowns that phenocopy disease state | PARTIAL | Strobl & Gamazon 2025 [S34] (Perturb-seq ordering applied to patient RNA). CMap concept (mem) | Matching a CELLxGENE case/control signature to genome-wide Perturb-seq was not found |
| SC-7 | set completion pretrained on experimental sets | PARTIAL | GSFM 2025 [S29] | Close to ML-3 and ML-5. The label-free experimental corpus is the stated difference |
| SC-8 | within-cell-type state programs | PRIOR-ART | sc-linker, Jagadeesh et al. Nat Genet 2022 [S37]: cellular-process programs within a cell type carry distinct heritability. DisCP-Atlas: 759 diseases | Not used to rank genes for modules, which is the only difference |
| SC-9 | perturbational coherence cutoff | NO-PRIOR-FOUND | — | No dedicated query (budget). Perturb-seq / capture queries found nothing on cutoffs. Low confidence |
| SG-1 | locus-space transfer + within-locus choice | PARTIAL | as IR-2 [S27] | Duplicate of IR-2 |
| SG-2 | dense phenome fingerprints | PARTIAL | DeGAs, Tanigawa Nat Commun 2019 [S38]: tSVD of 2,138-phenotype UKB z matrix, found PDE3B/GPR151 | Gene–gene profile similarity to seeds for disease ranking was not found |
| SG-3 | multivariate shrinkage forecast | PARTIAL | mash (Urbut 2019, mem). Holland 2016 [S5] | Same cluster as ML-4 and WC-XPOWER |
| SG-4 | HPO-decomposed quantitative-trait GWAS | PARTIAL | Freund et al. AJHG 2018 [S39]: Mendelian genes enriched near GWAS loci of phenotype-matched traits (~27×) | Strong support for the premise. As a per-rare-disease ranker it was not found |
| SG-5 | sub-threshold burden atlas | PARTIAL | Genebass [S25]. Spence 2025 [S6] | Same cluster as IR-10 |
| SG-6 | trans-QTL regulators of seeds | PARTIAL | as CI-02 [S3, S4] | Same cluster as CI-02 and SC-5 |
| SG-7 | ingestion-lag arbitrage | NO-PRIOR-FOUND | — | Same cluster as LT-3. Exploits the label pipeline |
| SG-8 | importance × discoverability | PARTIAL | Spence 2025 [S6] | Same cluster as CI-06 and WC-SELFN |
| SG-9 | rg donor weights | PARTIAL | as CI-08 [S9]. rg is standard (LDSC, mem) | Same cluster as CI-08 |
| WC-L2GEMU | emulate label generator | PARTIAL | Holland 2016 [S5]. OT L2G [S27] | Same cluster as SG-3 and IR-2 |
| WC-XPOWER | crossing forecast without own GWAS | PARTIAL | as CI-08/SG-3 | Same cluster |
| WC-BACKLOG | curation backlog forecast | NO-PRIOR-FOUND | — | Duplicate of LT-3 / IR-3 |
| WC-SELFN | channel-mixture selection function | PARTIAL | Spence 2025 [S6] | Same cluster as CI-06 |
| WC-HAWKES | self-exciting evidence flux | PARTIAL | temporal link prediction [S23] | No Hawkes process for gene–disease evidence found (low confidence) |
| WC-CAPTURE | capture-recapture completeness | PARTIAL | Capture-recapture for omics meta-analysis (bioRxiv 2023) [S40]: number of undetected causal genes for preterm birth. Classical saturation estimates | Neg: estimates are very imprecise when capture rates vary between genes [S40] |
| WC-DCA | inverse-Ising couplings over disease sets | NO-PRIOR-FOUND | method review: Stein, Marks, Sander 2015 [S41] | Query: "inverse Ising maximum entropy … disease gene sets…" |
| WC-UPPERLIM | measured nulls as negative evidence | NO-PRIOR-FOUND | as IR-4 [S33] | Duplicate of IR-4 |
| WC-MARKET | grants/patents as leading indicators | PARTIAL | Stoeger et al. PLoS Biol 2018 [S24]: publications, first-report year and NIH funding per gene are predictable from gene properties. Azoulay (NBER): half of NIH-linked patents are for another disease | Gene × disease-specific funding/patent flux as a forecaster was not found |

## Summary

- **Strong prior art (8):** CI-03 cis-MR/coloc (and circular with L2G), CI-09 allelic series (GPS family), EV-1 phenologs (Neg: only mouse helps), EV-3 ERC kernel, ML-3 gene-set LM (GSFM 2025), LT-8 text two-tower (GenePT), SC-4 (Ota 2025), SC-8 (sc-linker).
- **Partial with a solid mechanism precedent:** CI-01/SC-1, CI-02/SC-5/SG-6 (eQTLGen convergence, Downstreamer), SG-4 (Freund 2018), SG-2 (DeGAs), EV-5 (Neg: Raymond 2020), CI-06/SG-8/WC-SELFN/EV-4 (Spence 2025, Mostafavi 2023).
- **No prior found (13):** CI-04 directed donor graph, ML-2 ascertainment-simulator PFN, ML-6 tabular ICL (Neg: TFM limits on genetic data), ML-7 set diffusion, IR-9 discovery-sequence model, IR-4/WC-UPPERLIM powered nulls, WC-DCA inverse Ising, LT-6, SC-9, and the label-pipeline cluster LT-3/IR-3/WC-BACKLOG/SG-7.
  The label-pipeline ideas are novel only because nobody publishes models of a benchmark's own curation lag. A gain there would be real on the benchmark and empty biologically.
- **Duplicate clusters to merge before round 2:**
  - (LT-3, IR-3, WC-BACKLOG, SG-7)
  - (IR-4, WC-UPPERLIM)
  - (ML-1, IR-8)
  - (IR-2, SG-1)
  - (CI-02, SC-5, SG-6)
  - (CI-01, SC-1, SC-4)
  - (CI-08, SG-9, WC-XPOWER, SG-3, ML-4)
  - (CI-06, SG-8, WC-SELFN, EV-4)
  - (LT-7, ML-9, IR-6, WC-HAWKES)
  - (ML-3, ML-5, SC-7)
- **Negative prior results to respect:**
  - distant-organism phenologs add nothing [S11b]
  - human→cattle priors help only without own data [S14]
  - RERconverge false positives on null traits [S15]
  - LLM gene rankers driven by gene fame and candidate order [S22, S22b]
  - LLM entity normalisation is poor [S20]
  - TabPFN degrades with many features [S31]
  - trans-eQTLs replicate poorly [S3]
  - capture-recapture estimates are imprecise [S40]

## Sources (all accessed 2026-11-12)
- S1 Ota et al. 2025, Nature, https://www.nature.com/articles/s41586-025-09866-3
- S2 Schnitzler et al. 2024, Nature, https://www.nature.com/articles/s41586-024-07022-x ; preprint https://www.biorxiv.org/content/10.1101/2022.11.01.514606v1
- S3 Võsa et al. (eQTLGen), https://www.doi.org/10.1101/447367 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC8432599
- S4 Downstreamer, https://www.medrxiv.org/content/10.1101/2021.10.21.21265342.full.pdf
- S5 Holland et al. 2016, Front Genet, https://www.frontiersin.org/articles/10.3389/fgene.2016.00015/full
- S6 Spence et al. 2025, Nature, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12823407/
- S7 Mostafavi et al. 2023, Nat Genet, https://www.nature.com/articles/s41588-023-01529-1
- S8 O'Donoghue et al. 2021, https://arxiv.org/abs/2111.07348 ; FairPRS https://pmc.ncbi.nlm.nih.gov/articles/PMC10804441
- S9 MTAR 2020, Nat Commun, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7275056/
- S10 COAST allelic series test, https://www.biorxiv.org/content/10.1101/2022.12.23.521658.full.pdf ; GPS/allelic-series direction https://pmc.ncbi.nlm.nih.gov/articles/PMC12488494 ; rare-to-common allelic series https://www.vascular-proteomics.com/wp-content/uploads/2025/10/41588_2025_Article_2355.pdf
- S11 McGary et al. 2010, PNAS, https://pmc.ncbi.nlm.nih.gov/articles/PMC2851946 ; S11b model-organism contribution preprint 2021, https://www.biorxiv.org/content/10.1101/2021.12.24.474099.full.pdf
- S12 Lal et al. 2020, Genome Med, https://doaj.org/article/45a19bc663e847f68db119e08f35c85f ; S12b Paralogue Annotation, https://www.biorxiv.org/content/10.1101/2023.08.07.552236.full.pdf
- S13 Priedigkeit et al. 2015, PLoS Genet, https://journals.plos.org/plosgenetics/article/info:doi/10.1371/journal.pgen.1004967 ; ERC2.0 https://genome.cshlp.org/content/genome/early/2025/08/07/gr.280586.125.full.pdf
- S14 Raymond et al. 2020, PLoS Genet, https://pmc.ncbi.nlm.nih.gov/articles/PMC7514049
- S15 RERconverge, https://www.biorxiv.org/content/10.1101/451138.full.pdf ; null-phenotype caveat https://www.biorxiv.org/content/10.1101/2020.10.14.338608v1.full.pdf
- S16 Edgetic structural interactome 2019, https://link.springer.com/article/10.1038/s41467-019-11180-2 ; MutPred-PPI https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12767517/
- S17 MetaDome 2019, https://pubmed.ncbi.nlm.nih.gov/31116477/ ; HMC 2024 https://link.springer.com/10.1186/s13073-024-01358-9
- S18 Temporally enabled word embeddings, https://nference.com/publications/YWZd4xIAACoA3gEu ; https://www.biorxiv.org/content/10.1101/627513.full.pdf
- S19 DeepProphet2, https://arxiv.org/abs/2208.01918
- S20 DisGeNET UKB phecode mapping note, https://blog.disgenet.com/?p=104 ; S20b GENPHIRE https://www.medrxiv.org/content/10.64898/2025.12.03.25341576.full.pdf
- S21 GenePT, https://www.biorxiv.org/content/10.1101/2023.10.16.562533.full.pdf ; GenePT revisited https://www.iclr.cc/virtual/2026/10021585
- S22 LLM gene prioritisation survey, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12263109/ ; S22b https://neurips.cc/virtual/2026/161116
- S23 Link prediction review, https://arxiv.org/pdf/2312.01275 ; Rosalind https://link.springer.com/10.1038/s41598-020-74922-z
- S24 Stoeger et al. 2018, https://amaral.northwestern.edu/publications/large-scale-investigation-reasons-why-potentially-important-genes-are-ignored/ ; Azoulay NBER https://www.nber.org/papers/w20889.pdf
- S25 Genebass / burden PheWAS, https://pmc.ncbi.nlm.nih.gov/articles/PMC9903662 ; https://blog.opentargets.org/how-open-targets-integrates-gene-burden/
- S26 Disease gene prediction with privileged information and heteroscedastic dropout 2021, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8275341/
- S27 Open Targets Genetics L2G, https://www.biorxiv.org/content/10.1101/2020.09.16.299271.full.pdf ; HyPrColoc https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7858636/ ; SigNet https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11741684/
- S28 MapPFN, https://arxiv.org/abs/2601.21092
- S29 Gene Set Foundation Model 2025, https://www.biorxiv.org/content/10.1101/2025.05.30.657124.full.pdf ; Rummagene https://pubmed.ncbi.nlm.nih.gov/38643247/
- S30 PhenoGnet, https://arxiv.org/pdf/2509.14037 ; SmuDGE https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6129260/
- S31 TabPFN limits, https://www.biorxiv.org/content/10.1101/2025.08.15.670537.full.pdf ; https://neurips.cc/virtual/2026/poster/139799
- S32 DigNet, https://genome.cshlp.org/content/35/2/340
- S33 Negative pharmacogenetic studies, https://www.nature.com/articles/tpj201158 ; Connally 2022 eLife https://ibg.colorado.edu/cdrom2025/4.4.FinemappingColocalization_RanCui/refs/elife_Connally_2022_missingregulation.pdf
- S34 Strobl & Gamazon, eLife, https://elifesciences.org/articles/100949
- S35 BioGRID ORCS, https://academic.oup.com/nar/article/47/D1/D529/5204333
- S36 PrEDiCT, eLife 2024, https://elifesciences.org/articles/84613
- S37 sc-linker, Nat Genet 2022, https://www.doi.org/10.1038/s41588-022-01187-9
- S38 DeGAs, Nat Commun 2019, https://www.ukbiobank.ac.uk/publications/components-of-genetic-associations-across-2138-phenotypes-in-the-uk-biobank-highlight-adipocyte-biology/
- S39 Freund et al. AJHG 2018, https://pubmed.ncbi.nlm.nih.gov/30290150/
- S40 Capture-recapture for omics, https://www.biorxiv.org/content/10.1101/2023.04.24.537481.full.pdf ; screen saturation https://pmc.ncbi.nlm.nih.gov/articles/PMC1448089
- S41 Stein, Marks, Sander 2015, https://pmc.ncbi.nlm.nih.gov/articles/PMC4520494
- From memory, not fetched: SMR (Zhu et al. 2016 Nat Genet), MTAG (Turley et al. 2018 Nat Genet), mash (Urbut et al. 2019 Nat Genet), DropoutNet (Volkovs et al. NeurIPS 2017), ExpoMF (Liang et al. WWW 2016), GWAX (Liu et al. 2017 Nat Genet), Tabach et al. 2013 Nature, Ferrero et al. 2017 J Transl Med, CMap (Lamb 2006 / Subramanian 2017).
