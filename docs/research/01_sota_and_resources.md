# 01 State of the art (2022-2026) and local resources for next-generation disease module prediction

Written 2026-11-12 by a research subagent. Every web source was accessed 2026-11-12.
Task: given a disease, rank the ~19k human protein-coding genes by membership in its disease module, and call the module.
Companion file: `docs/research/02_local_data_inventory.md` (the lead's inventory of `/mnt/msadb`). Part B below does not repeat it. It adds only what 02 leaves out: tool environments, what is inside the larger folders, and evidence stores outside `/mnt/msadb`.

**How sure each claim is.** Each claim carries one of three tags:
- **[v]**: checked this session against a primary page or abstract.
- **[s, unverified]**: taken only from a search snippet or secondary coverage.
- **[m]**: from model background knowledge, not re-checked this session. Check [m] claims before you cite them.

**Leakage classes** (from modeng `CLAUDE.md`):
- `temporal-clean`: every input is dated at or before the seed release.
- `transductive`: modern inputs, so labels can leak in through the inputs.
- `holdout-current`: cross-validation on current labels.

The leakage notes in Part A use these classes.

---

## Part A. State of the art, 2022-2026

### A1. Network propagation, disease modules and degree bias

**Methods.**
- Classic: RWR, DIAMOnD, network expansion, NetCore, HotNet2, DOMINO, ROBUST.
- The largest recent application is Barrio-Hernandez et al. 2023 (Nat Genet, doi 10.1038/s41588-023-01327-9). It seeds with L2G > 0.5 genes from 1,002 GWAS traits and runs network expansion. It finds 73 pleiotropic modules [s, unverified]. Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10858647/ (2024 practical guide, Brief Bioinform), plus the search snippet above.
- The 2024 guide reports that seeding with gene-level GWAS p-values beats unweighted seed sets when the summary statistics are good [s, unverified]. This matches modet's own direction (MAGMA z as input).

**Critiques.** Propagation scores track node degree. Known corrections:
- DADA degree-aware statistics (2011): https://pmc.ncbi.nlm.nih.gov/articles/PMC3143097
- Normalising against degree-preserving rewired networks (Frontiers Genet 2019): https://www.frontiersin.org/articles/10.3389/fgene.2019.00004/text
- Coreness in place of degree (NetCore 2020) [s].

**Recent evidence of degree and study bias.**
- Brechtmann et al. 2023 (NAR Genom Bioinform lqad095): functional gene embeddings built from literature and low-throughput data look strong mainly because the benchmarks favour well-studied genes. On GWAS gene prediction they do not beat other embeddings [s, unverified].
- Link-prediction evaluation for under-studied proteins: https://www.biorxiv.org/content/10.1101/2022.10.13.511953 [s].
- A 2025 GRN-inference preprint: degree heuristics match GNNs under standard negative sampling (https://www.biorxiv.org/content/10.64898/2025.12.19.695616) [s, unverified].
- Imperfect gold standards distort AUCs (bioRxiv 2023.05.04.539407) [s].
- Wright et al. 2025, Mol Syst Biol, "State of the interactomes": compares about 45 interactomes on gene-set recovery. It ships post-2023-07-27 GWAS gene sets and PubMed citation counts for a citation-bias analysis. Code: https://github.com/sarah-n-wright/Network_Evaluation_Tools [m; files are on disk, see B2].

**What it means for us.**
- Any propagation or GNN result must be reported in two extra ways:
  - against degree-preserving (XSwap) and citation-matched nulls;
  - with performance split by gene citation decile.
- subex already holds XSwap rewirings and modet's matched null queries.

### A2. GNNs and knowledge-graph embeddings (PrimeKG, TxGNN, PyKEEN)

**PrimeKG** (Chandak et al., Sci Data 2023):
- About 129k nodes and 4M edges, 17,080 diseases [m].
- Its disease-protein edges come from DisGeNET, OMIM and similar curated or text-mined sources.
- Leakage: as an input for disease-gene prediction it is fully **transductive**, because labels are inside the graph. You must delete the query disease's disease-gene edges, and also those of its ontology family.

**TxGNN** (Huang et al., Nat Med 2024):
- An R-GCN pretrained on PrimeKG, with a disease-similarity metric-learning module for zero-shot diseases.
- Reports +49.2% on indications and +35.1% on contraindications against 8 baselines in a zero-shot split [s].
- Source: https://ai.stanford.edu/~jure/pubs/foundation-naturemedicine24.pdf
- The target is drug-disease links, not disease genes. The reusable idea is the **disease-similarity transfer module**: for a disease with few seeds, borrow embeddings from similar diseases.
- An independent 2026 preprint (OREVID, https://www.preprints.org/manuscript/202610.1918) reports TxGNN zero-shot Recall@100 of 0.412 against 0.812 for its own LightGBM ranker [s, unverified]. A second snippet describes an October 2026 LightGBM ranker on PrimeKG + Open Targets + STRING + PubTator3, evaluated on held-out disease neighbourhoods. TxGNN had higher AUPRC in 5 of 9 areas [s, unverified]. The general lesson, which matches modet's finding, is that gradient-boosted trees on graph-derived features compete with end-to-end GNNs.

**Other KG work:**
- PrimeKG++/BioMedKG, multimodal contrastive KG embeddings (Front Syst Biol 2025): https://public-pages-files-2025.frontiersin.org/journals/systems-biology/articles/10.3389/fsysb.2025.1651930/epub [s].
- KG + LLM for disease prediction: https://arxiv.org/pdf/2512.01210 [s].

**Code and data.**
- PrimeKG `kg.csv` is on disk (modet/data/raw/primekg, 948 MB).
- The TxGNN package is installed in conda env `txgnn_env` (txgnn 0.0.3, DGL 2.4, torch 2.11+cu128; see B1).
- PyKEEN and PyG are in `bio-net`.

### A3. GWAS-to-gene: MAGMA, PoPS, FLAMES, cS2G, Open Targets L2G

**MAGMA** (de Leeuw 2015) [m]:
- Gene-level z-scores from summary statistics plus an LD reference.
- No disease-gene labels go in, so it is **temporal-clean** if the GWAS predates the seed release.
- Pitfall: gene size and LD inflate z. Use the MHC-excluded, size-adjusted z from the gene model.

**PoPS** (Weeks et al., Nat Genet 2023, 55:1267, https://www.nature.com/articles/s41588-023-01443-6) [v]:
- Ridge regression of MAGMA z on 57,543 gene features: expression, PPI, pathways.
- PoPS and closest-gene are each best alone, and combining them gives high-precision calls. That yields 10,642 gene-trait pairs across 113 traits.
- Code: https://github.com/FinucaneLab/pops. Features: Zenodo 12635505.
- Leakage: the full feature set includes MGI/KEGG/GO/Reactome pathway features, and some of those encode disease genes. The `pathway_naive` variant removes pathway and PPI features, and it is on disk.

**FLAMES** (Schipper et al., Nat Genet 2025, 57:323, doi 10.1038/s41588-025-02084-7) [s]:
- XGBoost over 22 SNP-to-gene annotation types plus PoPS convergence, run within fine-mapped credible sets.
- Reported to beat L2G, cS2G and PoPS on accuracy and robustness.
- Code: https://github.com/Marijn-Schipper/FLAMES. Annotation bundle: Zenodo 12635505, already on disk as `Annotation_data.tar.gz`.
- Leakage: the training gold standards are curated causal genes. Applied to a disease whose genes were in the gold standard, scores are **transductive**.

**Open Targets L2G** (Gentropy):
- Retrained for each release on gold-standard gene-trait pairs.
- The 25.12 release has more than 1.4M GWAS credible sets, COLOC-PIP colocalisation, PICS for multi-ancestry studies, and ENCODE-rE2G enhancer-gene links [s].
- Development models `locus_to_gene_26.06*` are on Hugging Face [s]: https://huggingface.co/opentargets/locus_to_gene_26.06
- Docs: https://platform-docs.opentargets.org/gentropy/locus-to-gene-l2g and https://blog.opentargets.org/open-targets-platform-25-12-has-been-released/
- Leakage: L2G's gold standard overlaps the curated (ClinGen/G2P/ChEMBL) labels we evaluate against. So L2G > 0.5 genes partly **are** labels. Moreover, OT's association score for a disease already contains the L2G evidence. Using L2G as an input while evaluating on OT associations is circular unless the GWAS-credible-set datasource is removed from the labels, or the labels are restricted to non-GWAS sources.

**cS2G** (Gazal 2022) [m]: SNP-to-gene linking strategies combined. On disk.

**Common benchmarks:**
- Fine-mapped coding variants (Weeks).
- OT gold standards.
- ExWAS genes as an orthogonal truth for GWAS loci, a strong design.
- Drug targets (Minikel 2024).

### A4. Rare-variant exome burden

- **Genebass** (Karczewski et al., Cell Genomics 2022): 394,841 UKB exomes, gene burden for 4,529 phenotypes. Browser: https://app.genebass.org [m]. Gene-level results are downloadable as Hail tables; the summary download is about 10-30 GB, estimate [m]. Source for the counts: https://ibg.colorado.edu/cdrom2025/5.2.RareVariantAssociation_DuncanPalmer/refs/ [s].
- **Backman et al. 2021** (Nature, about 455k exomes, Regeneron): 564 genes with health associations [s].
- **AstraZeneca PheWAS Portal** (https://azphewas.com): collapsing analyses, now v5 on about 470k UKB exomes [s]. Open Targets ingests its gene-burden evidence: https://blog.opentargets.org/how-open-targets-integrates-gene-burden/ [s].
- UKB WES final release covers about 470k participants: https://www.ukbiobank.ac.uk/learn-more-about-uk-biobank/news/final-data-release-from-the-world-s-largest-whole-exome-sequencing-project [s].
- **Why it matters:** burden hits are close to causal-gene truth and nearly free of network or literature bias. That makes them the best orthogonal **evaluation** source and a strong **input** for common diseases.
- Leakage: OT's `gene_burden` datasource is the same data. Use it as an input or as a label, never both.

### A5. Drug genetics and target validation

Minikel et al. 2024 (Nature 629:624): drug mechanisms with genetic support have a 2.6x higher probability of success. The effect rises with confidence in the causal gene [v]. Source: https://ideas.repec.org/a/nat/nature/v629y2024i8012d10.1038_s41586-024-07316-0.html

The data are on disk (`minikel2024`), and modet already has a drug-phase ground truth. This gives a non-OT, utility-oriented evaluation axis: does the module rank successful targets?

### A6. Gene and cell foundation models (single-cell)

Models:
- **Geneformer** (Theodoris, Nature 2023). V2 is about 316M parameters with 1,152-d embeddings [s]. In-silico deletion has been used to nominate cardiomyopathy targets [m].
- **scGPT** (Cui, Nat Methods 2024) [m].
- **scFoundation** (Hao, Nat Methods 2024; 3,072-d) [s].
- **UCE** (Rosen et al. 2023, https://ai.stanford.edu/~jure/pubs/uce-bioarxiv23.pdf). Its gene tokens are ESM2 protein embeddings [s]. The ESM2-15B gene embeddings from UCE are on disk.
- **TranscriptFormer** (CZI 2025): up to 112M cells, 12 species. Variants tf-sapiens, tf-exemplar and tf-metazoa; `pip install transcriptformer` [s]. Sources: https://www.biorxiv.org/content/10.1101/2025.04.25.650731 and https://biohub.org/blog/transcriptformer-model-overview/

**Critiques.** Several benchmarks find these models no better than simple baselines on perturbation prediction:
- Ahlmann-Eltze et al. 2025 [s].
- PertEval-scFM, ICML 2025: https://proceedings.mlr.press/v267/wenteler25a.html [s].
- PerturBench: https://arxiv.org/abs/2408.10609 [s].
- A 2026 pre-registered external evaluation finds a "partial-replication" signal in frozen Geneformer, scGPT and UCE features: https://arxiv.org/pdf/2608.26170 [s, unverified].

I found **no** study that benchmarks these models' gene embeddings for disease-gene ranking against plain expression-specificity features. That gap is cheap to test.

Leakage: pretraining is on expression only, so these embeddings are clean of disease-gene labels. Atlases include disease samples, which is acceptable.

### A7. Protein language models and structure

- **ESM C** (EvolutionaryScale, December 2024): 300M, 600M (960 / 1,152-d) and 6B (2,560-d). 300M and 600M weights are open. The 6B is reachable only through the Forge/SageMaker API according to the later page [s]. Source: https://evolutionaryscale.ai/blog/esm-cambrian
- **ESM3** conditions on function keywords and InterPro annotations [m]. Embeddings that use function tokens are therefore **not** clean. Use sequence-only ESM C.
- **ProtT5-XL-U50** (Elnaggar 2021) [m]: UniProt ships per-protein embeddings; on disk.
- AlphaFold DB v4 human, Foldseek clusters (Barrio-Hernandez 2023), and AF2 interactomes (Burke 2023, HuRI subset) are on disk.
- **Zhang et al. 2025 Science**, RF2-PPI + DCA proteome-wide PPI screen: 17,849 PPIs at 90% precision [s]. Not downloaded; Dryad needs a browser.
- Typical gain: protein LM embeddings add little beyond networks for disease-gene ranking. They help most for under-studied genes, which is exactly where degree-biased methods fail [m; modet experience].

### A8. Genomic sequence-to-function models

- **AlphaGenome** (Nature 2026, 649:1206, doi 10.1038/s41586-025-10014-0): 1 Mb input, base-pair resolution output. Matches or beats the best external model in 25 of 26 variant-effect evaluations [v]. Non-commercial API. Source: https://pubmed.ncbi.nlm.nih.gov/41606153/
- **Borzoi** (Linder, Nat Genet 2025) and **Enformer** (Avsec 2021) [m] are local. gReLU wraps Enformer; Borzoi `replicate-0` weights are in the HF cache.
- Use for gene ranking: score fine-mapped credible-set variants for effects on nearby gene expression in disease-relevant tissues. This is a mechanistic variant-to-gene link that is independent of L2G's curated training.
- Leakage: trained on ENCODE/GTEx tracks, not disease labels, so **temporal-clean apart from model date**.
- Cost: AlphaGenome API calls (quota) or local Borzoi at about 4-5 GB of GPU per pass.

### A9. Perturbation atlases and dependency maps

- **Replogle 2022** (Cell): genome-wide CRISPRi Perturb-seq in K562 plus essential-gene screens in K562 and RPE1, >2.5M cells [v]. Pseudobulk is on disk.
- **Nadig et al. 2025**: HepG2 and Jurkat essential-gene Perturb-seq (GEO GSE264667). On disk; subex built pseudobulk views.
- **Zhu, Dann et al.**: genome-scale CRISPRi Perturb-seq in **primary human CD4+ T cells**. 22M cells, 12,748 targets, 4 donors x 3 conditions. Published in Cell on 2026-08-28 (doi 10.1016/j.cell.2026.08.002) [s]. Preprint: https://www.biorxiv.org/content/10.64898/2025.12.23.696273 [s]. On disk (pseudobulk plus DE stats).
- **X-Atlas/Orion** (Xaira, June 2025): 8M cells, all protein-coding genes, 2 cancer lines. About 520 GB, non-commercial licence, figshare 10.25452/figshare.plus.29190726 [s]. Follow-on X-Atlas/Pisces (March 2026): 25.6M cells, 7 contexts. I did not find whether it is public [s]. Sources: https://www.businesswire.com/news/home/20250617146938/en and https://www.businesswire.com/news/home/20260317710096/en
- **Tahoe-100M** (drug perturbations, 50 cancer lines, 379 drugs). Hugging Face `tahoebio/Tahoe-100M`, about 89 GB [s]. It is useful only indirectly, as drug-target signatures.
- **DepMap**: 20Q4 (pre-T0) and 26Q1/26Q3 are on disk. Co-essentiality modules are a strong, literature-free gene-gene similarity [m].
- Leakage: perturbation data are experiment-derived and clean of labels. Target-gene selection (expressed or essential genes) is a coverage bias, not leakage.
- Evidence: perturbation-signature similarity recovers complexes and pathways (Replogle). Using it **as a disease-module network layer** is rarely benchmarked. The DisMech third-party analysis of the CD4 data tested GWAS-gene enrichment across 111 regulator clusters: https://dismech.monarchinitiative.org/elements/gwas-tcell-validation-report/ [s, unverified].

### A10. LLMs: zero-shot prioritisation, text embeddings, agents

**Causal genes at GWAS loci.** medRxiv 2024.05.30.24308179 (https://www.medrxiv.org/content/10.1101/2024.05.30.24308179v1.full.pdf) [s]:
- GPT-4 and Claude rival or beat L2G/PoPS-style methods at picking causal genes at loci.
- Accuracy correlates with the number of publications on the gene, which is literature bias.
- Leakage was controlled with loci published after the training cutoff and unpublished 23andMe loci.

**Rare disease.** Neeley et al. 2025: GPT-4 ranks the causal gene first about 30% of the time. It is biased toward well-studied genes and sensitive to input order [s]. Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12263109/

**GenePT** (Chen & Zou 2023/2025): 1,536-d ada-002 embeddings of NCBI gene summaries (also text-embedding-3-large). Matches or beats Geneformer on gene-property tasks [s]. Sources: https://www.biorxiv.org/content/10.1101/2023.10.16.562533 and data https://zenodo.org/records/10833191
- **Leakage: severe.** NCBI and UniProt summaries name diseases ("mutations cause X"). Gene text embeddings are **transductive by construction** unless sentences that mention diseases or phenotypes are stripped, or the text is date-gated (an archived gene_info / GeneRIF snapshot at or before T0).

**GeneAgent** (Nat Methods 2025, doi 10.1038/s41592-025-02748-6): self-verifying gene-set agent. About 92% of verification decisions were correct; tested on 1,106 gene sets [v]. Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12328209/ It does function naming, not ranking.

**Biomni** (bioRxiv 2025.05.30.656746): general agent, includes causal-gene prioritisation tasks. The later Biomni-Eval1 claims state of the art [s]. Source: https://www.biorxiv.org/content/10.1101/2025.05.30.656746

**GSFM**, gene-set foundation model (Ma'ayan lab 2025): denoising autoencoder pretrained on Rummagene/RummaGEO gene sets, beats transformer variants [s]. Sources: https://pubmed.ncbi.nlm.nih.gov/40501705/ and https://gsfm.maayanlab.cloud
- Leakage: Rummagene is mined from paper supplements, so it contains published disease gene lists. Transductive.

**General LLM lesson:** LLMs memorise the labels. They are only evaluable on (a) disease-gene associations that first appeared after the model's training cutoff, or (b) inputs stripped of the disease name. Local option: Qwen3.8-27B (B1). Its training cutoff is unknown; it must be pinned before any temporal claim.

### A11. Cross-disease transfer, set models, contrastive embeddings, calibration

**Multi-task / transfer:**
- GenePlexus trains one classifier per gene set on network embeddings; installed locally [m].
- TxGNN's disease-similarity pooling (A2).
- Open Targets' own "indirect" ontology propagation is the simplest transfer. It is also a leakage channel: parent and child labels.
- Evaluation must hold out whole **disease families**. modet's `folds_v1.json` cluster folds do this, and its sealed fold must stay unused.

**Set-based models.**
- I found **no** published Set-Transformer or Deep-Sets model that takes a variable-size seed set and scores all genes for disease-gene prediction [s; search negative].
- Closest: GSFM (autoencoder over gene sets), DeepProphet2 (transformer recommending genes from a gene set, trained on co-citation; https://ar5iv.labs.arxiv.org/html/2208.01918) and GSI (MSigDB gene-set features) [s].
- This gap is clear.

**Contrastive gene-disease embeddings:**
- BioMedKG (multimodal contrastive) [s].
- CLIP-style pairing of gene embeddings (ESM/expression/perturbation) with disease text embeddings (MONDO definitions) is a natural fit for zero-shot new diseases. I found no benchmarked disease-gene instance.

**Conformal / calibrated module calls:**
- No search hit for conformal prediction in disease-gene prioritisation [s; search negative].
- A NeurIPS 2026 cell-fate paper shows marginal coverage hides poor conditional coverage, 0.966 against 0.858 [s]: https://neurips.cc/virtual/2026/169462
- For module calls the relevant guarantees are FDR control on membership (conformal p-values with Benjamini-Hochberg, Bates et al. 2023 [m]) calibrated per disease using **null queries**. modet's matched null-query exports on disk fit this directly.

### A12. Summary table

| Family | Representative (year) | Reported gain | Benchmark | Main leakage risk | Code/data |
|---|---|---|---|---|---|
| Propagation / expansion | Barrio-Hernandez 2023 | modules for 1,002 traits | L2G seeds, enrichment | degree, L2G seeds | public; local NetMedPy/NeDRex |
| KG GNN | TxGNN 2024 | +49% indications (zero-shot) | PrimeKG splits | disease-gene edges in KG | txgnn_env local; PrimeKG on disk |
| GWAS to gene | PoPS 2023, FLAMES 2025, L2G | best single and combined at loci | fine-mapped coding variants, OT gold std | pathway features, curated gold std = labels | on disk (PoPS, FLAMES annot., OT 26.09 L2G) |
| Exome burden | Genebass 2022, AZ PheWAS v5 | near-causal genes | — | same data in OT gene_burden | download |
| scFMs | Geneformer, scGPT, UCE, TranscriptFormer | mixed, often ≈ baselines | perturbation and cell tasks | low | UCE-ESM2 gene emb. on disk; models not installed |
| Protein LMs | ESM C, ProtT5 | helps under-studied genes | — | ESM3 function tokens | on disk (ESM C 300M, ProtT5, ESM2-15B) |
| Seq-to-function | AlphaGenome 2026, Borzoi | SOTA variant effects | 25 of 26 VEP evals | none for labels | API key + local Borzoi |
| Perturbation | Replogle 2022, CD4 2026, Nadig 2025, X-Atlas | complex/pathway recovery | — | coverage only | mostly on disk |
| LLM | GPT-4 at GWAS loci 2024, GenePT, GeneAgent, Biomni | ≈ or > L2G at loci | post-cutoff loci | severe memorisation | local Qwen3.8-27B; GenePT Zenodo |
| Set / contrastive / conformal | GSFM 2025; none for conformal | — | — | Rummagene = published lists | gap |

---

## Part B. Local resources (read-only check, 2026-11-12)

The disk-level inventory of `/mnt/msadb/modet/downloads`, `modet/data/processed` and `/mnt/msadb/subex` is in **02_local_data_inventory.md**. Only the gaps are below.

### B1. Tool environments relevant to this task

Wrappers live in `~/biotools/bin`. All GPU use goes through `~/biotools/bin/gpu-run --mem <GB> -- ...`. Free space on `/mnt/msadb`: 683 GB of 1.6 TB.

| Resource | Path / call | Content | Version | Notes |
|---|---|---|---|---|
| bio-net (main) | `~/biotools/venvs/bio-net/bin/python` | PyG 2.8, PyKEEN, NetMedPy, NeDRex client, networkx, igraph, leidenalg, duckdb, polars, gseapy; torch 2.11.0+cu128 | 2026-10 | default env for modeng |
| txgnn_env | `~/miniforge3/envs/txgnn_env` | txgnn 0.0.3 (editable), DGL 2.4, PyG 2.8, torch 2.11+cu128 | — | TxGNN baseline possible |
| ESM | `esmc-embed`, `esm-python`, venv `esm` | ESM C 300M (weights `biohub/ESMC-300M`, 1.3 GB in HF cache) | — | 600M not local; 6B API only |
| bio-sc | venv `bio-sc` | scanpy, rapids-singlecell, scvi-tools, pertpy, decoupler, cellxgene-census | — | **Geneformer, scGPT, UCE, scFoundation, TranscriptFormer are NOT installed** (no package found in any venv) |
| pytdc | venv `pytdc` | PyTDC, cellxgene_census, transformers | — | |
| bio-genomics | `mamba run -n bio-genomics` | gget, pysam, bcftools, Borzoi (weights `johahi/borzoi-replicate-0`, 710 MB), AlphaGenome client (`ALPHAGENOME_API_KEY` set) | — | Borzoi about 4-5 GB GPU |
| grelu | venv `grelu` | gReLU 1.1 + Enformer | — | |
| MAGMA | `~/biotools/opt/magma/magma` | v1.10; 1000G EUR ref in modet/data/raw/magma | — | not redistributable |
| PoPS | `~/biotools/opt/pops`, venv `pops` | FinucaneLab PoPS | — | features on disk |
| geneplexus / obnb / pecanpy | venvs | PyGenePlexus, OBNB benchmark, node2vec | — | baselines |
| gbm-challengers | venv | LightGBM, XGBoost 3.2, CatBoost | — | modet's strongest head is LightGBM |
| tabpfn | venv | TabPFN 2.5 | — | weights need a licence click (blocked) |
| amim / amim-domino / cdlib | venvs | ROBUST, DOMINO, overlapping communities | — | module baselines |
| subex-sbm | mamba | graph-tool 3.9 (SBM), pcst_fast | — | numpy<2 pinned |
| multixrank | venv | MultiXrank (multiplex RWR) | 0.3 | multilayer propagation baseline |
| Local LLM | `llm-serve start` (vLLM 0.31) | Qwen3.8-27B INT4 at http://127.0.0.1:8011/v1, about 23 GB GPU | — | training cutoff unknown; `ATHENA-R1-Qwen3-8B` (16 GB) also cached |
| Text embedders | HF cache | Qwen3-Embedding-0.6B/8B, bge-m3, harrier-0.6b, MedCPT | — | **only metadata stubs (KB-MB) in `~/.cache/huggingface/hub`; weights must be downloaded** (Qwen3-Embedding-8B about 16 GB, 0.6B about 1.2 GB) |
| MCP | `opentargets`, `optimuskg` (190k nodes / 21.8M edges KG), `biomcp`, `duckdb`, ToolUniverse | live lookups | — | optimuskg = another transductive KG |

### B2. Data contents that 02 does not detail

| Resource | Path | Content | Date / version | Size |
|---|---|---|---|---|
| Per-trait MAGMA (current) | `modet/data/processed/gwas_magma/*.parquet` | **92** trait files, one per OT disease id. Columns `gene` (ENSG), `entrez`, `z`, `p`, `nsnps`, `n`, `gcst`, `disease_id`, `chr`, `start`, `stop`. Example EFO_0000274: 17,506 genes from GCST90429795 (N≈19k). Run log `magma_runs.tsv` (85 runs, lambda_GC); selection `studies_selected_test.tsv` (86) | GWAS Catalog harmonised sumstats, 2026-11 | 38 GB incl. `work/` |
| MAGMA at T0 | `modet/data/processed/gwas_magma_t0/` | 108 studies x {`leads`, `w0`, `w10`} (0 kb / 10 kb windows) + `pops/` + `lead_links.parquet` | pre-2021-04-30 GWAS | 4.4 GB |
| OT gene sets | `modet/data/processed/groundtruth/` | `ot_current_sets_2609` (7,799 diseases; disease_id, name, genes), `ot_t0_sets` (4,592), `ot_temporal` (2,958 diseases with t0/t1/new genes), panels (260 diseases per network, dev/test split), `arrival_*` (per new gene: evidence class, sources, 2025/26 scores) | OT 19.04 to 26.09 | 19 MB |
| Gene universe | `modet/data/processed/hgnc_protein_coding.parquet` | 19,255 genes (Entrez/Ensembl) | — | — |
| T0 features | `modet/data/processed/features/` | ESM C 300M 960-d (19,137 genes), GO T0 propagated (9,994 terms), GTEx v8 54 tissues, Reactome v72 without its disease branch | T0 = 2021-04-30 | 110 MB |
| HumanBase | `modet/downloads/humanbase/` | `global_top.gz` (237 MB) + **38 tissue networks** `<tissue>_top.gz` (amygdala ... uterus, incl. blood, brain, colon, heart, kidney, liver, lung, macrophage, lymphocyte, pancreas, skin) | GIANT/HumanBase (pre-2021 build, [m]) | 13 GB |
| PoPS / FLAMES features | `modet/downloads/pops_features/` | `pops_features_full` (57,543 features), `pops_features_pathway_naive`, FLAMES `Annotation_data.tar.gz` (unused so far), extracted in `extract/` | Zenodo 12635505 v5 (content pre-T0) | 25 GB |
| OT evidence rows | `modet/downloads/ot_evidence/` | `evidence_2104` (20 datasources, 4.0 GB), `evidence_2509` (only `gwas_credible_sets`, 172 MB), `study_2509` (90 MB) | OT 21.04, 25.09 | 4.2 GB |
| OT 26.09 genetics | `~/biotools/data/evidence/raw/opentargets/26.09/` | `l2g_prediction_slim.parquet` (3,293,342 rows: studyLocusId, geneId, score), `credible_set_gwas_slim.parquet` (1,519,507 rows: studyLocusId, studyId, variantId, position, finemappingMethod ...), study, disease | OT 26.09 | 88 MB |
| OT 26.09 associations | `~/biotools/data/evidence/raw/opentargets_assoc/` | overall direct/indirect, by-datatype indirect, disease | 26.09 (fetched 2026-10-08) | 5.6 GB |
| Other evidence (outside /mnt/msadb) | `~/biotools/data/evidence/raw/` | ClinVar variant_summary, GenCC submissions, HPO (genes_to_disease/phenotype), MGI, IMPC, Monarch per-disease JSON, Orphanet XML, MONDO obo, MeSH 2025, **gnomAD v4.1 constraint**, PubTator3 (gene2pubtator3, disease2pubtator3, relation2pubtator3; 3.1 GB), NCBI gene_info + orthologs, BioGRID 5.0.262, STRING v12.0/12.5, CELLxGENE cell-type h5ads (3.3 GB), DisGeNET API pages | 2025-2026 | ≈14 GB |
| Tabula Sapiens v2 | `/mnt/msadb/subex/raw/tabula_sapiens_v2/` | 29 h5ad files, one per tissue (`<Tissue>_TSP1_30_version2d_10X_smartseq_scvi_Nov122024.h5ad`), per-cell, scVI-integrated | 2024-11-12 build | 54 GB |
| CD4 Perturb-seq | `/mnt/msadb/subex/raw/cd4_perturbseq/` | `GWCD4i.pseudobulk_merged.h5ad` (44.6 GB), `GWCD4i.DE_stats.h5ad` (16.8 GB), guide KD efficiency, sgRNA library, sample metadata, DE suppl. table; per-cell files not downloaded (118-173 GB each). subex built `ps_cd4_2025{,_rest,_stim48hr}` signature views | Zhu/Dann 2025-26 (Cell 2026) | 58 GB |
| Nadig 2025 | `/mnt/msadb/subex/raw/nadig2025/` | GSE264667 per-cell raw h5ad: HepG2 (5.6 GB), Jurkat (9.4 GB) + series matrix; pseudobulk views in `subex/derived/ps_{hepg2,jurkat}_nadig*` | GEO public 2024-05-05 | 14 GB |
| Zhang 2025 PPI | `/mnt/msadb/subex/raw/zhang2025_humanppi/` | **README only, not downloaded** (Dryad needs a browser). Wanted: final_predictions (14 MB), RF2-PPI_scores (444 MB, about 47M pairs), AF_scores (32 MB), DCA_scores (778 MB), screened_pairs (336 MB) | Dryad 2025-09-16, CC0 | ≈1.6 GB to fetch by hand |
| Burke 2023 AF2 interactome | `/mnt/msadb/subex/raw/burke2023_huintaf2/` | Suppl. Tables S1-S6 (xlsx): 65,484 FoldDock pair models with pDockQ (HuRI 55,277 / hu.MAP 9,898). Use only the HuRI rows; hu.MAP is CORUM-derived. Full models (15.3 GB) not downloaded | NSMB 2023 | 14 MB |
| Wright 2025 | `/mnt/msadb/subex/raw/wright2025/` (+ modet copy) | `gwas_20230727.genesets` (48 post-2023-07-27 GWAS trait gene sets), `gwas.genesets` (699 GWAS sets, 2024-01-22; in modet's copy), `experimental.genesets` (17 Jan-2024 papers), `gene_citation_counts_Dec20_2023.txt`, CORUM pairs | 2024 | 2.8 MB |
| Gene embeddings (sequence) | `subex/raw/{uniprot_prott5_embeddings, uce_esm2_protein_embeddings, esmc_sprot2021_02}` | ProtT5 1,024-d (19,086 genes), ESM2-15B 5,120-d from UCE (18,807 symbols), ESM C 300M | 2021-2026 | 0.5 GB |
| PrimeKG | `modet/data/raw/primekg/kg.csv` | KG with disease-protein edges (transductive) | 2022-2023 | 948 MB |
| Modet benchmark scores | `/mnt/msadb/subex/modet_exports/{oof,null_queries}` | out-of-fold devall scores (4 modet models x 3 networks x 3 seeds), matched null queries (≈3.8k per network), production nulls on OT 26.09 | 2026-11-10/11 | 3.5 GB |

**Constraints that apply.** modet's sealed fold (`folds_v1.json`, `sealed: true`) is off-limits. `gwas_magma` covers only 92 traits, so GWAS-input claims hold only for that subset.

---

## Top 10 opportunities (combinations I found no published instance of)

In every row, "on disk" means usable today. Downloads give a URL and a size estimate.

**1. Permutation-invariant seed-set transformer over frozen multi-view gene embeddings, trained multi-task across about 7.8k OT diseases**
- Idea: a Set Transformer / Deep Sets encoder takes the seed set and outputs a disease query vector, then scores all 19k genes by cross-attention against gene tokens.
- Gene tokens are concatenated ESM C, ProtT5, ESM2-15B, GTEx/HPA/Tabula Sapiens specificity, DepMap and Perturb-seq signatures, plus node2vec over STRING/HuRI/BioGRID.
- Train episodically with random seed/target splits, holding out disease families.
- Combines the A11 gap with the modet LightGBM finding (trees beat GNNs) by testing a learned set-conditioned head against modet-U.
- Data: all on disk (02 + B2). No download.

**2. Perturbation-response networks as a disease-module layer, matched to disease context**
- Build gene-gene similarity from knockdown signatures: CD4 Stim8hr/48hr/Rest for immune diseases, HepG2 for liver/metabolic, K562 GWPS generic, Jurkat.
- Propagate MAGMA z or seeds on it. Compare against STRING with degree-matched nulls.
- Perturbation edges are literature-free, so they should shrink the citation bias in A1.
- Data: on disk (subex `derived/ps_*`, raw CD4/Nadig/Replogle).
- Optional download: X-Atlas/Orion pseudobulk, figshare 10.25452/figshare.plus.29190726. Full release about 520 GB; fetch only pseudobulk/DE files if they are offered, size unknown, check first.

**3. A genetics-only module: MAGMA z + fine-mapped credible sets + AlphaGenome/Borzoi variant-to-gene, without L2G's curated gold standard**
- Score each OT 26.09 credible-set variant with Borzoi (local) or AlphaGenome (API) for effects on gene expression in matched tissues.
- Aggregate to gene level with PIP weighting. Combine with MAGMA z as node evidence for propagation.
- This gives a label-independent GWAS-to-gene step, so the result is temporal-clean when the GWAS is dated, unlike L2G or FLAMES.
- Data on disk: OT 26.09 credible sets, MAGMA (92 traits), Borzoi weights, AlphaGenome key.
- Missing: per-variant tissue tracks are computed, not downloaded. Budget is about 1.5M credible-set variants. Restrict to the 92 traits with lead PIP > 0.1, likely ≈50-100k variants.

**4. Exome burden as both orthogonal truth and input, crossed with GWAS-derived modules**
- Use Genebass / AZ PheWAS gene-burden p-values to (a) evaluate modules built from GWAS and networks without OT labels, and (b) act as a second genetic seed channel.
- Nothing published uses ExWAS genes systematically as an independent **module** benchmark (A4).
- Data: download.
  - Genebass gene-level burden results, https://app.genebass.org / `gs://ukbb-exome-public/500k/results/` [m]. Gene-level tables are about 10-30 GB, estimate.
  - Or AZ PheWAS portal v5 gene collapsing download, https://azphewas.com, about 1-5 GB, estimate.
  - OT `gene_burden` evidence from the 26.09 evidence dump (≈100-200 MB, https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.09/output/) is the quickest proxy. It is absent locally; only `gwas_credible_sets` is in `evidence_2509`.

**5. A leakage-controlled text channel: disease-text query embeddings + gene text with disease sentences removed**
- A CLIP-style contrastive model aligns (a) a disease query built from MONDO/EFO definitions and synonyms (on disk) with (b) gene descriptions dated at or before T0 with every disease or phenotype mention removed.
- This allows zero-shot ranking for diseases with 0-2 seeds and directly measures how much of GenePT's gain is label leakage. That ablation has not been published.
- Data:
  - On disk: MONDO/EFO obo; local LLM to rewrite and strip text.
  - Download: GenePT summaries + embeddings, https://zenodo.org/records/10833191, about 1-2 GB [s].
  - Download: Qwen3-Embedding-8B weights (about 16 GB, HF `Qwen/Qwen3-Embedding-8B`); only stubs are cached.
  - Download: a dated NCBI gene summary snapshot (gene_info carries no summaries; Entrez E-utilities `esummary`, about 50 MB).

**6. Learning from discovery arrivals (temporal self-supervision)**
- Use the OT snapshots on disk (19.04, 21.04, 23.06, 25.09, 26.09) to train on "which genes were added between release t and t+1" rather than on random holdouts.
- The model learns what predicts future discovery, separated from what predicts current curation, which is study bias.
- Evaluate on 25.09 to 26.09 arrivals, by evidence class. modet's `arrival_*` tables already label genetic versus literature arrivals.
- Data: on disk (`ot_temporal`, `arrival_*`, OT 1904/2104/2306/2509/2609 assoc).

**7. Citation-debiased training and evaluation as a first-class objective**
- Add gene citation count as an adversarial nuisance (gradient reversal) or as a stratification covariate.
- Report AUPRC within citation deciles and on Wright's post-2023 GWAS sets.
- Combine with modet's degree- and promiscuity-matched null queries.
- Data: on disk (Wright citation counts, Dec 2023; PubTator3 gene2pubtator3 for dated counts; modet null queries).

**8. Conformal FDR-controlled module calls from null-calibrated scores**
- Turn any ranker's scores into conformal p-values against the matched null queries for the same disease (modet exports, 20 nulls per disease, about 3.8k per network). Apply Benjamini-Hochberg so the module is a set with a stated false-discovery guarantee.
- Optionally use Mondrian conformal by degree or citation stratum to fix conditional coverage, the failure noted in A11.
- Pass the called set to subex for connected-module extraction (PCST/SBM).
- Data: on disk (modet null_queries + oof scores). No download.

**9. Structure-predicted, literature-free interactome layer**
- Union of HuRI, Burke AF2 pDockQ re-scores (HuRI subset), Zhang 2025 RF2-PPI de-novo pairs and AFDB Foldseek cluster co-membership.
- Used as a propagation layer, or as a bias-free edge set inside idea 1.
- Tests whether unbiased experimental and predicted edges beat STRING once citation strata are controlled.
- Data:
  - On disk: HuRI, Burke S1, AFDB clusters.
  - Download: Zhang 2025 Dryad https://datadryad.org/dataset/doi:10.5061/dryad.15dv41p84 (≈1.6 GB of the listed files; needs a browser, so ask the user).

**10. Single-cell foundation-model gene embeddings against plain cell-type specificity, in disease-relevant tissues**
- Compute gene x cell-type specificity from Tabula Sapiens v2 (on disk, 29 tissues) as a strong cheap baseline.
- Compare it with frozen gene embeddings from TranscriptFormer (`pip install transcriptformer`, tf-sapiens weights about 1-4 GB, estimate) and Geneformer V2 (HF `ctheodoris/Geneformer`, about 1-2 GB).
- Feed both to idea 1. No published comparison exists for disease-gene ranking (A6).
- Data:
  - On disk: Tabula Sapiens v2, HPA single-cell, GTEx v8/v10, UCE ESM2 gene embeddings.
  - Download: model weights, ≈5 GB total.
  - GPU: about 8-12 GB via gpu-run for embedding passes.

**Suggested order** (cheap and on disk first): 8 → 6 → 7 → 2 → 1 → 9 → 10 → 3 → 5 → 4.

**Discipline that applies to every idea:**
- Label each result `temporal-clean`, `transductive` or `holdout-current`.
- Never use L2G/FLAMES/PoPS-full, PrimeKG, GenePT-raw or Rummagene-derived inputs in a `temporal-clean` claim.
- Never touch modet's sealed fold.

---

## Sources (all accessed 2026-11-12)

- Network propagation for GWAS guide (2024): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10858647/
- DADA degree-aware prioritisation: https://pmc.ncbi.nlm.nih.gov/articles/PMC3143097
- Propagation normalisation: https://www.frontiersin.org/articles/10.3389/fgene.2019.00004/text
- Under-studied proteins in link prediction: https://www.biorxiv.org/content/10.1101/2022.10.13.511953.full.pdf
- GRN sampling bias: https://www.biorxiv.org/content/10.64898/2025.12.19.695616.full.pdf
- Core disease genes / inspection bias: https://pmc.ncbi.nlm.nih.gov/articles/PMC10632370
- TxGNN: https://ai.stanford.edu/~jure/pubs/foundation-naturemedicine24.pdf
- OREVID preprint: https://www.preprints.org/manuscript/202610.1918
- BioMedKG: https://public-pages-files-2025.frontiersin.org/journals/systems-biology/articles/10.3389/fsysb.2025.1651930/epub
- PoPS: https://www.nature.com/articles/s41588-023-01443-6
- FLAMES: https://research.vu.nl/en/publications/prioritizing-effector-genes-at-trait-associated-loci-using-multim/ ; preprint https://medrxiv.org/content/10.1101/2023.12.23.23300360v2.full
- Open Targets L2G docs: https://platform-docs.opentargets.org/gentropy/locus-to-gene-l2g
- Open Targets 25.12 release: https://blog.opentargets.org/open-targets-platform-25-12-has-been-released/
- Open Targets 25.03 release: https://blog.opentargets.org/open-targets-platform-25-03-release/
- Open Targets L2G 26.06 models: https://huggingface.co/opentargets/locus_to_gene_26.06
- Open Targets gene burden: https://blog.opentargets.org/how-open-targets-integrates-gene-burden/
- Genebass paper copy: https://ibg.colorado.edu/cdrom2025/5.2.RareVariantAssociation_DuncanPalmer/refs/
- UKB final WES release: https://www.ukbiobank.ac.uk/learn-more-about-uk-biobank/news/final-data-release-from-the-world-s-largest-whole-exome-sequencing-project
- AZ 281k exomes: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8458098/
- Minikel 2024: https://ideas.repec.org/a/nat/nature/v629y2024i8012d10.1038_s41586-024-07316-0.html
- TranscriptFormer: https://www.biorxiv.org/content/10.1101/2025.04.25.650731.full.pdf ; https://biohub.org/blog/transcriptformer-model-overview/
- UCE: https://ai.stanford.edu/~jure/pubs/uce-bioarxiv23.pdf
- scFM pre-registered evaluation: https://arxiv.org/pdf/2608.26170
- PerturBench: https://arxiv.org/pdf/2408.10609
- PertEval-scFM: https://proceedings.mlr.press/v267/wenteler25a.html
- ESM C: https://evolutionaryscale.ai/blog/esm-cambrian
- AlphaGenome: https://pubmed.ncbi.nlm.nih.gov/41606153/ ; https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/
- Replogle 2022: https://ntc.hms.harvard.edu/publications/mapping-information-rich-genotype-phenotype-landscapes-genome-scale-perturb-seq
- CD4 Perturb-seq: https://www.biorxiv.org/content/10.64898/2025.12.23.696273.full.pdf
- DisMech CD4 report: https://dismech.monarchinitiative.org/elements/gwas-tcell-validation-report/
- X-Atlas/Orion: https://www.businesswire.com/news/home/20250617146938/en
- X-Atlas/Pisces: https://www.businesswire.com/news/home/20260317710096/en
- Tahoe-100M: https://www.biorxiv.org/content/10.1101/2025.02.20.639398.full.pdf
- LLMs at GWAS loci: https://www.medrxiv.org/content/10.1101/2024.05.30.24308179v1.full.pdf
- LLM gene prioritisation survey: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12263109/
- GenePT: https://www.biorxiv.org/content/10.1101/2023.10.16.562533.full.pdf ; https://zenodo.org/records/10833191
- GeneAgent: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12328209/
- Biomni: https://www.biorxiv.org/content/10.1101/2025.05.30.656746.full.pdf
- GSFM: https://pubmed.ncbi.nlm.nih.gov/40501705/
- DeepProphet2: https://ar5iv.labs.arxiv.org/html/2208.01918
- Conformal cell fate (NeurIPS 2026): https://neurips.cc/virtual/2026/169462
