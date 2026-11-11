# R1 ideation — lens: single-cell and perturbation biology (2026-11-12)

Baseline numbers used throughout are from `docs/ideation/brief.md`: `tr_max` alone nuAP 0.122, headline transfer LightGBM
nuAP 0.177 dev / 0.168 devall (temporal, STRING); zero-transfer-support positives reach the top 100 in 0.4 % of cases; 57 % of new
positives are GWAS-only (16 % found); minimum detectable effect ~0.01 nuAP. `results/leaderboard.tsv` did not exist when I looked
(`results/` empty), so no modeng-harness numbers are quoted.

## What I inspected (read-only)
- `/mnt/msadb/subex/derived/ps_*.npy`: perturbation x readout-gene signature matrices. K562 genome-wide (9852 x 8248), CD4 T 2025
  (12315 x 8871; rest 12264 x 8690; stim48hr separate), HepG2 and Jurkat (2391 x 9624 / 8882), RPE1 essential (2390 x 8749),
  K562 essential; pc10 / systema variants. Target tables: `ps_cd4_2025_targets.parquet` (on-target significance and effect size per
  perturbation), `raw/perturbseq/targets.parquet`.
- Coverage against the modeng universe (`modeng.data.universe()`, 19,255 genes): **13,165 genes perturbed in at least one atlas**
  (CD4 12,258; K562 gwps 9,820; HepG2/Jurkat/RPE1 ~2,380 each); **11,166 genes measured as readout** in CD4/HepG2/Jurkat bulk files.
  So about 6,100 universe genes have no perturbation row at all: any perturbation idea needs a fallback for them.
- `depmap20q4.npy` (17,518 genes x 946 lines; GLS / PCA / robust-PCA variants), `raw/depmap_current/chronos26q1_gene_effect.csv`,
  `expr_hpa_sc.npy` (19,484 genes x 154 single-cell clusters; HPA version not checked, subex holds both v20 and v23),
  `raw/tabula_sapiens_v2/*.h5ad` (per-tissue files, 0.2-4.9 GB each, 54 GB in total per the inventory).
- Harness: `modeng.tasks.load(bench)`; dev sizes: temporal 53 tasks (median 38 seeds, 23 targets), forward 842, holdout 303.
  Context matrix rows: 3,700 / 6,164 / 7,799 diseases. Baselines in `src/modeng/methods/baselines.py` (`transfer_jaccard`,
  `transfer_onto`, `modet_headline`).

Dating note that applies to every perturbation idea: Replogle K562/RPE1 (2022), Nadig HepG2/Jurkat (2025) and the CD4 atlas
(preprint Dec 2025) are all dated after 2021-04, so on `temporal` they are **transductive by the brief's rule**, even though they are
label-free experiments. The CD4 atlas is also dated after the `forward` seed date (2025-09); it contains no disease labels, but this
must be declared. Only DepMap 20Q4/21Q1 is strictly T0-clean.

---

## Step 1 — eight assumptions of transfer LightGBM and the baselines

| # | Assumption | If it is false |
|---|---|---|
| A1 | **A gene must already be a disease gene somewhere.** Evidence flows only through the disease x gene matrix, so the donor universe is "diseases". | Donors can be any experiment that produces a specific gene set (a CRISPR screen, a knockdown signature, a cell-state program). Genes unknown to any disease get transfer support for the first time. |
| A2 | **Related diseases are those that share known genes** (seed overlap picks the donors). | Two diseases can share a cellular locus (same cell type, same state program) with zero shared genes. Donor choice by cell context opens support for zero-support genes and for rare diseases with 3-5 seeds. |
| A3 | **A gene is one static, context-free vector.** | The same gene is a regulator in one cell type and silent in another. Scoring must be routed to the cell context of the disease; pooled features (GTEx, HPA, PoPS) wash this out, which is consistent with their +0.000-0.007. |
| A4 | **Seeds and targets are related by similarity** (guilt by association, undirected). | The relation is causal and directed: targets are upstream regulators of the seed program, or its convergent downstream effectors. Similarity in any embedding misses regulators, which look nothing like their targets. |
| A5 | **The module is a flat, unordered set**; the output is a ranking. | A module has layers (regulators -> core program -> effectors) and a natural size. A model that represents the layers can output structure and a cutoff, not only ranks. |
| A6 | **Evidence has no sign.** Loss of function, gain of function, risk and protection are all "associated". | Direction-consistent evidence (knockdown moves the program the same way as loss-of-function burden moves the trait) is a much sharper filter than unsigned co-membership. |
| A7 | **All positives arrive by the same process**, so one scorer fits all. | GWAS-only arrivals (57 %) are power events on common regulatory variants; they sit in cell-type state programs and trans-regulatory hubs, not in the Mendelian/clinical neighbourhood that transfer encodes. They need their own scorer. |
| A8 | **Absence of a label means absence of knowledge**; the only disease-specific input is the seed set. | Disease-specific measurements exist that are not gene labels: sub-threshold GWAS/burden statistics, case-control single-cell states, trans-QTL effects on seed genes. A disease can be described by data, not only by its seeds. |

---

## Step 2 — ideas

### SC-1
- **id / title:** SC-1 — Virtual Perturb-seq screen with the seed module as readout
- **mechanism:** For each disease, treat the seed set as the phenotype of an in-silico pooled screen: for every perturbed gene g,
  compute the competitive enrichment of seeds in g's knockdown signature (upstream score: "g regulates the module"), and for every
  readout gene the consensus response across seed knockdowns (downstream score: "the module converges on g"). The atlas is chosen
  per disease by where its seeds are expressed (CD4 for immune, HepG2 for hepatic/metabolic, K562 for erythroid/myeloid, RPE1 for
  epithelial/ciliary), so the score is directed and context-routed rather than a similarity. Output is a three-layer module
  (regulators / seeds / effectors).
- **p_typical:** 0.20
- **data:** on disk, `/mnt/msadb/subex/derived/ps_{k562_gwps,cd4_2025_rest,cd4_2025_stim48hr,hepg2_nadig,jurkat_nadig,rpe1_ess}.npy`
  plus `*_bulk.npz` (readout gene ids) and `expr_hpa_sc.npy` for routing. **Leakage class:** transductive on temporal (atlases dated
  2022-2025), temporal-clean on forward for K562/RPE1/HepG2/Jurkat, holdout-current on holdout. **Main route:** none through labels
  (no disease annotation enters); the residual route is library design (essential-gene libraries) correlating with studiedness.
- **minimal test (<= 2 h, CPU):** `score(tasks, B)`: load the five matrices once (float32, < 1 GB), z-score columns; per task build
  the seed indicator on readout genes; upstream = `(X @ s - n_s * X.mean(1)) / (X.std(1) * sqrt(n_s))` per atlas; downstream =
  mean of rows of seed perturbations; route by cosine of the seed mean HPA-sc profile to each atlas' cell type; rank-average with
  `transfer_jaccard`. Run on temporal dev (53), forward dev (842), holdout dev (303); matrix products only, ~10 min. Report nuAP/AP
  overall, on zero-transfer-support positives, and on the routed subset (immune, hepatic, blood) versus the unrouted rest.
- **expected effect:** alone nuAP 0.03-0.05 (above RWR's 0.01-0.02 because it is not degree-driven); fused with transfer
  **+0.010 nuAP, AP +0.003** overall; on routed immune/hepatic/blood diseases +0.02-0.03; recall@100 of zero-support positives from
  0.4 % to 2-3 %. **Kill rule:** fused gain < +0.005 nuAP on forward dev AND zero-support recall@100 < 1.5 %.
- **most likely failure:** knockdown signatures in five cell lines are dominated by generic stress and proliferation programs, so
  the upstream score re-ranks ribosome/mediator/proteasome genes for every disease; and 6,100 genes have no perturbation row.

### SC-2
- **id / title:** SC-2 — Experiments as donor "diseases" in the transfer matrix
- **mechanism:** Keep the mechanism that is proven (specific co-association transfer) but change the donor universe: append rows
  to the disease x gene matrix that are experimental gene sets — every Perturb-seq knockdown's top responders and each readout
  gene's top regulators, DepMap lineage-selective dependency sets, cell-type and cell-state marker sets, and published genome-wide
  CRISPR screen hit lists (BioGRID ORCS). A disease whose seeds overlap "hits of a FOXP3-regulator screen" or "responders to
  knockdown of gene X" then transfers that row's other genes, including genes no disease has ever carried.
- **p_typical:** 0.08
- **data:** on disk for a first version: the `ps_*.npy` matrices (about 30k rows after thresholding at |z| top 50-200 per
  perturbation), `depmap20q4_rpca.npy`, `expr_hpa_sc.npy`. Download for the full version: BioGRID ORCS all-screens archive,
  https://downloads.thebiogrid.org/BioGRID-ORCS (a few hundred MB; ~2,000 curated screens with phenotype text and publication
  date). **Leakage class:** temporal-clean if restricted to DepMap 20Q4 + ORCS screens published <= 2021-04; transductive with the
  2022-2025 atlases; fair on forward. **Main route:** ORCS screens designed around a known disease gene set (focused libraries);
  use genome-wide libraries only and drop screens whose phenotype text names the task disease or a relative.
- **minimal test (<= 2 h, CPU):** build a sparse experimental donor matrix E (rows x 19,255) from on-disk data only; call the
  existing `_transfer(tasks, B, k=50)` logic with context `vstack([B.context_for(t), E])`, with row weights IDF-normalised so 30k
  experimental rows do not swamp 3,700-7,800 disease rows; compare `transfer_jaccard` versus augmented, and run the modet control
  (degree-preserving rewiring of E must kill the gain). ~30 min. Ablate by donor family.
- **expected effect:** **+0.010-0.015 nuAP** over `transfer_jaccard` on forward/holdout dev, AP ~0 (experimental rows carry few
  hubs); zero-support positives recall@100 0.4 % -> 3 %; the unknown-to-any-disease pool AUC from ~0.60-0.63 to >= 0.66.
  **Kill rule:** < +0.005 nuAP on forward dev, or the rewiring control keeps more than half of the gain.
- **most likely failure:** Jaccard between a 30-gene seed set and experimental rows is almost always 0 or 1 gene, so donor
  selection becomes noise; the experimental rows that do match are the generic ones (cell cycle, translation).

### SC-3
- **id / title:** SC-3 — Cell-address donor selection for diseases without gene overlap
- **mechanism:** Give each disease a "cellular address": the enrichment vector of its seeds over single-cell clusters and, within
  the top clusters, over state programs. Transfer then draws donors by address similarity instead of seed overlap, so a
  3-seed podocyte disease borrows from every podocyte disease even with no shared gene. This relaxes exactly the condition that
  defines the zero-support stratum.
- **p_typical:** 0.10
- **data:** `expr_hpa_sc.npy` (19,484 x 154; on disk) for the first test; `raw/tabula_sapiens_v2` (54 GB) or CELLxGENE Census
  pseudobulk for the finer version. **Leakage class:** temporal-clean if the HPA v20.1 single-cell table is used (verify which
  version built `expr_hpa_sc.npy`); otherwise transductive on temporal. **Main route:** the donor matrix itself (same as transfer),
  already handled by `B.context_for(task)`; none added by the expression atlas.
- **minimal test (<= 2 h, CPU):** specificity z per gene across clusters; disease address = mean z of seeds minus a
  degree/expression-matched null; address computed for all context rows from their own genes with the task's hidden pairs masked;
  score = address-cosine-weighted top-k donor vote; fuse with seed-overlap transfer by max. ~15 min for all three dev sets. Report
  by seed-count bin (3-5, 6-15, > 15) and for zero-support positives.
- **expected effect:** overall **+0.005 nuAP** (most diseases already have overlapping donors); in the <= 5-seed bin and for
  zero-overlap-support positives +0.02 nuAP and recall@100 from 0.4 % to ~4 %; AP +0.002. **Kill rule:** no bin gains >= +0.01
  nuAP and zero-support recall@100 stays < 2 %.
- **most likely failure:** an address built from few seeds is noisy and collapses to broad tissue classes (immune, brain, liver),
  so the donors are just "all immune diseases" and the vote is a popularity prior within a tissue — the same result as the banned
  tissue-context weighting.

### SC-4
- **id / title:** SC-4 — Signed regulator-to-program-to-trait model from burden statistics
- **mechanism:** Following the regulator -> program -> trait causal graph of Ota et al. (Nature 2025), learn for each trait the
  signed weight of each expression program from genome-wide loss-of-function burden effects, then score every perturbed gene by
  its signed knockdown effect on those programs. The disease is described by sub-threshold exome statistics over all genes
  (assumption A8) and the match is direction-consistent (A6), so a gene with no disease annotation and no significant test can
  still score high.
- **p_typical:** 0.15
- **data:** Perturb-seq matrices on disk; programs = cNMF or the existing `*_pc10` variants. Burden statistics: Genebass
  (394,841 UKB exomes, ~4,500 phenotypes, pLoF burden), https://app.genebass.org/downloads (hosted as Hail tables on cloud
  buckets; full gene-level results are tens of GB, a per-phenotype pLoF extract for ~100 mapped traits is < 1 GB). MAGMA z on disk
  (`modet/data/processed/gwas_magma`, `gwas_magma_t0`) is the unsigned fallback. **Leakage class:** transductive on temporal
  (Genebass 2022); temporal-clean on forward. **Main route:** the burden statistics of the task trait are the same signal that
  later creates OT burden evidence (legitimate forecasting on forward, circular on holdout — mask genes with own-trait burden
  p < 1e-4 there).
- **minimal test (<= 2 h, CPU):** restrict to dev diseases with a mapped quantitative or binary trait in the on-disk MAGMA set
  (unsigned first, no download): ridge of gene z on K = 60 regulator-effect program loadings (rows of the routed atlas), leave-one-
  chromosome-out; score = out-of-fold prediction; compare with own-trait raw z and with transfer. Then repeat signed with Genebass
  for blood/immune/lipid traits. ~40 min.
- **expected effect:** on the GWAS-only arrival subset of routed diseases recall@100 16 % -> 20 %; **+0.005-0.010 nuAP** fused on
  that subset, ~0 overall on unmapped diseases; AP ~0. **Kill rule:** out-of-fold prediction adds < 0.02 AUC over raw own-trait z
  for later positives.
- **most likely failure:** this is PoPS with perturbation features, and PoPS-like heads gave +0.000-0.007; burden effects are
  powered for only a few hundred genes per trait, and Ota et al. needed a trait-matched cell type (HSPCs for blood traits) to make
  it work.

### SC-5
- **id / title:** SC-5 — Trans-QTLs as an in-vivo, genome-wide perturbation atlas
- **mechanism:** Every cis-regulatory variant is a natural knockdown of its cis gene, and trans-eQTL/trans-pQTL maps record what
  that perturbation does to other genes in human tissue. Run the SC-1 virtual screen on this natural atlas: score a gene by whether
  its cis variants shift the disease's seed genes or seed proteins in trans. The perturbed genes are those with common regulatory
  variation — the population from which GWAS-only arrivals are drawn — and the perturbation happens in primary human blood, not in
  a cancer line.
- **p_typical:** 0.06
- **data:** eQTLGen phase 1 trans-eQTLs (31,684 blood samples; public since the 2018 preprint), https://www.eqtlgen.org
  (significant trans file ~10 MB, full ~60 GB); UKB-PPP plasma pQTLs (2,923 proteins, Sun et al. 2023 supplementary tables,
  ~50 MB). cis-gene assignment from `cs2g` on disk. **Leakage class:** eQTLGen is temporal-clean by date; UKB-PPP is transductive
  on temporal, clean on forward. **Main route:** eQTLGen phase 1 tested only ~10k trait-associated SNPs for trans effects, so
  "having a trans-eQTL" already encodes GWAS knowledge of 2018; and the cis-gene assignment may use curated biology. Control:
  gene-level permutation within matched GWAS-hit-count bins.
- **minimal test (<= 2 h, CPU):** build sparse T (cis gene x trans gene, signed z); score = hypergeometric / z-sum enrichment of
  seeds among the trans targets of g, plus the reverse (g is a trans target of seed cis variants); fuse with transfer. Evaluate on
  forward and temporal dev, stratified GWAS-only versus other arrivals and immune/blood versus rest. ~20 min after download.
- **expected effect:** GWAS-only recall@100 16 % -> 19 % on blood/immune diseases; overall **+0.003-0.006 nuAP**, AP +0.002.
  **Kill rule:** GWAS-only recall@100 gain < 2 points on immune/blood dev diseases, or the matched permutation retains the gain.
- **most likely failure:** sparsity — a few hundred trans hubs (and mostly in blood); for most disease/gene pairs the score is zero.

### SC-6
- **id / title:** SC-6 — Knockdowns that phenocopy the diseased cell state
- **mechanism:** Describe the disease by data instead of seeds: a case-versus-control signature per cell type from CELLxGENE
  Census (datasets carry a disease ontology label). Match each Perturb-seq knockdown signature against it, gene-level
  connectivity-map style: genes whose loss reproduces the disease state are candidate causal genes with a predicted direction;
  genes that are merely differentially expressed are treated as effectors, not causes.
- **p_typical:** 0.15
- **data:** CELLxGENE Census via the installed env (pseudobulk by donor x cell type; a few GB per disease, 20-40 benchmark diseases
  expected to have case/control data — to be counted first); Perturb-seq matrices on disk. **Leakage class:** transductive on
  temporal unless datasets are filtered by publication date <= 2021-04 (Census records the publication); clean on forward with
  <= 2025-09 datasets. **Main route:** Open Targets has an expression-evidence data type, so differential expression can be the
  very evidence that creates a label; score only with the knockdown-match term or exclude expression-only labels.
- **minimal test (<= 2 h, CPU):** pick the 10 dev diseases with the most case/control donors in immune or hepatic cell types;
  pseudobulk DE (donor-level, limma-like) -> signature d; score_g = cosine(knockdown signature of g, d) in the matching atlas (CD4
  or HepG2); compare with DE-rank alone and with transfer on those 10 tasks. ~1.5 h, dominated by Census queries.
- **expected effect:** on covered diseases **+0.010 nuAP** fused (new, label-free information), ~0 elsewhere so overall
  +0.001-0.003; AP ~0. **Kill rule:** on the 10 pilot diseases the knockdown-match score has AUROC < 0.55 for targets, or does
  not beat the DE rank.
- **most likely failure:** case-control signatures are consequences of disease, treatment and sampling, not causes; coverage is
  too small for the benchmark to detect anything (10-40 diseases).

### SC-7
- **id / title:** SC-7 — Set-completion model pretrained on experimental gene sets
- **mechanism:** Reformulate the task as few-shot set completion and pretrain it on a corpus that contains no disease label:
  ~50k experimental sets (responders per knockdown, regulators per readout gene, state programs per cell type, lineage dependency
  sets, co-essential clusters). A set transformer over fixed gene tokens learns "what completes a functional set"; disease seeds
  are then just another prompt, optionally fine-tuned on historical disease sets. Unlike a LightGBM on features, the prior over
  set structure comes from experiments, so it is not bounded by which genes the literature has touched. (EASE-style set completion
  is on the banned list; the new reason is the label-free experimental pretraining corpus.)
- **p_typical:** 0.10
- **data:** on disk (Perturb-seq, DepMap, HPA-sc, Tabula Sapiens); gene tokens initialised from ESM C embeddings
  (`modet/data/processed/features`). **Leakage class:** temporal-clean only for the DepMap-only corpus; transductive on temporal
  with the atlases; clean on forward. **Main route:** fine-tuning on disease sets must use `B.context_for(task)`; pretraining has
  no label route.
- **minimal test (<= 2 h, GPU via `~/biotools/bin/gpu-run --mem 8 -- python ...`):** 4-layer set encoder, masked-member
  objective with degree-matched negatives, 30 min pretraining; zero-shot scoring of the dev tasks, then a 15-min fine-tune on
  context rows; compare zero-shot, fine-tuned and fused with transfer.
- **expected effect:** zero-shot nuAP 0.03-0.04; fused **+0.005-0.010 nuAP**, AP ~0; unknown-gene pool AUC +0.03. **Kill rule:**
  zero-shot nuAP < 0.025 (no better than RWR) or fused gain < +0.005 on forward dev.
- **most likely failure:** modet's "GNN = MLP" result generalises: the model learns degree and expression level, and experimental
  sets differ too much in size and nature from disease sets for the prior to carry over.

### SC-8
- **id / title:** SC-8 — Within-cell-type state programs as the unit for GWAS arrivals
- **mechanism:** Common-variant genes are rarely cell-type markers; they are members of state programs (activation, interferon
  response, lipid handling) that vary between cells of one type. Factorise within-cell-type residual variation in Tabula Sapiens
  into programs, represent a disease by the programs its seeds load on in its addressed cell types, and score genes by loading on
  those programs. This is a different axis from the between-tissue specificity that failed (GTEx/HPA +0.000-0.007).
- **p_typical:** 0.25
- **data:** `/mnt/msadb/subex/raw/tabula_sapiens_v2/*.h5ad` (54 GB; work per tissue, subsample 20k cells per cell type).
  **Leakage class:** transductive on temporal (Tabula Sapiens v2 is dated 2024); clean on forward and holdout. **Main route:** none
  through labels.
- **minimal test (<= 2 h, CPU, check `gpu-run status` and machine load first):** four tissues (Blood, Liver, Lung, Pancreas);
  per cell type regress out depth and donor, NMF with k = 20 on 5k variable genes; pooled ~600 programs; disease loading = seed
  enrichment per program (competitive, expression-matched); gene score = loading-weighted membership; fuse with transfer; stratify
  GWAS-only versus others. ~1.5 h.
- **expected effect:** GWAS-only recall@100 16 % -> 18 %; overall **+0.003-0.005 nuAP**, AP ~0. **Kill rule:** < +0.003 nuAP
  fused on forward dev and no GWAS-only gain >= 1.5 points.
- **most likely failure:** within-cell-type programs from sparse 10x counts are mostly technical or generic (ribosomal,
  mitochondrial, stress); only variable genes get a score, so coverage of the universe is ~25 %.

### SC-9
- **id / title:** SC-9 — Module cutoff and layers from perturbational coherence
- **mechanism:** Use the perturbation atlas as an external, label-free judge of where a module ends: walk down any ranking and
  track the coherence of the growing set (mean pairwise similarity of knockdown signatures, and the mutual regulation density
  among members) against a degree- and expression-matched null; cut where the marginal coherence falls to the null. The same
  matrix orders members into regulators and effectors, so the output is a sized, layered module rather than a ranked list — the
  part of the task statement no ranking metric rewards.
- **p_typical:** 0.07
- **data:** Perturb-seq matrices and `depmap20q4_gls.npy` on disk; any score matrix from the harness (e.g. `transfer_jaccard`).
  **Leakage class:** inherits that of the ranking; the cutoff adds only the atlas dating (transductive on temporal, clean on
  forward). **Main route:** none through labels; the cutoff never sees targets.
- **minimal test (<= 2 h, CPU):** for each dev task take the transfer ranking, compute cumulative coherence for k = 1..500 over
  the genes that have a perturbation row, choose k* by the null-crossing rule; compare module F1 and precision at k* with a fixed
  k = 100 and with the oracle k; also use coherence-with-seeds as a re-ranker of the top 500. ~20 min.
- **expected effect:** as a cutoff, nuAP and AP unchanged by construction; module F1 +10 % relative versus fixed k = 100 and
  Spearman(k*, true number of positives in the top 500) >= 0.3. As a re-ranker of the top 500: **+0.003 nuAP**. **Kill rule:**
  Spearman < 0.15 and F1 no better than the best fixed k.
- **most likely failure:** disease modules are not perturbationally coherent in a cell line (they span cell types and pathways),
  so the coherence curve is flat and k* is arbitrary; one third of ranked genes have no perturbation row.

---

## Bets
- **I would bet on SC-2 (experiments as donor diseases).** It keeps the one mechanism modet measured as working (specific
  co-association transfer) and removes its hard ceiling — a gene must already be a disease gene — with data that is already on
  disk; it has a decisive, cheap control (rewiring the experimental donor matrix). SC-1 is its natural partner: SC-1 is the
  directed, context-routed special case and tells us whether the atlases carry disease-specific signal at all.
- **Most underrated: SC-5 (trans-QTLs as an in-vivo perturbation atlas).** It is the only idea here whose "perturbed" genes are
  exactly the population that GWAS-only arrivals come from (82 % of strictly-new positives), in primary human tissue, and eQTLGen
  phase 1 is old enough to be temporal-clean. Its weakness is sparsity, which is an empirical question a 20-minute test settles.
- Honest prior: perturbation atlases cover five cell contexts and 13,165 of 19,255 genes, and modet found that almost nothing adds
  to transfer. I expect at most two of these nine to clear +0.005 nuAP; the stratified read-outs (zero-support, unknown-gene pool,
  GWAS-only) matter more than the overall number.

## Sources (all accessed 2026-11-12)
- Ota et al., "Causal modelling of gene effects from regulators to programs to traits", Nature 2025 (preprint Jan 2025):
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11785173/
- Genome-scale Perturb-seq in primary human CD4+ T cells (bioRxiv Dec 2025; 22 M cells, 4 donors, 3 conditions):
  https://www.biorxiv.org/content/10.64898/2025.12.23.696273.full.pdf ; GEO record GSE314342 via https://www.omicsdi.org/dataset/geo/GSE314342
- Genebass (Karczewski et al. 2022, 394,841 UKB exomes, ~4,500 phenotypes): https://pmc.ncbi.nlm.nih.gov/articles/9903662 ;
  downloads https://app.genebass.org/downloads (file layout and presence of burden betas not verified by me)
- X-Atlas/Orion genome-wide Perturb-seq, HCT116 and HEK293T, 8 M cells, ~521 GB processed, non-commercial licence (possible extra
  cell contexts for SC-1/SC-2): https://www.biorxiv.org/content/10.1101/2025.06.11.659105.full.pdf ;
  https://doi.org/10.25452/figshare.plus.29190726
- TranscriptFormer (cross-species single-cell foundation model; weights and embeddings via the `transcriptformer` CLI) — looked
  at, not proposed, because generic gene embeddings fall under the banned "embedding as features" result:
  https://www.biorxiv.org/content/10.1101/2025.04.25.650731.full.pdf ; https://pypi.org/project/transcriptformer/
- Not searched, cited from memory and to be verified before use: eQTLGen phase 1 (https://www.eqtlgen.org), UKB-PPP (Sun et al.
  2023), BioGRID ORCS download site and its screen count, HPA v20.1 release date, sizes of all downloads marked "~".
