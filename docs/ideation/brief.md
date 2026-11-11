# Problem brief for ideators (modeng, 2026-11-12)

## Task
Given a disease (an Open Targets / EFO disease id, its name, its ontology position) and a few known **seed genes**, rank all 19,255 human
protein-coding genes by how likely they belong to the disease's **disease module** (genes with genetic, clinical or functional evidence for the disease),
and ideally output the module itself (a gene set with a calibrated cutoff and its internal structure). Goal of this sprint: an **absolutely novel,
next-generation** model that clearly beats the current state of the art, using any modern data or tool (foundation models, perturbation atlases, LLMs,
biobank statistics, knowledge graphs, sequence models). No restriction to data released before a cut-off date: but a model is only credible if its
evaluation is not circular (see benchmarks).

## Benchmarks (built in `src/modeng`, results in `results/leaderboard.tsv`)
- `temporal` (temporal-clean, comparable with modet): seeds = Open Targets 21.04 genes; targets = genes newly associated by 25.09. ~190 non-sealed diseases;
  dev (53) / confirm. Inputs dated after 2021-04 can leak future knowledge into this benchmark: then the result counts as `transductive`.
- `forward`: seeds = OT 25.09; targets = new in 26.09 (one year of discoveries). Modern inputs (up to 2025-09) are fair here.
- `holdout`: current knowledge completion on OT 26.09: hide 20 % of a disease's genes, predict them (ontology relatives masked).
- Metrics: AP, nuAP (AP with positives weighted 1/degree: rewards finding non-hub genes), AUROC, recall@100, nDCG@100. Paired cluster bootstrap.

## What the predecessor project (modet) established (measured facts, STRING / devall unless noted)
1. **Transfer between related diseases dominates.** Score a gene by its association with diseases whose known genes overlap the seeds (`tr_max`):
   nuAP 0.122 alone; the headline LightGBM (`modet-U-cov-Lb`: transfer + gene covariates + network-propagation features) 0.177 dev / 0.168 devall.
   Transfer is specific co-association, not popularity (degree-preserving rewiring of the donor matrix kills 87-94 %).
2. **The protein network adds little beyond node degree.** GNN = MLP; dc-RWR / DIAMOnD / GenePlexus alone ~0.01-0.02 nuAP; dense STRING with text mining hurts;
   knowledge-graph embeddings (PrimeKG RotatE, T0-clean) reach 0.109 alone but add nothing on top of transfer; HumanNet/functional networks, PoPS-like
   ridge heads on 57k gene features, GTEx/HPA tissue context add +0.000-0.007.
3. **Success is decided by transfer support.** Positives with zero transfer support (no related disease carries the gene) are found 0.4 % of the time
   in the top 100, those with high support 69 %. Genes not associated with ANY disease at the seed date are essentially unreachable (within-pool AUC of
   the best method ~0.60-0.63).
4. **57 % of new positives are GWAS-only** (credible-set / L2G evidence); only 16 % of those are found in the top 100. 82 % of the strictly-new
   (first evidence 2022+) positives are GWAS-only: their arrival is mainly a statistical-power event (bigger studies). Own-trait sub-threshold
   GWAS z (before significance) has AUC ~0.8 for later positives in 4 diseases.
5. **Label quirks:** ~40 % of "new" positives in 25.09 were old evidence (ClinVar/curated records dated <= 2020) newly attached, identifier migration
   accounts for a sizable share, 18 % are re-scoring of the same evidence. Ontology propagation means genes of a child disease appear in the parent.
6. Popularity/degree biases are strong: AP rewards hubs, nuAP penalises them. Seeds are few (median ~11), positives per disease ~20-30.
7. A minimum detectable effect between two good models on ~70-100 diseases is ~0.01 nuAP.

## Resources available locally (details: `docs/research/01_sota_and_resources.md` when written)
RTX 5090 32 GB GPU (shared, queue via gpu-run), 32 cores, 90 GB RAM, 680 GB disk. Envs with ESM C, Boltz-2, scanpy/scvi, PyG, PyKEEN, CELLxGENE Census,
AlphaGenome, Borzoi, gget, LightGBM. Data on disk: OT 19.04/21.04/25.09/26.09 label sets + ontologies, STRING/BioGRID/HuRI/HumanNet/BioPlex networks,
MAGMA gene z for many GWAS traits (pre- and post-2021), GWAS catalog, GTEx v8, HPA, DepMap 21Q1 / 20Q4, Perturb-seq K562/RPE1 (subex), GO/Reactome
(T0 and current), ESM C embeddings, gnomAD constraint, IMPC, MSigDB, PrimeKG, PoPS features, CORUM, Complex Portal, cS2G, Minikel 2024, Duffy 2024 GPS.
Downloads of any size are allowed (Open Targets platform/genetics full releases, UKB/FinnGen summary stats, CELLxGENE, scGPT/Geneformer weights,
LLM text embeddings of gene summaries, etc.).

## What counts as "absolutely novel / next-gen"
Not another feature column in the same LightGBM. A different *problem formulation*, *learning paradigm* or *information source* that could plausibly
move the hard strata: zero-transfer-support genes, GWAS-only arrivals, genes unknown to any disease, rare diseases with 3-5 seeds. Ideas must still be
testable on the benchmarks within hours to a day on this machine.
