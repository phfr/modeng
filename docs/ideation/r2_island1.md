# Round 2, fresh island 1: first principles + far analogies (2026-11-12)

Blind to measured round-1 results on purpose. Read: `brief.md`, `02_local_data_inventory.md`, `banned_r1.md`, the 73 titles in `docs/ideas.md`,
`implementer_brief.md`, `data/donors/MANIFEST.tsv`, `src/modeng/channels.py` (`donor_transfer`, `view_sim`). Nothing was run; every number below is a
prior, not a measurement. Web: 3 searches (sources at the end, accessed 2026-11-12).

## Step 1: eight assumptions and their flips

| # | Assumption the field makes | Flip |
|---|---|---|
| A1 | A module is a region of the **protein interaction** graph. | For a large class (inborn errors, mitochondrial, lipid, glycosylation) the module is a region of the **metabolite / flux** graph: two enzymes belong together because they carry the same flux, with no physical contact. -> F1-FLUX |
| A2 | One disease = one seed vector, matched to donors **one donor at a time** (`tr_max`, Jaccard to the whole seed set). | Seeds are a **sparse superposition** of a few mechanism atoms. Recover the atoms jointly (explaining away, additivity), as in compressed sensing, instead of taking independent matches. -> F1-CSDONOR, F1-ENDO |
| A3 | Prediction is **discriminative**: score(gene given disease). | Run it as an **inverse problem**: learn the forward map gene set -> phenotype description, then ask which added gene moves the predicted phenotype towards the disease (protein-design "hallucination"). -> F1-INVERT |
| A4 | Perturbation evidence comes from the **laboratory** (CRISPR, mouse). | Two huge perturbation screens ran **in humans, in vivo**: pathogens evolved proteins that bind exactly the host modules they need, and approved drugs hit known targets and produce recorded phenotypes (adverse events). -> F1-PATHOGEN, F1-ADR |
| A5 | Expression context is **adult, steady state** (GTEx, HPA: measured +0.000-0.007). | A module is switched on in an **organ x developmental window**; adult bulk tissue averages that window away. Timing is the missing coordinate, not tissue. -> F1-DEVTIME |
| A6 | The transfer kernel (Jaccard, top-50, linear sum) is a **fixed human design**; learning happens only in the stacker on top. | Transfer carries most of the signal, so the **kernel itself** is where search should happen: program search over kernel forms, scored on the 1,786 `past` episodes. -> F1-KSEARCH |
| A7 | Module membership is **binary and unsigned**. | Membership depends on dosage direction and on which partial mechanism a gene serves; a gene can be a member of one endotype and irrelevant to another. Output = a set of atoms with member lists, not one list. -> F1-ENDO, F1-CSDONOR (structure output) |
| A8 | Unreachable genes (no disease at T0) need **more gene features**. | They need **different rows**, not columns: sets defined by a process that never looked at disease literature (virus binding, flux coupling, developmental trajectory) so a dark gene has co-members. -> F1-PATHOGEN, F1-FLUX, F1-DEVTIME |

## Step 2: ideas

Common conventions: channel = `src/modeng/ext/<name>.py`, `CHANNELS = {name: fn(B, split_tag) -> float32 [n_tasks, 19255]}`, NaN where the source has
no coverage, gene order `modeng.data.universe()`. Decisive test = `python -m modeng.stack --bench <b> --channels base,donor,view,<new>` against
`base,donor,view` (temporal: `--train past`), paired with `modeng.compare`. "Expected effect" is the nuAP increment over that learned stacker.
Negative control for every idea: same channel with gene ids permuted within degree deciles (must give ~0).

### F1-FLUX: Flux phenocopy in a genome-scale metabolic model
- **Mechanism.** In Human-GEM, knock out each of the ~3,000 metabolic genes in silico and record the change of the flux-capacity vector (flux
  variability range per reaction, or simply the set of blocked reactions / lost metabolites). Score a gene by the similarity of its knock-out
  signature to the seeds' signatures, plus a metabolite-hypergraph proximity (shared non-currency metabolites, 1-2 reaction steps). Two enzymes at
  opposite ends of a pathway, with no PPI and different GO terms, get the same signature.
- **p_typical** 0.12 (Reactome donors partly cover it; only ~15 % of genes are in the model).
- **Data.** On disk: `/mnt/msadb/subex/raw/human_gem/Human-GEM-v1.7.0.zip` (check release date in README; if after 2021-04 the content is still
  textbook biochemistry, but report it as `transductive` on temporal unless an older tag, v1.6 or earlier, is downloaded from the Human-GEM GitHub
  releases). cobrapy via pip in a scratch venv.
- **Leakage / route.** Temporal-clean with a pre-2021-04 model tag; holdout-current otherwise. Route: channel + virtual donor family
  (`data/donors/gem_blocked.npz`: one set per metabolite = genes whose knock-out blocks its production).
- **Test (<= 2 h, CPU, 4 threads).** Single-gene deletion with FBA on the generic model is minutes; cheaper first pass without any solver: gene x
  metabolite incidence (currency metabolites removed by degree cut-off), cosine to seeds via `view_sim`. Stack as above; report separately the stratum
  "tasks with >= 3 seeds in the model" (expect ~10-15 % of tasks).
- **Expected effect.** +0.001-0.003 nuAP overall; +0.01-0.03 in the metabolic stratum; mostly rare diseases with 3-5 seeds.
- **Kill rule.** Alone, nuAP in the metabolic stratum not above `msig_reactome` donor transfer on the same tasks, or stack increment CI includes 0 in
  that stratum.
- **Likely failure.** Metabolic disease genes are the best annotated genes there are; Reactome / GO already place them, and the stratum is too small
  to move the overall mean above the 0.01 detectable effect.

### F1-CSDONOR: Compressed-sensing donor decomposition (joint sparse recovery instead of per-donor matching)
- **Mechanism.** Treat the seed indicator as a few noisy measurements of a sparse non-negative code over ALL donor rows at once (context diseases
  plus every virtual family in `data/donors`): solve `min ||W(s - M^T a)||^2 + lam*||a||_1, a >= 0` with seeds as ones and a random sample of
  non-seeds as zeros (columns scaled by degree^-0.5), then score `M^T a`. Unlike `donor_transfer` (independent Jaccard, top-50 sum) the solve
  explains away redundant donors, lets three partial donors add up, and penalises a donor for the non-seed genes it would wrongly predict.
  The non-zero atoms are the module's internal structure (which donors, which members), delivered for free.
- **p_typical** 0.08. Stated difference from the banned "EASE set completion": EASE learns one global item-item matrix; this is a per-disease
  donor-side sparse solve across heterogeneous families, with held-out-seed selection of lambda.
- **Data.** Only what the harness has: `B.context_for(task)`, `modeng.channels.load_family`. No download.
- **Leakage / route.** Temporal-clean (same inputs as base + pre21/seq donors; exclude families tagged `modern` on temporal). Route: channel
  `cs_donor`; lambda by leave-one-seed-out inside each task (no access to targets).
- **Test (<= 2 h).** Stack M (context rows + families, ~25k rows x 19,255, sparse); prefilter rows with >= 1 seed; `sklearn.linear_model.Lasso(positive=True)`
  or 200 steps of projected ISTA per task, 4 jobs. Works on `past` unchanged. Compare alone to `transfer_jaccard`, then stack. Second arm: elastic
  net (alpha 0.5) in case pure L1 is too sparse with ~11 seeds.
- **Expected effect.** +0.004-0.010 nuAP; the gain should sit in positives with LOW but non-zero transfer support (several weak donors), not in the
  zero-support stratum.
- **Kill rule.** Alone < `transfer_jaccard` - 0.01 nuAP AND stack increment < +0.003 on temporal confirm.
- **Likely failure.** With a median of 11 measurements the recovery is under-determined; L1 picks one large popular donor and the LightGBM stacker
  already approximates additivity across the family-wise transfer channels.

### F1-ENDO: Endotype unmixing: one transfer problem per seed sub-module, combined by noisy-OR
- **Mechanism.** A disease is a mixture of mechanisms (immune arm, barrier arm). Bicluster the seed x donor incidence matrix (which seeds does each
  donor contain) into 2-4 seed groups, run transfer separately per group with Jaccard computed against the GROUP, and combine with noisy-OR; a donor
  that contains 3 of 3 seeds of a minor arm now scores as a perfect match instead of 3 of 14. Output: the module as named sub-modules, each with its
  own donors and candidates.
- **p_typical** 0.07. Round-1 ideators named the mixture assumption (llmtext A4, evostruct A8) but the registered ideas act on text queries
  (LT-6) and selection signatures (EV-9); none changes the transfer kernel itself. subex found annotation Leiden the best decomposition: use it as
  the second grouping arm.
- **Data.** Harness only (+ `msig_go`, `msig_reactome` families for the annotation arm).
- **Leakage / route.** Temporal-clean. Route: channels `endo_max` (max over groups) and `endo_nor` (noisy-OR), plus `endo_k` (number of groups) as a
  task-level feature for the stacker.
- **Test (<= 1.5 h).** For each task: donor rows with >= 2 seeds -> binary seed x donor matrix -> spectral co-clustering or Leiden on seed-seed
  co-donor counts (k chosen by modularity, k = 1 allowed) -> `donor_transfer` per group -> combine. Negative control: random seed partition with the
  same group sizes (isolates the effect of splitting from the effect of smaller denominators).
- **Expected effect.** +0.003-0.008 nuAP, concentrated in tasks with >= 10 seeds and broad (parent-like) diseases; nothing for 3-5 seed tasks.
- **Kill rule.** Not better than the random-partition control by >= +0.003 nuAP in the stack on dev OOF.
- **Likely failure.** Smaller groups make Jaccard noisier (2-of-2 matches by chance against 25k donor rows); the max over groups raises false donors
  faster than true ones. A hypergeometric p-value in place of Jaccard is the planned rescue.

### F1-INVERT: Forward-model inversion ("hallucinate the gene that completes the phenotype")
- **Mechanism.** Train the FORWARD map, a permutation-invariant set encoder f(gene set) -> text embedding of the disease definition (Qwen3-Embedding
  on disk; names and definitions only), on all context diseases and virtual donor sets with their term names. At test time score each gene by the
  marginal gain `cos(f(seeds + g), e_disease) - cos(f(seeds), e_disease)`: the gene that makes the seed set "look more like this disease", not like
  its neighbours. The disease text enters as the target of an inverse problem, so a 3-seed disease is still constrained by what the disease IS.
- **p_typical** 0.06. Differs from ML-8 (disease encoder predicting future overlap), LT-8 (two-tower retrieval) and ML-5/ML-7 (set completion /
  diffusion): there the model outputs genes; here it outputs a phenotype and the genes come from inverting it.
- **Data.** Disease names / definitions: `/mnt/msadb/modet/downloads/ot2104` disease index (temporal) or `v2/ot2609_disease` (holdout).
  Gene inputs: `data/views/{esmc,go_t0,depmap20q4,phylo_npp}.npz`. Encoder weights: HF cache `Qwen3-Embedding-0.6B` (CPU is enough for ~25k short
  texts; if GPU: `~/biotools/bin/gpu-run --mem 4 -- ...`).
- **Leakage / route.** Temporal-clean if the text is the 21.04 definition and views are pre21/seq (the text encoder is a 2025 model: note as a mild
  transductive risk, test with bge-m3 as a second encoder). Route: channel `invert_gain`; training rows = `B.context_for(task)` with the task's
  cluster held out (5 disease-grouped folds, same as the stacker).
- **Test (<= 2 h).** DeepSets (mean + max pooling over a 256-d gene projection, 2-layer head, cosine loss, random subsets of 3-15 genes per row as
  augmentation), 5 folds, a few minutes each on CPU; scoring is one batched forward pass per task (19,255 sets). Control: shuffle disease texts
  across rows (must fall to the content-only baseline).
- **Expected effect.** +0.003-0.010 nuAP; should help 3-5 seed tasks and genes with zero transfer support, because the score does not need a donor
  that carries the gene.
- **Kill rule.** Shuffled-text control within 0.002 nuAP of the real model (text contributes nothing), or stack increment < +0.003 on confirm.
- **Likely failure.** The forward map is dominated by a few coarse axes (therapeutic area), so marginal gains reproduce "gene looks like the area"
  = what `view` channels already give; the difference of two cosines is also noisy for single-gene additions (mitigate: add g with weight 3).

### F1-PATHOGEN: Pathogen proteins as evolved probes of host modules
- **Mechanism.** Each viral or bacterial effector protein binds a small set of host proteins chosen by selection to control one host process
  (interferon signalling, cell cycle, trafficking). Use every pathogen protein (and every pathogen) as a virtual donor row of its host targets:
  a set defined by evolution and AP-MS, not by disease literature, which reaches immune, cancer and neurological genes that no disease row carried at T0.
  Gulbahce et al. 2012 showed host targets of EBV / HPV proteins lie in the network vicinity of genes of the virally implicated diseases [S1].
- **p_typical** 0.10.
- **Data.** On disk: `/mnt/msadb/subex/raw/intact/intact_2021-03-10_psimitab.zip` (filter taxid pairs human x non-human virus / bacteria). Optional
  download: HPIDB 3.0, VirHostNet 3 (use entries with publication year <= 2020).
- **Leakage / route.** Temporal-clean with the 2021-03-10 IntAct file (includes the 2020 SARS-CoV-2 maps: fine, dated before T0).
  Route: donor family `data/donors/pathogen_ppi.npz` (rows: per pathogen protein, per pathogen species, per virus family), consumed by the existing
  `donor_transfer`; channel name `don_pathogen_ppi`.
- **Test (<= 1 h).** Parse MITAB, map UniProt -> ENSG, keep rows with 3-500 host genes, write the npz in the MANIFEST format, run the stack with and
  without the family. Control: degree-matched random host sets of equal sizes (viral targets are hubs: this control is essential under nuAP).
- **Expected effect.** +0.001-0.004 nuAP overall; visible in infection-linked, autoimmune and cancer tasks.
- **Kill rule.** Increment not above the degree-matched control, or family gets < 0.5 % of stacker gain.
- **Likely failure.** Viral targets are high-degree, well-studied proteins; after the deg^-0.5 weighting and with `base` in the stack, what is left is
  PPI-degree information that modet already showed to be worthless.

### F1-ADR: Adverse-event phenocopy: approved drugs as human in-vivo perturbations
- **Mechanism.** A drug with known targets that produces a phenotype as a side effect (drug-induced parkinsonism, long QT, neutropenia, lupus-like
  syndrome) is a dose-controlled human perturbation experiment linking target -> phenotype. Build donor rows "side effect -> targets (and
  off-targets) of the drugs that cause it", weighted by how specific the target is to that side effect across drugs (target-level disproportionality),
  and match rows to the query disease both by seed overlap (as any donor) and by name / MedDRA-to-EFO mapping (works with zero seeds).
  Indications are deliberately excluded (they are the ChEMBL label source).
- **p_typical** 0.12.
- **Data.** Download SIDER 4.1 (2015, `meddra_all_se.tsv.gz`, 1,430 drugs, 5,880 side effects, CC BY-NC-SA) [S3]; targets from on-disk
  `/mnt/msadb/modet/downloads/ot_chembl_2104` or ChEMBL 28 mechanisms + binding (pChEMBL >= 6) for off-targets. Forward benchmark arm: OnSIDES /
  FAERS-derived tables (post-2021).
- **Leakage / route.** Temporal-clean (2015 labels, 21.04 targets). Route: donor family `data/donors/adr_targets.npz` -> `donor_transfer`; second
  channel `adr_named` (text match of side-effect term to disease name via bge-m3 cosine > threshold, then target specificity score).
- **Test (<= 2 h).** Join STITCH/PubChem ids -> ChEMBL -> targets, compute per side effect a target enrichment (Fisher, drugs with vs without the
  event), keep targets with q < 0.1 as the row; write npz; stack. Control: permute drug -> side-effect assignment preserving drug degree.
- **Expected effect.** +0.001-0.004 nuAP; the interesting stratum is zero-transfer-support positives in cardiac-rhythm, movement-disorder,
  haematological and metabolic tasks.
- **Kill rule.** Not above the permuted-drug control, or all gain disappears when genes that are ChEMBL targets for the SAME disease at T0 are removed
  from scoring (then it only re-reads indication labels).
- **Likely failure.** Only ~700 genes are drug targets, and they are the most studied genes: almost no dark genes are reachable; target promiscuity
  (GPCR panels, hERG) yields the same few genes for every side effect.

### F1-DEVTIME: Developmental window matching (organ x time as the module's address)
- **Mechanism.** Describe each gene by its trajectory across 7 organs x ~20 developmental stages (4 weeks post conception to old age) and summarise
  it as: organ of peak, time of peak, on/off switch time, trajectory shape (spline coefficients). Seeds define a window (e.g. cerebellum, late fetal);
  score genes by trajectory correlation with the seeds within the organs where the seeds are coherent, and add k-means trajectory clusters as a
  virtual donor family. Adult atlases cannot see genes whose disease-relevant expression ends before birth.
- **p_typical** 0.25 (expression context has failed repeatedly; time is the only new element).
- **Data.** Download: Cardoso-Moreira et al. 2019 human series, ArrayExpress E-MTAB-6814 (processed RPKM tables via the Kaessmann lab resource),
  Nature 2019, doi 10.1038/s41586-019-1338-5 [S2]; optional BrainSpan developmental transcriptome (2011-2014) for 16 brain regions.
- **Leakage / route.** Temporal-clean (2019 / 2014 data). Route: view `data/views/devtime.npz` (-> `view_sim`), donor family `km_devtime`, and a
  seed-coherence-gated channel `devtime_window` (score only in organs where seed-seed trajectory correlation exceeds a permutation null).
- **Test (<= 2 h).** Download (< 200 MB), median per organ-stage, log, z-score per gene, write view + k-means k=50/200/800 family with the existing
  script, stack. Report the neurodevelopmental / congenital-heart / kidney-malformation tasks separately.
- **Expected effect.** +0.000-0.003 nuAP overall; +0.005-0.015 in congenital and neurodevelopmental tasks if anything.
- **Kill rule.** Stack increment over `base,donor,view` (which contains `gtex`, `hpa_sc`) < +0.002 in the congenital / neurodevelopmental stratum.
- **Likely failure.** Developmental trajectories are highly correlated with adult tissue specificity and with constraint; the stacker already has
  both. Bulk organ data mixes cell types exactly as GTEx does.

### F1-KSEARCH: Program search over the transfer kernel (LLM-guided, fitness on past episodes)
- **Mechanism.** Define a small language for transfer kernels: donor weight = g(overlap, seed count, donor size, donor IDF, ontology distance,
  donor age, donor evidence mix), aggregation = (sum | max | noisy-OR | top-k power mean), gene-side normalisation = (none | degree^-a | donor-count
  IDF), second-order hops allowed. An LLM agent proposes kernel programs, sees only aggregate fitness (nuAP on the 1,786 `past` episodes, by stratum)
  and mutates the best ones (FunSearch-style evolutionary loop); the winners are frozen and evaluated once on temporal dev / confirm. The LLM never
  sees a gene or disease name, so memorisation cannot leak.
- **p_typical** 0.05 (kernel details usually wash out under a learned stacker; but transfer is where 70 % of the signal is and its form was never searched).
- **Data.** Harness only. LLM: local Qwen / gemma weights from the HF cache via `gpu-run --mem 24`, or no LLM at all for the first pass.
- **Leakage / route.** Temporal-clean by construction (search on 19.04 -> 21.04 episodes only; pre-register that confirm is touched once).
  Route: channels `ks_best1..3` replacing / joining `transfer_jaccard` in `base`.
- **Test (<= 2 h, no LLM).** Vectorise the kernel family on sparse matrices; random search + successive halving over ~300 configurations on a 300-
  episode subsample of `past`, validate the top 5 on the remaining `past` episodes, then one stack run on temporal dev. Only if the best kernel beats
  Jaccard top-50 alone by >= 0.01 nuAP on held-out `past` episodes is the LLM loop worth building.
- **Expected effect.** +0.003-0.008 nuAP in the stack; +0.01-0.02 for the transfer channel alone.
- **Kill rule.** Best searched kernel < +0.01 nuAP over `transfer_jaccard` alone on held-out `past`, or the stack gain on temporal dev < +0.003.
- **Likely failure.** Overfitting to the 19.04 -> 21.04 transition, whose arrivals are dominated by pipeline changes; LightGBM over several
  hand-made transfer variants already spans the useful part of the kernel family.

## Ranking by expected value per hour (own prior)
1. **F1-CSDONOR** and 2. **F1-ENDO**: no download, <= 2 h each, act on the dominant signal, give the module's internal structure as a by-product.
3. **F1-INVERT**: the only one here aimed at zero-support genes and 3-seed diseases through the disease description.
4. F1-KSEARCH (cheap no-LLM gate first), 5. F1-PATHOGEN (1 h, on disk), 6. F1-ADR, 7. F1-FLUX, 8. F1-DEVTIME (stratum-only hopes).

## Sources (accessed 2026-11-12)
- [S1] Gulbahce N. et al., Viral perturbations of host networks reflect disease etiology, PLoS Comput Biol 8(6): e1002531 (2012).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3386155 (abstract-level check only).
- [S2] Cardoso-Moreira M. et al., Gene expression across mammalian organ development, Nature 2019, doi 10.1038/s41586-019-1338-5; human data
  E-MTAB-6814: https://omicsdi.org/dataset/biostudies-arrayexpress/E-MTAB-6814 (processed-table file names not verified; check the Files tab).
- [S3] SIDER 4.1 (2015-10-21): https://zenodo.org/records/7877720 ; paper https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4702794/ ; download
  http://sideeffects.embl.de/download/ (licence CC BY-NC-SA 4.0; column layout not verified).
- Not verified by search (from memory): Human-GEM release dates, HPIDB / VirHostNet versions, FunSearch as the name of the LLM program-search method.
