# 05 Model families for "the best, most ambitious model" (research note, 2026-11-12)

Written 2026-11-12 by a research subagent. Web sources were accessed 2026-11-12. No installs; no GPU used; local checks are read-only.
Tags: **[v]** checked this session against the page itself; **[s, unverified]** from a search snippet or summary only;
**[m]** model background knowledge, not re-checked; **[local]** checked on this machine today.

**Starting point** (from `docs/reports/FINAL_overnight_2026-11-12.md`, `docs/insights.md`, `docs/invalidations.md`, `docs/ideas.md`):
- **Current best:** modeng-v1, a LightGBM stacker over about 20 witness channels.
- **What has already failed:**
  - seed-set transformer: memorised arrivals; its self-supervised twin reached transfer level and added nothing to the stack;
  - learned disease encoder (ML-8): mixed;
  - evolved transfer kernel: it only re-weighted degree;
  - gating, class-A head and context features did not move strict-a;
  - in modet, the GNN over PPI was no better than an MLP (F-10/F-27/F-80);
  - in modet, CatBoost, XGBoost and TabPFN did not beat LightGBM (F-70).
- **What the gains are made of:**
  - out of window, much of the gain is label-process forecasting (gwas_catlag, L2G priors);
  - the robust in-window gains come from disease-relational channels (ridge, cs_donor), literature and phenotypes.
- **Hard stratum:** strict-a is limited by T0 GWAS coverage, not by the learner.

**Consequence for this note:** a new model family is worth trying only if it either
- (a) brings a **new witness**, i.e. information the stacker's channels do not hold; or
- (b) turns **disagreement between witnesses** into a product: candidate new biology, or an honest "label-process vs biology" split.

Swapping the learner on the same features is not on the list: modet F-70 and the r3 gating runs already answer it.

## 0. Witness taxonomy (used by every family below)
The modeng-v1 channels (`docs/preregistration_modeng_v1.md`) fall into six witness classes:

| class | channels | what it knows |
|---|---|---|
| **R** relational / transfer | base transfer_*, dc_ridge(_pos), cs_donor, donor families (GO/Reactome/complex/embedding modules) | what related diseases already have |
| **L** literature | dA_pt, pt_ridge(_pos) (PubTator, PMID <= T0) | what papers co-mention |
| **P** phenotype | ph_mean, ph_top3, pri_hpo_n, HPO donor family | clinical-feature overlap |
| **G** genetics / label process | gwas_catlag, pri_l2g_*, hz_gwas | what the GWAS pipeline is about to attach |
| **M** Mendelian / curation | pri_mend_n, hz_mend | Mendelian prior |
| **B** gene biology | view_* (ESM C, ProtT5, GO, GTEx, perturbation, ...), rwr_string, degree, popularity | what the gene is like |

## 1. Families, one by one

### F1. Heterogeneous graph models over a date-safe knowledge graph (TxGNN, PrimeKG, path-based GNNs, graph transformers)

**What it would add over the stacker**
- The stacker sees the graph only through one-hop summaries: donor overlap, ridge, view similarity.
- A path-based KG model scores (disease, gene) through multi-relation paths, for example disease -> phenotype -> gene, disease -> drug -> target, or disease -> MeSH/literature -> gene -> pathway. That is a witness the channels do not compute.
- It is **inductive on the disease side**: a disease with zero seed overlap can still reach genes through its phenotypes, drugs or literature. This covers family F6.

**Evidence**
- **TxGNN** (Nat Med 2024): an R-GCN on PrimeKG with disease-similarity pooling. It reports +49.2 % on zero-shot indications. Its target is drug-disease, not disease-gene [s, unverified] https://ai.stanford.edu/~jure/pubs/foundation-naturemedicine24.pdf
- **TxGNN against a LightGBM ranker:** an independent 2026 preprint reports TxGNN zero-shot Recall@100 of 0.41 against 0.81 for a LightGBM ranker [s, unverified] (`01_sota` A2).
- **BioPathNet** (NBFNet-based, Nat Biomed Eng 2026, doi 10.1038/s41551-025-01598-z): beats shallow embeddings and R-GCNs on gene function, drug indication, synthetic lethality and lncRNA tasks. It gives path explanations and uses a PPI "background graph" [s, unverified] https://www.biorxiv.org/content/10.1101/2024.06.17.599219.full.pdf , https://idw-online.de/de/news864585
- **ULTRA** (ICLR 2024): a KG foundation model that transfers zero-shot to unseen graphs and relation vocabularies. On 57 KGs it is often on par with per-graph baselines [s, unverified] https://proceedings.iclr.cc/paper_files/paper/2024/hash/85dd09d356ca561169b2c03e43cf305e-Abstract-Conference.html
- **EchoLink** (NeurIPS 2026 poster): a heterogeneous graph transformer with sparse top-K attention on the Open Targets KG. It is "frozen on the 2021 release", reports Recall@100 +5.6 points over the best baseline and 28.6 % of future clinically connected targets in the top 100. The page names no baselines, gives no split details and does not discuss leakage [v] https://neurips.cc/virtual/2026/164238
- **COMET:** a metapath heterogeneous graph transformer for gene-disease prediction (arXiv 2501.07970). It initialises nodes with BioGPT, which is a text leak path [s, unverified] https://arxiv.org/pdf/2501.07970v1.pdf
- **Canastra & Pesquita 2025:** a systematic KGE evaluation for gene-disease association, finding that link-prediction methods beat node-pair classification [s, unverified] https://arxiv.org/abs/2504.08445
- **Temporal splits are rare:** biomedical KG link prediction is mostly evaluated on random transductive edge splits. No biomedical KG lets you recover the evidence subgraph as of a past cutoff (THBKG, arXiv 2608.05982) [s, unverified] https://arxiv.org/pdf/2608.05982 . Time-aware construction reduces leakage but does not remove it (MedKGent) [s, unverified] https://arxiv.org/pdf/2508.12393
- **Degree shortcut:** link-prediction benchmarks reward high-degree pairs, and a degree-corrected benchmark trains GNNs better (Aiyappa et al., ICML 2025) [s, unverified] https://proceedings.mlr.press/v267/aiyappa25a.html
- **Own prior, modet:**
  - RotatE on **PrimeKG-T0** alone reaches dev nuAP 0.109, against transfer tr_max 0.122 and modet-U 0.177 (F-46). The degree-permuted KG collapses to 0.02.
  - Gene-context KG embeddings added to the stacker give nothing (F-49).
  - **The disease-layer question ("version B") is still open (O-13).**

**Building a date-cut KG from on-disk sources** [local]
- **Start from modet's `primekg_t0` builder** (`/home/ph/code/modet/src/modet_bench/external/kge.py`):
  - PrimeKG v2 (sources about May 2021) with its DisGeNET `disease_protein` relation removed;
  - OT 21.04 genetic edges as `disease_protein_t0`;
  - OT 21.04 ontology `ot_parent` edges, and MONDO `same_as` links.
- **Add dated layers already on disk:**
  - HPO annotations 2021-01 (wayback): disease -> phenotype;
  - GO T0 and Reactome v72 without the disease branch: gene -> term;
  - STRING 11.5 physical: gene -> gene;
  - OT 21.04 ChEMBL / known-drug: drug -> target and drug -> indication. These are T0-dated and are not the genetic label;
  - PubTator3 disease-gene and MeSH co-mentions with PMID <= T0: disease -> literature -> gene. This needs PMID dates, which `pubmed_mesh` / `pubmed_status` can give;
  - GTEx v8 tissue nodes.
- **Leave out, or mark transductive:**
  - CELLxGENE and Tabula Sapiens 2024 cell types (modern, but not label-derived);
  - NCBI gene_info text (2026).
- **For forward:** the same builder with OT 25.09 edges.

**Compute**
- Size: PrimeKG-T0 has 4.3 M triples and 134 k entities.
- PyKEEN RotatE takes 400-2,000 s per config on this GPU [local, modet HPO table].
- NBFNet-style models run one message pass per query (head) over the whole graph. With batches of about 32 diseases, 6 layers and 32-64 dims, expect 8-16 GB VRAM ([m]; edge dropout or neighbour sampling if needed) and 2-6 GPU-h per training run. `gpu-run --mem 16`.
- Tooling: PyG 2.8 (`HGTConv`, RGCN) and PyKEEN are in `bio-net`; txgnn_env exists. NBFNet / BioPathNet / ULTRA code and weights are **not** on disk. NBFNet is about 300 lines in PyG; ULTRA checkpoints would be a download.

**Leakage traps**
1. PrimeKG layers that encode gene-disease knowledge indirectly:
   - DisGeNET (dated 2021-05, one month after T0, "DL1");
   - HPO disease-phenotype annotations, which are curated from the same OMIM gene papers;
   - drug indications whose target is the arrival gene.
   Every layer needs a dated source and its own ablation.
2. The own row and descendants: 287 temporal targets sit in the disease's own 21.04 row (modet propagates with the 25.09 ontology). These edges must be removed per query.
   - Path-based models can drop the query's edges at inference cheaply.
   - Shallow KGEs (RotatE) need fold-wise retraining, which is why version B is "fold-wise".
3. The training objective: reconstructing T0 edges teaches "what OT says", which includes label-process regularities.
   - Train on masked-reconstruction episodes, holdout-style, with ancestors re-propagated (the `test_holdout_no_mask_tell` lesson).
   - Never put 21.04 -> 25.09 arrivals into the graph.
4. Node features from text or LMs (BioGPT, GenePT) re-import modern literature, so they are transductive. Use ESM C / ProtT5 only.
5. Degree: KG degree tracks study intensity. Always run a degree-permuted KG control (modet's `primekg_t0_perm`) and report nuAP and degree-matched AP.
6. Ontology vintage: use the 21.04 ontology for temporal edges; modern MONDO merges collapse diseases.

**Cheapest decisive test (no GPU, about 1 h)**
- modet already stores **RotatE score matrices** for all 260 panel diseases on the STRING-T0 genes:
  - real KG: `/home/ph/code/modet/results/cache/kge/scores_kge_rotate_primekg_t0_t0_string115_phys_notm_400.npz` (12,743 genes x 260 diseases);
  - **degree-permuted control:** `..._primekg_t0_perm_...npz` [local].
- Import both as channels for the modeng temporal tasks only. Drop modet sealed ids; the modeng bench already excludes them.
- Add them to modeng-v1 on **temporal dev** (cv), plus the `past` bench if its diseases are covered.
- **Criterion:**
  - increment in AP AND nuAP with CI > 0;
  - the permuted control near 0;
  - the gain concentrated in the zero-transfer-support stratum.
- Caveats:
  - modet tuned these hyper-parameters on the same 53 dev diseases, so the result is mildly optimistic;
  - the KG puts panel seeds as edges, so the 287 own-row targets must be checked: drop those pairs from the evaluation.
- If the shallow KGE gives nothing beyond transfer, a path-based model is the second and last shot: one NBFNet/BioPathNet-style run on the date-cut KG with per-query edge masking, evaluated the same way.

**Kill rule**
- Kill if all of the following hold:
  - the RotatE channel adds < +0.005 nuAP or < +0.005 AP over modeng-v1 on temporal dev;
  - **and** one path-based run (<= 6 GPU-h) adds < +0.005 on both, on temporal dev plus past;
  - **or** the degree-permuted KG reproduces >= 80 % of the gain.
- If killed, record it as "KG disease layer = transfer" and close O-13.

### F2. Mixture of experts, specialist models per disease type, witness-subset ensembles, disagreement

**What it would add**
- On accuracy, little directly. The stacker with context features is already an implicit task-level mixture (r3 gate_ctx: confirm AP +0.006). Class-A heads and area flags overfit the 53 dev tasks.
- The new value is in two places:
  1. **Witness-partitioned experts.** Train one LightGBM per witness class (R, L, P, G+M, B), each blind to the others. Each expert's ranking is an independent "testimony".
  2. **Disagreement as output:**
     - genes ranked high by the biology and phenotype experts (B, P) but low by literature and label-process (L, G) are *candidate new biology*;
     - genes ranked high only by G are *label-pipeline forecasts*.
  This directly gives the no-label-process arm that the R3 red team asked to make co-primary, and a ranked "novel biology" list per disease. Neither exists today.
- **Learned gate:** a per-task softmax over experts from task context (n_seeds, seed evidence mix, own-GWAS flag, HPO coverage, therapeutic area).
  - Train it on the **`past` bench (about 1,786 tasks, 19.04 -> 21.04) plus forward dev (842)**, not on the 53 temporal dev tasks. That fixes the sample-size failure of E-B3 and r3.
  - Use Louvain leak folds (`settx.groups.leak_folds`) for any CV within past.

**Evidence**
- I found no published mixture of experts over disease categories for disease-gene prediction. Gene-MOE (pan-cancer RNA-seq, arXiv 2311.17401) routes samples, not diseases [s, unverified] https://arxiv.org/abs/2311.17401v3
- Stacking gains come from base learners with different inductive biases (H2O on TabPFN stacking) [s, unverified] https://h2o.ai/blog/2026/using-tabpfn-v2-in-h2o-driverless-ai/
- Ensemble disagreement as an uncertainty and novelty signal (deep ensembles, query-by-committee) is standard ML [m]. Using *witness-disjoint* experts so that disagreement means "different evidence" rather than "different seeds" is, as far as I found, not published for disease genes [s; search negative].

**Compute**
- CPU only: 5 experts x 5 folds of the existing stacker, about 20 GB RAM per stacker run ("one stacker per agent when RAM < 20 GB").
- About 1-3 h per bench on 8 threads. The gate is a small MLP or multinomial logit.

**Leakage traps**
1. Experts trained on same-window arrivals inherit the cv-training caveat. Report past-trained twins.
2. The G expert is label-process by design. Its "agreement" with the truth on forward is not biology; keep it out of any biology claim.
3. A disagreement list scored on the same benchmark used to pick the disagreement threshold is circular. Pre-register the threshold (for example the top 50 by rank(B,P) - rank(L,G) within the top 500 of the full stack).
4. The gate must not see target-derived features. Its inputs are T0 seeds and context only, which is already the r3 rule.

**Cheapest decisive test (CPU, about half a day)**
- **H1 (pre-register first):** among each disease's top-500 of modeng-v1, genes with high D = rank_L,G - rank_B,P (i.e. favoured by biology and phenotype) are enriched for arrivals that had **no** pre-T0 literature co-mention with the disease family (PubTator, PMID <= T0) and no sub-threshold genetic evidence. Compare the top D-tercile against the bottom one.
- **H2:** the gated mixture of witness experts is at least as good as modeng-v1 on the no-label-process arm (AP and nuAP).
- Run on temporal dev, past and forward dev. Confirm has been burnt (59 variants), so do not use it.

**Kill rule**
- Kill the "discovery" product if, on two of three benches, the top D-tercile does not show precision on no-prior-evidence arrivals >= 1.5x the bottom tercile with CI > 1.
- Kill the gate if it loses > 0.005 AP or nuAP to modeng-v1 on the no-label-process arm.
- The experts themselves are cheap and stay as features either way.

### F3. LLM-based approaches: agents or judges over time-locked evidence, and text embeddings

**Local models** [local, HF cache resolved through symlinks]
- **Complete:**
  - Qwen3-Embedding-0.6B (1.2 GB);
  - harrier-oss-v1-0.6b (1.2 GB);
  - bge-m3 (2.3 GB);
  - MedCPT **Cross**-Encoder (0.44 GB). The query and article encoders are not present;
  - **Qwen3.8-27B-INT4** (18.6 GB + 0.85 GB MTP). It is served by `llm-serve` at about 23 GB VRAM, about 64 tok/s single and about 230 tok/s with 8 parallel (`~/biotools/README.md`);
  - **gemma-4-31B-it-NVFP4** (23.3 GB). Serving it on vLLM 0.31 / sm_120 is unverified and would take nearly the whole GPU;
  - ATHENA-R1-Qwen3-8B (16 GB).
- **Missing:** Qwen3-Embedding-8B has config and tokenizer only, no weights (about 16 GB to download).

**What it would add**
1. **A literature-reasoning witness** that is richer than PubTator co-mention counts: mechanism, model-organism and experimental statements, as in F2-7 and LT-6.
2. **Text-embedding towers** for genes and diseases, which give a disease representation for zero-overlap diseases (F6).
3. **A judge** that re-ranks the top-k with an explanation.

**Evidence**
- At GWAS loci, LLMs are competitive with L2G-type methods and add value when integrated. Accuracy grows with how well studied the gene is (except for nearest gene and L2G). Contamination was handled with loci published after 2023-04-30 plus unpublished 23andMe loci (Shringarpure et al., medRxiv 2024 / PSB 2026) [s, unverified] https://www.medrxiv.org/content/10.1101/2024.05.30.24308179.full.pdf , https://d2ygbh7lwzeoju.cloudfront.net/psb-online/proceedings/psb26/shringarpure.pdf
- Rare disease: GPT-4 ranks the causal gene first about 30 % of the time, with bias toward well-studied genes and sensitivity to input order (Neeley et al. 2025) [s, unverified] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12263109/
- Text embeddings: GenePT's newer backbones gain 1-17 % on gene-property, interaction and perturbation tasks; disease tasks were not tested (ICLR 2026) [s, unverified] https://www.iclr.cc/virtual/2026/10021585 . Small open embedders are viable substitutes [s, unverified] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12359258/
- **Time-locked LLMs exist:** ChronoBERT/ChronoGPT, one model per cutoff year from 1999 to 2024, built for finance. Lookahead bias there was "modest" and application-specific [s, unverified] https://arxiv.org/abs/2502.21206 . An instruction-tuned follow-up exists [s, unverified] https://arxiv.org/html/2510.11677v1 . No biomedical time-locked LLM was found.

**Compute**
- **Judge** (Qwen3.8-27B, 1-token logprob verdict on a dossier of about 1.5 k tokens): prefill-bound, roughly 3-5 k tok/s [m]. That is about 1 h per 10 k pairs: enough for the top-50 of about 200 tasks, not for all genes.
- **Embeddings** (0.6B): 19 k gene documents plus about 8 k disease documents in under 10 min at about 4 GB VRAM.

**Leakage traps (severe)**
1. Qwen3.8 and Gemma 4 are 2026 models. Their pretraining surely covers papers behind most 21.04 -> 25.09 arrivals and many 25.09 -> 26.09 arrivals. **Neither the temporal nor the forward bench can evaluate a generative LLM witness.** Only a prospective arrival set dated after the model's release (the next OT release, already candidate fresh fold 1 in the pre-registration) can.
2. Text embedders also encode post-T0 associations whenever the *input text* is modern: NCBI and UniProt summaries say "mutations cause X". Build input documents only from dated material:
   - genes: pre-T0 GO terms, pre-T0 MeSH terms of their PubTator PMIDs, ESM-free;
   - diseases: MONDO / OT 21.04 definitions and HPO 2021 phenotypes, with gene symbols stripped.
   The embedder's weights are still 2025-2026, so add a **gene-identity-masked control** (symbols replaced by random tokens) and a **shuffled-document control**.
3. **Memorisation probe before anything else:**
   - ask the LLM, with no evidence, to list the genes for disease d;
   - compare its recall of 19.04 -> 21.04 arrivals with its recall of 21.04 -> 25.09 arrivals, both normalised by how well studied the genes are;
   - if the recall of "future" arrivals is at least that of past ones, or above modeng-v1, the model knows the answers.
4. Popularity: LLM scores correlate with citation count. Report the result by citation decile, by nuAP and by degree-matched AP.

**Cheapest decisive test**
- **(a) Embedding witness (GPU about 1 h, `gpu-run --mem 6`):**
  - set up a two-tower setting with Qwen3-Embedding-0.6B gene and disease vectors from dated documents;
  - channels: cosine(gene, disease text) and cosine(gene, seed centroid);
  - test them in the stacker on temporal dev and past, with the identity-masked and shuffled controls.
- **(b) Generative judge:** only the memorisation probe now (about 1 h, `gpu-run --mem 23`, when no modet run is queued). If it passes, pre-register an anonymised-dossier judge (LT-5) for the prospective fold; do not tune it on existing benches.

**Kill rule**
- **Embeddings:** kill if the increment over modeng-v1 is < +0.005 AP or nuAP on temporal dev plus past, or if the masked control keeps >= 70 % of the gain.
- **Generative judge:** kill for retrospective use if the probe shows future-arrival recall >= 0.5x its past-arrival recall or > modeng-v1 top-50 precision. In that case keep it prospective-only.

### F4. Prior-fitted networks (TabPFN, TabICL) as the judge

**Local** [local]
- `tabpfn` venv: tabpfn 9.2.0 with **v3.5 weights** (`~/.cache/tabpfn/tabpfn-v3.5-20260909.safetensors`, 876 MB). The user accepted the licence on 2026-11-09 (modet `docs/manual_downloads_to_try.md`).
- Licence: non-commercial, research benchmark only.
- TabICL is **not** installed (no install allowed today).

**What it would add**
- In-context learning is attractive in the small-n regime, where LightGBM overfits:
  - per-disease-type specialists (F2 experts with a few dozen tasks each);
  - a per-disease "context = neighbour-disease arrivals" learner (ML-6).
- As a general replacement of LightGBM it adds nothing.

**Evidence**
- TabPFN v2 (Nature 2025) is the best default up to about 10 k rows and 500 features [s, unverified] https://www.nature.com/articles/s41586-024-08328-6
- TabPFN-2.5 handles 50 k rows and 2,000 features [s, unverified] https://arxiv.org/abs/2511.08667
- TabICL handles 500 k rows [s, unverified] https://arxiv.org/abs/2502.05564 ; TabICLv2 scales to million-row tables [s, unverified] https://arxiv.org/html/2602.11139
- **Own prior:** modet F-70. The TabPFN v3.5 pilot had nuAP -0.024, reproduced by an *unweighted* CatBoost, so the deficit is the missing degree weighting, not the model class.

**Compute**
- The stacker's training table is about 53 x (targets + 3,000 negatives), roughly 160 k rows, with about 40 features. Subsample to 50 k rows; inference over 19 k genes x tasks is batched. `gpu-run --mem 10`, minutes per fold.

**Leakage traps**
- TabPFN has no sample weights. Emulate deg^-0.35 by resampling positives; otherwise it loses on nuAP by construction (F-70).
- The context set must come from training tasks only (cluster folds).
- Choosing among PFN variants on confirm is the winner's-curse trap of the night.

**Cheapest decisive test (about 2 GPU-h)**
- Fit the modeng-v1 feature matrix with TabPFN v3.5 (positives resampled ∝ deg^-0.35, 50 k rows) on temporal dev cv and past.
- Then test the only place it could win: the F2 per-class experts with fewer than 300 tasks.

**Kill rule**
- Kill if it is within ±0.005 of LightGBM on AP and nuAP on both benches. This is the expected outcome: then the learner is not the bottleneck, consistent with F-70.

### F5. End-to-end neural set / listwise learners, self-supervised on gene-set families, then fine-tuned

**What it would add**
- In principle, one model that learns the set-completion structure across many gene-set families: a GSFM-like denoiser plus a seed-conditioned listwise head.
- In practice, ML-5 already showed two things:
  - training on arrivals memorises;
  - the self-supervised twin reaches transfer level, with no stacker increment, and about 70 % of its gain survives gene permutation (set structure, not biology).

**Evidence**
- GSFM (Ma'ayan lab, bioRxiv 2025, *Patterns* 2026): a denoising autoencoder on about 626 k Rummagene/RummaGEO sets. It beats transformer variants and includes a **date-cutoff validation** (train on sets up to a date, predict later discoveries) [s, unverified] https://www.biorxiv.org/content/10.1101/2025.05.30.657124.full.pdf , https://huggingface.co/maayanlab/gsfm-rummage/blob/main/README.md
- Rummagene sets come from paper supplements with PMC dates, so a **date-cuttable corpus** exists. It is not on disk.

**Compute**
- A GSFM-size denoiser trains in a few GPU-h at about 8 GB [m].

**Leakage traps**
- Supplement tables contain published disease gene lists, so the PMC date cut must be at or before T0.
- GSFM's released weights are trained on modern sets, so they are transductive and cannot be used for temporal claims.
- Fine-tuning on arrivals memorises (ML-5).

**Cheapest decisive test**
- Repair E-A5: residual (offset) training of `settx_auxonly` on top of the stacker's logit, with release-matched twins. Run on past and temporal dev.

**Kill rule**
- Same as ML-5: kill if the increment is < +0.005 on AP and nuAP, or if the gene-permuted control keeps >= 70 %.
- **Prior: low. Not in the top recommendations.**

### F6. Disease-side representation for diseases with zero gene overlap

**What it would add**
- About 60 % of temporal targets have zero transfer support (insights 06:00).
- Diseases whose seeds overlap no donor get nothing from R witnesses. Today only phenotype (HPO) and literature (dA_pt) rows reach them.
- A disease representation from **phenotypes, text definitions, ontology position, drugs and literature** gives donor weights without seed overlap. TxGNN's disease-similarity pooling is the reference design [s, unverified].
- This also enables a **seedless arm** (F2-5): predict a disease's genes from its description alone. That is a new product; the current bench requires at least 5 seeds.

**Evidence**
- INDIGENA (Hoehndorf group, Bioinformatics 2026) embeds phenotype ontologies so that unseen diseases can be scored [s, unverified] https://borg.kaust.edu.sa/articles/2026/06/01/new-publication-indigena-predicts-disease-gene-associations-phenotype
- BioTranslator (Nat Commun 2023): text-to-biology contrastive zero-shot, including phenotype-gene association [s, unverified] https://www.nature.com/articles/s41467-023-36476-2
- Own: the learned disease encoder ML-8 was "mixed". Ridge and HPO are what work today.

**How to realise it**
- Not as a separate family. It falls out of **F1** (inductive KG: diseases attach through phenotype, drug and literature edges) and **F3a** (disease text tower).
- Evaluate on the **zero-transfer-support stratum** and on a **seedless arm**: hide all seeds of temporal dev diseases and score against seeds plus targets, with transfer = 0 by construction.
- Kill rule: as F1 and F3a, restricted to that stratum.

## 2. Ranked recommendation

| rank | family | why first | cost | decisive test | prior of a real gain |
|---|---|---|---|---|---|
| **1** | **F2 witness-partitioned experts + disagreement (+ gate trained on past + forward dev)** | turns the night's main caveat (label process vs biology) into the product; gives the co-primary no-label-process arm and a "novel biology" list; CPU only | about half a day CPU | H1 enrichment of no-prior-evidence arrivals in the high-D tercile; H2 gate vs modeng-v1 on the no-label-process arm | medium for H1, low for accuracy |
| **2** | **F1 date-cut heterogeneous KG with a path-based inductive GNN (NBFNet / BioPathNet / HGT-style); covers F6** | the only family that brings a new multi-hop witness (phenotype -> gene, drug -> target, literature paths) and handles zero-overlap diseases; answers modet's open O-13 | about 1 h CPU (RotatE import) then <= 6 GPU-h | RotatE and permuted-RotatE channels from modet's cached scores on temporal dev; then one NBFNet run with per-query masking | low-medium (the shallow KGE alone is at transfer level) |
| **3** | **F3a dated-text embedding towers (Qwen3-Embedding-0.6B) for genes and diseases; covers F6. Generative judge only after a memorisation probe, prospective only** | a cheap new witness; the disease tower supplies zero-overlap diseases and the seedless arm | about 1 GPU-h + 1 h probe | stacker increment with identity-masked and shuffled controls; LLM future-vs-past arrival recall probe | low-medium (embeddings), unknown (judge) |
| 4 | F4 TabPFN v3.5 as a small-n specialist judge inside F2 | weights and licence already in place | about 2 GPU-h | parity check with deg-resampling | low (F-70) |
| - | F5 set / listwise end-to-end | parked: ML-5 negative; only the E-A5 residual repair | - | - | low |

**Order of work:**
- Run F2 first. It is pure CPU and can run while modet's runs hold the GPU.
- Run the F1 RotatE import alongside it (also CPU).
- Queue the F3a embedding and the F1 NBFNet run through `gpu-run`.

**Rules for every family:**
- Every decisive comparison above is written into a pre-registration and committed *before* it runs (project rule).
- Development uses temporal dev, past and forward dev only. Temporal confirm is spent.
- Claims wait for the fresh fold, the next OT release, with the no-label-process arm co-primary.

**The "most ambitious model"** is the composition of ranks 1-3:
- witness experts (R, L, P, G+M, B, plus a KG-path expert and a text-tower expert);
- a gate trained on about 2.6 k past and forward tasks;
- two outputs per disease: a calibrated module (as modeng-v1) and a disagreement-ranked "biology-only" list.

Its credibility rests on each new expert first passing its own kill rule above.

## Sources (all accessed 2026-11-12)
- TxGNN, Huang et al., Nat Med 2024: https://ai.stanford.edu/~jure/pubs/foundation-naturemedicine24.pdf [s]
- BioPathNet preprint: https://www.biorxiv.org/content/10.1101/2024.06.17.599219.full.pdf ; press release: https://idw-online.de/de/news864585 [s]
- ULTRA (ICLR 2024): https://proceedings.iclr.cc/paper_files/paper/2024/hash/85dd09d356ca561169b2c03e43cf305e-Abstract-Conference.html [s]
- EchoLink (NeurIPS 2026): https://neurips.cc/virtual/2026/164238 [v]
- COMET: https://arxiv.org/pdf/2501.07970v1.pdf [s]
- Canastra & Pesquita 2025: https://arxiv.org/abs/2504.08445 [s]
- THBKG temporal biomedical KG: https://arxiv.org/pdf/2608.05982 [s]
- MedKGent: https://arxiv.org/pdf/2508.12393 [s]
- Aiyappa et al., ICML 2025: https://proceedings.mlr.press/v267/aiyappa25a.html [s]
- Gene-MOE: https://arxiv.org/abs/2311.17401v3 [s]
- H2O TabPFN stacking: https://h2o.ai/blog/2026/using-tabpfn-v2-in-h2o-driverless-ai/ [s]
- Shringarpure et al., LLMs at GWAS loci: https://www.medrxiv.org/content/10.1101/2024.05.30.24308179.full.pdf ; https://d2ygbh7lwzeoju.cloudfront.net/psb-online/proceedings/psb26/shringarpure.pdf [s]
- Neeley et al. 2025: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12263109/ [s]
- GenePT Revisited (ICLR 2026): https://www.iclr.cc/virtual/2026/10021585 [s]
- Small open embedders for genes: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12359258/ [s]
- Chronologically consistent LLMs: https://arxiv.org/abs/2502.21206 ; follow-up https://arxiv.org/html/2510.11677v1 [s]
- TabPFN (Nature 2025): https://www.nature.com/articles/s41586-024-08328-6 [s]
- TabPFN-2.5: https://arxiv.org/abs/2511.08667 [s]
- TabICL: https://arxiv.org/abs/2502.05564 ; TabICLv2 https://arxiv.org/html/2602.11139 [s]
- GSFM: https://www.biorxiv.org/content/10.1101/2025.05.30.657124.full.pdf ; https://huggingface.co/maayanlab/gsfm-rummage/blob/main/README.md [s]
- INDIGENA: https://borg.kaust.edu.sa/articles/2026/06/01/new-publication-indigena-predicts-disease-gene-associations-phenotype [s]
- BioTranslator: https://www.nature.com/articles/s41467-023-36476-2 [s]
- Local: modet `paper/FINDINGS.md` F-10, F-27, F-46, F-49, F-70, F-80, O-13; `results/frontier_kge_summary.md`; `results/cache/kge/*.npz`; `~/biotools/README.md`; HF cache under `~/.cache/huggingface/hub`; `~/.cache/tabpfn`.
