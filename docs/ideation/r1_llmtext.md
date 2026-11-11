# Round 1 ideation: lens "language models, literature and biocuration" (2026-11-12)

Sources for claims about our data: `docs/ideation/brief.md` (= "brief"), `docs/research/02_local_data_inventory.md` (= "inventory"),
`src/modeng/tasks.py` docstring (= "tasks.py"), and files I inspected read-only (marked "inspected"). `docs/research/01_sota_and_resources.md`
and `results/leaderboard.tsv` did not exist when I wrote this, so all baseline numbers are the brief's (transfer LightGBM nuAP 0.177 dev / 0.168
devall, `tr_max` alone 0.122; MDE ~0.01 nuAP). Expected effects below are my estimates, not measurements.

## What I found on disk that matters for this lens (inspected)
- **A date-stamped sentence corpus already exists**: `/mnt/msadb/modet/downloads/ot_evidence/evidence_2104/sourceId=europepmc` (3.4 GB, 200 parquet
  parts). First part: 52,770 rows, 6,859 genes, 2,972 diseases, one PMID per row, and `textMiningSentences` with the sentence text plus character
  offsets of the gene span (`tStart/tEnd`) and the disease span (`dStart/dEnd`). Extrapolated ~10 M gene-disease-PMID rows (not counted). It is the
  OT 21.04 release, so everything in it predates the temporal benchmark's seed date. The 25.09 evidence folder holds only `gwas_credible_sets`.
- `uniprot_sprot-only2021_02.tar.gz` contains `uniprot_sprot.dat.gz` (function / disease / tissue comment text as of Feb 2021).
- `hpo_annotations_2021-01_wayback/genes_to_phenotype_ALL_SOURCES_20210115.txt` (gene, HPO term name, HPO id).
- `/mnt/msadb/subex/raw/ncbi/gene2pubmed.gz`: 3,355,029 human gene-PMID rows (current snapshot; needs a PMID/date cut for T0 use).
- `~/biotools/data/evidence/raw/pubtator3` (3.1 GB): `gene2pubtator3.gz` (758 MB), `disease2pubtator3.gz` (2.2 GB), `relation2pubtator3.gz` (300 MB),
  PMID-level, so date-filterable. Also current `hpo` (hp.obo, phenotype.hpoa, genes_to_phenotype), `mondo.obo`, `orphanet`, `gencc`, `mesh` (2025
  descriptors), `clinvar` (inventory section "~/biotools/data/evidence/raw").
- HF cache: Qwen3-Embedding-0.6B / 8B, bge-m3, bge-reranker-v2-m3, MedCPT cross-encoder, Qwen3.8-27B-INT4, gemma-4-31B (inventory says: check
  that weights are complete). No pre-2021 encoder (PubMedBERT, SapBERT) is cached; both are ~440 MB downloads.
- Not checked: whether a gene with only a weak europepmc co-mention at T0 counts as "unknown (weak evidence) excluded" in the temporal panel
  (tasks.py says such genes are excluded). This decides how much of the own-disease literature signal is even rankable; every test below must
  report it.

## Leakage ground rules I use for text
Three separate routes, each needs its own control:
1. **Corpus date**: text written after the seed date can name the target disease. Control: only corpora with a release date or PMID cut at the seed date.
2. **Model weights**: an encoder or LLM trained after the seed date carries later associations in its parameters. Control for `temporal`: encoders
   with a training cut before 2021-04 (PubMedBERT 2020, SapBERT 2020), or chronologically consistent models (ChronoBERT / ChronoGPT have yearly
   vintages 1999-2024, https://arxiv.org/abs/2502.21206, accessed 2026-11-12; general-domain and small, so weak on biomedical text). Any modern
   LLM on `temporal` = `transductive`. On `forward`, a model with cutoff <= 2025-09 is a fair input (brief: inputs up to 2025-09 are fair there).
3. **Annotation tools**: PubTator3 tags were produced by taggers trained ~2022-23 (https://arxiv.org/abs/2401.11048, accessed 2026-11-12). They
   map mentions to ids; they cannot add a gene-disease pair that the dated paper does not contain. I treat this as clean, and say so.

---

## Step 1: eight assumptions of transfer LightGBM and the baselines, and what follows if each is false

| # | Assumption | If false |
|---|---|---|
| A1 | A gene's relevant knowledge is its row in the OT association matrix (which diseases already carry it). | Genes with no row have other dated knowledge (sentences, function text, phenotypes) that never reached a curated association. Then zero-support genes are reachable through text that is not about any disease. |
| A2 | Donor diseases must share seed genes with the target (similarity = gene-set overlap). | Diseases with 3-5 seeds have useless donors. If similarity can be defined from disease meaning (name, definition, phenotype profile), donors can be chosen with zero seeds, and the donor universe can include label spaces that are not diseases (HPO terms, MeSH headings, GO terms). |
| A3 | The label is a property of biology. | Fact 5 of the brief says ~40 % of "new" positives are old evidence newly attached, plus identifier migration and re-scoring. If a large share of arrivals is curation and mapping lag, the best predictor of that share is a model of the curation pipeline, not of biology. |
| A4 | A disease is one point (one seed set, one score vector). | Seed sets mix mechanisms (e.g. an immune arm and a barrier arm). If modules are mixtures, a single centroid or a single donor weighting washes out the minor mechanisms, exactly where new non-hub genes arrive. |
| A5 | Numeric features (expression, essentiality, constraint) can be learned from ~11 seeds per disease. | They cannot (PoPS-like ridge added +0.000-0.007). If the mapping "gene property -> disease relevance" is supplied as prior knowledge by a language model instead of being learned from 11 examples, the same data may carry signal. |
| A6 | The feature set is static at the seed date; the history before it does not matter. | Arrival is a process in time. If the 19.04 -> 21.04 transition (on disk: `ot_t19_sets.parquet`) looks like 21.04 -> 25.09, a model can learn "what a gene looks like shortly before it arrives" (rising literature, weak tiers, sub-threshold GWAS). |
| A7 | Evidence about the disease itself is fully used once it is in the seed set. | Sub-threshold evidence exists (own-trait GWAS z has AUC ~0.8, brief fact 4; weak co-mentions; "limited" curation tiers; unmapped free-text conditions). If it is systematic, there is a large pool of "almost positive" pairs that the binary seed set throws away. |
| A8 | Popularity (degree, number of diseases) is a nuisance to be controlled by a covariate. | In text, popularity is the amount of text. If text representations are built with equal budget per gene, the studiedness channel can be closed at the source, which is what nuAP rewards. |

---

## Step 2: ideas

### LT-1 Masked-context cloze: which disease does this gene's literature sound like
- **Mechanism**: Take the T0 sentences, replace the gene span by `[GENE]` and the disease span by `[DISEASE]` (offsets are in the data), and train a
  model to predict the masked disease from the remaining context. For a candidate gene, pool its masked sentences that are about *other, unrelated*
  diseases and score the target disease by how strongly those contexts predict it. This is transfer at the level of experimental and mechanistic
  context ("complex I assembly defect in patient fibroblasts") instead of the level of shared disease labels, so it can reach genes with zero
  transfer support as long as they have any sentence.
- **p_typical**: 0.05
- **Data**: `evidence_2104/sourceId=europepmc` (3.4 GB, on disk, inspected). Encoder: PubMedBERT (2020 weights, ~440 MB,
  https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext). **Leakage class: temporal-clean** on `temporal`.
  Main route: sentences that co-mention the target disease or an ontology relative with the target gene (legitimate T0 information, but it must be
  dropped per task so the idea is tested as context transfer and not as "weak evidence lookup"); second route: residual disease words left in the
  sentence outside the tagged span (e.g. an abbreviation), controlled by also masking all strings of the target family's names and synonyms.
- **Minimal test (<= 2 h)**: (1) `python scripts/lt1_build.py`: read 200 parts, keep <= 30 sentences per (gene, disease), sample 3 M sentences,
  write masked text + gene + disease id (CPU, ~15 min). (2) `~/biotools/bin/gpu-run --mem 8 -- python scripts/lt1_embed.py` frozen PubMedBERT,
  mean pool, fp16 (~3 M short sentences, ~20 min). (3) Train a linear + softmax head sentence -> disease on context diseases (dev diseases and their
  `fam_rows` removed from the label set; they are scored through a disease-name embedding tower instead, or through the nearest retained diseases
  weighted by the task's seed profile), 10 min GPU. (4) `score(tasks, ctx)`: per task, gene score = mean over the gene's sentences (excluding
  family-disease sentences) of predicted affinity to the seed profile; evaluate alone, and as `rank-average(transfer, LT-1)` on temporal dev.
  Report separately for the zero-transfer-support stratum.
- **Expected effect**: alone nuAP 0.05-0.08 (text covers fewer genes than the label matrix). Combined with transfer: **+0.010 to +0.020 nuAP**,
  AP +0.005; top-100 recall of zero-support positives from 0.4 % to 3-5 %. Why: the signal is orthogonal by construction (no shared labels), and
  mean pooling per gene with a sentence cap removes much of the studiedness advantage. **Kill rule**: combined gain < +0.005 nuAP on temporal dev
  AND zero-support recall@100 < 2 %.
- **Most likely failure**: the corpus only contains sentences that co-mention some disease, so a gene's contexts are dominated by its already known
  disease area and the head simply re-learns transfer; genes unknown to any disease may have too few sentences.

### LT-2 Semantic transfer over virtual donors (MeSH, HPO, GO terms as donor rows)
- **Mechanism**: Keep the mechanism that works (transfer) but change both of its limits: the donor universe becomes every dated label space that
  attaches genes to a named concept (PMID-level PubTator3 gene x MeSH disease co-occurrence cut at the seed date, HPO term -> genes 2021-01, GO T0
  terms, mouse phenotype terms), and donors are selected by *meaning* (text embedding of the concept name and definition against the disease name,
  definition and phenotype profile), blended with seed overlap when seeds exist. A disease with 3 seeds, or a gene that has a MeSH/HPO row but no
  OT row, becomes reachable.
- **p_typical**: 0.08
- **Data**: `~/biotools/data/evidence/raw/pubtator3/{gene2pubtator3,disease2pubtator3}.gz` (3 GB, on disk); PMID -> year from a PMID threshold
  (first PMID of 2021-04, conservative: cut at 2021-01) or PubMed baseline dates; `hpo_annotations_2021-01_wayback` (6.8 MB); GO T0 features
  (modet `features/`); `mesh/d2025.bin` for names (names only). Embeddings: SapBERT 2020 for the clean run, Qwen3-Embedding-8B (cached) for the
  forward run. **Leakage class: temporal-clean** with the date cut and a 2020 encoder; `transductive` if a modern embedder selects donors on
  `temporal`. Main route: the current PubTator3 dump contains post-T0 PMIDs (must be cut; test with a canary: no PMID above the threshold survives);
  second route: MeSH disease rows that are the target disease itself or a relative (map MeSH -> EFO/MONDO via xrefs and mask like `fam_rows`).
- **Minimal test (<= 2 h)**: (1) `zcat gene2pubtator3.gz | awk` PMID cut, join with disease2pubtator3 on PMID in DuckDB, count distinct PMIDs per
  (gene, MeSH) -> sparse matrix ~19k x ~5k, PMI-normalised (CPU, ~30 min; the 2.2 GB file is the slow part). (2) Embed ~5k MeSH + ~8k HPO names
  and the task disease names: `gpu-run --mem 6 -- python scripts/lt2_embed.py` (2 min). (3) `score`: for each task, donor weight = softmax(cosine /
  tau) x (1 + overlap term), score = weighted max / top-k mean over donor rows, exactly the `_transfer` form in `methods/baselines.py` with a new
  donor matrix and new weights. Evaluate alone and added to `transfer_jaccard`; stratify by number of seeds (<= 5, 6-20, > 20).
- **Expected effect**: **+0.010-0.015 nuAP** on temporal dev overall when combined; in the <= 5-seed stratum relative gain of 30 %+; AP roughly flat
  or slightly up (literature co-occurrence is hub-biased, PMI controls part of it). **Kill rule**: combined < +0.005 nuAP overall AND < +0.01 in
  the few-seed stratum.
- **Most likely failure**: brief fact 2 ("dense STRING with text mining hurts"): co-occurrence matrices are popularity matrices. If PMI and
  per-gene budget normalisation do not remove that, it is a degree feature in disguise.

### LT-3 Curation debt: predict what the pipeline will attach next
- **Mechanism**: Treat the label as the output of a curation and ontology-mapping pipeline (brief fact 5: ~40 % of new positives are old evidence
  newly attached, plus identifier migration). Build the "debt" at the seed date: (a) T0 evidence rows whose free-text condition
  (`diseaseFromSource` in ClinVar/EVA, UniProt, G2P, PanelApp rows) is a synonym, subtype or phenotype-level description of the target disease
  but was mapped to another id or to none; (b) weak tiers that precede promotion (ClinGen Limited, G2P limited, PanelApp amber/red, GenCC
  "Limited"/"Supportive"; category definitions: https://search.thegencc.org/faq, accessed 2026-11-12). A text matcher (embedding + cross-encoder)
  links these strings to the target disease more permissively than the OT mapper did.
- **p_typical**: 0.07 (the banned list has "identifier-repair label audit" as an evaluation audit; the new reason is to use the same phenomenon
  as a predictor, which was never run)
- **Data**: `evidence_2104/sourceId={eva,eva_somatic,uniprot_literature,uniprot_variants,gene2phenotype,genomics_england,clingen}` (on disk; the
  column `diseaseFromSource` exists in the europepmc part I inspected, to be confirmed for the others); UniProt 2021_02 `.dat` DISEASE comments;
  optional download: PanelApp panel version history via https://panelapp.genomicsengland.co.uk/api/v1/ (small, date-stamped versions).
  Current `gencc`/`clinvar` on disk are 2025-26 snapshots: usable only with their submission dates, and only for `forward`.
  **Leakage class: temporal-clean** when restricted to the 21.04 evidence files and a 2020 encoder. Main route: using a later ontology (26.09
  parent/child or merged ids) to decide what counts as "the same disease" imports the future mapping; use the 21.04 ontology and plain text only.
- **Minimal test (<= 2 h)**: (1) one DuckDB query over `evidence_2104` (non-europepmc sources): distinct (gene, diseaseFromSource, diseaseId,
  source, score) (~10 min). (2) Embed distinct strings (expected < 100k) and the 191 temporal disease names + synonyms with SapBERT
  (`gpu-run --mem 4`, 5 min); rerank top-20 matches with the cached MedCPT cross-encoder. (3) `score`: gene score = max over its T0 evidence rows
  of match(string, target disease) x source weight, for rows whose mapped id is outside the task's seed row. Evaluate alone and combined; report by
  arrival class using modet's `arrival_*.parquet` (does it find the "old evidence newly attached" class?).
- **Expected effect**: concentrated: recall@100 on the old-evidence-newly-attached class from the transfer level to 40 %+; overall
  **+0.010-0.025 nuAP, AP +0.01-0.02** on temporal dev (these positives are often non-hub Mendelian genes). **Kill rule**: < 10 % of that class
  in the top 100 by the debt score alone, or combined gain < +0.005 nuAP.
- **Most likely failure**: those rows already sit in a near-duplicate or relative disease row, so transfer already uses them (or `fam_rows`
  masking deliberately zeroes them). Then the gain is real only on the unmasked part and may be small. Also scientifically it predicts database
  behaviour, not new biology; it must be reported as its own stratum, not sold as discovery.

### LT-4 Equal-budget data-to-text dossiers for dark genes
- **Mechanism**: Write every gene's dossier in one fixed schema with a fixed token budget from non-literature data at the seed date (protein
  domains and family, ortholog descriptions and mouse phenotypes, top tissues and cell types, top co-essential genes, Perturb-seq neighbours,
  constraint, subcellular location), so a gene nobody studied gets as much text as TP53. Embed dossier and disease description with an
  instruction-tuned embedder and score by a small two-tower head trained on context diseases. The language model supplies the prior that a ridge
  on 57k numeric features could not learn from 11 seeds (A5), and equal budgets close the studiedness channel (A8).
- **p_typical**: 0.12
- **Data**: all on disk (inventory): GTEx v8, HPA v20.1, DepMap 20Q4/21Q1, IMPC 13.0, `ensembl103_compara`, UniProt 2021_02, gnomAD 2.1.1, Replogle
  Perturb-seq signatures, GO T0. Embedder: Qwen3-Embedding-8B (cached). **Leakage class: transductive on `temporal`** (inputs are T0, the embedder
  is not: it knows that a named co-essential partner is a later disease gene); **temporal-clean on `forward`**. A clean temporal variant uses a
  2020 encoder (weaker). Main route: the embedder's parametric knowledge of gene symbols appearing in the dossier (partners, orthologs); control:
  an ablation that replaces all gene symbols in dossiers by family/function phrases.
- **Minimal test (<= 2 h)**: `python scripts/lt4_dossier.py` (templating from existing feature tables, 19,255 x ~250 tokens, CPU 20 min);
  `gpu-run --mem 20 -- python scripts/lt4_embed.py` (19k dossiers + disease texts, ~10 min with the 8B model, or 0.6B in 1 min); two-tower ridge
  / bilinear head on context diseases (CPU 10 min). Run on `forward` dev (fair) and `temporal` dev (labelled transductive), alone and combined.
  Key readout: within-pool AUC for genes unknown to any disease (brief: best method 0.60-0.63).
- **Expected effect**: unknown-gene pool AUC 0.62 -> 0.68-0.70; overall combined **+0.005-0.012 nuAP**, AP flat. **Kill rule**: unknown-pool AUC
  < 0.65 and combined gain < +0.005.
- **Most likely failure**: it is the PoPS feature set in words; if the embedder adds no usable prior, the result repeats +0.00x.

### LT-5 Anonymised-dossier LLM judge with a measured memorisation gap
- **Mechanism**: A local LLM reads, for each of the top-300 candidates of a task, a T0 dossier (LT-4 text, T0 sentences from LT-1, sub-threshold
  GWAS z, which related diseases carry the gene) and returns a mechanism-based probability; scores are fused with the retriever's rank. Every
  candidate is judged twice, once named and once with the symbol replaced by a random code and identifying strings removed: the anonymous run is
  the claimable result, and named minus anonymous is a direct estimate of parametric-memory leakage that the whole sprint can reuse.
- **p_typical**: 0.30 for "LLM reranker"; 0.08 for the paired anonymised design as a leakage instrument
- **Data**: dossiers from LT-4 / LT-1; LLM: Qwen3.8-27B-INT4 or gemma-4-31B in the HF cache (inventory: verify weights). I do not know these
  models' training cutoffs; they must be looked up before any `forward` claim. **Leakage class: transductive** on `temporal` (named);
  the anonymous arm is "temporal-clean up to re-identification"; on `forward` clean only if the model cutoff is <= 2025-09. Main route:
  re-identification of the gene from its dossier (domain + tissue + partners identify many genes); control: ask the model to guess the gene from
  the anonymous dossier and report the hit rate, and stratify results by it.
- **Minimal test (<= 2 h)**: 20 temporal dev diseases x 300 candidates x 2 arms = 12,000 prompts of ~600 tokens, batched listwise (10 candidates
  per prompt -> 1,200 calls). `gpu-run --mem 26 -- python scripts/lt5_judge.py --model <cached 27B> --arms named,anon` with vLLM; at a guessed
  1.5-2k tokens/s batched throughput this is 60-90 min. Fuse by reciprocal rank with transfer; paired bootstrap on the 20 diseases (underpowered:
  it is a go/no-go, not a claim).
- **Expected effect**: named arm +0.02-0.04 nuAP (mostly memory); anonymous arm **+0.005-0.015 nuAP**, AP +0.005. Only the top 300 are touched,
  so zero-support genes outside it are not helped. **Kill rule**: anonymous arm < +0.005 nuAP on the 20 diseases, or re-identification rate > 50 %
  (then the arm is not clean and the idea has no claimable form on `temporal`).
- **Most likely failure**: once the name is removed, the LLM's judgement collapses to the features the LightGBM already has.

### LT-6 Mechanism cards: the module as a set of text queries
- **Mechanism**: Cluster the seeds in text-embedding space, have a model write one short "mechanism card" per cluster from the seeds' T0
  descriptions only (process, compartment, cell type, phenotype), and retrieve genes per card with late interaction (max over cards of
  card-dossier similarity, then a learned combination). The output is not just a ranking: each predicted gene is attached to a card, which is
  the module's internal structure asked for in the brief, and minor mechanisms are not averaged away (A4).
- **p_typical**: 0.15
- **Data**: UniProt 2021_02 function text + GO T0 term names for seeds; gene dossiers from LT-4 or UniProt text; cached Qwen3-Embedding and a
  cached LLM for card writing (an extractive variant needs no LLM: card = top TF-IDF terms of the cluster). **Leakage class: temporal-clean** for
  the extractive variant with a 2020 encoder; **transductive** with a modern LLM writer/embedder on `temporal`; clean on `forward` (cutoff
  caveat as LT-5). Main route: the card writer adds knowledge that is not in the supplied seed texts (e.g. names a pathway discovered later);
  control: the extractive variant, and a check that every card term occurs in the input.
- **Minimal test (<= 2 h)**: `python scripts/lt6_cards.py --variant extractive` (k = 2-5 clusters per disease by silhouette; CPU 10 min);
  embed cards and gene texts (`gpu-run --mem 6`, 5 min); `score` = max-sim over cards vs. baseline "mean seed embedding" (the comparison that
  isolates the multi-vector formulation); then combined with transfer on temporal dev. Compare cluster assignment with subex's GO/Reactome Leiden
  decomposition (banned list says it is the best decomposition; reuse it as the clustering if it wins).
- **Expected effect**: multi-vector vs. single-centroid text retrieval +20-30 % relative nuAP (alone, from ~0.04 to ~0.05); combined
  **+0.005-0.010 nuAP**; AP flat. **Kill rule**: multi-vector does not beat single-centroid by >= 0.005 nuAP alone, or combined gain < +0.005.
- **Most likely failure**: with a median of ~11 seeds (brief fact 6), clusters have 2-4 genes and cards are noise; function similarity to seeds
  is also what GO "function users" (banned, no gain) already tested in another form.

### LT-7 Arrival-hazard learner trained on the previous transition, with literature momentum
- **Mechanism**: Change the supervision: instead of learning "is this gene in the disease set", learn "what does a (disease, gene) pair look like
  shortly before it arrives", trained on 19.04 -> 21.04 arrivals and applied to 21.04 -> 25.09. Features are precursors that only make sense for
  this target: yearly counts and acceleration of co-mention PMIDs with the disease and with its semantic neighbours, first-mention year, weak-tier
  status, sub-threshold MAGMA z, number of other diseases the gene arrived in during the previous window.
- **p_typical**: 0.20 (rolling-origin windows are in the banned list as an *evaluation* idea; the new element is training on a past transition
  with time-series precursors from literature)
- **Data**: `groundtruth/ot_t19_sets.parquet`, `ot_t0_sets.parquet` (on disk, inspected by name); PubTator3 PMID-level files with a year cut
  (as LT-2); `gwas_magma_t0` (4.4 GB; 102 entries in the directory). **Leakage class: temporal-clean** if all precursor features use PMIDs <= the
  respective window start. Main route: off-by-one in the date cut (papers of 2021 used for the 21.04 seed date); use a hard cut at end of 2020 for
  the test window and end of 2018 for the training window. Second route: sealed diseases in the 19.04 -> 21.04 training rows: remove
  `data.excluded_task_ids()` there as well.
- **Minimal test (<= 2 h)**: reuse the LT-2 PMID-cut join with a year column (30 min); build per-(disease-neighbourhood, gene) yearly counts for
  candidate pairs (top-2000 per disease by transfer, to bound size); LightGBM on 19.04 -> 21.04 arrivals of non-dev, non-sealed diseases (CPU
  10 min); apply at 21.04; evaluate on temporal dev vs. the same model without momentum features (isolates the information) and vs. the
  transfer LightGBM (isolates the paradigm).
- **Expected effect**: **+0.008-0.015 nuAP**, AP +0.01 (momentum favours studied genes, so AP gains more than nuAP). **Kill rule**: momentum
  features add < +0.005 nuAP over the same hazard model without them.
- **Most likely failure**: 19.04 -> 21.04 arrivals differ in kind from 21.04 -> 25.09 (82 % of strictly new positives are GWAS-only, brief fact
  4, driven by study size, which literature momentum cannot see).

### LT-8 Time-locked two-tower text retrieval (the obvious baseline that must exist)
- **Mechanism**: Gene text strictly from dated sources (UniProt 2021_02 function + GO T0 names, optionally T0 sentences with all disease spans
  masked), disease text = name + definition + HPO profile; a two-tower contrastive head trained on context diseases; run once with a 2020 encoder
  (clean) and once with a modern embedder (transductive on `temporal`, fair on `forward`). The difference between the two runs measures how much
  of the "GenePT-style" gain is model-weight leakage (GenePT embeds current NCBI summaries with a hosted model and reports no disease-gene
  benchmark: https://pmc.ncbi.nlm.nih.gov/articles/PMC10614824, accessed 2026-11-12).
- **p_typical**: 0.60
- **Data**: `uniprot_sprot-only2021_02.tar.gz` (on disk), GO T0, `hpo_annotations_2021-01_wayback`; PubMedBERT/SapBERT download (~440 MB each);
  Qwen3-Embedding-8B cached. **Leakage class: temporal-clean** (2020 encoder arm) / **transductive** (modern arm). Main route: UniProt DISEASE
  and "involvement in disease" comment lines name diseases directly; they must be stripped for the function-only arm and reported separately.
- **Minimal test (<= 2 h)**: parse `.dat` (CPU 5 min), embed ~20k texts x 2 encoders (`gpu-run --mem 20`, 10 min), bilinear head (CPU 10 min),
  evaluate alone / combined on temporal and forward dev. Total < 45 min.
- **Expected effect**: alone 0.03-0.05 nuAP (clean arm), 0.06-0.09 (modern arm); combined clean **+0.003-0.008 nuAP** (probably below the MDE).
  Its value is as the reference every other text idea has to beat. **Kill rule**: none as a baseline; as a method, drop if combined < +0.005.
- **Most likely failure**: function text is GO in prose; brief fact 2 suggests such functional similarity adds little on top of transfer.

### LT-9 Language-selected proxy traits for sub-threshold GWAS
- **Mechanism**: Brief fact 4: own-trait sub-threshold z predicts later positives (AUC ~0.8) but was shown for 4 diseases, and 57 % of arrivals
  are GWAS-only. Use text to scale the trait matching: embed every MAGMA trait description and every disease, and let a reasoning step add
  *proxy* traits that are not synonyms (biomarkers, medications, family history, symptoms), with signed weights; the disease's GWAS score is the
  weighted combination of gene z over own and proxy traits. Donors are again chosen by meaning, not by gene overlap (the banned
  "T0 UKB MAGMA studies as donors" selected by overlap).
- **p_typical**: 0.20
- **Data**: modet `data/processed/gwas_magma_t0` (4.4 GB, 102 entries) for `temporal`; `gwas_magma` (38 GB, 328 entries) for `forward`
  (inventory); trait names from the study metadata. **Leakage class: temporal-clean** with T0 studies and a 2020 encoder for matching;
  proxy selection by a modern LLM is a small `transductive` route on `temporal` (the LLM knows which biomarker GWAS later explained the disease);
  control: restrict proxies to a pre-2021 source (HPO disease annotations 2021-01, UKB field categories). Main route otherwise: post-T0 studies in
  the MAGMA folder used for a temporal task.
- **Minimal test (<= 2 h)**: embed ~100 T0 trait names + 53 dev disease names (CPU/GPU 1 min); top-5 traits per disease by cosine with weights;
  load gene z matrix (19k x 102, small), `score` = weighted mean z, degree- and gene-length-residualised; evaluate alone, combined with transfer,
  and on the GWAS-only stratum (modet arrival classes). 30 min total.
- **Expected effect**: on diseases with a matched trait, GWAS-only recall@100 from 16 % to 22-25 %; overall **+0.005-0.012 nuAP** on temporal dev
  (limited by 102 T0 traits); larger on `forward` with 328 traits. **Kill rule**: < +0.005 nuAP on the subset of dev diseases with >= 1 trait at
  cosine above the chosen threshold.
- **Most likely failure**: too few T0 traits match the 53 dev diseases, so the effect is real but covers a handful of tasks.

---

## Summary table
| id | p_typical | leakage class (temporal bench) | expected combined nuAP gain | stratum aimed at |
|---|---|---|---|---|
| LT-1 | 0.05 | temporal-clean | +0.010-0.020 | zero transfer support |
| LT-2 | 0.08 | temporal-clean | +0.010-0.015 | few seeds, genes without OT row |
| LT-3 | 0.07 | temporal-clean | +0.010-0.025 | old evidence newly attached, rare disease |
| LT-4 | 0.12 | transductive (clean on forward) | +0.005-0.012 | genes unknown to any disease |
| LT-5 | 0.30 / 0.08 | transductive; anon arm near-clean | +0.005-0.015 | top-300 reordering; leakage instrument |
| LT-6 | 0.15 | clean (extractive) | +0.005-0.010 | module structure |
| LT-7 | 0.20 | temporal-clean | +0.008-0.015 | imminent arrivals |
| LT-8 | 0.60 | clean / transductive pair | +0.003-0.008 | reference baseline |
| LT-9 | 0.20 | temporal-clean | +0.005-0.012 | GWAS-only |

## My bet and the most underrated idea
- **Bet: LT-2 (semantic transfer over virtual donors).** It extends the one mechanism the brief shows to work, removes its two measured
  limits (needs seed overlap; needs an OT row), all data is on disk, and it is temporal-clean with a simple PMID cut. LT-1 is the more original
  idea and I would run it in the same session because both share the build step, but its failure mode (contexts re-encode the known disease area)
  is more likely.
- **Most underrated: LT-3 (curation debt).** It looks like bookkeeping, but the brief says ~40 % of the "new" positives are exactly this class, no
  biological model can predict a mapping event, and nobody has tried to predict it directly. It also separates cleanly what part of any future
  gain is database behaviour and what part is discovery, which the final paper needs anyway.

## Things I could not verify
- No leaderboard file yet, so no AP baseline number; all expected AP effects are relative guesses.
- Row count of the europepmc 21.04 evidence (only part 1 of 200 read), and whether `diseaseFromSource` is populated in the other 21.04 sources.
- How the temporal panel treats genes with only weak literature evidence at T0 (excluded as "unknown" or rankable).
- Training cutoffs and weight completeness of the cached Qwen3.8-27B and gemma-4-31B models.
- PubTator3 file sizes from the web (the search returned none); the sizes above are from the local copies.
