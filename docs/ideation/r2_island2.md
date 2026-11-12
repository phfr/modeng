# Round 2, fresh island 2: lens "large models as the engine" (2026-11-12)

Written without looking at measured round-1 results (on purpose). Read: `brief.md`, `02_local_data_inventory.md`, `banned_r1.md`, the 73 titles
in `docs/ideas.md`, `implementer_brief.md` (channel API), and the "on disk" and leakage sections of `r1_llmtext.md`.
None of the tests below was run; effect sizes are guesses, stated as such.

## What I checked on disk for this lens
| item | state |
|---|---|
| `Qwen3-Embedding-0.6B` | complete (1.2 GB, `model.safetensors`) |
| `Qwen3-Embedding-8B` | **NOT usable**: 16 MB, only `model.safetensors.index.json`, no weight shards (needs ~16 GB download) |
| `RedHatAI/Qwen3.8-27B-INT4` | complete (19 GB) |
| `RedHatAI/gemma-4-31B-it-NVFP4` | complete (22 GB) |
| `biohub/ESMC-300M`, `johahi/borzoi-replicate-0` | complete (1.3 GB, 710 MB; one Borzoi replicate only) |
| also cached | `mims-harvard/ATHENA-R1-Qwen3-8B` (16 GB, 4 shards), `Qwen3-1.7B`, bge-m3, bge-reranker-v2-m3, MedCPT cross-encoder |
| PubTator3 | `gene2pubtator3.gz` 758 MB, `disease2pubtator3.gz` 2.2 GB, `relation2pubtator3.gz` 300 MB (PMID-level) |
| `pubmed_status` | only `in_process.txt`, `not_medline.txt` (PMID status lists, no dates) |
| `paperqa_corpus` | 5.0 GB, 107 topic folders of `<PMCID>.txt` full texts (topic-selected, not a general corpus; dates not checked) |

Not checked: training cut-off dates of Qwen3.8-27B and gemma-4-31B (I treat both as later than 2025-09, so their *knowledge* is transductive on
every benchmark unless the design blocks gene identity; see "Leakage design" below).

## Leakage design shared by all ideas
Three routes (same as round 1): corpus date, model weights, annotation tools. What this lens adds:
- **Identity firewall.** A large model may see the disease, but its output must be an object with **no gene identifiers** (a program over numeric
  views, a text description, a list of metabolites or cell types). Gene scores are then computed by code from time-stamped data. The model's
  memory of "gene X causes disease D" cannot pass; only disease-level knowledge (which tissue, which process) can. Enforced by a regex over HGNC
  symbols/aliases/ENSG ids on every model output, and by a **disease-swap control** (run the object produced for disease A on disease B).
- **Vintage ladder.** Where identity cannot be blocked (F2-1), use the same model family at several training cut-offs and report the score as a
  function of cut-off. The clean number is the one with cut-off <= seed date; the rest of the curve measures the leak.
- Class per benchmark is stated per idea: `clean`, `weights-transductive` (model trained after the seed date) or `corpus-transductive`.

---

## Step 1: eight assumptions and their flips
| # | assumption in the transfer + donors + views stacker | flip | idea |
|---|---|---|---|
| A1 | The scoring function is written by people and fixed; only its weights are learned. | The function itself (similarity, aggregation, normalisation of the donor matrix) is searched by a code-writing model against 1,786 past episodes. | F2-3 |
| A2 | One model for all diseases; a disease differs only by its seeds. | Each disease gets its own hypotheses, found at test time and kept only if they recover held-out seeds. | F2-2 |
| A3 | A large language model is a feature extractor (embeddings) or a judge of pairs. | It is a generative prior over gene symbols whose training cut-off is an experimental variable. | F2-1 |
| A4 | A gene is reachable only through what was measured or written about it. | A protein is reachable from its sequence alone if a text-sequence model can match it to a description of the disease mechanism. | F2-4 |
| A5 | The module is found from seed genes. | The module is found from non-gene anchors of the disease (metabolites, cell types, organelles, lab findings) mapped to genes by non-label resources. | F2-5 |
| A6 | GWAS evidence enters as gene-level z from positional windows (MAGMA). | A sequence model decides which gene a sub-threshold variant acts on, and in which tissue, before any aggregation. | F2-6 |
| A7 | Literature enters as gene-disease co-mention or as embeddings of gene summaries. | Literature is read for experiment outcomes (perturb gene, observe cell phenotype), with disease words removed; the outcomes are the donor rows. | F2-7 |
| A8 | Ranking genes one by one and cutting at k gives the module. | A set-level critic, trained against the ranker's own confident errors, scores whole modules; search over set edits gives the module and the cutoff. | F2-8 |

---

## Step 2: ideas

### F2-1 Vintage ladder: a time-locked generative language model as gene prior, with the leak measured by cut-off year
- **Mechanism.** Take a chronologically consistent language model (ChronoGPT yearly vintages 1999-2024) and continue its pretraining on PubMed
  abstracts with publication date <= seed date. Score each of the 19,255 genes by the constrained-decoding log-likelihood of its symbol in
  prompts such as "New susceptibility genes for {disease} include", minus the same likelihood under a disease-free prompt (removes popularity).
  Repeat for vintages after the seed date: the rise of target-gene likelihood with vintage is a direct measurement of weight leakage.
- **p_typical.** 0.06 (round 1 has cloze over existing contexts, LT-1, and an anonymised judge, LT-5; a cut-off dose-response with a generative
  gene prior was not in the titles; search found no gene-disease work with year-locked models).
- **Data / model.** ChronoGPT vintages (https://arxiv.org/abs/2502.21206; instruction-tuned variants https://arxiv.org/pdf/2510.11677, both
  accessed 2026-11-12; download, size not checked). PubMed baseline abstracts with dates (download, ~40 GB XML) or, for the 2-hour test, the
  dated sentences in `/mnt/msadb/modet/downloads/ot_evidence/evidence_2104/sourceId=europepmc`. Gene symbol trie from `ncbi_gene_info`.
- **Leakage class and route.** `temporal`: clean for vintage <= 2020 plus text <= 2021-03 (route closed by construction); every later vintage
  is weights-transductive and is reported only as the leak curve. `forward`: vintage 2024 is clean. Residual route: PubMed date fields
  (use `PubDate`/first `PubMedPubDate`, not the revision date).
- **2-hour test.** No continued pretraining: ChronoGPT-2020 and ChronoGPT-2024 as downloaded, 53 temporal dev diseases, 3 prompts, prefix
  trie over symbols (multi-token symbols: sum of token log-probs), batch over genes via `gpu-run --mem 8`. Channel
  `src/modeng/ext/f2_vintage.py`, `CHANNELS={"lm_prior_2020":..., "lm_prior_2024":...}`; `modeng.bench.evaluate` alone, then
  `python -m modeng.stack --bench temporal --channels base,lm_prior_2020 --train past` (channel returns NaN on `past`, so use `--train cv` for
  this first look and say so).
- **Expected effect on nuAP vs stacker.** +0.000 to +0.004 for the clean vintage (small general-domain model, weak on gene symbols). The
  deliverable is the curve: if 2024 minus 2020 is, say, +0.02 alone, that number bounds what any modern-LLM channel on `temporal` owes to memory.
- **Kill rule.** Clean vintage alone has AUROC < 0.55 on dev AND the 2024 vintage alone is not better than the 2020 vintage (then the model
  family is too weak to measure anything).
- **Likely failure.** ChronoGPT is small and general-domain: gene symbols are rare tokens, likelihoods are dominated by symbol frequency and
  tokenisation. The continued pretraining then becomes the real work (a day, not two hours).

### F2-2 Hypothesis programs: per-disease proposer, data-only verifier, seed hold-out as referee
- **Mechanism.** For each disease, a local LLM (Qwen3.8-27B) writes 30-50 short programs in a restricted language over the numeric gene views
  (e.g. "top decile of GTEx pancreas specificity AND Perturb-seq K562 knock-down shifts the unfolded-protein-response program"); the language
  has no gene identifiers. A verifier with no LLM in it runs each program, keeps those that rank held-out seeds high (leave-seed-out, with a
  permutation null matched on degree), feeds the scores back to the proposer for two or three refinement rounds, and the channel is the
  verified-program ensemble. The accepted programs are also the readable internal structure of the module.
- **p_typical.** 0.07 (LT-6 "mechanism cards" are text queries from the module; here the object is executable, searched at test time and
  refereed by seeds).
- **Data / model.** `RedHatAI/Qwen3.8-27B-INT4` (complete, 19 GB; `gpu-run --mem 24`); views `data/views/*.npz` with column names from
  `docs/reports/views_manifest.tsv` (columns need human-readable names: tissue, cell type, GO/Reactome term, perturbed gene is NOT allowed
  as a name, replace by its GO slim class).
- **Leakage class and route.** Identity firewall holds if the view-column vocabulary has no gene names. Remaining route: the LLM knows the
  disease's biology as of its training date, including mechanisms published after the seed date: `weights-transductive` on `temporal`, and on
  `forward` unless the model cut-off is <= 2025-09 (not checked). Controls: (a) disease-name-blind proposer (sees only which view columns are
  enriched among seeds), (b) disease-swap, (c) views are all T0 (`go_t0`, `reactome_t0`, DepMap 20Q4...; `ps_cd4` 2025 is post-T0 data).
- **2-hour test.** 20 temporal dev diseases (10 with <= 6 seeds), 30 programs each, one refinement round, language = weighted sums and
  thresholds of named columns from `gtex`, `hpa_sc`, `go_t0`, `reactome_t0`, `depmap20q4`. Channel `f2_hypprog` (NaN for other tasks), compare
  `--channels base,donor,view` against `+f2_hypprog` on those 20 tasks with `modeng.compare`; also the blind-proposer variant.
- **Expected effect.** +0.002 to +0.008 nuAP overall; the place to look is the few-seed and zero-transfer-support strata, where a named
  mechanism can stand in for missing donors. Much of it may equal what the `view` channels already give (then the gain is interpretability only).
- **Kill rule.** Named proposer is not better than the blind proposer (the LLM adds nothing beyond seed enrichment), or verified programs
  do not beat unverified ones on the hidden targets (seed hold-out does not referee).
- **Likely failure.** With a median of 11 seeds the verifier is noisy: programs are accepted by chance and the ensemble is a worse copy of
  the learned view channels. Round 1 facts say function-based channels add +0.000-0.007; this may sit in the same band.

### F2-3 Evolved transfer kernel: LLM-driven program search over the function that dominates
- **Mechanism.** Transfer is the main driver, yet its form (overlap similarity, max aggregation) was hand-written. Run an AlphaEvolve-style
  loop: a code-writing LLM mutates one Python function `score(context_matrix, seeds, disease_meta) -> gene scores`, fitness is nuAP on the
  1,786 `past` episodes (19.04 seeds), with islands and an archive; the temporal dev/confirm folds are never used as fitness. The result is a
  small, readable program, tested once on dev and confirm.
- **p_typical.** 0.08 (open frameworks exist; I found no use on disease-gene ranking; searches listed below).
- **Data / model.** Harness only: `B.context_for(task)`, `B.context_ids`, ontology. Mutator: Qwen3.8-27B local, or an open framework
  (ShinkaEvolve, Apache-2.0, https://sakana.ai/shinka-evolve/; CodeEvolve reports good results with an open-weight Qwen coder model,
  https://arxiv.org/pdf/2510.14150; both accessed 2026-11-12).
- **Leakage class and route.** `clean` on all benchmarks as far as inputs go (label matrix at seed date only). Route to watch: the search
  optimises on `past`, whose targets are 21.04 labels = the seeds of `temporal`; that is allowed (it is what `--train past` does), but the
  evolved code must not hard-code gene or disease indices (lint: no integer literals > 50, no id strings; plus a row/column-permutation
  equivariance test of the function). Selection bias from hundreds of candidates is handled by the untouched confirm fold.
- **2-hour test.** Fitness on a fixed random 300 of the `past` episodes (seconds per candidate with sparse matrices), 150 candidates in 5
  islands seeded with `tr_max` and transfer_jaccard, 4 threads. Best program -> channel `f2_evo_transfer`; alone vs `tr_max`, then
  `--channels base` with the base transfer channel replaced by it vs `base`, `--train past`, dev OOF + confirm.
- **Expected effect.** +0.003 to +0.012 nuAP on the stacker: likely finds things like evidence-type-aware weights, IDF on donors, ontology
  distance discounts, shrinkage by donor size. Honest prior: half of it is already inside the LightGBM through other transfer-derived columns.
- **Kill rule.** Best evolved kernel beats `tr_max` alone on the 1,486 unused `past` episodes by < 0.005 nuAP, or the gain disappears in the
  stacker on confirm (paired bootstrap interval includes 0).
- **Likely failure.** The search rediscovers known recommender formulas and the stacker absorbs them; or it overfits quirks of the 19.04 ->
  21.04 transition (identifier migration) that do not repeat in 21.04 -> 25.09.

### F2-4 Text-to-sequence retrieval: reach proteins nobody has written about
- **Mechanism.** A tri-modal protein model (ProTrek: sequence, structure, text in one space) scores every human protein against a
  gene-free description of the disease mechanism. The description is written by an LLM from the seed genes' T0 function text (UniProt
  2021-02) and passes the identity firewall; the protein side uses only sequence (and AFDB structure). A protein with no literature and no
  annotation is then scored on the same footing as a famous one.
- **p_typical.** 0.05 (round 1 has protein-embedding views and homology transfer; a text query against a sequence tower was not in the titles;
  the ProTrek material I found shows only function-text queries, none on disease mechanisms).
- **Data / model.** `westlake-repl/ProTrek_650M_UniRef50` (download; text tower PubMedBERT, protein tower ESM-2 650M size; 40 M training
  pairs, 14 M from Swiss-Prot; bioRxiv https://www.biorxiv.org/content/10.1101/2024.05.30.596740, model card
  https://huggingface.co/westlake-repl/ProTrek_35M_UniRef50, accessed 2026-11-12). Sequences: `/mnt/msadb/subex/raw/uniprot`, structures
  `afdb_v4_human`. Seed text: `esmc_sprot2021_02` folder's Swiss-Prot 2021-02 (path named in `r1_llmtext.md`).
- **Leakage class and route.** `temporal`: weights-transductive (ProTrek was trained on Swiss-Prot text of ~2023-24, which contains
  disease comments for human proteins: a protein's sequence embedding may have been pulled towards its later disease text). `forward`:
  clean (training data < 2025-09). Control that sizes the route: restrict evaluation to proteins without any Swiss-Prot disease comment in
  the current release, and compare with the non-human-ortholog sequence of each protein (mouse sequence from `ensembl103_compara`), which was
  never paired with human disease text.
- **2-hour test.** Embed 19,255 canonical sequences (minutes on GPU, `gpu-run --mem 10`), template description per disease = concatenated
  T0 function sentences of the seeds with symbols removed (no LLM yet) plus the disease name; channel `f2_txt2seq` = cosine; also a
  `km`-style donor family from the same embedding as a control (does the text query add anything over "nearest to seeds in ProTrek space").
  Run on `forward` (fair) and `temporal` (transductive), alone and `--channels base,donor,view,f2_txt2seq`.
- **Expected effect.** +0.000 to +0.004 overall; target is the stratum "gene not associated with any disease at seed date" (within-pool AUC
  0.60-0.63 for the best method): +0.03 AUC there would be worth a keep even with zero overall gain.
- **Kill rule.** Text query does not beat the seed-centroid control in the same embedding, or within-pool AUC in the no-disease stratum
  improves by < 0.02.
- **Likely failure.** The joint space encodes molecular function, and the views (`esmc`, `prott5`, `esm2_15b`) already carry it; disease
  mechanism text is too coarse to select among functionally similar proteins.

### F2-5 Seedless anchors: an agent reads the clinic, code maps anchors to genes
- **Mechanism.** For each disease an LLM agent reads time-locked clinical text (case reports, pathology, biochemistry) and outputs only
  non-gene anchors: accumulating or depleted metabolites, affected cell types and organelles, abnormal lab values, drugs that worsen or
  relieve it. Code maps anchors to genes through resources that are not disease-gene labels: metabolite -> reactions -> enzymes and
  transporters within two steps in Human-GEM, cell type -> specific genes (HPA single cell, Tabula Sapiens), organelle -> proteome
  (Human Cell Map). The channel needs no seed gene, so it does not weaken when there are 3-5 seeds.
- **p_typical.** 0.06 (LT-2 uses ontology terms as donor rows and SC-3 a cell address; the metabolite route through a genome-scale metabolic
  model and an agent that is forbidden to name genes were not in the titles).
- **Data / model.** `/mnt/msadb/subex/raw/human_gem`, `human_cell_map`, `tabula_sapiens_v2`, views `hpa_sc`; text: PubTator3 PMIDs for the
  disease MeSH id with a date cut plus abstracts fetched for those PMIDs (download), or `paperqa_corpus` where a topic folder matches;
  HPO 2021-01 phenotype terms of the disease as a no-LLM baseline for anchors. Qwen3.8-27B or gemma-4-31B.
- **Leakage class and route.** Identity firewall on the output. Corpus: clean with PMID date cut. Weights: the agent may add anchors from
  memory that were published later: require a verbatim supporting quote from a dated document for every anchor (string match), else drop;
  with that, `clean` up to quote-checking errors; without it, weights-transductive. Human-GEM version must predate the seed date for
  `temporal` (version on disk not checked).
- **2-hour test.** Skip the agent: for the temporal dev diseases with an inborn-error, mitochondrial, lysosomal or lipid flavour (pick by
  ontology branch, about 10-15), take metabolite anchors from HPO 2021-01 "abnormal circulating/urinary X concentration" terms, map through
  Human-GEM (distance <= 2 reactions, currency metabolites removed), channel `f2_anchor_met` = inverse distance; evaluate alone and as an
  increment on that subset; negative control = metabolite labels shuffled among diseases.
- **Expected effect.** Large on the metabolic subset (possibly +0.02 nuAP there), about +0.001-0.003 over all diseases because the subset
  is small; the cell-type and organelle anchors are expected to add little (tissue context was measured as +0.000-0.007).
- **Kill rule.** On the metabolic subset the anchor channel alone is not better than its shuffled control by >= 0.01 nuAP, or all its top
  hits are already seeds/transfer hits (no gain in the zero-transfer-support stratum).
- **Likely failure.** Metabolic disease genes are the best-annotated genes there are; the channel finds what transfer already finds. For
  common complex diseases anchors are unspecific (inflammation, fibrosis).

### F2-6 Sequence-model verdicts at sub-threshold loci: which gene, in which tissue
- **Mechanism.** For the disease's own T0 GWAS, take sub-threshold lead variants (say 5e-8 < p < 1e-4) and their LD proxies, and let a
  sequence-to-function model predict each variant's effect on every gene's expression within its window, in the tissues where the seed genes
  are expressed. A gene's score is the z-weighted sum of predicted effects on it, so a locus contributes to the gene the sequence model
  points at rather than to every gene in a positional window. This targets the 57 % GWAS-only arrivals at the step where MAGMA is blind.
- **p_typical.** 0.08 (round 1 has locus-crossing forecasts, coloc/MR with measured eQTLs, and a host-gene prior; a sequence model as the
  per-variant gene chooser below significance was not in the titles).
- **Data / model.** `/mnt/msadb/modet/downloads/gwas_sumstats_t0` (90 GB), Borzoi replicate 0 (on disk; gene-level expression deltas via
  gReLU), AlphaGenome env on this machine. Search results state that precomputed AlphaGenome scores for ~9 billion SNVs were released in
  September 2026 (https://deepmind.google/blog/alphagenome-atlas-a-predictive-map-of-every-possible-dna-letter-change-in-the-human-genome/,
  accessed 2026-11-12; I did not open the page, so size, licence and whether gene-level expression scores are included are unverified).
- **Leakage class and route.** `clean` in spirit: the models are trained on functional genomics tracks, not on disease labels; GWAS files are
  T0. Formally the model weights are dated 2023-2025, after the `temporal` seed date, but they contain no gene-disease pairs; say so in the
  report. One real route: Open Targets L2G itself uses sequence-model-like features in recent releases, so this channel partly emulates the
  label generator for GWAS arrivals (useful, but it is "predicting the pipeline").
- **2-hour test.** 4 dev diseases with a T0 GWAS and many GWAS-only arrivals; 1,500 lead variants each from the sub-threshold band (no LD
  expansion), Borzoi ref/alt forward passes (about 12,000 passes, roughly an hour on the GPU if one pass is ~0.2-0.3 s; time 20 passes
  first), `gpu-run --mem 16`, run in background. Channel `f2_seq_v2g` (NaN elsewhere). Compare on those tasks against the existing own-trait
  MAGMA z channel and a nearest-gene version of the same variants: recall@100 of GWAS-only targets and nuAP increment.
- **Expected effect.** On diseases with a GWAS: +0.005 to +0.015 nuAP over the stacker if MAGMA z is not yet in it, +0.000 to +0.005 if it
  is. Overall less, in proportion to the share of diseases with usable T0 summary statistics.
- **Kill rule.** Not better than nearest-gene assignment of the same variants on GWAS-only targets (paired, 4 diseases, then 12 before a
  final no).
- **Likely failure.** Sequence models are weak at distal effects and at direction for single variants (a July 2026 preprint,
  https://www.preprints.org/manuscript/202607.1417, argues AlphaGenome effect scores are distance-dependent and miscalibrated; not peer
  reviewed, accessed 2026-11-12); nearest gene is right about two times in three, which leaves little room. One Borzoi replicate adds noise.

### F2-7 Literature as a perturbation atlas: an LLM extracts experiment outcomes, disease words removed
- **Mechanism.** An LLM reads dated abstracts and full texts sentence by sentence and extracts grounded tuples (perturbed gene, direction,
  measured cell or organism phenotype, system), keeping only tuples whose quote is found verbatim and dropping any sentence that names a
  disease. The tuples form a signed gene x phenotype matrix covering thousands of small-scale experiments that no screen atlas contains; its
  rows become virtual donors (a disease is matched to phenotype rows through its seeds, as with any donor family).
- **p_typical.** 0.09 (SC-2 uses real experiments as donors, LT-2 ontology terms, LT-1 masks diseases in contexts; LLM-built outcome tuples
  from small-scale literature as a donor family is a combination I did not see in the titles, but it is close to all three).
- **Data / model.** Gene-tagged PMIDs from `pubtator3/gene2pubtator3.gz`; PMIDs with a disease tag in `disease2pubtator3.gz` are excluded
  outright in the strict version; abstracts by download (PubMed baseline) with date cut; `relation2pubtator3.gz` as a no-LLM baseline for
  gene-gene and gene-chemical relations. Extractor: Qwen3.8-27B (throughput on this GPU not measured; assume a few thousand abstracts per
  hour, so a sample, not all of PubMed, in the first pass).
- **Leakage class and route.** Corpus: clean with date cut. Weights: an extractor can fabricate a tuple from memory; the verbatim-quote rule
  closes that (measure the rejected share). Disease removal closes direct co-mention; a remaining route is phenotype terms that are near
  synonyms of a disease ("insulin secretion" for diabetes): that is biology, not leakage, but report the share of signal that survives
  removing phenotype rows whose text embeds close to the target disease name.
- **2-hour test.** No LLM: build the donor family from `relation2pubtator3.gz` restricted to PMIDs <= the seed-date PMID boundary
  (gene-chemical and gene-gene relation types as rows, minimum 5 genes per row), save `data/donors/f2_litpert.npz` in the `load_family`
  format, run the standard donor increment `--channels base,donor,view` vs `+` this family on `temporal --train past`. Only if that moves
  anything is the LLM extraction of outcomes worth a day.
- **Expected effect.** +0.001 to +0.005 nuAP; literature-derived rows inherit study bias, so the gain, if any, sits on well-studied genes
  and may be negative for nuAP.
- **Kill rule.** PubTator-relation family gives < +0.002 nuAP on dev OOF and nothing in the zero-transfer stratum: do not build the LLM version.
- **Likely failure.** It is GO/MSigDB again by another route (the banned list says function-based channels add little), and popularity
  dominates: genes with many papers have many tuples.

### F2-8 Critic by self-play: a set-level verifier trained on the ranker's confident errors, then search over modules
- **Mechanism.** Proposer = the current stacker. On `past` episodes, collect its top-k proposals; the critic (a small set transformer over
  identity-blind per-gene features plus pairwise relations among the proposed set: same locus, same complex, paralogs, shared donors) learns
  to tell sets containing later arrivals from the proposer's high-ranked misses, the proposer is refit with the critic's score, and the loop
  repeats two or three times. At test time, search (beam over add/drop/swap edits) maximises the critic's score for the whole module, which
  yields the module, a calibrated size, and removes redundant picks (ten genes from one locus or one complex).
- **p_typical.** 0.10 (masked diffusion over gene sets and inverse-Ising couplings are in round 1; the adversarial training signal and
  test-time search over set edits are the new parts; the family is the same).
- **Data / model.** Harness channels and views; `past` episodes for training; relations from `data/donors/complexportal.npz`,
  `paralog30.npz`, gene coordinates (`ensembl104`). PyTorch in the bio-net env; CPU is enough for the pilot.
- **Leakage class and route.** `clean` (same inputs as the stacker). Route to watch: the critic trains on proposer outputs for `past`
  episodes, so those must be out-of-fold predictions, otherwise the critic learns the proposer's overfit rather than its errors.
- **2-hour test.** Skip self-play rounds: one critic pass as a listwise re-ranker of the stacker's top 300 per task, trained on OOF `past`
  predictions with set-context features (count of same-locus and same-complex genes ranked above, donor overlap with higher-ranked genes).
  Channel `f2_critic` = re-ranked score (stacker score elsewhere); compare with `modeng.compare` against `stackP_bdv` on dev and confirm:
  nuAP, recall@100, and module-level F1 at the critic's chosen size against the stacker's top-|seeds| baseline.
- **Expected effect.** +0.000 to +0.004 nuAP (re-ranking the top 300 cannot reach the hard strata); the real product is the module
  output: a per-disease cutoff and less redundant sets, measured by module F1 and calibration, not by nuAP.
- **Kill rule.** No nuAP or recall@100 gain on confirm AND module F1 at the critic's size is not better than a fixed-k cutoff tuned on dev.
- **Likely failure.** Redundancy is real biology: complexes and loci do arrive together, so a critic that spreads picks hurts. With 20-30
  positives per disease the set-level signal is thin.

---

## Summary
| id | paradigm | hard stratum aimed at | clean benchmark | p_typical | guess: nuAP increment on stacker |
|---|---|---|---|---|---|
| F2-1 | time-locked generative LM + cut-off ladder | measures LLM leak; dark genes little | temporal (vintage <= 2020), forward | 0.06 | +0.000-0.004 |
| F2-2 | test-time hypothesis search, seed-referee | few seeds, zero transfer | none strictly (weights); firewall + controls | 0.07 | +0.002-0.008 |
| F2-3 | LLM program evolution of the transfer kernel | all (the main driver) | all | 0.08 | +0.003-0.012 |
| F2-4 | text-to-sequence retrieval | genes unknown to any disease | forward | 0.05 | +0.000-0.004 |
| F2-5 | seedless non-gene anchors | rare diseases, 3-5 seeds | all, with quote rule | 0.06 | +0.001-0.003 (subset +0.02) |
| F2-6 | sequence-model variant-to-gene below threshold | GWAS-only arrivals | all (note on weights date) | 0.08 | +0.000-0.015 |
| F2-7 | LLM-read experiment outcomes as donors | zero transfer | all, with date cut + quote rule | 0.09 | +0.001-0.005 |
| F2-8 | self-play critic + set search | module output, cutoff | all | 0.10 | +0.000-0.004 |

**My top 3.** F2-3 (attacks the component that carries most of the signal, clean on every benchmark, cheapest to test, 1,786 episodes give a
fitness signal well below the 0.01 detection limit of the dev fold), F2-6 (the only idea here aimed at the largest block of misses, the
GWAS-only arrivals, with a clear control: nearest gene), F2-1 (small expected gain, but it turns "is the LLM channel circular?" from an
argument into a measured curve, which every other LLM idea in the register needs).
**Most underrated.** F2-5 on the metabolic subset: tiny scope, but it is the one channel that does not depend on seeds at all.

## Not verified
- Training cut-offs of Qwen3.8-27B, gemma-4-31B, ATHENA-R1; ChronoGPT download size and how it tokenises gene symbols.
- AlphaGenome precomputed-score release (seen only as a search-result summary); Borzoi seconds per pass on this GPU.
- Human-GEM version date on disk; whether `paperqa_corpus` files carry publication dates; LLM extraction throughput.
- Whether any existing `base`/`donor` channel already contains own-trait MAGMA z or PubTator relations (I did not read results on purpose).

## Web searches (5 of 6 used, 2026-11-12; no personal or project names)
1. ProTrek tri-modal protein-text model, weights and text-to-protein retrieval: https://www.biorxiv.org/content/10.1101/2024.05.30.596740 ,
   https://huggingface.co/westlake-repl/ProTrek_35M_UniRef50
2. Chronologically consistent / time-locked language models and lookahead bias: https://arxiv.org/pdf/2510.11677 ,
   https://icml.cc/virtual/2025/48685 . No gene-disease work with year-locked models found.
3. Open AlphaEvolve-style frameworks: https://sakana.ai/shinka-evolve/ , https://arxiv.org/pdf/2510.14150 . No use on gene prioritisation found.
4. Sequence models for GWAS variant-to-gene: https://deepmind.google/blog/alphagenome-atlas-a-predictive-map-of-every-possible-dna-letter-change-in-the-human-genome/ ,
   https://www.preprints.org/manuscript/202607.1417 (critical preprint). Nothing specific on sub-threshold loci found.
5. LLM agents for gene prioritisation with temporal controls: https://pmc.ncbi.nlm.nih.gov/articles/PMC12263109/ (bias to well-studied
   genes, order sensitivity). No time-split or verifier-based work on Open Targets found.
"No prior found" means these five queries, nothing more.
