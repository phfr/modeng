# R4 strategy review B: is the stacker a dead end, and what would be genuinely bigger? (Fable reviewer B, 2026-11-12 14:10-15:00)

Stance: adversarial to incrementalism. Independent of reviewer A (not seen). Read: FINAL report, r1-r5 reports, research 03/04/05,
R3 red team, insights, invalidations, ideas, pre-registration. New numbers below come from read-only checks on the saved modeng-v1
scores (`results/temporal/{dev,confirm}/stack_v3ptr_p0.35.scores.npy`) and the channel cache (`data/cache/channels/temporal_*`);
scripts and pair tables are in the session scratchpad, nothing was written to the repo except this file. **All numbers are
descriptive** (temporal confirm is spent); they size the problem, they decide no claim.

## Verdict in five lines
1. **Yes, the stacker over the present witnesses is near its ceiling.** It already puts 67 % of the "witnessed" targets into the top 200;
   what is left there is worth at most about +0.02-0.03 AP, i.e. 3-5 times the retrain noise (0.005-0.007).
2. **The real limit is evidence, not the learner: 54 % of future targets have no strong trace in any witness**, and on those the model is a
   gene prior (3.8 % in the top 200; chance 1.1 %). 71 % of the untraced, classed targets are GWAS-only arrivals; 30 % do not persist (label noise).
3. So the next step change can only come from **(a) data that creates evidence for the untraced half** (summary statistics, not hit lists)
   or **(b) a different target** (new-data events instead of Open Targets label arrivals). More similarity witnesses, gates and learners cannot do it.
4. The project is optimising a benchmark in which the cheapest gains are label-process forecasts. Split the product in two and say so.
5. Do not spend the sealed fold now. An **untouched temporal fold exists today** (about 700 off-panel diseases); use that first.

## 1. Ceiling of the stacker paradigm

### 1a. What fraction of future targets has any pre-seed trace? (temporal, modeng-v1 candidates; dev 1,886 / confirm 4,438 target pairs)
"Trace" = the target is in the **top 200 (about 1 %) of at least one seed-specific witness** among nine: transfer_jaccard, dc_ridge,
dc_ridge_pos, cs_donor (class R), dA_pt, pt_ridge (L), ph_mean, don_hpo2101 (P), gwas_catlag (G). The same rule flags 4.2 % of
non-targets (about 770 genes per task), so it is a generous definition. (The loose definition "any sparse witness non-zero" covers 76 % of
targets but also 34 % of non-targets; it does not separate.)

| quantity | dev | confirm |
|---|---|---|
| targets traced (top 200 of any witness) | 48 % | **46 %** |
| ... by class R / L / P / G | 33 / 25 / 19 / 11 % | 34 / 27 / 20 / **4 %** |
| modeng-v1 recall@200 | 36 % | 33 % |
| modeng-v1 top 200, given traced | 73 % | **67 %** |
| ... traced by 1 / 2 / 3 classes | 50 / 87 / 97 % | 48 / 72 / 94 % |
| modeng-v1 top 200, given untraced | 2.8 % | **3.8 %** (chance 1.1 %) |
| untraced: mean rank percentile, model vs popularity alone | 0.72 vs 0.70 | 0.73 vs 0.69 |
| untraced: median STRING degree (traced) / share off-network | 5 (11) / 26 % | 5 (12) / 26 % |
| strict-a: traced / model top 200 | 43 % / 27 % | 44 % / 28 % |
| strict-a traced by G (gwas_catlag) | 13 % | **4.6 %** |

By arrival class (modet arrival table, 4,981 of 6,324 pairs matched, dev + confirm pooled):

| class | n | traced | model top 200 |
|---|---|---|---|
| gwas_only | 2,863 | **38 %** | 26 % |
| clinvar_only | 827 | 59 % | 40 % |
| curated_only | 633 | 73 % | 52 % |
| mixed_non_gwas | 434 | 78 % | 69 % |
| burden_only | 201 | 44 % | 20 % |
| arrival A (new genetic, persistent) / B (old evidence) / X (not persistent) | 1,894 / 1,782 / 1,006 | 45 / 68 / **26 %** | 28 / 53 / 14 % |

Composition of the untraced targets: **gwas_only 71 %** (57 % of all), clinvar_only 14 %, curated 7 %, burden 5 %; by arrival A 42 %, **X 30 %**, B 23 %.

Reading:
- The model is a good combiner: when two evidence classes agree it is at 72-97 %. Its losses are single-witness traces (966 confirm pairs, 48 %).
- Where no witness speaks, the model adds 0.03-0.04 percentile over gene popularity. No learner change can alter that; this is the wall.
- Strict-a is not "less witnessed" than the rest (44 % vs 47 %), but its traces are transfer and literature, hardly ever genetics (G 4.6 %):
  the model has **no genetic witness for 95 % of the genetic discoveries it is asked to predict**. That is the precise form of "coverage-limited".
- 30 % of the untraced are class X (do not persist to 26.09): an irreducible noise floor of about 16 % of all classed targets.

### 1b. Oracle bounds on AP (confirm, modeng-v1 AP 0.153)
| bound | AP | what it means |
|---|---|---|
| best fixed single witness (transfer_jaccard) | 0.119 | - |
| oracle per-task choice of one of 16 witnesses | 0.188 | a perfect disease-type gate over single witnesses; optimistic (maximum of 16 noisy values, chosen on the truth) |
| oracle max(stack, best witness) per task | 0.202 | **upper bound for any gate / mixture of experts: +0.05**; 64 % of tasks have a single witness that beats the stack |
| each target at its best rank over stack + witnesses | 0.326 | unreachable; needs the answer |
| all traced targets ordered perfectly first, rest at stack rank | 0.57 | the information ceiling of the present witnesses with a perfect learner |

A realistic share of the gate bound is a fifth to a third (the r3 context gate obtained +0.006 of the +0.05). Adding the known free items (drop the 30 views
+0.006, hybrid past + dev training, context features +0.006) I put the **stacker family's reachable confirm AP at 0.17-0.18 and recall@200 at about 0.38-0.40
(now 0.33); the hard recall@200 limit of this witness set is about 0.46-0.50.** dev gives the same picture (0.176; oracle gate 0.214).

### 1c. Why I call it a dead end rather than "nearly finished"
- Every new witness since r2 has been redundant (cell type, KGE, gene kernels, catalog transfer, sub-threshold loci, power forecast): seven kills in a row.
- 60 weak features on 53 training diseases are fitted noise (the views hurt); the paradigm is now limited by its own width.
- Out of window, the gain without label-process channels is AP +0.018 (forward, in-window training) or zero (cross-window): the transferable biology
  content of the present model is small.
- The largest measured effect of the whole sprint (forward nuAP 0.137 -> 0.307) is curation backlog. A project that keeps the Open Targets arrival as its only
  target will drift into forecasting Open Targets, whatever it intends.

## 2. Directions that could break the ceiling (ranked)

### D1. Predict the outcome of the next GWAS: a genome-wide, sample-size-aware genetics forecaster on full biobank summary statistics
**Why this one.** The untraced half is 71 % GWAS-only, and G traces 4.6 % of strict-a. Hit lists (gwas_catlag, sub-threshold catalog leads, 20-trait
ML-4) were tried; **full summary statistics of the cohort that later produces the labels were not.** FinnGen is behind 47-65 % of strict-a GWAS arrivals,
Open Targets 21.04 contained none of it, and R4 was public on 2020-11-30 (sweep 04).

**Design** (three layers, each testable on its own):
1. *Atlas:* gene x endpoint statistics from FinnGen R4 (2,444 endpoints) plus Pan-UKBB 2020 for uncovered diseases: MAGMA gene z, lead-variant distance, credible-set
   mass if fine-mapping is available, PoPS (features on disk). Endpoint -> disease id by endpoint code through the OT `study` table (mapping vintage post-T0: declare it, as for gwas_catlag).
2. *Growth law, no learning:* a locus at z in R4 with n cases reaches expected z x sqrt(n_later / n_R4) in a later freeze. Score = P(crossing 5e-8 | z, growth factor),
   times a within-locus gene prior (distance, constraint LOEUF, PoPS, L2G-winner history). This is a physical forecast with two free parameters, not a 131-feature fit.
3. *Borrowing:* multivariate shrinkage of gene z across genetically correlated endpoints (rg from the same atlas; ideas SG-3 / SG-9), so that diseases without an own
   endpoint receive a posterior from relatives. This is the part that reaches the 37 confirm strict-a diseases with no own T0 study.
The output enters the stacker as two or three channels (own-endpoint crossing probability, borrowed posterior, coverage flag), and stands alone as a product:
"the loci your next freeze will find, and the gene at each".

**Its own benchmark (important).** FinnGen freeze-to-freeze: train on R2 -> R4 (2020-01 -> 2020-11) or R3 -> R4, test R4 -> R6..R12 (2022-2024), label = gene at a new
genome-wide-significant locus of the same endpoint. Thousands of endpoint tasks, labels that are a measurement, not curation, never looked at by this project.

**Leakage class.** Temporal-clean by date (R4 <= T0; R5 is 11 days after T0: sensitivity arm only). Partly label-process for the Open Targets target (OT ingested FinnGen
later), so report three strata: FinnGen-supported arrivals, non-FinnGen GWAS arrivals (the real test of biology), diseases without own endpoint (borrowing only).
On the freeze benchmark there is no label-process component; the risk there is the trivial baseline "largest sub-threshold z", which must be the comparator.

**Cheapest decisive test (CPU, about half a day, no large download).** P1: R4 genome-wide-significant lead tables only (small files) -> nearest / PoPS-prior gene ->
one channel. Measure on temporal dev (cv; 20 strict-a diseases) and, descriptively, the trace table of 1a.
**Kill rules (pre-register before the download):**
- P1 must raise the share of **dev strict-a targets traced by G from 13 % to >= 25 %**, and the stacker increment on dev must be strict-a onnet nuAP >= +0.010 with the
  same-width permuted control within +-0.003. If traced-by-G stays < 20 %: stop, do not download sumstats; record "strict-a is not reachable from T0 genetics".
- P2 (sumstats + growth law, about 300-500 mapped endpoints, estimated 150-400 GB, MAGMA about 1 CPU-day): on the freeze benchmark the growth-law score must beat
  "rank by R4 z" by >= +0.03 AP on new R4 -> R8 loci (cluster bootstrap over endpoints, CI > 0), and on temporal dev the non-FinnGen GWAS stratum must not get worse
  (CI lower bound > -0.005). If only the FinnGen-supported stratum gains, the channel is filed under label process and removed from the biology arm.
**Compute:** CPU only; disk a few hundred GB. **Risk:** medium-high. r4 found own-study MAGMA z weak (sa_AP 0.019) on 34 diseases, so layer 1 alone may repeat that;
the bet is on same-cohort growth and on borrowing. Endpoint mapping is the main engineering cost. Prior of passing P1: about 0.5; P2: about 0.3.

### D2. Change the target: an event-typed benchmark where "new data" is the primary endpoint and Open Targets ingestion is a separate product
**Why.** 37 % of confirm targets already had OT 21.04 evidence in the family; 47 % of forward arrivals had backlog evidence; 30 % of the untraced are non-persistent.
The present benchmark rewards three different skills (curation-lag forecasting, old-knowledge retrieval, discovery) with one number, and the first two win.
A more ambitious model is not definable until the target separates them.

**Design.**
- Every arrival gets an event type from the date of the underlying data, not of the OT release: **N** (supporting study / submission first public after T0),
  **L** (public before T0, ingested later: lag), **O** (ontology / threshold movement), plus persistence. Sources: OT evidence publication years and study ids (as used
  for modet's arrival classes), GWAS-Catalog added dates, ClinVar archive dates, FinnGen freeze dates.
- Primary endpoint for the *biology model*: N events that persist, scored with AP and nuAP, label-process channels forbidden. Primary endpoint for the *forecaster*:
  all arrivals, anything dated <= T0 allowed (backlog channels, catlag, GenCC), judged on calibration and lead time. Two leaderboards, two claims.
- Orthogonal outcome sets that are not OT labels, as secondary truths for the biology model: FinnGen freeze-to-freeze loci (D1), exome burden results published after T0
  (Genebass / AZ: an *outcome* for temporal, not an input), drug-programme outcomes (Minikel 2024, on disk). A model that gains on N events and on two of three
  orthogonal sets is evidence for biology; a model that gains only on L is a pipeline forecaster.

**Leakage class.** An evaluation change; the typing uses post-T0 information by construction (allowed for labels, never for inputs).
**Cheapest decisive test (CPU, 2-3 h, existing files):** type the temporal dev arrivals, recompute modeng-v1, v1clean, transfer and modet's headline on N-persistent only.
**Decision rule (no kill, a fork):** if on N-persistent events modeng-v1 minus transfer is < +0.010 AP on dev (R3's nearest proxy, "no 21.04 evidence in the family",
gave +0.025 on confirm, and a tie against modet's headline), declare the OT-arrival ranking paradigm saturated for biology, freeze modeng-v1 as the forecaster line, and
move all modelling effort to D1 / D3 under the N endpoint. If >= +0.010, the stacker stays the base model and D1 / D3 are judged as increments on N.
**Compute:** negligible. **Risk:** low technically; the real risk is that the honest N-endpoint numbers are small and unglamorous. That is the point of doing it first.

### D3. Reading agents that create evidence from time-locked text (extractor, not judge), with memorisation controls built into the design
**Why.** Literature already traces 27 % of targets through co-mention counts; co-mentions ignore what a paper found (direction, tissue, model, phenotype).
54 % of targets are untraced and low-degree; whatever exists about them before T0 is in a few papers that co-mention no disease. Only reading can use that.
**Design.**
- Corpus: abstracts with PMID <= T0 cut (and PMC open-access full text with dated versions) for each gene; the disease is never shown.
- A local LLM (Qwen3.8-27B via `llm-serve`, through `gpu-run`) extracts, per paper, structured **experiment statements** (perturbation, system / tissue / cell type,
  measured phenotype mapped to HPO / GO / cell-type terms, direction), each with a **verbatim supporting span**. Code, not the LLM, maps gene cards to diseases
  (HPO profile of the seeds, tissue, pathway), giving channels and a readable dossier per call. This is F2-7 / LT-6, chosen over a judge because a judge re-reads its own memory.
- Controls inside the method: (i) span check: a statement whose span is not in the dated text is dropped (blocks recall from weights); (ii) **entity-swap control**: gene
  symbols exchanged between papers before extraction; the card must follow the text, not the true gene; (iii) memorisation probe of the model without evidence
  (recall of 19.04 -> 21.04 vs 21.04 -> 25.09 arrivals, citation-matched) reported next to every result; (iv) citation-decile strata and nuAP.
- The generative judge (anonymised dossier, LT-5) is pre-registered for the next OT release only.
**Leakage class.** Input-clean by date; model weights are 2026, so `vintage / transductive` until the swap control and the span check show the output is text-bound.
**Cheapest decisive test (about 2-3 GPU-h, `gpu-run --mem 24`, after asking the user: GPU budget rule):** 2,000 genes sampled by citation stratum, not by outcome; up to 30
pre-T0 abstracts each (about 60k abstracts); build the card witness; evaluate on temporal dev.
**Kill rules:** the card witness must put **>= 8 % of the currently untraced dev targets among the sampled genes into its top 1 %** (popularity alone gives about 3 %, the
present model 2.8 %); the entity-swapped run must keep <= 30 % of that excess; stacker increment on dev >= +0.005 AP and nuAP over a same-width permuted control.
Fail any -> kill the retrospective use and keep only the prospective judge. Full run if passed: about 20-25 GPU-h (ask first).
**Risk:** high. Dark genes have little text; extraction noise; and the 2026 annotation / model vintage can never be fully excluded retrospectively. Prior about 0.25.
It is still the only route to the non-GWAS part of the untraced half (29 %).

### Considered and placed lower
- **Witness-class experts + disagreement (05, rank 1 there).** Useful as the reporting layer of D2 (biology arm vs forecaster arm, a "biology-only" list). It breaks no
  ceiling: the perfect-gate bound is +0.05 AP and experts use the same witnesses. Do it only as part of D2, not as a model programme.
- **Path-based KG model (NBFNet).** The shallow KGE is dead, 1a shows the binding limit is absent evidence, and KG paths are made of the same T0 edges. One 6 GPU-h run
  is defensible only for diseases with < 20 % traced targets (12 / 138 confirm tasks); not a priority.
- **Evidence-stream foundation model (hazard / point process over all dated OT evidence across ~30 releases).** The most promising *forecaster* (ML-9, IR-9); file it under
  the forecaster product of D2. It would likely win the next-release fold, and it would say nothing about biology.
- **Seed-set transformers, TabPFN, more gene views, single-cell foundation embeddings:** closed by the evidence (section 3).

## 3. What the project should stop doing
1. **Stop adding similarity / view / donor witnesses and biology-derived kernels.** Seven redundant in a row; the views already hurt. v2 = v1clean minus views minus weak donors, then freeze.
2. **Stop learner swaps and gating on 53 dev diseases** (class-A heads, area flags, TabPFN, set transformers). Bound +0.05, realised +0.006.
3. **Stop scoring anything on temporal confirm.** 59 stackers + r3-r5 arms have been seen there. It is now a descriptive set only.
4. **Stop strict-a attempts that reuse catalog hit lists** (five negatives). Only D1 (new data) may reopen strict-a.
5. **Stop reporting one mixed number.** Every table: forecaster arm and biology (clean, no-label-process) arm; AP and nuAP; N-persistent stratum once D2 exists.
6. **Stop treating non-persistent arrivals as targets** in the biology endpoint (30 % of the untraced; 32 % of GWAS-only arrivals vanish by 26.09).
7. **Stop single-seed deltas below 0.007.** Three seeds or no statement.
8. **Stop comparing with modet's headline on nuAP as the lead claim**: it becomes a tie once prior-evidence targets are removed (R3).

## 4. Confirmation plan and the sealed fold
**Finding: a fresh temporal fold exists now.** From the cached label matrices: 1,613 diseases have 5-1,000 genes at 21.04 and >= 3 new genes by 25.09; 1,168 of them are
outside the panel and outside `excluded_task_ids()` (sealed diseases and relatives); **713 are also not an ontology relative of any panel disease** (median 13 new genes,
about 70k new pairs; raw label sets, before modet-style "unknown" exclusion and arrival classing, so the final count will be lower). No model of this project has been
scored on them. Limits: same time window as the training arrivals (it confirms the in-window claim, not cross-window transfer); their rows served as donor context;
strict-a needs the arrival classing rebuilt from the evidence files.

Order:
1. **Now (CPU, about a day): off-panel temporal fold.** Build tasks with the temporal masking rules; freeze by commit hash before any score exists: modeng-v1 as
   pre-registered, co-primary v1clean, the no-label-process arm, and v2 (v1clean minus views) as a declared secondary; comparator transfer_jaccard; primary endpoint
   as in the pre-registration (AP and nuAP, both CIs exclude 0, cluster bootstrap over ontology-connected components); three LightGBM seeds averaged. One run, one table.
   Write an amendment to the pre-registration first (the fold is not among its listed candidates).
2. **In parallel: D2 typing**, so that the off-panel run also reports the N-persistent stratum (declared as secondary before the run).
3. **Next OT release (seeds 26.09):** the cross-window test, forecaster and biology arms separately. A single quarter gives few arrivals: fix the minimum in advance (for
   example >= 300 diseases with >= 3 new genes) and wait for the second release if it is not met. The LLM judge of D3 can only be tested here.
4. **D1 confirmation:** the FinnGen freeze benchmark (R4 -> R8 / R12) is its own untouched confirmation set; then the strict-a stratum of the off-panel fold, once.

**Sealed fold: do not spend it now, and do not spend it on the stacker at all.** Its primary endpoint is strict-a, where modeng-v1 ties in every variant; a tie costs the
only single-use asset of the sibling project and buys nothing. Spend it once, on a model with a genetic witness, when all of these hold:
(a) D1 passed its P1 and P2 kill rules; (b) the frozen D1 model shows a strict-a gain over modet's headline on the off-panel fold or the next OT release with CI > 0
(pre-registered, run once); (c) the gain is present in the non-FinnGen GWAS stratum, so that it is not ingestion lag; (d) recipe, seeds and endpoints are committed and
timestamped; (e) the user authorises it through modet's ledger. If D1 is killed, the sealed fold stays sealed for modet.

## 5. Checks behind section 1 (reproducible)
- Candidate set per task = universe minus `task.exclude`; ranks with average ties; NaN ranked last. Witness channels from `data/cache/channels/temporal_{dev,confirm}_*.npy`.
- Non-target rates from 2,000 random non-target candidates per task. Arrival classes from modet's `arrival_devall_t0_string115_phys_notm_400.parquet` (read only).
- Off-panel count from `D.label_matrix('2104' / '2509')`, `D.excluded_task_ids()`, `D.relatives()`.
- Not checked: whether traced-by-R strict-a targets are biology or hub prior (degree-matched trace rate); how many off-panel diseases survive modet-style filtering;
  FinnGen file sizes and endpoint mapping coverage (figures in D1 are taken from sweep 04, tagged [m] there).
