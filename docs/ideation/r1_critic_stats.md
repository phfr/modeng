# R1 critique — statistician (leakage and evaluation), 2026-11-12

Scope: the 73 ideas in `r1_{causal,evostruct,llmtext,ml,recsys,singlecell,statgen,wildcard}.md`. I read all of them, `brief.md`, `banned_r1.md`,
the data inventory, the views manifest, `src/modeng/{tasks,metrics,compare,data}.py`, `methods/baselines.py` and modet's `groundtruth.py`.
Everything marked **[measured]** I computed tonight with single-thread one-off scripts on the dev splits (no GPU, nothing written but this file).
Effect sizes are my priors for the paired delta in nuAP against `transfer_jaccard` when the idea is fused *properly* (see M2), on temporal dev
unless another benchmark is named. "conf" = my confidence in that number. T / F / H = temporal / forward / holdout.

## 0. Measured facts that change how every idea must be tested

- **M1. Label tell in the harness's "exact masking correction" [measured].** `_transfer` computes `L[top].T @ w` and then subtracts the masked pairs
  in float32. The cancellation is not exact: on holdout dev **8.5 % of targets carry a non-zero residue |s| < 1e-5 (4.3 % negative), against 0.005 % of
  all genes**. For `transfer_onto` on holdout the harness gives nuAP 0.090 where an explicit `B.context_for(task)` computation gives 0.036 (recall@100
  0.20 vs 0.16); for `transfer_jaccard` the effect is small (0.1753 vs 0.1746, AUROC 0.912 vs 0.918) because real scores dominate. Temporal: 0.13 % of
  targets, forward: 0. Any method that reuses this code path and then thresholds (`score > 0` = "has support"), ranks, takes logs, or feeds the score
  to a tree model can read hidden holdout targets off the residue. EV-1/2/3, SC-2, IR-6, LT-2 and the whole "virtual donor" cluster say they will reuse
  `_transfer` "including its exact family-mask correction". Fix before anything else: compute neighbour sums from the masked rows, or zero `out[i, mask_c]`
  entries whose remaining support is empty; add a unit test "explicit mask == fast path, exactly".
- **M2. Rank-average fusion destroys the baseline [measured].** `rank(transfer_jaccard) + rank(uniform noise)`: temporal 0.096 -> 0.017, forward
  0.098 -> 0.025, holdout 0.198 -> 0.029. With popularity as the partner: 0.031 / 0.051 / 0.075. Reason: ~90 % of genes are tied at 0 in transfer, so
  any dense partner reorders the whole list. At least 20 ideas specify "rank-average / rank-blend with transfer" as their fused arm (CI-01, CI-03,
  ML-2, ML-3, ML-4, IR-1, IR-3, SC-1, LT-1, LT-5, all SG ideas, WC-MARKET ...). With that fusion a real signal looks harmful: false kills.
  Required fusion: `transfer + lam * z(new)` with one lam cross-fitted across disease clusters, or a 2-3 parameter stacked model. Required free
  control: transfer with a **popularity tie-break** among its zeros: +0.0022 temporal, +0.0027 forward, -0.0002 holdout. A fused idea that does not
  beat this control has shown nothing.
- **M3. Noise [measured].** SD of per-task paired nuAP differences between close variants is 0.03-0.04 on temporal dev (n = 53; 35 clusters):
  SE 0.004-0.005, minimum detectable effect (80 % power) about **0.011-0.015**. Forward dev (n = 842, SD 0.04-0.06): nominal SE 0.0015-0.002, but the
  bootstrap unit is the task itself although 56 % of target pairs come from genes that are targets in >= 10 tasks (ontology propagation); realistic MDE
  ~0.006. Kill rules of "+0.003 to +0.005 on temporal dev" (most ideas) are coin flips. Subset tests are worse (M5).
- **M4. What nuAP measures [measured].** 70 % of the positive weight sits on targets with STRING degree <= 1 (24 % of targets; 34 % of all genes have
  degree 0 in the T0 network). Kish effective number of positives per task: 9.4 of 23 (T), 6.9 of 15 (F), 4.4 of 9 (H). Weight share of targets with
  zero transfer support: **63.5 % T, 50 % F, 15 % H**; with an empty context column: 18 % T, 5 % F, 0.5 % H. So (a) the "unknown to any disease" stratum
  hardly exists outside temporal; (b) a source that does not cover degree <= 1 genes cannot move nuAP. Coverage of degree <= 1 dev targets by the ready
  views: ESM/ProtT5/GTEx/HPA 97-100 %, DepMap 94 %, phylo 94 %, GO-T0 79 %, **CD4 Perturb-seq 56 %, K562 36 %, Reactome 38 %, HepG2/Jurkat/RPE1 3 %**.
- **M5. Own-trait GWAS coverage [measured].** Harmonised T0 GWAS: 15 of 53 temporal dev diseases (47 of 191; "direct" only 8 of 53). 52 % of dev
  GWAS-only arrivals are in those 15 diseases; 13 diseases carry 80 % of all GWAS-only arrivals; the *median* per-disease GWAS-only share is 21 %
  (the brief's 57 % is pair-weighted, the metric is disease-weighted). A subset test on 15 diseases has MDE ~0.025 nuAP on the subset.
- **M6. The temporal candidate set is built with future information.** modet `groundtruth.py`: `unknown = weak(T0) | weak(T1) | direct-on-ancestor(T1)
  | ...`, and `new = T1 - T0 - weak(T0)`. Median 1,566 genes per task are excluded [measured]. Consequences: (i) a pair with sub-threshold own-disease
  *genetic* evidence at T0 (L2G < 0.5, ClinVar < 0.5) is neither a target nor rankable, so ideas that read OT's own weak genetic rows (SG-7, IR-7
  genetic channel, CI-05, part of WC-L2GEMU) have their best signal removed on temporal by construction; non-genetic T0 evidence (literature, animal
  model, drugs) is NOT excluded; (ii) `Task.exclude` and `Task.alt` are T1-derived objects: a method must never read them (not as features, not to
  pick training negatives across tasks).
- **M7. Dev and confirm are not independent [measured].** Temporal: 18 of the 41 confirm clusters also occur in dev. Forward: 60 % of confirm tasks
  have an ontology relative in dev, 38 % a dev task with Jaccard >= 0.5. Holdout: 36 % / 22 %. Tuning on dev transfers to confirm by near-duplication.
  Several WC ideas additionally fit on "non-dev diseases", which in temporal *is* the confirm split.
- **M8. `forward` is mostly a pipeline change, not one year of discovery [measured].** On the 4,716 rows common to 25.09 and 26.09: 208,646 pairs
  added (+61 %) and 53,063 removed (15.5 %); 16 % of a task's 25.09 seeds are gone from its 26.09 row; 1,448 of 6,164 ids vanish. 9,066 distinct genes
  are forward-dev targets. Checked: zeroing the vanished-id rows costs `transfer_jaccard` only 0.006 nuAP, so identifier migration is not the main
  driver. But a gain on forward may be "predicts the OT 26.09 re-scoring", and must be read next to temporal.
- **M9. Cross-benchmark contamination.** The forward context (25.09) contains every temporal target; the holdout context (26.09) contains every forward
  target. Any fitted artefact (gene embedding, SVD, set-transformer weights, learned kernel, a heavily searched hyperparameter set) trained on a later
  context and reused on an earlier benchmark is a direct label leak. One artefact per benchmark, fitted from that benchmark's context only.
- **M10. Reference level.** `transfer_jaccard` (0.096 T dev) is below modet's `tr_max` (0.122) and far below the headline (0.177); k = 50 is not the
  problem (k = 200: -0.002, k = 1000: -0.011 [measured]). In modet a LightGBM with gene covariates added ~+0.05 to transfer. Therefore every idea that
  brings its own learner plus degree/constraint covariates (ML-6, IR-7, LT-7, WC-SELFN, WC-HAWKES, CI-06 ...) will "beat transfer_jaccard" for reasons
  unrelated to its claim. The control is always the same learner without the new ingredient. The leaderboard's holdout rows do not reproduce with the
  current cache (I get `transfer_jaccard` 0.198 on holdout dev, the file says 0.171; `transfer_onto` 0.09 vs 0.0012): rerun before quoting.
- **M11. Look-ahead inside the "previous transition" pseudo-split.** Ideas that fit weights on 19.04 -> 21.04 arrivals with features built at 21.04
  (T0 MAGMA z from studies published 2019-2021, 21.04 literature scores, 21.04 L2G rows) train on features that post-date their labels; the fitted
  weight on those features is inflated and is then applied at test time. Only LT-7 and ML-9 specify an as-of cut for the training window.
- **M12. Few-seed stratum is absent.** Seeds per dev task: median 38, 10th percentile 13 (T); 39.5 / 8 (F). Claims about "3-5 seed diseases"
  (ML-8, SC-3, SG-4, SG-9, LT-2) cannot be tested on these benchmarks as they stand.
- No tell found: on holdout, popularity restricted to genes absent from the task's relatives has nuAP 0.001 (the ancestor re-propagation does what
  it says).

## 1. Per-idea review

### Causal (CI)
- **CI-01 upstream-of-core (Perturb-seq).** Fatal: no; but perturbed genes cover 36 % (K562) / 56 % (CD4) of the degree <= 1 targets that carry 70 % of nuAP (M4).
  Leak: T transductive by date only (label-free); F: CD4 atlas (Dec 2025) post-dates the seed date; H none. Unperturbed genes "get the task median": then
  being-perturbed (= expressed, studied) is itself the signal. Test: rank-average arm invalid (M2); shows only whether a generic responsiveness score
  correlates with labels. Effect +0.001 (0 to +0.004), conf med. Fix: score only inside transfer's zeros with cross-fitted lam; controls =
  perturbed-indicator and mean |z| per gene; pre-register the immune/blood subset.
- **CI-02 trans-QTL convergence.** Fatal: nearly: eQTLGen tested ~10k GWAS SNPs only; after hub correction most genes score 0. Leak: T clean (eQTLGen),
  UKB-PPP transductive; H: if seed loci are taken per disease rather than per seed gene, the hidden gene's own locus enters in cis. Test: "applies to
  ~40 % of diseases" = ~20 dev tasks, MDE ~0.02, expected effect 0.005-0.015: cannot decide. Effect +0.001, conf med-high. Fix: one test for CI-02 +
  SC-5 + SG-6 on forward (n large) against out-degree-preserving rewiring.
- **CI-03 cis-MR / coloc.** Fatal: no. Coverage 15 dev diseases (M5). Leak: T clean with GTEx v8 + studies dated < 2021-04 (the T0 study table has rows up
  to 2021-04-23); F: 25.09 coloc tables only; H: direct (coloc -> L2G -> label). Test: top-eQTL SMR without HEIDI is MAGMA z at one SNP; the AUROC kill
  rule vs MAGMA is sound but on <= 15 diseases; "fuse with modet_headline" uses an unreliable row. Effect +0.004 overall (+0.015 covered), conf low.
  Fix: plain MAGMA-z arm first; endpoint = pooled GWAS-only recall@100 over all covered tasks with disease bootstrap.
- **CI-04 directed donor graph.** Fatal: no. Leak: T clean; the F variant fits on 21.04 -> 25.09 flows, which include the sealed diseases' targets
  (drop them). Asymmetry from one release pair is confounded with row size and with the 19.04 -> 21.04 pipeline change. Test: theta picked among 4 on
  ~26 diseases is noise. Effect +0.001 (0 to +0.003), conf med. Fix: one pre-registered theta, plus a placebo asymmetry built from row-size ratio only.
- **CI-05 power-jump gate.** Fatal for the 2-h test: the gate can only vary across the 15 covered dev diseases, and a, b are fitted on 19.04 -> 21.04
  with T0 z (M11 look-ahead: a 2020 study "predicts" its own 2020 arrivals). Leak: as they state (realised N forbidden). Effect +0.001, conf med.
  Fix: drop the gate, report the best constant weight; keep "predicted arrival count" as a separate read-out with n_t0 as the baseline.
- **CI-06 channel detectability.** Fatal: no. Leak: clean. Test: baseline is `modet_headline` (unreliable row); the stated control (pooled propensity)
  is right. Effect +0.003 vs transfer_jaccard, of which +0.002 is what any gene prior gives as a tie-break (M2); ~0 over the pooled control; conf med.
  Fix: merge CI-06 / SG-8 / WC-SELFN / IR-1 into one observation-model test with pooled-propensity and popularity-tie-break controls.
- **CI-07 invariant predictors.** Fatal as a sprint item: expected overall effect is inside the noise by the author's own numbers, and the kill rule
  (AUC gain < 0.02 AND loss > 0.005) can hardly fire. Leak: none. Effect 0.000 +- 0.004, conf high. Fix: run it only as a diagnostic table
  (per-environment sign/size of each feature), no claim.
- **CI-08 rg-borrowed z.** Fatal: no; close to the banned "T0 MAGMA studies as donors". Leak: T clean; the seed-regression arm fits LD (seed locus
  mates inherit z): leave-locus-out is mandatory; H circular. Effect +0.003, conf low. Fix: one "trait-space transfer" test with CI-08, SG-3, WC-XPOWER,
  LT-9, SG-9 as arms (own z / seed-regressed combination / rg-weighted / text-selected), same covered set.
- **CI-09 allelic series.** Fatal on T (transductive) and weak on F: Genebass is a static 2022 table, everything significant was ingested long ago,
  burden-only arrivals are 4 % of positives, binary-trait sub-threshold tails are near null. H: copy. Hail download does not fit 2 h. Effect +0.001
  (F), conf med-high. Fix: fold into one burden test with IR-10 / SG-5, starting with the sub-threshold AUROC diagnostic only.

### Evolution / structure (EV)
- **EV-1 phenolog donors.** Fatal: no. Leak: T clean with IMPC 13.0; F/H with Monarch: mouse models are built *after* the human gene is found, and
  Monarch mixes orthology-projected human knowledge: on holdout these rows are label copies that no family mask covers. M1 applies (reuses
  `_transfer`). Effect +0.003 (0 to +0.006), conf med. Fix: enter as one family in the unified donor model (section 2a) with a family weight
  and a size-preserving rewiring null.
- **EV-2 paralog / fold lift.** Fatal: no. Leak: T clean; H fine only if L.K is built from the masked context (stated). Test: a 4x4x3 grid tuned on
  ~26 diseases, expected effect below MDE: the winner will be noise. Effect +0.003 (0 to +0.007), conf med. Fix: one pre-registered K (paralogs
  >= 30 % identity, weight 0.3), judged on forward dev for power, with a family-size-preserving random K as null.
- **EV-3 co-evolution kernel.** Fatal: no. Leak: none. Test is well posed and cheap (premise check: kNN degree vs STRING degree). Effect +0.001
  (0 to +0.004), conf med. Fix: use it as a lift kernel in the unified model, not as an RWR network (RWR is at 0.005-0.015 here).
- **EV-4 selection-aware hazard.** Fatal: the interaction term is fitted "on half" of 15 covered diseases = 7 diseases. Leak: clean (check GeneBayes
  prior features). Effect: z part +0.003 overall (shared with the GWAS cluster), interaction 0, conf med. Fix: test z x constraint on trait-level
  release pairs (ML-4's table, thousands of rows), not on the benchmark.
- **EV-5 livestock / dog loci.** Fatal for the sprint: intervals are megabases wide, author candidates are circular (they say so), downloads
  unverified. H/F: OMIA links phenes to human diseases = label copy. Effect +0.000 (0 to +0.002), conf high. Fix: OMIA rows only, dated, as one
  family of the unified model.
- **EV-6 convergent evolution.** Fatal for the sprint: the on-disk fallback (pairwise dN/dS to human) is not a branch rate. Leak: none. The AUROC
  < 0.55 kill rule is fine. Effect +0.000 (0 to +0.002), conf high. Fix: test seed-profile correlation on the ready `phylo_npp` view instead; it is
  EV-3 by another name.
- **EV-7 interface-resolved edges.** Fatal: coverage is a few thousand well-studied genes, i.e. not the degree <= 1 genes that carry nuAP (M4); raw
  AlphaMissense is a ClinVar-calibrated label route. Effect 0.000, conf high. Fix: drop from this sprint.
- **EV-8 domain-level transfer.** Fatal: no. Leak: clean in the coarse version; `mis_z` is a gene prior, keep it as a separate control arm. It is
  function similarity to seeds (GO "function users" was banned as no gain). Effect +0.002 (0 to +0.004), conf med. Fix: Pfam co-membership as one
  more kernel in EV-2's lift, IDF-weighted.
- **EV-9 selection-space meta-learner.** Fatal: the 12 features are close to one-dimensional, so it is a gene prior; pHaplo is trained on known
  dosage genes (label route). Effect +0.002 = the popularity-tie-break level, conf med-high. Fix: only meaningful against the global-model control;
  otherwise drop.

### ML
- **ML-1 cold-start imputation.** Fatal: no. Leak: an SVD of the unmasked context contains every hidden pair on holdout (and the 287 own-row target
  pairs on temporal): it must be refitted per fold; Replogle transductive on T. Effect +0.003 (0 to +0.006), conf med. Fix: skip the SVD; impute a
  gene's column as the weighted mean of its content-neighbours' columns = EV-2's L.K with an embedding kernel; judge by pooled within-pool AUC on
  empty-column genes.
- **ML-2 ascertainment-simulator PFN.** Fatal for 2 h, and the simulator assumes compact modules in content space, which brief fact 2 contradicts.
  Leak: cleanest of all. A failure cannot be attributed (simulator vs model). Effect -0.003 to +0.003, conf med-high. Fix: meta-train on real
  19.04 -> 21.04 episodes instead (that is ML-5 / ML-6).
- **ML-3 gene-set LM on Rummagene.** Fatal: no. Leak: T clean with a per-paper date cut; the extra rule "drop papers cited in evidence_2509" selects
  inputs with future information (conservative, but not reproducible at deployment): do not use it. H: supplement tables of GWAS/exome papers are the
  labels: holdout unusable. Download unverified. Effect +0.004 (0 to +0.010), conf low. Fix: PPMI rows as donor family in the unified model first;
  no transformer unless the linear version passes.
- **ML-4 release-pair extrapolator.** Fatal: no. Leak: T transductive as stated; the on-disk post-T0 files are named after the panel diseases, so
  training traits = evaluation tasks unless cross-fitted by modet cluster. MAGMA significance is not the L2G label. Covered 15 dev tasks. Effect
  +0.005 overall (+0.02 covered), conf low. Fix: train on GWAS-Catalog same-trait study pairs both dated <= 2021-04 (fully clean), and make forward
  the decision benchmark.
- **ML-5 identity-blind set transformer.** Fatal: no. Leak: "identity-blind" is not true: a frozen 256-d embedding is a unique key per gene, so the
  model can memorise memberships; on holdout it must be trained with task families held out. MSigDB/HPO sets contain disease-gene knowledge (fine on
  T by date). Effect +0.002 (0 to +0.006), conf low-med. Fix: run the linear version (unified donor model with a content kernel) first.
- **ML-6 tabular ICL with neighbour arrivals.** Fatal: no. Leak: T clean if every feature of a context row is built as of 19.04 (T0 MAGMA z is not:
  M11). Test: the LightGBM control is right, but a third arm is needed (same LightGBM trained on stock labels), otherwise the covariate effect (M10)
  is booked as "arrival objective". Effect +0.01 to +0.03 vs transfer_jaccard, 0 to +0.005 vs the stock-trained control, conf med. Fix: three arms,
  cross-fitted by cluster inside dev.
- **ML-7 set diffusion.** Fatal as a ranking idea (equal to ML-5 by the author's account). The size-calibration deliverable needs a baseline:
  arrivals regressed on n_t0. Effect 0, conf high. Fix: first check whether anything beats the n_t0 count baseline (WC-CAPTURE test 1).
- **ML-8 disease encoder for donor selection.** Fatal as posed: its stratum (<= 5 seeds) does not exist (M12). Leak: modern text encoder = mild
  transductive route on T (stated). Effect 0 to +0.003 at full seeds, conf med. Fix: evaluate on a seed-subsampled variant (3 / 5 / 10 seeds).
- **ML-9 evidence-trajectory hazard.** Fatal: no; own-pair literature evidence is rankable (only weak *genetic* pairs are excluded, M6). Leak: T
  clean, training window correctly truncated; `publicationYear` is empty in the europepmc partition (needs PMID -> year); F needs the 25.09 evidence
  dump (only `gwas_credible_sets` is on disk). Effect: level features +0.01 (0 to +0.03), slope/recency +0.001; conf low-med. Fix: level-only arm
  first (= IR-7's direct channel).

### Recsys (IR)
- **IR-1 exposure-aware MF.** Fatal: no. Collapses to a per-disease scalar (author's own failure mode); pseudo-split look-ahead (M11); gene2pubmed
  needs a date cut. Effect +0.003, mostly the gene-prior tie-break, conf med. Fix: merge into the observation-model test (CI-06).
- **IR-2 locus slates.** Fatal: no. Leak: T clean; H: hidden genes are often L2G locus-mates of seeds, so a holdout gain would be the banned locus
  prior. On T, own-locus re-assignment is removed by M6, so the gain must come from donor loci. Effect +0.003 (0 to +0.008), conf low. Fix: SG-1's
  10-minute diagnostic first, with a matched random-block baseline.
- **IR-3 identity linking** (= LT-3 = WC-BACKLOG). Fatal: no. Leak: T clean with the 21.04 ontology and string/CUI matching; drop sealed diseases'
  evidence rows. ClinVar rows < 0.5 under the task's own id are excluded by M6; the signal is rows filed under another id. Effect +0.015
  (0 to +0.03) on the full temporal endpoint, ~0 on `sa`, conf low-med. Fix: exact-match arm only, reported by arrival class and on `sa`; treat as
  a mandatory control arm for every other temporal claim.
- **IR-4 powered nulls** (= WC-UPPERLIM). Fatal: no. Covered 15 dev tasks; demoting a few genes barely moves AP; a, b fitted with look-ahead (M11).
  Effect +0.001, conf med-high. Fix: only WC-UPPERLIM's diagnostic (positive rate of measured nulls in the transfer top-200).
- **IR-5 study baskets.** Fatal: no. Leak: T clean; H: baskets (PMIDs, panels, RCVs under other ids) contain the hidden pair and cannot be masked.
  It is item-kNN on a re-cut of the same evidence (EASE was banned as no gain), plus text-mining popularity. Effect +0.003 (0 to +0.008), conf low.
  Fix: genetic-study baskets only, and the control "gene-gene cosine on the context rows".
- **IR-6 trending.** Fatal: no. Leak: T clean. Good design (pre-registered beta = 1). Small perturbation, so paired SD is small and the test has
  power. Effect +0.001 (0 to +0.004), conf med. Fix: add the forward version with the 23.06 snapshot on disk (23.06 -> 25.09 delta, n = 842).
- **IR-7 multi-behaviour funnel.** Fatal: no. Leak: T clean with 21.04 per-datasource scores; the lambdarank fitted on the pseudo-split has
  look-ahead (M11), and cross-fitting "over non-dev tasks" burns confirm; H: circular (the hidden pair's own scores and the papers about it).
  Effect +0.015 (0 to +0.04) vs transfer_jaccard, conf med; unknown how much survives against a covariate learner. Fix: before any learner, the
  direct arm `transfer + lam * own-row non-genetic score` per channel (literature, animal model, drugs), cross-fitted lam.
- **IR-8 cold-start tower** (= ML-1). Same leak rule (retrain without hidden pairs per fold; Replogle transductive on T). Effect +0.002, conf med.
  Fix: judge by pooled within-pool AUC on empty-column genes, against a covariates-only model.
- **IR-9 next-item sequences.** Fatal-ish: first-evidence dates are missing for curated sources and for europepmc. The shuffled-order ablation is
  the right test. Effect +0.001, conf med-high. Fix: keep only IR-6.
- **IR-10 burden cross-domain.** Fatal: T transductive; H copy; F fair but the unknown-gene stratum is 5 % of weight there (M4). Effect +0.001 (F),
  conf med. Fix: burden top-k sets as one donor family in the unified model, with the p < 1e-4 exclusion they propose.

### Single cell (SC)
- **SC-1 virtual Perturb-seq screen.** Same as CI-01 (same data, same statistic, plus routing). Same flaws: coverage (M4), rank-average (M2).
  Effect +0.001 (0 to +0.004), conf med. Fix: one implementation for CI-01 + SC-1; pre-registered routed subset.
- **SC-2 experiments as donor rows.** Fatal: no. Statistical flaw: Jaccard of a ~38-gene seed set with ~30k small experimental sets is 0-2 genes;
  taking the top 50 of 33k rows selects chance overlaps (multiplicity) and pushes real disease donors out of the top 50. Leak: label-free; T
  transductive by date. The rewiring control is right. Effect -0.005 to +0.004, conf med. Fix: separate donor budget per family, similarity =
  shrunk hypergeometric / PMI, family weight cross-fitted.
- **SC-3 cell-address donors.** Fatal as posed: few-seed stratum absent (M12); otherwise the banned tissue-context weighting. Effect +0.001, conf
  med-high. Fix: seed-subsampled benchmark.
- **SC-4 regulator -> program -> trait.** Fatal: PoPS with perturbation features on (15 covered) x (routed) = ~5 dev diseases. T transductive; H
  circular. Effect +0.001, conf high. Fix: test in trait space (leave-one-chromosome-out R2 of z on program loadings over ~100 traits); only then
  touch the benchmark.
- **SC-5 trans-QTL atlas.** = CI-02 / SG-6. See CI-02.
- **SC-6 phenocopy of disease state.** Fatal for the sprint: 10 pilot diseases, signatures are consequences, and the stated leak route (OT expression
  evidence) does not exist here (labels are genetic only). Effect 0, conf high. Fix: drop.
- **SC-7 set completion on experimental sets.** = ML-5 with a label-free corpus. Fine-tuning per task on `context_for` is infeasible on holdout
  (needs fold-wise training). Effect +0.002, conf med. Fix: merge into ML-5's linear pre-test.
- **SC-8 within-cell-type programs.** Fatal for tonight: NMF on Tabula Sapiens is CPU-heavy while the machine is loaded (load ~32, another GPU job
  running), and only ~25 % of genes get a score. Effect +0.001, conf high. Fix: residualised `hpa_sc` view as a cheap proxy, or drop.
- **SC-9 perturbational cutoff.** Not a ranking idea. "Spearman(k*, true positives in top 500)" needs the baseline n_seeds (bigger diseases get more
  arrivals). Coverage 36-56 % of the relevant genes. nuAP effect 0. Fix: compare against the count baseline.

### LLM / text (LT)
- **LT-1 masked-context cloze.** Fatal: no. Leak: T clean (2020 encoder, 21.04 corpus, family-name masking); F needs the 25.09 europepmc dump.
  Unidentifiable as posed: gene contexts mostly restate the known disease area. Effect +0.004 (0 to +0.008), conf low. Fix: must beat the trivial
  own-pair co-mention count (IR-7 literature channel) and transfer together, not transfer alone.
- **LT-2 semantic virtual donors.** Fatal: no. Leak: T clean with a PMID cut and a 2020 encoder; HPO-2021 gene sets and MeSH co-occurrence encode
  gene-disease knowledge that OT 21.04 lacked (legitimate, but it is ingestion lag, not "semantic transfer"); H: all of these are label copies outside
  the family mask. M1 applies. Effect +0.008 (0 to +0.015), conf low-med. Fix: three separately reported arms: (a) the task's own MeSH/HPO row
  (lookup), (b) other rows chosen by seed overlap, (c) other rows chosen by text similarity. Only (b) and (c) are transfer.
- **LT-3 curation debt.** = IR-3 / WC-BACKLOG. The weak-tier part is mostly moot: curated sources count at any score in the label, so limited-tier
  genes are already seeds. Same numbers as IR-3.
- **LT-4 equal-budget dossiers.** Fatal as posed: transductive on T, and on its only fair benchmark (F) the unknown-gene stratum is 5 % of the weight
  (M4). Effect +0.002, conf med. Fix: symbol-ablated dossiers with a 2020 encoder on T, read out as pooled within-pool AUC on empty-column genes.
- **LT-5 anonymised LLM judge.** Fatal for claims: the cached LLMs' training cut-offs are unknown and probably after 2025-09, so the named arm is
  transductive on all three benchmarks; 20 diseases give MDE ~0.025. Effect (anonymous arm) +0.002, conf med. Fix: a cheaper leakage instrument:
  ask the LLM for the disease's genes with no dossier at all; its nuAP on temporal targets is the parametric-memory ceiling for every text idea.
- **LT-6 mechanism cards.** Fatal: no. Leak: clean in the extractive variant. Multi-vector vs single-centroid is a clean ablation, but it is
  function similarity to seeds. Effect +0.001, conf med-high. Fix: evaluate the structure claim directly (do targets that arrive together share a
  card more often than chance?).
- **LT-7 arrival hazard with momentum.** Fatal: no. Best as-of design of the hazard ideas (hard cuts end-2018 / end-2020); T0 MAGMA z in the training
  window still has look-ahead. Candidates limited to the transfer top 2,000 exclude most zero-support genes. Effect: momentum +0.002; whole pipeline
  +0.01 to +0.03 vs transfer_jaccard for the M10 reason; conf med. Fix: one shared flow-training table for ML-6 / LT-7 / WC-SELFN / WC-HAWKES with
  ablation columns.
- **LT-8 time-locked two-tower.** Fatal: no (it is a reference). Leak: UniProt DISEASE lines are label-like (stripping stated); the bilinear head
  must not be trained on hidden pairs on holdout. Effect clean arm +0.002, conf med-high. Keep as the measure of weight leakage (modern minus 2020).
- **LT-9 language-selected proxy traits.** Fatal: no. Covered subset small; "threshold chosen" is a forking path; LLM proxy choice transductive.
  Effect +0.003, conf low-med. Fix: arm of the trait-space test (CI-08), top-5 and threshold fixed in advance.

### Statistical genetics (SG)
- **SG-1 locus-space transfer.** = IR-2. The diagnostic (share of zero-support GWAS positives with a supported block-mate) is the right first step
  but needs a random-block baseline (blocks with many genes are "supported" by chance). Effect +0.003, conf low.
- **SG-2 phenome fingerprints (PhenomeXcan).** Fatal: no. Leak: T clean (2020-09); H circular (old GWAS signal of hidden pairs). Profile similarity
  mostly returns LD mates of seeds (banned locus prior) unless regressed out. Download size unverified. Effect +0.003 (0 to +0.006), conf low. Fix:
  its real value is coverage: 4,091 traits as the own/proxy-trait z source for the whole GWAS cluster instead of 108 studies.
- **SG-3 mash forecast.** Fatal: no. Leak: T clean with the date filter stated. Mash vs raw z is the right comparison. Effect: GWAS channel +0.004
  overall, mash minus raw z ~0, conf med. Fix: arm of the trait-space test.
- **SG-4 HPO-decomposed traits.** Fatal as posed: dev tasks with <= 10 seeds are fewer than 5 on temporal (M12). The permuted-map control is good.
  Effect +0.001 overall, conf med. Fix: forward dev (about 100 tasks with <= 10 seeds) or the seed-subsampled benchmark.
- **SG-5 burden atlas.** = IR-10 / CI-09. The T0-clean mini-test (DDD/ASC/Epi25) touches <= 3 dev tasks: no test. Effect +0.001, conf med.
- **SG-6 trans-QTL regulators.** = CI-02 / SC-5. The rewiring null is right.
- **SG-7 ingestion lag.** Fatal: no. Leak: T clean only with an archived catalog release (they say so); H copy; F fair with a <= 2025-09 archive. On
  T, M6 removes pairs that OT already held as weak. Effect +0.008 T (0 to +0.015), +0.015 F (0 to +0.03), conf low-med. Fix: 5-minute ceiling first:
  share of dev GWAS-only arrivals with `fy_gwas` <= 2020 in the arrival table.
- **SG-8 channel hazard.** = CI-06 / WC-SELFN. Pooled-prior control stated. Effect ~0 over that control, conf med.
- **SG-9 rg donor weights.** Fatal as posed: few-seed stratum absent; a few hundred mappable UKB phenotypes. Effect +0.001, conf high. Fix: arm of
  the trait-space test.

### Wild card (WC)
- **WC-L2GEMU.** Fatal: no. Leak: T clean; "w chosen on non-dev diseases" and the L2G model "trained outside dev" = fitted on confirm (M7); H copy.
  Clumping 90 GB of summary statistics conflicts with the current machine load. Covered 15 dev tasks. Effect +0.005 overall (+0.02 covered), conf
  low-med. Fix: cross-fit inside dev by cluster; plain z arm first; pooled GWAS-only recall@100 as the endpoint.
- **WC-XPOWER.** Fatal: no. For a single trait the crossing probability is a monotone function of |z|, so "crossing vs linear" cannot differ in
  ranking; it only matters inside the multi-trait sum, where it is a soft threshold. Effect +0.003, conf low. Fix: state the claim as "soft-threshold
  before summing beats linear sum"; arm of the trait-space test.
- **WC-BACKLOG.** = IR-3 / LT-3. Its exact-match-first design is the best of the three.
- **WC-SELFN.** = CI-06 / SG-8 with flow training; trained "on non-dev diseases" (confirm). Effect +0.01 to +0.02 vs transfer_jaccard (M10), ~0 to
  +0.004 mixture over pooled flow model, conf med.
- **WC-HAWKES.** = IR-6 inside a LightGBM, fitted on non-dev tasks (confirm). One difference of two snapshots. Effect of velocity features +0.001,
  conf med.
- **WC-CAPTURE.** Fatal: no. Test 1 is well posed, but on 53 dev diseases a Spearman has SE ~0.14, so "0.05 better than n_t0" is undecidable.
  Sources are not independent captures. nuAP effect 0 to +0.002, conf med. Fix: run test 1 on forward dev (n = 842).
- **WC-DCA.** Fatal: ~3.7k heavily duplicated samples for ~10k spins: it regularises to item-kNN / EASE (banned, no gain); on holdout it must be
  refitted per fold. Effect +0.002, conf med-high. Fix: keep only stage 2 (predict couplings from pair features), which is kernel learning for the
  unified lift.
- **WC-UPPERLIM.** = IR-4. Its diagnostic is the only part worth running.
- **WC-MARKET.** Fatal: symbol matching in grant abstracts is noisy and what survives is the T0 text-mining signal. Effect +0.001, conf high.
  Fix: drop; if IR-7's literature channel pays, add grants as one more channel.

## 2. Synthesis

### (a) Same idea in different words
1. **Universal donor / lifted transfer** (SC-2, EV-1, EV-2, EV-3, EV-5, EV-6, EV-8, LT-2, ML-3, IR-5, IR-10, SG-2 profiles; in learned form ML-1,
   IR-8, ML-5, SC-7, WC-DCA stage 2). Two different operators are being mixed up:
   - *extra rows E*: score = sum_r sim(seeds, E_r) E_r = seeds x (E' W E). This is a gene-gene co-membership kernel applied to the **seed vector**:
     guilt by association with a new kernel. It does not pass through disease rows at all. modet measured this family as ~0 on top of transfer
     (GO/Reactome "function users", EASE, propagation).
   - *column lift K*: score = transfer(L) x K = the transfer output smoothed over a gene-gene kernel. Disease-specific, and never run.
   Is one unified model the right test? **Yes as a harness, no as a single fused number.** Build one function
   `donor_transfer(context, row_families, kernels)` with: per-family donor budgets (never one top-50 over 100k+ rows: chance overlaps win, see SC-2);
   similarity = shrunk hypergeometric / PMI, not Jaccard; one non-negative weight per family, cross-fitted by cluster; a size- and degree-preserving
   rewiring null per family; the popularity-tie-break control; M1 fixed. Report each family alone and leave-one-out; a pooled "universal" number hides
   which source works and invites tuning 10+ weights on 53 diseases. Label-derived families (HPO, MeSH/PubTator, Monarch, OMIA, burden, supplement
   tables) are barred from holdout. Before building it: the **headroom diagnostic** (30 min) — for zero-support targets, how often is the target
   among the top-100 kernel neighbours of any seed, against degree-matched random genes, per kernel. If no kernel reaches 2x random, the cluster is
   dead without any model.
2. **Own-trait / trait-space genetics** (CI-03, CI-08, EV-4, ML-4, SG-2, SG-3, SG-9, LT-9, WC-L2GEMU, WC-XPOWER; nulls: IR-4, WC-UPPERLIM; gate: CI-05).
   One channel with 15 covered dev diseases. One test, arms in increasing complexity, plain MAGMA z first.
3. **Observation model / discoverability** (CI-06, SG-8, WC-SELFN, IR-1, EV-9, WC-CAPTURE's mixing weight): all reduce to a gene prior unless they
   beat the pooled-propensity control.
4. **Train on flows, not stocks** (ML-6, LT-7, ML-9, WC-SELFN, WC-HAWKES, IR-6, IR-9, CI-04): one shared 19.04 -> 21.04 table built strictly as of
   19.04, three arms (stock / flow / flow + precursors).
5. **Curation backlog and lag** (IR-3, LT-3, WC-BACKLOG; SG-7; own-row weak channels in IR-7, ML-9 level, LT-2 arm a): information-source arbitrage.
   Probably the largest effects on the list and the least "biology"; they define the stratum every other claim must be reported without.
6. **Directed perturbation edges** (CI-01, SC-1, SC-4, SC-9; natural version CI-02, SC-5, SG-6).
7. **Few-seed donor selection** (ML-8, SC-3, SG-4, SG-9, LT-2 arm c): untestable until a seed-subsampled benchmark exists.
8. **Module as output** (ML-7, SC-9, LT-6, WC-CAPTURE): all lack the trivial baseline (count model on n_t0; fixed k).

### (b) Directions nobody proposed
- **A strong in-harness reference**: a cross-fitted stacked learner on transfer + degree + popularity + constraint (what the modet headline is).
  Without it, no learned idea can be interpreted (M10).
- **Seed-subsampled benchmark** (3 / 5 / 10 seeds from the same tasks): creates the rare-disease stratum with full power and paired design.
- **Hierarchical prediction**: labels are propagated, so a parent's arrivals are the union of its descendants' arrivals. Score each descendant from
  its own seeds and combine upward (noisy-OR). Nobody uses the ontology as a structured output.
- **Label instability as a target**: 15.5 % of 25.09 pairs are gone in 26.09. Predict which seeds are unstable, down-weight them (evidence-score-
  weighted seeds and donor matrix instead of 0/1), and report on persistent targets only.
- **De-duplicated donors**: use direct (unpropagated) rows as donors; propagated ancestors are counted many times in top-50.
- **Placebo tests as standard**: the same pipeline predicting 19.04 -> 21.04 with inputs as of 19.04; predicting *removed* pairs; rewiring per source.
- **Calibrated cutoff by split-conformal across diseases**: a module size with coverage guarantee from dev tasks, no new data.
- **Multiplicity plan**: ~70 ideas x 3 benchmarks x 3-4 strata at +0.005 thresholds yields several false "passes". Shortlist on dev, one
  pre-registered primary endpoint each, Holm across the shortlist on confirm, confirm used once; bootstrap forward/holdout by ontology component,
  not by task.

### (c) Top 8 by expected information per hour
Prerequisites (30 min, not an idea): fix M1 with an equality test; add `transfer + popularity tie-break` and the cross-fitted `transfer + lam*z`
fusion helper; rerun the holdout leaderboard rows.

| # | Test (ideas decided) | Time | Kill rule |
|---|---|---|---|
| 1 | **Backlog, exact-match arm** (IR-3, LT-3, WC-BACKLOG), temporal dev, by arrival class and on `sa` | 45 min | < 5 % of class-B (old-evidence) dev arrivals in the top 100 by the match score alone, or fused delta < +0.010 nuAP on the full endpoint. If it passes it becomes a mandatory control arm, not a model. |
| 2 | **Own-row non-genetic T0 channels**, direct arm `transfer + lam*score` per channel (IR-7, ML-9 level, LT-2 arm a, LT-1's bar) | 30 min | best single channel < +0.010 nuAP on temporal dev AND < +0.006 on forward dev (25.09 per-datasource scores). |
| 3 | **Ingestion-lag ceiling** (SG-7): share of dev GWAS-only arrivals with `fy_gwas` <= 2020, then the archived-catalog arm | 5 + 45 min | ceiling < 15 % of GWAS-only arrivals, or catalog arm < +0.006 nuAP on forward dev. |
| 4 | **Stock vs flow learner** (ML-6 control, LT-7, WC-SELFN, WC-HAWKES, IR-6 as a column), features as of 19.04, cross-fitted by cluster in dev | 1 h | flow-trained minus stock-trained LightGBM < +0.008 nuAP on temporal dev: the whole "arrival objective" cluster stops. Precursor columns: each must add >= +0.005 over the flow model. |
| 5 | **Universal-donor headroom diagnostic, then two on-disk families** (paralog/ESM-kNN lift = EV-2, ML-1; HPO-2021 + IMPC rows = EV-1, LT-2 arm b) in the unified harness | 30 + 45 min | no kernel reaches 2x the degree-matched random hit rate for zero-support targets -> cluster dead. Otherwise: delta < +0.006 on forward dev, or the rewiring null keeps > 50 % of the gain. |
| 6 | **GWAS channel, plain z first, pooled endpoint** + SG-1's locus diagnostic (CI-03, CI-08, SG-3, EV-4, LT-9, WC-L2GEMU, WC-XPOWER, IR-2) | 45 min | plain MAGMA z: pooled GWAS-only recall@100 gain < 5 points on the 15 covered dev diseases (disease-bootstrap CI includes 0) -> stop the cluster on temporal, move it to forward. Any refinement (SMR, mash, crossing, host model) must beat plain z by >= 0.02 pooled AUROC with CI excluding 0. Locus: < 10 % of zero-support GWAS targets with a supported block-mate, or < 2x random blocks. |
| 7 | **Content cold-start, pooled AUC** on empty-column genes with the ready views (ML-1, IR-8, LT-4, LT-8, EV-9, ML-5, SC-7, ML-2) | 30 min | best view (kNN or ridge to the masked context) beats a covariates-only model (degree, constraint, expression breadth) by < 0.02 pooled within-pool AUC -> no transformer, no dossiers. |
| 8 | **Perturb-seq upstream score** (CI-01, SC-1; decides SC-2's experimental rows, SC-4, SC-9), forward dev, pre-registered immune/blood subset, covered genes only | 20 min | AUROC of zero-support targets against covered, expression-matched negatives < 0.55, or no gain over the perturbed-indicator control. |

Order: 3 (ceiling), 1, 2, then 4 and 5 in parallel on CPU; 6-8 after. Nothing in this list needs the GPU or the sealed fold.

## 3. The one warning to carry into round 2
On holdout (and partly forward) almost every external source proposed as a "donor" or feature is a re-encoding of the hidden label that the family
mask cannot see (HPO gene sets, MeSH/PubTator co-occurrence, Monarch, OMIA, GWAS catalog / MAGMA / PhenomeXcan / coloc, burden tables, supplement
gene sets, per-datasource OT scores, evidence baskets), and the harness itself currently leaks hidden holdout pairs through float residue (M1).
Holdout is valid only for label-matrix-only methods and strictly label-free inputs; claims go on temporal (dated inputs) and are cross-read on forward.
