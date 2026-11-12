# R2 F2-3: evolved transfer kernel (LLM-as-mutation-operator program search)

## 1. Pre-registration (written and committed 2026-11-12 ~04:30 CET, before the first evaluation)

- **Search space.** Python programs `src/modeng/evolve/cand_<n>.py` exposing `score(tasks, B) -> float32 [n_tasks, n_genes]`.
  The mutation operator is the implementing agent (Claude), proposing 4-6 mutations per generation of the current top-3
  programs (similarity: Jaccard / cosine / overlap coefficient / hypergeometric / IDF-weighted overlap; k, softmax temperature,
  size penalties, multi-hop transfer, ontology-relative boosts, popularity normalisation of label rows, deg^-a reweighting,
  combinations). Start: `cand_0` = re-implementation of `transfer_jaccard` (k = 50, sum of Jaccard x label row).
  Budget: <= 40 candidates in total; every candidate is logged (failures included, never deleted).
- **Inputs a candidate may use.** Only through `modeng.evolve.harness.SafeBench`: task seeds, disease id, the exactly masked
  context matrix `B.L(t)` (= `Benchmark.context_for`, no float correction), own context row, ontology relatives
  (release-matched: past/temporal 21.04 ontology only, forward <= 25.09, holdout all), network degree. Targets, exclude
  lists, alt sets and the mask pairs are not exposed. Lint (logged): no disease/gene id strings, no integer literals > 1000,
  no private harness access.
- **Fitness.** Frozen evaluator `scripts/evolve_eval.py` (committed with this section, never edited afterwards): mean nuAP over
  a fixed random subset of 600 `past` episodes (`np.random.default_rng(0).choice(1786, 600, replace=False)`); tie-break mean AP
  (ties = |delta nuAP| < 0.0005). Log: `results/evolve/log.tsv`.
- **Selection of the final program.** The candidate with the highest fit600 nuAP among all logged `ok` candidates with
  lint `ok` (past fitness alone). No temporal / forward / holdout number is looked at before the choice is fixed and written
  into this file.
- **Within-past overfitting check.** The final program and `cand_0` are scored once on the 1,186 complement `past` episodes
  (`--complement`); reported, not used for selection.
- **Held-out tests (once, final program only), vs `transfer_jaccard`.** Channel `evo_transfer` (`src/modeng/ext/evolved_k.py`)
  alone on temporal dev and confirm (all-gene nuAP and AP, onnet nuAP) and forward dev and confirm; paired cluster bootstrap
  (`modeng.compare.paired_bootstrap`, n = 10,000).
  Primary claim-deciding comparison: **temporal confirm, nuAP, evo_transfer vs transfer_jaccard** (gain if CI lower bound > 0).
  Secondary: forward confirm nuAP; stacker increment `stackP_bdv_evo` (`--train past`, channels base,donor,view,evo_transfer)
  vs `stackP_bdv` on temporal confirm nuAP (reference 0.151), paired bootstrap.
- **Overfitting statistic.** (past fit600 gain over cand_0) vs (past complement gain) vs (temporal confirm gain); the
  shrinkage ratio is reported.
- **Kill rule (from the idea card).** Complement-past gain < 0.005 nuAP, or the stacker gain on confirm has a CI including 0
  -> "no usable gain".
