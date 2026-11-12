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

## 2. Selection (fixed 2026-11-12 04:35 CET, before any held-out number)

40 candidates logged (cand_0..cand_39, 8 generations, all `ok`, all lint `ok`). Best fit600 nuAP = cand_32 (0.07039);
the tie set (within 0.0005) is {17, 32, 33, 19, 36, 37}; tie-break by AP selects **cand_17** (nuAP 0.06990, AP 0.02874):
cosine similarity, top-200 neighbours weighted cos^2, plus 1e-3 x all-row cosine transfer (tail), times (degree+1)^-0.6.

## 3. Candidate log (all 40, `results/evolve/log.tsv`; 8 generations, mutator = Claude)

Generations: g1 = 1-6 (from 0), g2 = 7-11, g3 = 12-16, g4 = 17-21, g5 = 22-25, g6 = 26-30, g7 = 31-35, g8 = 36-39 (parents =
current top-3). Every candidate ran (no failures), all lint-clean, 1-18 s each on 600 episodes (6 threads).

| cand | parent | fit600 nuAP | AP | AUROC | mutation |
|---|---|---|---|---|---|
| 0 | - | 0.0406 | 0.0470 | 0.677 | cand_0: transfer_jaccard baseline (top-50 Jaccard-similar context diseases, sum of sim x label row). |
| 1 | 0 | 0.0415 | 0.0473 | 0.694 | cosine similarity instead of Jaccard, k=50. |
| 2 | 0 | 0.0419 | 0.0469 | 0.725 | Jaccard^2 weights (sharper), k=100. |
| 3 | 0 | 0.0402 | 0.0463 | 0.676 | Jaccard top-50, label rows scaled by row^-0.5 (large diseases spread less mass per gene). |
| 4 | 0 | 0.0405 | 0.0468 | 0.680 | IDF-weighted Jaccard (seed genes weighted log(N/df), df = masked column count), k=50. |
| 5 | 0 | 0.0584 | 0.0335 | 0.676 | transfer_jaccard x (degree+1)^-0.3 (nuAP favours low-degree genes); zero scores stay zero. |
| 6 | 0 | 0.0403 | 0.0435 | 0.676 | transfer_jaccard / (1 + gene popularity)^0.3 (popularity = masked column count). |
| 7 | 5 | 0.0629 | 0.0266 | 0.675 | transfer_jaccard x (degree+1)^-0.6 (stronger degree discount). |
| 8 | 5,2 | 0.0586 | 0.0369 | 0.724 | Jaccard^2 weights, k=100, x (degree+1)^-0.3. |
| 9 | 5,1 | 0.0610 | 0.0325 | 0.693 | cosine similarity, k=50, x (degree+1)^-0.3. |
| 10 | 5 | 0.0605 | 0.0352 | 0.774 | cand_5 plus a weak fallback for genes no neighbour carries: 1e-3 x global transfer over ALL rows with sim>0 |
| 11 | 5,2 | 0.0632 | 0.0345 | 0.757 | Jaccard^2, k=200, x (degree+1)^-0.45. |
| 12 | 11,9,10 | 0.0688 | 0.0320 | 0.770 | cosine^2, k=200, x (degree+1)^-0.45, plus 1e-3 all-row tail. |
| 13 | 11,10 | 0.0636 | 0.0347 | 0.774 | Jaccard^2, k=200, x (degree+1)^-0.45, plus 1e-3 all-row tail. |
| 14 | 11,7 | 0.0651 | 0.0315 | 0.756 | Jaccard^2, k=200, x (degree+1)^-0.6. |
| 15 | 11 | 0.0635 | 0.0332 | 0.755 | Jaccard^2 k=200 x deg^-0.45, ontology relatives' rows get +0.05 similarity before ranking. |
| 16 | 11 | 0.0652 | 0.0234 | 0.733 | Jaccard^2 k=200 transfer (max-normalised) + 0.3 x item-item CF (max-normalised), x deg^-0.45. |
| 17 | 12,14 | 0.0699 | 0.0287 | 0.767 | cosine^2, k=200, x (degree+1)^-0.6, plus 1e-3 all-row tail. |
| 18 | 12 | 0.0647 | 0.0335 | 0.770 | cosine^3 over ALL rows with overlap (no k cut), x (degree+1)^-0.45, plus 1e-3 tail. |
| 19 | 12,16 | 0.0702 | 0.0278 | 0.745 | cand_12 (max-normalised) + 0.1 x item-item CF (max-normalised), x deg^-0.45. |
| 20 | 12 | 0.0684 | 0.0349 | 0.774 | weights inter^2 / row^1.5 (stronger large-row penalty than cosine^2), k=200, x deg^-0.45, plus tail. |
| 21 | 12 | 0.0630 | 0.0285 | 0.771 | hypergeometric similarity -log10 P(X >= inter) (gene universe = genes in any context row), weight = that^2, |
| 22 | 19,17 | 0.0693 | 0.0238 | 0.737 | cosine^2 k=200 transfer + 0.1 item-item CF + 1e-3 tail, x (degree+1)^-0.6. |
| 23 | 17 | 0.0596 | 0.0311 | 0.760 | softmax neighbour weights exp(cos / 0.05) over top-200 cosine rows, + 1e-3 tail, x deg^-0.6. |
| 24 | 17 | 0.0651 | 0.0254 | 0.763 | two-hop query expansion (Rocchio): first pass cosine^2 k=200 transfer; the query becomes |
| 25 | 17 | 0.0681 | 0.0274 | 0.765 | cand_17 / (1 + gene popularity)^0.2 (masked column count). |
| 26 | 17 | 0.0695 | 0.0251 | 0.762 | cosine^2, k=200, x (degree+1)^-0.8, plus 1e-3 tail. |
| 27 | 17 | 0.0684 | 0.0296 | 0.767 | cosine^2.5, k=400, x (degree+1)^-0.6, plus 1e-3 tail. |
| 28 | 17 | 0.0697 | 0.0290 | 0.766 | seed-count-adaptive neighbourhood k = clip(8 x n_seeds, 30, 300), cosine^2, x deg^-0.6, + tail. |
| 29 | 17 | 0.0667 | 0.0279 | 0.767 | degree-weighted cosine: seed overlaps weighted (deg+1)^-0.5 (a shared rare/low-degree gene is stronger |
| 30 | 17,15 | 0.0635 | 0.0269 | 0.766 | cosine^2 k=200 x deg^-0.6 + tail; ontology ANCESTOR/descendant rows' weights doubled (relatives |
| 31 | 17 | 0.0634 | 0.0248 | 0.762 | max aggregation: score = max_r cos_r^2 L[r,g] + 0.1 x sum (k=200), + 1e-3 tail, x deg^-0.6. |
| 32 | 17 | 0.0704 | 0.0283 | 0.765 | cosine^2 k=200 transfer + 0.05 x all-row tail (larger tail weight), x deg^-0.6. |
| 33 | 19,32 | 0.0702 | 0.0278 | 0.746 | cosine^2 k=200 transfer + 0.1 item-item CF + 0.05 tail (all max-normalised), x deg^-0.45. |
| 34 | 28,19 | 0.0696 | 0.0262 | 0.741 | adaptive k = clip(8 n_seeds, 30, 300), cosine^2 transfer + 0.1 item-item CF + 1e-3 tail, x deg^-0.5. |
| 35 | 17 | 0.0674 | 0.0299 | 0.771 | similarity = sqrt(Jaccard x cosine) (combined notion), weights sim^2, k=200, + 1e-3 tail, x deg^-0.6. |
| 36 | 32 | 0.0701 | 0.0271 | 0.763 | cosine^2 k=200 transfer + 0.2 x all-row tail, x deg^-0.6. |
| 37 | 32 | 0.0702 | 0.0264 | 0.762 | cosine^2 k=200 transfer + 0.05 tail, x deg^-0.7. |
| 38 | 32 | 0.0683 | 0.0260 | 0.764 | cosine^1.5 k=200 transfer + 0.05 tail, x deg^-0.6. |
| 39 | 32 | 0.0680 | 0.0268 | 0.766 | neighbour rows weighted cos^2 x (fraction of the row outside the seeds)^0.5 (rows that add new genes |

What the search found, in order of size:
1. **Degree discount** (x (deg+1)^-a): 0.041 -> 0.058 (a = 0.3) -> 0.063 (a = 0.6). This is nuAP alignment (nuAP weights
   positives by 1/deg), and it costs AP (0.047 -> 0.027). Beyond a = 0.6 no further gain (0.8: 0.0695, 0.7: 0.0702).
2. **Cosine instead of Jaccard, squared weights, wider neighbourhood (k = 200)**: +0.004 to +0.006 on top of the discount,
   and AUROC 0.68 -> 0.77.
3. **Tail** (weak all-row transfer that orders the genes no top-k neighbour carries): +0.001 to +0.002 nuAP, +0.05-0.10 AUROC.
4. Neutral or negative: IDF-weighted overlap, row-size penalties, popularity normalisation, hypergeometric similarity,
   softmax temperature, max aggregation, ontology-relative boosts (+0.05 or x2; both -0.006), Rocchio two-hop expansion,
   neighbour-novelty weights, degree-weighted overlap; item-item CF (+0.1 weight) was +0.0003 (inside the tie band).
   The fitness plateaued at ~0.070 from generation 4 on.

## 4. The winning program (cand_17 -> channel `evo_transfer`, `src/modeng/ext/evolved_k.py`)

    for each task: L = masked context; cos_r = |seeds & row r| / sqrt(|seeds| |row r|), own row excluded
    s(g)  = sum over the 200 rows with highest cos of cos_r^2 * L[r, g]
    T(g)  = sum over all rows with cos > 0 of cos_r * L[r, g];   T /= max T
    score(g) = (s(g) + 0.001 * T(g)) * (deg(g) + 1)^-0.6

Read as: a sharper, wider transfer (cosine normalises for large context diseases less harshly than Jaccard; squaring
concentrates weight on the closest diseases while k = 200 still lets distant relatives contribute), a fallback ordering
for genes outside the neighbourhood, and an explicit prior against high-degree (well-studied) genes. Works on
past/temporal/forward/holdout (smoke test holdout dev: nuAP 0.228 vs transfer_jaccard 0.198, AP 0.085 vs 0.205; not a
pre-registered claim). Leakage class = that of the context; exact masking via `context_for` (no float-correction route);
inputs = seeds, masked context, network degree (no ontology used by the winner).

## 5. Held-out results (run once, `scripts/evolve_heldout.py`, `results/evolve/heldout.tsv`)

Alone vs transfer_jaccard (paired cluster bootstrap, 10,000; delta [95 % CI]):

| bench / split | nuAP evo vs tj | delta nuAP | AP evo vs tj | delta AP | onnet nuAP delta | AUROC delta |
|---|---|---|---|---|---|---|
| temporal dev (53) | 0.1298 vs 0.0961 | +0.0337 [+0.022, +0.049] | 0.0644 vs 0.1229 | -0.058 [-0.087, -0.033] | +0.071 [+0.050, +0.093] | +0.119 |
| **temporal confirm (138)** | **0.1395 vs 0.1042** | **+0.0354 [+0.027, +0.047]** | 0.0667 vs 0.1189 | -0.052 [-0.068, -0.035] | +0.053 [+0.036, +0.073] | +0.109 |
| forward dev | 0.1364 vs 0.0964 | +0.0401 [+0.034, +0.047] | 0.0642 vs 0.0936 | -0.029 [-0.035, -0.024] | +0.060 [+0.052, +0.068] | +0.092 |
| forward confirm | 0.1225 vs 0.0831 | +0.0394 [+0.033, +0.045] | 0.0572 vs 0.0849 | -0.028 [-0.034, -0.022] | +0.055 [+0.048, +0.062] | +0.089 |

Primary pre-registered comparison (temporal confirm nuAP): **met** (+0.035, CI lower bound +0.027). AP falls by ~half on
every split: the gain is a re-ranking towards low-degree genes, which is what nuAP rewards.

Post-hoc decomposition (labelled, not pre-registered): cand_7 = transfer_jaccard x (deg+1)^-0.6 alone gives +0.030
(temporal confirm), +0.030 (forward confirm) of the +0.035 / +0.039; i.e. ~85 % (temporal) / ~75 % (forward) of the
single-channel nuAP gain is the degree discount; the cosine / k / tail changes add +0.006 to +0.009 nuAP and all of the
AUROC gain (cand_7 AUROC is 0.002-0.004 below transfer_jaccard).

**Stacker** (`scripts/evolve_stack.py` = memory-lean replica of `modeng.stack.run_past`, same rows / seeds / params,
LightGBM 6 threads; the first attempt via `python -c 'stack.run_past'` was killed (OOM, machine at 88/90 GB from parallel
stackers) after the dev line, which was bit-identical to the rerun): `stackP_bdv_evo` vs `stackP_bdv`:

| split | nuAP | delta nuAP | AP delta | onnet nuAP delta | AUROC delta |
|---|---|---|---|---|---|
| dev | 0.1234 vs 0.1237 | -0.0003 [-0.009, +0.007] | +0.0002 [-0.006, +0.005] | +0.012 [+0.001, +0.021] | +0.002 [-0.003, +0.009] |
| **confirm** | **0.1480 vs 0.1506** | **-0.0026 [-0.008, +0.002]** | -0.0008 [-0.005, +0.003] | -0.0002 [-0.007, +0.007] | +0.0048 [+0.001, +0.008] |

`evo_transfer` becomes the top stacker feature (22 % of gain, its rank copy 9 %), displacing transfer_jaccard and the
degree/popularity ranks, but adds nothing net: the stacker already learns the degree trade-off (degree and popularity are
base channels, positives are weighted deg^-0.5) and the sharper transfer form. Reference reproduced: stackP_bdv confirm
nuAP 0.1506.

## 6. Overfitting check

| | delta nuAP vs cand_0 / transfer_jaccard |
|---|---|
| past fit600 (selection set) | +0.0293 (0.0699 vs 0.0406) |
| past complement (1,186 unused episodes) | +0.0291 (0.0770 vs 0.0479) |
| temporal dev | +0.0337 |
| temporal confirm | +0.0354 |
| forward confirm | +0.0394 |

Shrinkage ratio (temporal confirm / fit600) = 1.21: no selection overfitting. Expected, because the found structure is a
handful of smooth, low-dimensional knobs (one exponent, one k, one similarity), not episode-specific code; 40 candidates on
600 episodes is a mild multiple-comparison load. The tie band mattered: the nominal fit600 maximum (cand_32, +0.0005)
was not chosen.

## 7. Verdict

- Kill rule part 1 (complement-past gain >= 0.005): passed (+0.029).
- Kill rule part 2 (stacker gain on confirm with CI excluding 0): **failed** (-0.0026 [-0.008, +0.002]).
- **Discard as a stacker channel; keep as the best single hand-readable scorer.** As a standalone baseline, evo_transfer
  is a much stronger transfer reference for nuAP (temporal confirm 0.140 vs 0.104; forward confirm 0.123 vs 0.083), and
  it is the honest "what nuAP rewards" control: anyone claiming a nuAP gain over transfer_jaccard should beat this
  (or cand_7) first. It is not an improvement in AP.
- Lesson for the evolve idea: with nuAP as fitness and degree available, the search spends its gain on the metric's degree
  weighting, which the stacker already has. A useful next run would use stacker-residual fitness (gain of the stacker with the
  candidate added, or AP on the stacker's residual ranking) or forbid degree as an input; both cost ~minutes per candidate.

## 8. Files, commands, runtime

- `scripts/evolve_eval.py` (frozen evaluator), `src/modeng/evolve/{harness,ops,cand_0..39}.py`, `results/evolve/log.tsv`
- `src/modeng/ext/evolved_k.py` (`evo_transfer`), `scripts/evolve_heldout.py` -> `results/evolve/heldout.tsv`,
  per-task parquet `results/{temporal,forward}/{dev,confirm}/{evo_transfer,evo_c7_jacdeg}.parquet`
- `scripts/evolve_stack.py base,donor,view,evo_transfer bdv_evo` -> `results/evolve/stackP_bdv_evo.txt`,
  `results/temporal/{dev,confirm}/stackP_bdv_evo.parquet` (model `results/evolve/model_stackP_bdv_evo.txt`, not committed)
- Rows appended to `results/leaderboard.tsv` (shared file, not committed by me: other agents have uncommitted rows in it).
- Runtime: search 40 x ~3 s; held-out ~1 min; stacker ~12 min.
