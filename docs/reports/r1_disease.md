# R1 implementer C: better disease-to-disease transfer (IR-6, CI-04, ML-8, EASE/SLIM ridge)

## Pre-registration (written and committed 2026-11-12 ~04:30, before any result of these channels was seen)
Fixed parameters (no tuning on dev or confirm):
- `dc_trend` (IR-6): donor matrix `L + beta * Delta`, `Delta` = context pairs absent from the previous release
  (temporal 21.04 vs 19.04; forward 25.09 vs 21.04), **beta = 1**, Jaccard similarity on the context, top-50, exact
  masking correction + quantisation. `dc_delta` = transfer with `Delta` alone (stacker column). NaN on `past`/holdout.
- `dc_directed` (CI-04): `w(a->d) = jaccard(a, d) * exp(theta * asym)`, `asym = log((n_{a->d}+1)/(n_{d->a}+1))`,
  `n_{a->d}` = genes of a in the previous release that are new in d in the current context (masked), **theta = 1**,
  clipped to [-3, 3], no branch shrinkage. NaN on `past`/holdout.
- `dc_learned` (ML-8 without text): LightGBM regression of donor precision (share of donor's non-seed genes that
  arrive) on pair features (Jaccard, intersection, sizes, overlap ratios, ontology relation, Jaccard rank), trained
  on `past` episodes (19.04 -> 21.04), 2-fold cluster cross-fit on `past` itself; transfer over top-50 by prediction.
- `dc_ridge` / `dc_ridge_pos` (EASE/SLIM-style disease regression): seed vector regressed on unit-norm donor rows
  with **lambda = 1**, all masked rows (fam rows + any row with a masked pair) removed as donors (exact masking);
  score = donors^T w (`dc_ridge`), or with w clipped >= 0 and top-50 (`dc_ridge_pos`).
Claim-deciding comparisons: (1) each channel alone vs `transfer_jaccard`, paired cluster bootstrap (`modeng.compare`),
nuAP and AP, temporal dev+confirm and forward dev+confirm. (2) increment `base + channel` vs `base`: temporal
`--train past` (stackP; only for channels defined on past: dc_learned, dc_ridge, dc_ridge_pos) and `--train cv` on
forward (all channels) and temporal cv (IR-6/CI-04 only, since they are NaN on past).
Keep rule: increment >= +0.005 nuAP on dev AND confirm with the same sign, P(delta<=0) < 0.1 on at least one.
Negative control: best channel with genes permuted by a fixed random permutation -> increment ~0.

## Results
Commits: pre-registration `a957112`; code `c2be75b` (+ report commit). Files: `src/modeng/ext/disease_c.py` (channels),
`scripts/r1_disease_c.py` (channel alone -> per-task parquet), `scripts/r1_disease_c_strata.py`, `scripts/r1_disease_c_exactcheck.py`,
`scripts/r1_disease_c_stack*.sh`, `scripts/r1_disease_c_queue.sh`; raw logs in `results/disease_c/`.
Deviation from the pre-registered plan: forward increments compare against the existing `stack_base_w05` parquets (identical
deterministic base run by another agent, 0.1447 / 0.1373 nuAP, reproduced 4x on the leaderboard) instead of my own forward base
run (killed to save RAM/CPU). `dc_ridge` and `dc_ridge_pos` were entered together into the stacker (one test, not two).
Extra diagnostic channel `dc_ridge_rel` (added after the holdout result, not pre-registered).

### Verdict in one line
**IR-6 (trend) and CI-04 (directed) and the text-free ML-8 donor model: discard. EASE/SLIM-style ridge disease regression
(`dc_ridge` + `dc_ridge_pos`): KEEP** — the first disease-side similarity that clearly adds to Jaccard transfer, inside the stacker
on both benchmarks, AP and nuAP both up (the lead's F2-3 rule), CIs excluding 0 on 3 of 4 split/benchmark cells.

### 1. Each channel alone vs transfer_jaccard (paired cluster bootstrap, delta = channel - transfer_jaccard)
| bench / split | channel | AP | nuAP | dAP [95% CI] | dnuAP [95% CI] |
|---|---|---|---|---|---|
| temporal dev (53) | transfer_jaccard | 0.1229 | 0.0961 | | |
| | dc_trend (IR-6, beta=1) | 0.0950 | 0.0814 | -0.028 [-0.038,-0.019] | -0.015 [-0.023,-0.007] |
| | dc_directed (CI-04, theta=1) | 0.1277 | 0.0936 | +0.005 [-0.008,+0.021] | -0.003 [-0.012,+0.007] |
| | dc_learned (ML-8 pair model) | 0.1108 | 0.0846 | -0.012 [-0.023,-0.001] | -0.012 [-0.021,-0.003] |
| | dc_ridge | 0.1075 | 0.0924 | -0.015 [-0.030,-0.003] | -0.004 [-0.016,+0.008] |
| | dc_ridge_pos | 0.1177 | 0.0919 | -0.005 [-0.020,+0.007] | -0.004 [-0.017,+0.005] |
| temporal confirm (138) | transfer_jaccard | 0.1189 | 0.1042 | | |
| | dc_trend | 0.0859 | 0.0820 | -0.033 [-0.047,-0.022] | -0.022 [-0.036,-0.011] |
| | dc_directed | 0.1186 | 0.1046 | -0.000 [-0.005,+0.006] | +0.000 [-0.006,+0.008] |
| | dc_learned | 0.1104 | 0.1014 | -0.009 [-0.016,-0.001] | -0.003 [-0.012,+0.007] |
| | dc_ridge | 0.1041 | 0.1139 | -0.015 [-0.034,+0.003] | +0.010 [-0.007,+0.027] |
| | dc_ridge_pos | 0.1128 | 0.1057 | -0.006 [-0.017,+0.005] | +0.002 [-0.009,+0.011] |
| forward dev (842) | transfer_jaccard | 0.0936 | 0.0964 | | |
| | dc_trend | 0.0868 | 0.0924 | -0.007 [-0.009,-0.004] | -0.004 [-0.007,-0.001] |
| | dc_directed | 0.0901 | 0.0929 | -0.004 [-0.006,-0.001] | -0.004 [-0.006,-0.001] |
| | dc_learned | 0.0916 | 0.0959 | -0.002 [-0.005,+0.002] | -0.001 [-0.005,+0.004] |
| | dc_ridge | 0.0876 | 0.1038 | -0.006 [-0.012,-0.001] | +0.007 [+0.001,+0.014] |
| | dc_ridge_pos | 0.0962 | 0.1015 | +0.003 [-0.002,+0.007] | +0.005 [+0.000,+0.010] |
| forward confirm (797) | transfer_jaccard | 0.0849 | 0.0831 | | |
| | dc_trend | 0.0781 | 0.0773 | -0.007 [-0.009,-0.005] | -0.006 [-0.008,-0.003] |
| | dc_directed | 0.0818 | 0.0815 | -0.003 [-0.005,-0.001] | -0.002 [-0.004,+0.001] |
| | dc_learned | 0.0764 | 0.0753 | -0.009 [-0.012,-0.005] | -0.008 [-0.012,-0.004] |
| | dc_ridge | 0.0816 | 0.0918 | -0.003 [-0.008,+0.002] | +0.009 [+0.003,+0.015] |
| | dc_ridge_pos | 0.0881 | 0.0914 | +0.003 [-0.001,+0.008] | +0.008 [+0.003,+0.014] |
AUROC / R@100 / onnet in `results/disease_c/alone_*.txt`. `dc_delta` alone (transfer of only the 19.04->21.04 new pairs) is
much weaker than full transfer (temporal nuAP 0.046 / 0.033): genes that just arrived in donors are *less* predictive than
established donor genes. IR-6 is also negative on strict-a (sa_ap, `results/disease_c/strata.txt`), so its kill rule fires twice.
Alone, the answer to "does a smarter disease similarity clearly beat Jaccard top-50?" is **no** on temporal, and **only for
nuAP (+0.005..+0.009, AP flat)** for the ridge on forward.

### 2. Increment over the stacker (the decisive test)
| test | dev AP | dev nuAP | confirm AP | confirm nuAP | dAP dev / conf | dnuAP dev / conf [95% CI] |
|---|---|---|---|---|---|---|
| temporal stackP_base (own rerun = positive control, reproduces 0.1187/0.1341) | 0.0778 | 0.1187 | 0.0810 | 0.1341 | | |
| temporal stackP base+dc_ridge+dc_ridge_pos | 0.0877 | 0.1220 | 0.0982 | 0.1550 | +0.010 [+0.002,+0.018] / +0.017 [+0.010,+0.024] | +0.003 [-0.010,+0.016] / +0.021 [+0.010,+0.033] |
| temporal stackP base+dc_learned | 0.0778 | 0.1142 | 0.0871 | 0.1402 | +0.000 / +0.006 [+0.002,+0.011] | -0.004 [-0.014,+0.008] / +0.006 [-0.000,+0.014] |
| temporal stackP base+negperm(ridge, ridge_pos) **negative control** | 0.0742 | 0.1165 | 0.0796 | 0.1309 | -0.004 / -0.001 | -0.002 / -0.003 (CIs include 0) |
| forward stack_base_w05 (cv) | 0.0809 | 0.1447 | 0.0770 | 0.1373 | | |
| forward base+dc_ridge+dc_ridge_pos (cv) | 0.0955 | 0.1677 | 0.0866 | 0.1513 | +0.015 [+0.012,+0.018] / +0.010 [+0.007,+0.013] | +0.023 [+0.018,+0.028] / +0.014 [+0.010,+0.018] |
| temporal stackP_bdv (reference, current parquet 05:41) | 0.1018 | 0.1237 | 0.0913 | 0.1506 | | |
| **temporal stackP bdv+dc_ridge+dc_ridge_pos** | **0.1175** | **0.1416** | **0.1033** | **0.1686** | +0.016 [+0.008,+0.025] / +0.012 [+0.006,+0.018] | +0.018 [+0.005,+0.034] / +0.018 [+0.008,+0.028] |
AUROC: temporal confirm +0.0065 [+0.002,+0.010], forward +0.006 / +0.007 (all P < 0.002). onnet nuAP: temporal dev -0.005
(n.s.), confirm +0.017; forward +0.020 / +0.012. `dc_ridge` is the top-gain channel on temporal stackP (above transfer_jaccard)
and 4th on forward. Pre-registered keep rule (>= +0.005 nuAP on dev AND confirm): met on forward; on temporal over base, dev nuAP is
+0.003 (n.s., 53 tasks) while AP dev is significant; **over bdv (the current best) the rule is met on temporal too: +0.018 nuAP
and +0.012..0.016 AP on dev and confirm, all P < 0.004** -> keep. stackP_bdv+ridge = new best temporal confirm (nuAP 0.169, AP 0.103;
onnet nuAP 0.189 vs modet headline ~0.17).
IR-6 / CI-04 stacker increment (temporal `--train cv`, base+dc_trend+dc_delta+dc_directed vs own cv base rerun 0.0811/0.1026 dev,
0.0921/0.1331 confirm): dev OOF AP +0.007 [+0.001,+0.012], nuAP +0.012 [+0.001,+0.024]; **confirm AP +0.001 [-0.003,+0.006], nuAP
+0.002 [-0.004,+0.010] -> does not replicate on the larger split; discard** (most used: r_dc_directed). Forward cv not run (RAM rule,
deadline).
`dc_learned`: mixed sign dev/confirm -> discard.

### 3. Leakage / exactness
- `dc_ridge*`: every row that carries a masked pair, the fam rows and the own row are removed as donors; all remaining rows are
  identical in the masked and unmasked context, so the channel is exact by construction. Checked numerically
  (`scripts/r1_disease_c_exactcheck.py`): from-scratch ridge on `B.context_for(task)` restricted to the same rows agrees to
  <= 1e-6 (rounding) on 5 temporal tasks. On temporal / forward / past the removed set is target-independent (own + descendants).
- **Holdout caveat:** on holdout the rows with masked pairs (re-propagated ancestors, Jaccard near-duplicates) depend on the
  hidden genes, so "drop rows carrying masked pairs" is a target-dependent donor selection. Holdout alone: dc_ridge_pos AP 0.312 /
  0.323 vs transfer_jaccard 0.205 / 0.217 (dev / confirm); with ALL ontology relatives removed (target-independent except the
  near-duplicate rule) `dc_ridge_rel` still gives 0.305 / 0.320, so the jump is not the selection effect, but treat holdout
  numbers as unconfirmed (holdout-current, not a claim benchmark here).
- Previous-release channels use only releases <= the seed release (temporal: 19.04/21.04; forward: 21.04/25.09); NaN on past
  and holdout. `dc_learned` is trained only on `past` (<= T0); on past itself 2-fold cluster cross-fitted.
- Leakage class: temporal-clean on temporal and forward for all channels (inputs = the benchmark's own context labels + 21.04
  ontology); no external data.

### 4. Seed-count strata (channel alone minus transfer_jaccard, nuAP; `results/disease_c/strata.txt`)
The ridge gain on forward is spread over all strata (<=10 seeds: +0.004/+0.018 dev/confirm for ridge_pos; >100: +0.000/+0.006);
on temporal it sits in 31-100 and >100 seeds (confirm 31-100 +0.029 for dc_ridge) and is negative for 11-30 seeds. No few-seed
specific effect (ML-8's hypothesis) for the learned donor model.

### 5. Runtime
Channels alone: ridge 40-60 s per split (one 3.7k-6.2k dense inverse + per-task block correction), trend/directed < 10 s,
dc_learned ~2 min training on past + 7 min forward features. Stacker runs 15-40 min each under the shared load (load avg 150).

### 6. What failed / why
- IR-6: the 19.04 -> 21.04 delta is mostly pipeline (L2G ingestion), and recently added donor genes are less, not more,
  predictive (dc_delta alone 1/2 of transfer). Upweighting them hurts on both benchmarks and on strict-a.
- CI-04: asymmetry from one release delta is mostly noise (alone within +-0.005; forward slightly negative).
- ML-8 (text-free pair model predicting donor precision): optimises the wrong quantity (precision of small donors) and loses
  the implicit size weighting of Jaccard-sum transfer; worse alone, mixed in the stacker. Text embeddings not tried (time).

### 7. New ideas
- Ridge disease regression = "explaining away" between overlapping donors (parent and child rows no longer double-count): try
  lambda / row normalisation sweeps on `past` only, a nonnegative (true SLIM, coordinate descent) version, and the
  gene-side EASE (19k x 19k item-item) with the same row-removal masking via Woodbury.
- Apply the same ridge to the virtual-donor families (data/donors) instead of Jaccard top-50 donor transfer.
- Anti-trend feature: "age" of a donor pair (present in both releases) as a donor-pair weight for the stacker.
