# R1 implementer E: curation backlog (IR-3 identity linking, LT-3 / WC-BACKLOG curation debt, IR-7 weak-evidence funnel, SG-7)

Date 2026-11-12 (04:15-06:40). Module `src/modeng/ext/backlog_e.py` (commit 2989b3f). Scripts in the agent scratchpad
(`bl_mass.py`, `bl_temporal.py`, `bl_bins.py`, `runstack.py` = `modeng.stack.run/run_past` with LightGBM threads from `OMP_NUM_THREADS`).

**What this is.** These channels forecast what the Open Targets *label pipeline* attaches next from evidence that is already in
the seed release but does not count as a label yet (sub-threshold genetic scores, other evidence types, evidence filed under
another disease id with the same name). They predict the label process (curation, L2G re-scoring, ingestion), **not biology**.
Every number below is a statement about OT's pipeline. Per the stats critic, the result is a mandatory **control arm / stratum**
for other temporal and forward claims, not a model of disease genes.

## Channels (all from the SEED release only; OT 26.09 never read)
| bench | input | ontology | leakage class |
|---|---|---|---|
| forward | OT 25.09 `association_by_datasource_direct` | 25.09 | temporal-clean (inputs = seed release) |
| temporal | OT 21.04 `associationByDatasourceDirect` (sealed + relatives' rows dropped) | 21.04 | temporal-clean |
| past (training) | OT 19.04 (genetic sources only) | 21.04 | <= T0; non-genetic channels NaN |
| holdout | none: all NaN (per-datasource scores are label re-encodings on holdout; barred) | | |

Family F(d) = task + its descendants in the seed-release ontology; channel = max datasource score over F(d):
`bl_genweak` (GWAS L2G / credible sets, ClinVar, phewas: the sub-threshold part; strong ones are seeds), `bl_lit` (europepmc),
`bl_animal` (phenodigm / impc), `bl_drug` (chembl), `bl_somatic` (CGC, intogen, eva_somatic, cancer_biomarkers),
`bl_func` (expression atlas, reactome, slapenrich, progeny, sysbio, crispr), `bl_any`, `bl_nsrc` (number of groups),
`bl_xid` (IR-3 exact-match arm: any evidence, label sources counted as 1, filed under other disease ids that are not in F(d) and not
ancestors, whose normalised name or exact synonym equals the task's name / an exact synonym, seed-release disease table; no fuzzy
or embedding matching), `bl_any_perm` (negative control: bl_any with gene identities permuted).

## 1. How much target mass is backlog
**Forward dev (842 tasks, 53,749 targets):** 46.9 % of 26.09 arrivals already had some 25.09 evidence for the disease family
(46.1 % of nuAP weight). Almost all of it is sub-threshold genetic evidence: **39.0 % of targets** (40.2 % of nuAP weight) had a
25.09 genetic score in (0, 0.5) in the family, 38.2 % from `gwas_credible_sets`, 0.9 % from ClinVar. Literature 10.7 %, animal
1.8 %, exact-match other ids 2.1 %. Permuted control: 11.4 % (chance overlap with dense columns).

Conversion (forward dev, candidates = non-seed genes): the weak-genetic score is a near-calibrated forecast of crossing the 0.5
threshold within one release:

| bl_genweak bin | 0 | (0,0.05) | 0.05-0.1 | 0.1-0.2 | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 |
|---|---|---|---|---|---|---|---|
| candidates | 15.9 M | 51 k | 56 k | 49 k | 36 k | 27 k | 32 k |
| P(target in 26.09) | 0.002 | 0.016 | 0.023 | 0.048 | 0.102 | 0.168 | **0.259** |

Literature, in contrast, is flat (0.5-0.8 % at every score). `bl_xid` >= 0.5: 16 % conversion (3,952 candidates).

**Temporal (21.04 -> 25.09):** own-family weak genetic genes are `unknown` and not ranked, so that part of the backlog is
invisible here by construction (4 % of targets have weak genetic evidence left, through 21.04 vs 25.09 ontology differences).
Any 21.04 evidence: 21.1 % of dev targets (17.0 % nuAP weight), 31.4 % confirm. Exact-match other ids (`bl_xid`): **14.1 % of dev
targets** (12.3 % nuAP weight), but only 3.9 % on confirm (dev is dominated by a few tasks). By modet arrival class (dev; B = old
evidence newly attached): evidence share A 22 %, **B 25 %**, X 16 %, U 29 %; class-B arrivals in the top 100 by the channel alone:
bl_xid 6.8 %, bl_lit 7.0 %, bl_nsrc 7.0 % (confirm: 3.4 %, 9.9 %, 12.2 %).

## 2. Channels alone (no fusion)
| bench/split | channel | AP | nuAP | AUROC | R@100 |
|---|---|---|---|---|---|
| forward dev | transfer_jaccard (ref) | 0.094 | 0.096 | 0.732 | 0.247 |
| forward dev | stack_E_base (ref, OOF) | 0.081 | 0.145 | 0.823 | 0.221 |
| forward dev | **bl_genweak** | **0.160** | **0.180** | 0.735 | **0.383** |
| forward dev | bl_any | 0.127 | 0.139 | 0.786 | 0.330 |
| forward dev | bl_xid | 0.021 | 0.021 | 0.515 | 0.029 |
| forward dev | bl_lit | 0.016 | 0.017 | 0.571 | 0.057 |
| forward dev | bl_any_perm (neg. control) | 0.004 | 0.004 | 0.497 | 0.004 |
| temporal dev | bl_genweak / bl_lit / bl_xid / bl_any | 0.015 / 0.013 / 0.011 / 0.009 | 0.018 / 0.011 / 0.010 / 0.008 | | |
| temporal dev | bl_any_perm | 0.003 | 0.003 | 0.473 | 0.008 |

Stratified (forward dev, pooled top-100 recall): transfer_jaccard finds 16.6 % of backlog targets (bl_genweak > 0, n = 20,959)
and 13.5 % of the others (n = 32,790); bl_genweak finds 44.8 % of backlog targets and by construction 0 % of the others.

On forward a single lookup (max sub-threshold genetic score in the family) beats the stacked base learner by +0.036 nuAP and
doubles AP.

## 3. Increment over the stacker (LightGBM, modeng.stack; channels base + all bl_* except perm)
FORWARD_TABLE

Temporal (`--train cv`, 5-fold cluster OOF on dev, dev-trained model on confirm; paired cluster bootstrap, 5,000 draws):
| split | metric | stack_E_base | + bl_* | delta [95 % CI] | P(delta<=0) |
|---|---|---|---|---|---|
| dev | nuAP | 0.1026 | 0.1228 | **+0.020 [+0.007, +0.035]** | 0.001 |
| dev | AP | 0.0811 | 0.0979 | +0.017 [+0.007, +0.028] | <0.001 |
| dev | onnet nuAP | 0.1601 | 0.1789 | +0.019 [+0.002, +0.034] | 0.013 |
| dev | sa AP (strict-a) | 0.0275 | 0.0327 | +0.005 [-0.009, +0.022] | 0.25 |
| confirm | nuAP | 0.1331 | 0.1432 | **+0.010 [+0.001, +0.019]** | 0.011 |
| confirm | AP | 0.0921 | 0.1039 | +0.012 [+0.004, +0.019] | <0.001 |
| confirm | onnet nuAP | 0.1684 | 0.1791 | +0.011 [-0.000, +0.021] | 0.025 |
| confirm | sa AP | 0.0334 | 0.0341 | +0.001 [-0.005, +0.005] | 0.37 |
Top bl features by gain: r_bl_xid, bl_xid (the exact-match arm), then r_bl_animal, r_bl_func, r_bl_somatic, r_bl_lit.
As predicted by the critics: positive on the full endpoint, ~0 on strict-a (`sa`).

Temporal `--train past` (train on 19.04 -> 21.04 episodes; 19.04 has only genetic sources, so only bl_genweak / bl_xid carry
values in training, the rest are NaN): stackP_E_base_bl vs stackP_base: dev nuAP 0.103 vs 0.119 (**-0.016** [-0.032, -0.002]),
confirm 0.126 vs 0.134 (-0.008 [-0.019, +0.001]); AP dev -0.000, confirm +0.011. Reason: on past, own weak-genetic genes are
rankable and the model leans on bl_genweak (top feature by gain); on temporal those genes are `unknown` (excluded), so the
learned weight is wasted and the residual weak-genetic genes (ontology mismatch) are false positives. A train/test candidate-set
mismatch, not a property of the signal. With only `bl_xid` added (stackP_E_base_blxid, trained on past):
dev nuAP +0.003 [-0.006, +0.012], AP +0.008 [+0.001, +0.015], sa AP +0.007 [-0.001, +0.022]; confirm nuAP -0.002 [-0.008, +0.005],
AP +0.004 [-0.003, +0.011]. Small, not significant on confirm: the exact-match arm learned on 19.04 genetic evidence alone does
not transfer much (in cv on temporal it is trained on 21.04 evidence of all sources).

## 4. Controls
- Positive control: stack_E_base reproduces the references (temporal dev nuAP 0.1026, confirm 0.1331 for `--train cv`).
- Negative control (bl_any with permuted gene identities): temporal cv stack_E_base_blperm vs stack_E_base: dev nuAP +0.003 [-0.006, +0.012],
  confirm -0.004 [-0.009, +0.002]; AP dev +0.003, confirm -0.002 (all CIs include 0). Forward: the stacked perm arm was
  cancelled (lead's RAM limit: one stacker at a time); alone, bl_any_perm gives forward dev AP 0.004 / nuAP 0.004 / AUROC 0.497.

## 5. Verdict against the R1 statistician's kill rules (#1 backlog exact-match)
- Rule "< 5 % of class-B dev arrivals in the top 100 by the match score alone": bl_xid 6.8 % on dev (passes narrowly), 3.4 % on
  confirm (fails there).
- Rule "fused delta < +0.010 nuAP on the full endpoint": temporal +0.020 dev / +0.010 confirm (passes, CI excludes 0 both).
- `sa`: ~0, as expected for a curation-pipeline effect.
VERDICT

## 6. Leakage
- forward: inputs are 25.09 files only (seed release); targets are 26.09-only genes. Temporal-clean. The weak-genetic genes are
  *not* excluded on forward (temporal excludes them as `unknown`), so forward is where this channel is visible and fair.
- temporal: 21.04 files + 21.04 ontology; sealed fold + relatives dropped from evidence. bl_xid matches on the task name (the
  panel name is 25.09) and 21.04 names / exact synonyms only (no 25.09 xrefs or obsolete-id maps).
- holdout: NaN (barred).
- The bl_genweak forward effect is mostly L2G / credible-set re-scoring around the 0.5 threshold between releases: it predicts
  a pipeline threshold crossing. Any forward claim by another method should be reported with and without the
  `bl_genweak > 0` stratum, or against a stack that includes bl_genweak.

## 7. Runtime and what failed
- Channels: 14 s for forward dev (842 tasks), ~1 min for temporal. Forward stack runs ~55-60 min each on the loaded machine
  (load ~140 from other agents), which limited the number of forward arms.
- `--train past` cannot test the non-genetic channels (19.04 file has genetic sources only) and penalises bl_genweak through the
  exclusion mismatch. A past-trained model should mask own-family weak genetic genes in training too (mirror of `unknown`).
- Not done: SG-7 GWAS-catalog ingestion lag (no archived catalog arm), fuzzy / embedding identity linking (only exact match),
  forward with the bdv stack (donor/view channels not built on forward in time), forward non-genetic-only arm (queued, see above).

## 8. New ideas
1. **Threshold-crossing forecaster**: the conversion curve is steep and monotone (0.2 % -> 26 %); a calibrated per-source model of
   score drift (L2G 0.4-0.5 -> >= 0.5) is the strongest single forward predictor found so far. Report it as a *label-process
   baseline* that every forward method must beat on the bl_genweak == 0 stratum.
2. **Stratified forward endpoint**: forward targets split into backlog (39 % with sub-threshold genetic evidence) and de-novo;
   claims on biology should be made on the de-novo stratum (or with `unknown`-style exclusion as temporal does).
3. **Past-training with mirrored exclusion**: build `past` episodes with weak 19.04 genetic genes excluded (as temporal does)
   so a past-trained stacker can use backlog channels honestly.
4. IR-3 extension: exact-match on dbXRefs (OMIM / Orphanet / MedGen) with the seed-release table; bl_xid already carries the most
   gain of the bl family on temporal.
