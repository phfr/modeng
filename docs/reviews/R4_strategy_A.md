# R4 strategy review A: what to build next (independent reviewer, 2026-11-12)

Read: FINAL report (incl. 9-10), r3/r4/r5 reports, witness audit, sweeps 03/04, model families 05, R3 red team, insights, invalidations,
pre-registration + amendment 1, NEXT. Two read-only checks were run (section 0). Nothing was trained; no result file was changed.

**Verdict in one paragraph.** The next step is a larger benchmark, and a pruned model retrained on it, before any new witness or model
family. Every increment tested since 08:00 was smaller than the retrain noise (0.005 AP) on 53 dev diseases, so those kill rules could not
decide anything. The same window (21.04 -> 25.09) holds about 1,170 eligible diseases that were never tasks here. They give roughly 10x
the training tasks, enough strict-a tasks to select on, and a fresh confirmation fold today, without the sealed fold. FinnGen R4 is second,
judged on that benchmark. Witness-class experts are third, cut down to two arms.

## 0. Checks run for this review
1. **Untouched diseases in the temporal window.** From the raw OT 21.04 and 25.09 label matrices (`modeng.data.label_matrix`), with
   5..1000 seeds and >= 3 new genes: 1,613 diseases. Of these 141 are temporal tasks, 304 are sealed or sealed relatives, and
   **1,168 are neither**. Of the 1,168:
   - 713 are not an ontology relative (any release) of any of the 191 temporal tasks; 414 of those have >= 10 new genes;
   - 825 have target-set Jaccard < 0.2 with every temporal task;
   - 630 are forward tasks (their 21.04 -> 25.09 arrivals were forward inputs, never labels);
   - median 29 seeds, 17 new genes.
   Limits of this count: raw sets, so no weak-evidence ("unknown") exclusion, no correction for genes that arrive only through new
   descendant links (12 % on forward), no arrival classes. Only 141 of the 191 panel tasks pass the raw rule, so the final number will differ.
2. **Recall of modeng-v1 on temporal confirm** (saved scores, mean per disease): recall@20 0.19, @100 0.32, @500 0.51, @2000 0.68.
   Strict-a targets in the full ranking (71 diseases): @100 0.21, @500 0.44.

## 1. Where the headroom is
| source of headroom | evidence | size I expect |
|---|---|---|
| **Training-set size** (same-window tasks) | 17 training diseases: AP 0.123 (= transfer 0.119); 53: 0.153. Forward, 842 in-window diseases vs the 53-disease model from the other window: +0.027 AP, +0.072 nuAP. Past + dev hybrid: +0.019 nuAP over v1. The 30 views hurt because 53 diseases cannot fit 60 weak features. | +0.01 to +0.03 AP; the largest lever, and not yet measured directly |
| **Pruning** | views removed: +0.006 AP / +0.008 nuAP (dev and confirm, 3 seeds); context features +0.006 AP; cs_donor, dA_pt, priors, hz_* removable | +0.005 to +0.01, not additive with the row above |
| **GWAS-dominated diseases** | 39/138 confirm diseases, AP 0.092 vs 0.174 (Mendelian). Lifting them by a third adds +0.008 to overall AP; lifting them to Mendelian level adds +0.023. | below +0.01 on overall AP; the effect would be on strict-a |
| **Strict-a** | tie with modet's headline in every variant (onnet nuAP 0.110 vs 0.109); 34/71 strict-a diseases have an own T0 study; FinnGen is behind 47-65 % of strict-a GWAS pairs and OT 21.04 had none | unknown; FinnGen R4 is the only untested lever |
| **Targets no witness reaches** | half of the targets are outside the top 500; about 60 % have zero transfer support, where every witness tested has AP <= 0.005 | not reachable with similarity witnesses (KGE, cell type, perturbation, structure all redundant) |

Two things limit what any of this can claim. Half of v1's AP margin over transfer comes from same-window training (past-trained: +0.018,
n.s.). Out of window the gain is carried by the label-process channels (forward without them: AP 0.082 vs 0.085).

## 2. What to build, ranked

### Rank 1. `temporal-wide` benchmark + modeng-v2 (pruned, trained at scale)
- **Why it can exceed +0.01.** The learning curve is not flat between 17 and 842 training diseases (table above), and v1 sits at 53.
  It is also the only way to make dev selection work: strict-a has 28 dev tasks today, and dev strict-a gains failed to replicate three
  times (r3, powerfc, subthreshold).
- **Design.**
  1. Rebuild the temporal task rule for non-panel diseases inside modeng (weak-evidence exclusion, descendant re-propagation as in
     `build_forward`, arrival classes for strict-a from the 25.09 / 26.09 evidence dates; modet is read, never edited).
  2. Cluster by ontology component and seed Jaccard >= 0.5. Split by cluster hash, before any label is loaded:
     `wide-train` (all diseases related to the 191 burnt tasks, plus unrelated ones up to about 600), `wide-dev` (about 200 unrelated),
     `wide-confirm` (about 300 unrelated, target Jaccard < 0.2 to every old task). Write the confirm ids to a manifest and add a guard
     in `tasks.py` that refuses to load their targets without a flag.
  3. v2 recipe = clean arm minus views, minus the 11 weak donor families, minus cs_donor, dA_pt, hz_*; one L2G prior, one ph channel;
     plus the 9 context features. Fit with 3 LightGBM seeds, averaged.
- **Data.** All on disk.
- **Leakage class.** temporal-clean labels; same-window training (state it in every claim); vintage caveats unchanged. Past-trained twin
  reported beside it.
- **Cheapest decisive test** (pre-register first). On `wide-dev`: v2 trained on `wide-train` vs the v1clean recipe trained on the 53 old
  dev diseases, same features computed for both.
- **Kill rule.** Scale is not the lever if the gain is < +0.010 AP or nuAP is lower (3-seed means, cluster bootstrap CI). Pruning is kept
  only if v2 is not below v1clean (both wide-trained) by more than 0.003 on AP and nuAP.
- **Compute.** CPU. Channels for about 1,100 tasks: under 1 h (pt_ridge took 15 min for 900 past episodes). Fits: minutes
  (about 2-3 M rows), about 20 GB RAM.
- **Risk.** Non-panel diseases differ from modet's panel (smaller, more ancestors with very large target sets; cap targets, report by size).
  modet's headline has no scores there, so that comparison stays on the old panel. It is not prospective.

### Rank 2. Genetics block for GWAS-type diseases: FinnGen R4 + constraint
- **Why.** It is the only identified lever for strict-a, and strict-a is the sealed fold's primary. 196 of 341 strict-a pairs in confirm
  diseases without an own study involve FinnGen.
- **Design.** P0: map the R4 endpoint list to task diseases (now including wide tasks); count strict-a targets in newly covered diseases.
  P1: `fg4_lead` (R4 leads at p <= 5e-8 and at p <= 1e-5, nearest gene and cS2G; reuse `subthr.py`). P2 (only if P1 passes): MAGMA on
  matched endpoints. Add `gc_loeuf`, `gc_pli` (gnomAD v2.1.1) in the same run with a permuted control.
- **Data.** gnomAD on disk. FinnGen R4 (public 2020-11-30): endpoint manifest < 1 MB; summary statistics about 0.5-1 GB per endpoint
  (size from sweep 04, unverified). Stream and keep p <= 1e-4 only for matched endpoints: about 100-200 GB transient, < 2 GB kept.
- **Leakage class.** temporal-clean by date, **label-process by mechanism** (an earlier freeze of the cohort OT ingests later). The
  endpoint -> disease mapping borrowed from OT 25.09 is post-T0 vintage; run a sensitivity arm with ICD cross-references from EFO 21.04.
- **Decisive test and kill rules.** P0: stop if < 100 strict-a targets (old dev + wide-dev) sit in newly covered diseases. P1 on
  `wide-dev` strict-a tasks: keep only if sa_nuAP rises by >= +0.010 over the same-width permuted control with CI > 0, and overall AP and
  nuAP do not drop by more than 0.003. Constraint: drop if within the permuted control.
- **Compute.** CPU, about half a day plus download time.
- **Risk.** High. Own-study MAGMA z added nothing even on covered diseases (sa_AP 0.019 vs transfer 0.056). I put the chance of passing
  P1 at about one in three. If it passes, it is forecasting of the label pipeline, not new biology, and must be reported as such.
  I expect constraint to add nothing (popularity and the Mendelian prior carry it).

### Rank 3. Two-arm expert model (biology arm and label-process arm) with a small gate
- **Why, and why only third.** Sweep 05 itself expects little accuracy; the context features already gave +0.006. The value is the product:
  the no-label-process arm that R3 made co-primary, and a list of genes the biology arm ranks high and the label-process arm does not.
- **Design.** Two LightGBM experts instead of five: `Bio` = relational + literature + phenotype + Mendelian + base, blind to
  gwas_catlag / L2G / FinnGen; `LP` = those genetics channels + base. Fusion = logistic gate on the 9 context features.
- **Train the gate on `wide-train`, not on past + forward dev.** R3 showed that a model moved across windows loses the non-label-process
  gain (forward: AP 0.082 vs 0.085), and the label-process share differs between a one-year and a four-year window. The 2.6k cross-window
  tasks are a sensitivity arm.
- **Data.** On disk. **Leakage class.** As rank 1; the LP arm is label-process by design.
- **Test and kill rules.** On `wide-dev`: kill the gate if it loses > 0.005 AP or nuAP to v2. Kill the disagreement list if, among each
  disease's top 500, the Bio-favoured tercile does not hold >= 1.5x the precision of the LP-favoured tercile on arrivals without pre-T0
  literature or genetic evidence (CI > 1).
- **Compute.** CPU, half a day. **Risk.** Low cost, low accuracy gain; the disagreement list may be mostly popularity.

## 3. What not to do
- **No more similarity witnesses** (expression, cell type, perturbation, structure-predicted PPI, embeddings): five in a row carried
  signal and added nothing.
- **No deeper KG model now.** Shallow RotatE is at transfer level, modet's GNN equalled an MLP, 586 target pairs leak through descendant
  and synonym nodes, and per-query masking plus 6 GPU-h buys a low prior. Revisit only on the zero-support stratum of `wide-dev`.
- **No generative LLM judge on existing benchmarks**: 2026 models have seen the papers behind both windows. Run the memorisation probe
  (1 h) and, if wanted, enter a judge in the prospective lockbox only (section 4). Dated-text embedding towers: park; pt_ridge already
  holds the literature signal.
- **No TabPFN, set transformer, class-A head, area flags, more power forecasting, Borzoi, or degree-power tuning.**
- **No further variant scored on temporal confirm.** It has seen 59 stackers overnight plus more than 30 since (r3-r5 variants, controls and audit arms).
- **No kill rule below the noise floor.** Rules of +0.003 on 53 diseases (cell type, sub-threshold) were decided inside a
  control-to-control spread of +-0.005 to 0.007.

## 4. Confirmation plan
**Tier A, available now: `wide-confirm`** (about 300 untouched, unrelated diseases, same window).
- Arms, each scored once, pre-registered before the labels are loaded: v1clean (frozen recipe, trained on `wide-train` as the
  pre-registration words it), v2, v2 + genetics block (only if it passed), and each without label-process channels.
- Comparators: transfer_jaccard and dc_ridge_pos alone.
- Primary: paired AP and nuAP vs transfer, cluster bootstrap over ontology components, both CIs must exclude 0 (unchanged rule).
  Co-primary: the same for the no-label-process arm. Secondary: v2 minus v1clean; strict-a nuAP; GWAS-dominated stratum;
  targets with no 21.04 evidence in the family; degree-matched AP.
- Claim wording: "untouched diseases, same time window". It is not a prospective confirmation.

**Tier B, prospective: the next OT release.**
- Before it appears, write the rankings of all arms for all eligible diseases at seeds 26.09 to disk, hash them and timestamp the manifest
  (OpenTimestamps, as in modetor). Include the past-trained twin and, if the probe is run, the LLM judge.
- Power: one release adds far fewer arrivals than the one-year forward window. Fix in advance that the primary is read at the first
  release with >= 400 tasks with >= 3 new genes, else cumulatively at the following one. (I assumed quarterly releases; not checked.)

**Sealed fold: no.** Its primary is strict-a, where v1 ties, so the expected outcome is a spent fold and a tie. Ask the user to authorise
it only if the genetics block gains >= +0.010 strict-a nuAP on `wide-dev` and that replicates on `wide-confirm`; then one run, one arm,
primary strict-a onnet nuAP vs modet's headline, through modet's ledger.

## 5. Process risks
- **Selection on confirm.** `wide-dev` will be burnt too. Set a budget in the pre-registration (for example <= 15 variants), report
  3-seed means, and keep `wide-confirm` to <= 6 arms scored once.
- **Fresh diseases are not a fresh window.** 630 wide diseases were forward tasks and many share target genes with the old panel
  (median 67 % of a wide disease's new genes are a target of some old task). Keep related diseases in training only, and state the
  same-window qualifier.
- **Label-process forecasting.** gwas_catlag, L2G priors and FinnGen forecast OT's pipeline; 37 % of confirm targets had 21.04 evidence
  in the family. Every headline number needs the no-label-process arm and the no-prior-evidence stratum beside it.
- **Vintage.** PubTator3 annotations and GWAS-Catalog mappings are 2026 versions; STRING 11.5 is 2021-08; HPO 2021-01 on past episodes
  post-dates their seeds; the FinnGen endpoint mapping would be post-T0. None has an archived vintage on disk; say so in each claim.
- **Benchmark construction.** The wide task rule is new code that defines labels. Check it by rebuilding the 191 panel tasks with it
  and comparing seeds and targets to modet's panel before any model is scored.
