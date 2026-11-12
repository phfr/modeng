# R2: final review of `docs/reports/FINAL_overnight_2026-11-12.md` (Fable reviewer, 2026-11-12 07:04-07:15)

Scope: every number in sections 1, 1b, 1c, 3, 4, 4b against `results/leaderboard.tsv`, `results/review/leader_endpoints*.tsv`,
`results/stack_*.txt`, `docs/insights.md`, `docs/reviews/R1_fable_redteam.md`; doubted deltas recomputed with `modeng.compare`.
The report was not edited.

## Verdict: NUMBERS CORRECT, FRAMING OVERCLAIMS IN SIX PLACES (fix before anyone reads section 1)

All point estimates and CIs in sections 1, 1c, 3, 4, 4b match the sources (recomputed: 14 deltas, all identical to the 3rd decimal).
The problems are which model a number belongs to, what the red team actually reviewed, and how much confirm was looked at.

1. **Section 1b reports `stack_v3cv`, not the leader and not the final model.** `results/review/leader_endpoints.tsv` has
   AP 0.1396 / nuAP 0.1764 / onnet nuAP 0.2120 = `stack_v3cv` (leaderboard 06:05), while section 3 names `stack_v3cv_ptr`
   (0.1434 / 0.1841 / 0.2187) as leader. So +0.050, +0.039, +0.077, +0.067, the strict-a pair 0.108 vs 0.109, +0.056, +0.033 and
   hub -0.026 are all `stack_v3cv` numbers. No strict-a onnet endpoint exists for `stack_v3cv_ptr` at all.
2. **The headline sentences (section 1, items 2-3) quote the leader, not the final model `modeng-v1`.** For `modeng-v1` the gain on
   modet's candidate set is nuAP +0.031 [+0.012, +0.054], not +0.046; nuAP vs the leader is -0.007 [-0.013, -0.002] (all genes)
   and -0.015 [-0.023, -0.007] (onnet). The power change bought AP (+0.010 [+0.005, +0.014]) with nuAP.
3. **"About 20 stacker variants viewed on confirm" is wrong by a factor of 3-4.** `results/leaderboard.tsv` has 78 methods scored on
   temporal confirm, 59 of them stackers (20 are the cv `stack_*` family alone); R1 already counted 69 / 59 at 06:25.
4. **The red team did not review the leader or the final model.** R1 (06:04-06:25) tested the past-trained round-1 stacker
   (AP 0.111, nuAP 0.175). `stack_v3cv_ptr` (06:10), `modeng-v1` (06:26), cv training on dev, pt_ridge, HPO matching and the
   channel priors were never red-teamed. "No leakage, masking or selection artefact was found" may only be said about that
   round-1 model. Section 1 item 5 "decides what may be claimed" has the same problem.
5. **Wrong statement about power 0.25.** R1 table: power 0.25 has AP 0.127 vs 0.111 (better) and nuAP 0.151 vs 0.175 (worse).
   "Better on both metrics" in R1 means AP and recall@100. The report says "better on both AP and nuAP".
6. **The power for `modeng-v1` was chosen on dev, but the grid (0.25 / 0.35 / 0.5) and the rule (mean of AP and nuAP) were set
   after the red team had shown confirm results by power** (`docs/insights.md`, 06:55 entry says so). The confirm numbers of
   `modeng-v1` are therefore not a clean confirmation. The same holds for the calibration factor 0.7 in 4b: it is estimated
   on confirm.

Checked and fine: the strict-a tie is stated (1b, 1c); forward CIs are not quoted and section 4 says why (compare shows
clusters = 797 = one task per cluster); section 4 names gwas_catlag and L2G priors as label-process channels; the
temporal CIs use 41 clusters for 138 diseases; `modeng-v1` hub AP vs transfer -0.006 [-0.029, +0.019] is correct.

## Missing caveats (not in the report)
- **cv-trained = trained on arrivals of the same label window.** The stacker is fitted on the 21.04 -> 25.09 arrivals of the dev
  diseases and tested on the same window for confirm diseases. Gene-level features (degree, popularity, priors, views) can
  carry "which genes arrive in this window" across diseases. The prospective version is the past-trained `stackP_v2`:
  AP 0.119 (= transfer), nuAP 0.182. The whole AP gain over transfer exists only with same-window training; forward is cv
  too, so it does not test this.
- **In the forward model gwas_catlag is the 2nd most important feature** (gain 6.6e5 vs transfer_jaccard 8.4e5,
  `results/stack_forward_v1_p0.35.txt`), with `r_gwas_catlag` and `pri_l2g_n` in the top 12. "Partly forecast the label pipeline"
  understates it; no forward run without these channels exists.
- **Strict-a on all genes is negative vs modet's headline**: `stack_v3cv` nuAP -0.027 [-0.050, -0.005], `modeng-v1`
  -0.019 [-0.038, +0.006], `stack_v3cv_ptr` -0.009 [-0.025, +0.015]. The report shows only the onnet tie.
- **"Each a Pareto gain" (section 1 item 4) is not true for priors + HPO**: `docs/insights.md` gives nuAP +0.006 [-0.001, +0.011].
  PubTator ridge is real but small: nuAP +0.008 [+0.005, +0.011], AP +0.004 [+0.001, +0.007].
- **AP vs transfer for the leader is borderline** (+0.025 [+0.002, +0.048]; onnet +0.022 [-0.001, +0.045]) and comes after
  59 looks at confirm.
- Section 4b module calls come from `stack_v3cv`, not `modeng-v1`. Section 3 has no row for `modeng-v1`.
- Forward `modeng-v1` is worse than forward `stack_v3cv` on nuAP (0.214 vs 0.222, -0.008 [-0.011, -0.005]) and better on AP.

## Corrections (exact old text -> new text)

1. Section 1 item 1
   - old: `diseases never used for model fitting) and replicates on the forward benchmark`
   - new: `diseases never used for model fitting, but the stacker is trained on dev arrivals of the same 21.04 -> 25.09 window) and replicates on the forward benchmark`
2. Section 1 item 2 (append after `+0.060 [+0.037, +0.088].`)
   - new: ` Final model modeng-v1 (section 1c): nuAP +0.031 [+0.012, +0.054], AP +0.063 [+0.049, +0.082].`
3. Section 1 item 4
   - old: `(each a Pareto gain on AP and nuAP with a same-width permuted control at ~0)`
   - new: `(Pareto gains on AP and nuAP against same-width permuted controls; priors + HPO only +0.006 nuAP [-0.001, +0.011], PubTator ridge +0.008 [+0.005, +0.011])`
4. Section 1 item 5
   - old: `about 20 stacker variants were viewed on temporal confirm during the night (no winner's-curse correction)`
   - new: `59 stacker variants (78 methods) were scored on temporal confirm during the night (no winner's-curse correction)`
5. Section 1 item 5
   - old: `and a Fable red-team review (`docs/reviews/R1_fable_redteam.md`) decides what may be claimed.`
   - new: `and the Fable red-team review (`docs/reviews/R1_fable_redteam.md`) covers the past-trained round-1 stacker only; the cv-trained leader and modeng-v1 are not red-teamed.`
6. Section 1b title
   - old: `(after the Fable red-team review `docs/reviews/R1_fable_redteam.md` and its endpoints on the leader)`
   - new: `(red-team endpoints of `docs/reviews/R1_fable_redteam.md` recomputed for `stack_v3cv`, the leader without PubTator ridge; `results/review/leader_endpoints.tsv`)`
7. Section 1b bullet 1
   - old: `No leakage, masking or selection artefact was found by the red team.`
   - new: `The red team found no leakage, masking or selection artefact in the past-trained round-1 stacker; the v3 channels (priors, HPO, PubTator ridge) and cv training were not reviewed.`
8. Section 1b bullet 2
   - old: `tie** (nuAP 0.108 vs 0.109, AP +0.010 n.s.).`
   - new: `tie** on modet's candidate set (nuAP 0.108 vs 0.109, AP +0.010 n.s.); on all genes strict-a nuAP is lower (-0.027 [-0.050, -0.005]).`
9. Section 1b bullet 3
   - old: `power 0.25 was better on both AP and nuAP in the red team's run but was chosen after seeing confirm (needs fresh data).`
   - new: `power 0.25 gave higher AP (0.127 vs 0.111) and recall@100 but lower nuAP (0.151 vs 0.175) in the red team's run on confirm; any power chosen after that needs fresh data.`
10. Section 1b bullet 4
    - old: `About 20 stacker variants were viewed on temporal confirm overnight`
    - new: `59 stacker variants (78 methods) were scored on temporal confirm overnight`
11. Section 1c title
    - old: `positive weight deg^-0.35 chosen on dev mean of AP and nuAP)`
    - new: `positive weight deg^-0.35 chosen on dev mean of AP and nuAP; grid and rule were set after the red team showed confirm results by power, so the confirm numbers below are not a clean confirmation)`
12. Section 1c
    - old: `strict-a: AP +0.023 [+0.006, +0.040], nuAP tie (+0.001).`
    - new: `strict-a: AP +0.023 [+0.006, +0.040], nuAP tie (+0.001 [-0.016, +0.025]). vs the leader `stack_v3cv_ptr`: AP +0.010 [+0.005, +0.014], nuAP -0.007 [-0.013, -0.002].`
13. Section 1c
    - old: `same recipe and power): AP 0.151`
    - new: `v3 recipe without pt_ridge, same power; no CIs because every forward task is its own cluster; gwas_catlag is the 2nd most important feature): AP 0.151`
14. Section 3 table (add row after the leader row)
    - new: `| stack_v3ptr_p0.35 = modeng-v1 (final) | 0.153 | 0.177 | 0.203 (onnet AP 0.160) |`
15. Section 4
    - old: `gwas_catlag and L2G-winner priors partly forecast the label pipeline.`
    - new: `gwas_catlag (2nd most important feature of the forward model) and L2G-winner priors forecast the label pipeline; no forward run without them exists, so the forward gain is not evidence for new biology.`
16. Section 4b
    - old: `Calibration is optimistic by ~1.4x: multiply predictions by 0.7.`
    - new: `Calibration is optimistic by ~1.4x; the factor 0.7 is estimated on confirm and the map belongs to `stack_v3cv`, not modeng-v1 (refit needed).`

## The 3 sentences for the top of the report
1. On the temporal benchmark (138 confirm diseases) the final stacker `modeng-v1` ranks future Open Targets disease genes better than
   raw transfer (AP 0.153 vs 0.119, nuAP 0.177 vs 0.104) and better than modet's headline on modet's candidate set for all
   arrivals (AP +0.063 [+0.049, +0.082], nuAP +0.031 [+0.012, +0.054]), but it ties modet on modet's primary strict-a endpoint
   (nuAP +0.001 [-0.016, +0.025]).
2. These are development numbers, not a confirmation: 59 stacker variants were scored on confirm, the degree power was fixed
   after confirm results by power were known, the stacker is trained on arrivals of the same time window, and the red team
   reviewed only the earlier past-trained model (which had AP equal to transfer).
3. The forward replication (AP 0.151 vs 0.085, nuAP 0.214 vs 0.083, 797 diseases) has no valid CIs and leans on channels that
   forecast the Open Targets label pipeline (GWAS-Catalog lag, L2G priors), so the next step is one pre-registered run on
   untouched diseases with the recipe frozen.
