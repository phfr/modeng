# r7 P4: the one pre-registered scoring of the frozen `wide_confirm` fold, 2026-11-12 15:16-15:30
**Pre-registration:** `docs/preregistration_wide.md` (frozen part: tag `wide-confirm-frozen`, commit `e6f00fa`) + **Amendment 1 — confirmatory
scoring plan**, commit **`2fb1b77e818d5b94dc978a1e3bf8f295af4ee36c`, 2026-11-12 15:16:30 +0100**, committed before the flag was set
(first command with `MODENG_SCORE_WIDE_CONFIRM=1` started 15:16:49; no `temporal_wide_wide_confirm_*` channel cache existed before).
Pre-check: `sha256sum -c docs/preregistration_wide.sha256` OK for `data/wide_confirm_ids.txt` (and the 4 other frozen files); the scored
task ids were asserted equal to that file. The user approved spending the fold. **wide_confirm is now spent.**
Leakage class: temporal-clean (seeds / context OT 21.04, targets new in 25.09); 35 diseases, 28 ontology components, 297 targets.

## Runs (one per arm, seed 0, no changes, no failures)
Driver `results/p4/run.sh`, log `results/p4/run_wide_confirm.log`; scores `results/temporal_wide/wide_confirm/*.scores.npy`.
- **A modeng-v1** (`stack_v1`, 65 channels, deg^-0.35, 400 rounds, 3000 neg.) and **B v1clean** (`stack_v1clean`, 51 channels): trained on all
  53 temporal dev diseases (`scripts/p4_score.py`). The same models reproduce their recipes' temporal-confirm rows exactly
  (AP 0.1532 / 0.1497), i.e. they are the frozen recipes.
- **C W_v1clean** (`stackW_P4C_v1clean`) and **D W_v1clean without label-process channels** (`stackW_P4D_v1clean_nolab`, 47 channels: minus
  gwas_catlag, pri_l2g_n, pri_l2g_max, hz_gwas): `python -m modeng.stack --train wide --score-confirm --pos-deg-power 0.35`, guard on
  (12,287 positive pairs dropped in 508 train tasks; 85 more than in r6 because the guard now also covers wide_confirm relatives).
- `modet_headline`: **unavailable** (0 / 35 wide_confirm diseases are in modet's headline run); no modet comparison.

## Primary endpoint: vs transfer_jaccard, paired cluster bootstrap 10,000 (clusters = 28 ontology components)
| arm | AP | ΔAP [95 % CI] | nuAP | ΔnuAP [95 % CI] | both CIs > 0 |
|---|---|---|---|---|---|
| transfer_jaccard | 0.091 | | 0.084 | | |
| A modeng-v1 (53 dev) | 0.105 | +0.014 [-0.016, +0.043] | 0.147 | +0.063 [+0.005, +0.145] | no |
| B v1clean (53 dev) | 0.106 | +0.015 [-0.017, +0.050] | 0.142 | +0.059 [-0.004, +0.149] | no |
| C W_v1clean (wide_train) | 0.115 | +0.024 [-0.001, +0.047] | 0.119 | +0.036 [+0.001, +0.073] | no (AP CI touches 0) |
| **D W_v1clean, no label-process ch.** | **0.118** | **+0.027 [+0.004, +0.049]** | **0.126** | **+0.042 [+0.006, +0.086]** | **yes** |

Multiplicity (stated in the amendment: four arms, no correction): D's confirmation is narrow. With a Bonferroni-width interval (98.75 %)
its CIs are AP [-0.002, +0.054] and nuAP [-0.002, +0.097], so it would **not** survive a correction for four arms. Bootstrap P(Δ <= 0):
AP 0.010, nuAP 0.009.

## Secondary endpoints (descriptive, no claims)
| arm | deg-matched AP (Δ vs transfer) | P@20 | onnet AP | onnet nuAP | AUROC | R@100 |
|---|---|---|---|---|---|---|
| transfer_jaccard | 0.172 | 0.053 | 0.097 | 0.093 | 0.68 | 0.27 |
| A | 0.239 (+0.067 [+0.023, +0.112]) | 0.063 (+0.010 [-0.003, +0.024]) | 0.101 (+0.004, CI ∋ 0) | 0.112 (+0.019, CI ∋ 0) | 0.86 | 0.37 |
| B | 0.233 (+0.061 [+0.025, +0.097]) | 0.064 (+0.011 [-0.003, +0.026]) | 0.105 (+0.008, CI ∋ 0) | 0.112 (+0.019, CI ∋ 0) | 0.87 | 0.36 |
| C | 0.259 (+0.087 [+0.044, +0.130]) | 0.076 (+0.023 [+0.008, +0.038]) | 0.147 (+0.050 [+0.007, +0.099]) | 0.151 (+0.058 [+0.011, +0.110]) | 0.88 | 0.40 |
| D | 0.259 (+0.087 [+0.043, +0.133]) | 0.076 (+0.023 [+0.008, +0.039]) | 0.152 (+0.055 [+0.013, +0.103]) | 0.156 (+0.064 [+0.018, +0.115]) | 0.87 | 0.39 |

Strict-a (4 tasks with >= 5 sa targets; means only, not evaluable): sa AP A 0.118 / B 0.158 / C 0.092 / D 0.103 / transfer 0.053.
By target count (AP): 3-5 targets (13 tasks) transfer 0.157, A 0.160, B 0.161, C 0.179, D 0.196; 6-10 (18) transfer 0.056, A 0.080,
B 0.082, C 0.090, D 0.083; > 10 (4) transfer 0.037, A 0.035, B 0.039, C 0.023, D 0.025. (`results/p4/strata_wide_confirm.tsv`)

## Between arms (descriptive only; MDE ~0.03-0.06 AP, ~0.04-0.08 nuAP at 28 clusters)
C vs B: AP +0.009 [-0.021, +0.037], nuAP -0.023 [-0.084, +0.013]; B vs A: AP +0.002, nuAP -0.004; D vs C: AP +0.003 [-0.007, +0.016],
nuAP +0.007 [-0.005, +0.022]; C vs A: AP +0.011, nuAP -0.028. All CIs include 0, as expected: the fold cannot separate the arms.

## Allowed claim sentences (exactly these)
- **D:** "W_v1clean without label-process channels (trained on wide_train, temporal-clean) ranks newly arriving disease genes better than
  transfer_jaccard on 35 fresh, pre-registered off-panel diseases (wide_confirm; OT 21.04 -> 25.09): AP +0.027 [+0.004, +0.049],
  nuAP +0.042 [+0.006, +0.086]." Must be accompanied by: "one of four pre-registered arms, uncorrected; does not survive a Bonferroni
  correction for four arms."
- **A, B, C:** "X is not confirmed better than transfer_jaccard on wide_confirm" with their numbers (A: AP +0.014 [-0.016, +0.043],
  nuAP +0.063 [+0.005, +0.145]; B: AP +0.015 [-0.017, +0.050], nuAP +0.059 [-0.004, +0.149]; C: AP +0.024 [-0.001, +0.047],
  nuAP +0.036 [+0.001, +0.073]).
- Not allowed: any ranking of the arms against each other; "the label-process channels hurt" (D vs C n.s.); any modet comparison.

## Plain-language verdict
On 35 new diseases nobody had looked at, every stacked model ranked the later-discovered disease genes above the simple
"similar diseases share genes" baseline on average (AP 0.105-0.118 vs 0.091; on most diseases), but the fold is small and only one arm
cleared the pre-registered bar: the model trained on the 712 wide training diseases **without** the GWAS/L2G label-process features.
That is reassuring on leakage (the win does not come from features that anticipate how Open Targets labels arrive), but it is a narrow
pass that would not survive correcting for having tried four arms. The two frozen 53-disease recipes (modeng-v1, v1clean) gain on nuAP
(non-hub genes) but are **not confirmed** on AP. Honest summary: "directionally confirmed, statistically marginal; the wide-trained
clean recipe is the one to carry forward, and a larger fresh fold (next OT release) is needed for a firm claim."

## Files
`docs/preregistration_wide.md` (Amendment 1), `scripts/p4_score.py`, `scripts/p4_analyze.py`, `results/p4/` (`run.sh`, logs,
`per_task_ / summary_ / primary_ / secondary_ / between_ / strata_wide_confirm.tsv`), `results/temporal_wide/wide_confirm/*.parquet`
(+ `.scores.npy`). C / D also re-scored temporal dev / confirm / wide_dev (rows in `results/leaderboard.tsv`; descriptive).
