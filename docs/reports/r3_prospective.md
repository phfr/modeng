# r3: prospective (past-trained) modeng-v1 (2026-11-12, 07:25-08:15)

**Question.** modeng-v1 (`stack_v3ptr_p0.35`) is trained by grouped CV on the 53 temporal dev diseases, i.e. on 21.04 -> 25.09
arrivals, the same window as the test (FINAL report caveat "same-window training"). The earlier past-trained stacker `stackP_v2`
(power 0.5, no priors/HPO/PubTator ridge) had AP equal to transfer. Does the AP gain over raw transfer survive when the v1 recipe is
trained only on `past` episodes (19.04 -> 21.04, all <= T0 = 21.04)?

**Verdict (short).** Partly. Past-trained v1 (`stackP_v1past`) keeps the full nuAP gain over transfer (confirm +0.082 [+0.057, +0.110])
and beats modet's headline on every endpoint except strict-a, but its **confirm AP gain over transfer is +0.018 [-0.005, +0.043]: not
significant** (the CI includes 0; the gain is in non-hub genes, hub AP -0.031). On temporal dev (53 diseases, also never trained on
by this model) AP +0.035 [+0.016, +0.055]; pooled dev+confirm (191 diseases, post hoc, not pre-registered) AP +0.023 [+0.005, +0.041].
About half of v1's AP margin over transfer (+0.034 on confirm) is therefore attributable to same-window training. Adding the 53 dev
diseases to the past episodes (hybrid) restores confirm AP to +0.028 [+0.003, +0.053] over transfer with the highest nuAP of all
variants (0.196), but that model again sees 21.04 -> 25.09 arrivals.

## 1. What was built
- `src/modeng/ext/past_fill.py`: channels `<name>_pf` for the 8 v1 channels that were NaN on `past`
  (pri_l2g_n, pri_l2g_max, pri_mend_n, pri_hpo_n, hz_gwas, hz_mend, ph_mean, ph_top3). Off `past` each `_pf` channel returns the cached
  original channel (checked: `pri_l2g_n_pf == pri_l2g_n` on temporal dev, exact), so evaluation inputs are exactly v1's.
  On `past` (seed release 19.04):
  - GWAS-winner priors (pri_l2g_*, hz_gwas, the seed mix in hz_mend): OT 21.04 `ot_genetics_portal` evidence filtered to
    `publicationYear <= 2018` (date filter exists for this source; 2019 studies dropped because the seed release is 2019-04).
    L2G scores are still from the 21.04 L2G model.
  - Mendelian prior (pri_mend_n): OT 21.04 eva(>=0.5)/curated evidence (no reliable date column).
  - HPO channels (pri_hpo_n, hz_mend, ph_mean, ph_top3): HPO 2021-01-15 (the only pre-T0 HPO vintage on disk).
- Not wrapped, because they already have 19.04-dated inputs on `past`: `gwas_catlag` (GWAS Catalog, date added <= 2019-04-30;
  non-NaN for 40 % of past tasks vs 45 % of dev tasks), `dA_pt`, `pt_ridge`, `pt_ridge_pos` (PubTator PMID <= 2019-03, `a_pt19`;
  pt_ridge had simply never been computed on past), `cs_donor`, `dc_ridge(_pos)`.
- Scripts: `scripts/r3_prospective.sh` (past-trained runs), `scripts/r3_hybrid.py` (past + dev), `scripts/r3_compare.py`
  (paired cluster bootstrap, `modeng.compare`, 10,000 resamples). Outputs: `results/r3/` (logs, `compare.tsv`,
  `pooled_dev_confirm.txt`), `results/review/leader_endpoints_stackP_v1past.tsv`, `..._stackH_v1hyb.tsv`, per-task parquet + saved scores
  in `results/temporal/{dev,confirm}/`.

**Leakage class: "training-side transductive".** On `past` the filled inputs (OT 21.04 evidence, HPO 2021-01, 21.04 L2G model)
post-date the past seed release and may already contain the 19.04 -> 21.04 arrivals that are the past training targets (family
exclusion removes only the task's own disease family; GWAS winners are year-cut, Mendelian and HPO are not). This can only distort the
weights the stacker learns (likely over-trusting priors/HPO); it cannot leak temporal test labels, because every input used at
evaluation is dated <= 21.04 = the temporal seed release, and no temporal arrival (21.04 -> 25.09) enters training.
Remaining selection caveats (unchanged from the FINAL report): the v1 channel list and power 0.35 were chosen after confirm results
were known; 59 stacker variants were scored on confirm overnight.

## 2. Commands
```
MODENG_SAVE_SCORES=1 python -m modeng.stack --bench temporal --train past --pos-deg-power 0.35 --tag v1past \
  --channels base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n_pf,pri_l2g_max_pf,pri_mend_n_pf,\
pri_hpo_n_pf,hz_gwas_pf,hz_mend_pf,ph_mean_pf,ph_top3_pf,pt_ridge,pt_ridge_pos          # -> stackP_v1past
... same without the 8 _pf channels --tag v1past_nopri                                    # -> stackP_v1past_nopri (ablation)
MODENG_SAVE_SCORES=1 python scripts/r3_hybrid.py --dev-oof                                # -> stackH_v1hyb
python scripts/r3_compare.py stackP_v1past stackP_v1past_nopri stackH_v1hyb
python scripts/leader_endpoints.py stackP_v1past ; python scripts/leader_endpoints.py stackH_v1hyb
```
`run_past` settings as in `stack.py` (900 of 1,786 past episodes, 1,500 negatives, <= 60 positives per task, 600 rounds; 1.37 M rows,
21,887 positives). Hybrid: the same past rows + all temporal dev rows (3,000 negatives per task, 161 k rows), 600 rounds; its dev
numbers are out-of-fold (past + 4/5 of dev clusters, the folds of `stack.run`). Runtime: past channels ~15 min (pt_ridge on 900
episodes dominates), each fit ~2-3 min, 8 threads.

## 3. Results (mean over diseases; temporal dev 53, confirm 138)
| model | training | dev AP | dev nuAP | conf AP | conf nuAP | conf onnet AP | conf onnet nuAP |
|---|---|---|---|---|---|---|---|
| transfer_jaccard | - | 0.123 | 0.096 | 0.119 | 0.104 | 0.128 | 0.128 |
| modet_headline | modet | - | - | (0.081) | (0.080) | 0.096 | 0.173 |
| stackP_v2 (power 0.5 default, v2 channels) | past | 0.143 | 0.168 | 0.119 | 0.182 | 0.125 | 0.205 |
| stackP_v1past_nopri (p0.35) | past | 0.157 | 0.156 | 0.132 | 0.175 | 0.136 | 0.190 |
| **stackP_v1past (p0.35)** | **past** | **0.158** | **0.164** | **0.137** | **0.186** | **0.142** | **0.205** |
| stackH_v1hyb (p0.35) | past + dev | 0.163 (OOF) | 0.170 | 0.146 | 0.196 | 0.153 | 0.218 |
| stack_v3ptr_p0.35 = modeng-v1 | cv on dev | 0.176 (OOF) | 0.150 | 0.153 | 0.177 | 0.160 | 0.203 |

Paired cluster bootstrap, delta [95 % CI] (`results/r3/compare.tsv`):

| comparison | split | AP | nuAP | onnet AP | onnet nuAP |
|---|---|---|---|---|---|
| v1past vs transfer | dev | +0.035 [+0.016, +0.055] | +0.068 [+0.043, +0.098] | +0.048 [+0.025, +0.074] | +0.090 [+0.049, +0.136] |
| v1past vs transfer | **confirm** | **+0.018 [-0.005, +0.043]** | +0.082 [+0.057, +0.110] | +0.014 [-0.010, +0.039] | +0.077 [+0.043, +0.115] |
| v1past vs transfer | dev+confirm pooled (191, post hoc) | +0.023 [+0.005, +0.041] | +0.078 [+0.056, +0.102] | +0.024 [+0.004, +0.044] | +0.081 [+0.051, +0.113] |
| v1past vs modeng-v1 (cv) | confirm | -0.016 [-0.029, -0.005] | +0.009 [-0.008, +0.026] | -0.018 [-0.031, -0.006] | +0.002 [-0.017, +0.020] |
| v1past vs modet headline | confirm | (+0.056) | (+0.106) | +0.046 [+0.030, +0.066] | +0.032 [+0.003, +0.066] |
| v1past vs stackP_v2 | confirm | +0.019 [+0.013, +0.024] | +0.004 [-0.005, +0.014] | +0.017 [+0.009, +0.024] | -0.000 [-0.017, +0.014] |
| v1past vs v1past_nopri (value of _pf) | confirm | +0.006 [-0.001, +0.012] | +0.011 [-0.001, +0.022] | +0.006 [-0.002, +0.014] | +0.015 [-0.002, +0.031] |
| hybrid vs transfer | confirm | +0.028 [+0.003, +0.053] | +0.091 [+0.066, +0.120] | +0.025 [+0.000, +0.050] | +0.091 [+0.059, +0.125] |
| hybrid vs modeng-v1 (cv) | confirm | -0.007 [-0.015, +0.002] | +0.019 [+0.007, +0.031] | -0.007 [-0.016, +0.002] | +0.015 [+0.002, +0.029] |
| hybrid vs v1past | confirm | +0.009 [+0.004, +0.016] | +0.010 [+0.002, +0.020] | +0.011 [+0.006, +0.017] | +0.013 [+0.005, +0.022] |
| hybrid vs modet headline | confirm | (+0.065) | (+0.116) | +0.057 [+0.041, +0.077] | +0.045 [+0.019, +0.076] |

(modet headline all-gene AP/nuAP in parentheses: its ranking is defined on modet's candidate set, use the onnet columns.)

Red-team endpoints for `stackP_v1past` on confirm (`results/review/leader_endpoints_stackP_v1past.tsv`; 3,000 cluster resamples):
vs transfer: recall@100 +0.048 [+0.018, +0.076], P@20 +0.026 [-0.007, +0.063], degree-matched AP +0.056 [+0.028, +0.085],
**hub AP -0.031 [-0.052, -0.006]**, non-hub AP +0.024 [+0.000, +0.050], strict-a AP -0.002 (n.s.), strict-a nuAP +0.032 [+0.007, +0.059].
vs modet headline: onnet AP +0.046, onnet nuAP +0.032 [+0.004, +0.067], degree-matched AP +0.077, hub AP +0.027; strict-a onnet nuAP
tie (-0.000 [-0.031, +0.031]), strict-a all-gene nuAP -0.025 [-0.055, +0.005]. Hybrid (`results/review/leader_endpoints_stackH_v1hyb.tsv`),
confirm vs transfer: recall@100 +0.056 [+0.026, +0.084], degree-matched AP +0.063, hub AP -0.025 [-0.047, -0.000], non-hub AP +0.034
[+0.010, +0.061], strict-a AP 0.000; vs headline: strict-a onnet nuAP +0.003 [-0.024, +0.031] (tie).

Top features of `stackP_v1past` (gain): gwas_catlag (largest by 2x), transfer_jaccard, ph_mean_pf, ph_top3_pf, hz_gwas_pf,
r_pri_mend_n_pf, dc_ridge, pt_ridge, don_hpo2101. The filled priors/HPO are used heavily, but their measured value on confirm is
small (+0.006 AP, +0.011 nuAP, both CIs touch 0), consistent with weights learned on training-side-transductive inputs.

## 4. Interpretation
- **nuAP gain is robust to prospective training** (every past-trained variant: +0.07 to +0.08 over transfer on confirm, CIs well above 0).
- **AP gain is not established prospectively on confirm.** Past-trained v1 moves AP from stackP_v2's 0.119 (= transfer) to 0.137
  (+0.019 [+0.013, +0.024] vs stackP_v2, so the v3 channels and power 0.35 do help prospectively), but the margin over transfer is
  +0.018 with a CI including 0. The deficit vs cv-trained v1 (-0.016 [-0.029, -0.005]) estimates the same-window-training share of v1's
  AP. AP is significant on dev and pooled dev+confirm, so the honest statement is "AP gain smaller (~+0.02) and fragile".
- The AP shortfall sits in hub genes (hub AP -0.031 vs transfer), i.e. the degree-weighted objective still trades hubs for non-hubs
  when trained on 19.04 -> 21.04 arrivals; the 21 -> 25 window's arrival mix (learned only by cv/hybrid) favours hubs more.
- Hybrid (past + dev) beats cv-trained v1 on nuAP (+0.019 [+0.007, +0.031]) with AP tie (-0.007 [-0.015, +0.002]): adding 900 past
  episodes to the dev rows is at least as good as dev-only training. It is NOT prospective (dev arrivals are 21.04 -> 25.09).
- For the pre-registered fresh-fold run (`docs/preregistration_modeng_v1.md`): "trained on all available training tasks of the fresh
  benchmark's training split" corresponds to the hybrid/cv regime; the expected prospective AP margin over transfer should be taken
  from `stackP_v1past` (~+0.02), not from the confirm numbers of modeng-v1 (+0.034).

## 5. What was not done / limits
- No 19.04 OT evidence or older HPO on disk, so Mendelian/HPO channels on past are 21.04/2021-01 vintage (documented above).
- No negative control specific to `_pf` (the ablation `v1past_nopri` serves as the null for their value).
- Only 900 of 1,786 past episodes are used (stack.run_past default, kept for comparability with stackP_v2).
- Pooled dev+confirm comparisons were added after seeing the confirm result: descriptive only.
