# R2 E-A4: backlog transfer (donor weights x relatives' GWAS Catalog hits), channel `catxfer_*`

Implemented 2026-11-12 06:22-07:10. Code: `src/modeng/ext/catxfer.py`. Target endpoint: strict-a (`sa_*`, genetic discoveries
first seen 2022+) on temporal, where the leader `stack_v3cv_ptr` only ties modet's headline.

## Verdict
**Discard as a leader addition (kill criterion met); keep the channel file.** Alone, `catxfer_ridge` is a real signal on strict-a
(confirm sa_AP 0.038 vs gene-permuted 0.001; on sa it is as strong as `gwas_catlag` and `dc_ridge_pos` alone) and covers 100 % of
diseases. In the stacker it is the 9th most-used feature, but the increment is split: **dev (53, OOF) slightly negative, confirm
(138) positive** (sa_AP +0.006 [-0.000, +0.012], P 0.03 vs the leader; +0.010 [+0.003, +0.017] vs the same-width permuted control).
The E-A4 kill rule ("no gain in the no-own-GWAS stratum on dev") is met (dev no-GWAS AP -0.007). Over the newer
leader `stack_v3ptr_p0.35` the confirm gain shrinks to sa_AP +0.002 [-0.004, +0.009], sa_nuAP +0.002, nuAP -0.002 and dev nuAP / onnet
nuAP fall. Forward (1,639 diseases): no increment (confirm AP -0.002 [-0.005, +0.000], nuAP -0.001). Strict-a is not fixed by
this channel.

## What was built
- `C` [context rows x genes]: 1 if the GWAS Catalog (release 2026-10, same table / p <= 1e-6 / MAPPED_GENE parsing / DATE ADDED filter
  as `gwas_catlag`, `gwas_d._assoc` + `gwas_d.CUTOFF`: past <= 2019-04-30, temporal <= 2021-04-30, forward <= 2025-09-30) has a hit
  for a trait mapped to the context disease or one of its ontology descendants. Sizes: past 1,745 traits, 726/2,696 rows with hits;
  temporal 2,363 traits, 1,118/3,700 rows, 229k pairs; forward 12,748 traits, 2,978/6,164 rows, 734k pairs.
- Per task donor weights w; score = C^T w, quantised to 1e-5. Donors blocked: own row, fam rows, every row carrying a masked pair
  (as `_ridge_w`), and every row whose trait set intersects the task's own trait set (own id + descendants), so the task's own catalog
  hits (= `gwas_catlag`) never enter via an ancestor row.
- Channels: `catxfer_ridge` (w = clip(dc_ridge weights, 0)), `catxfer_jac` (Jaccard top-50), `*_new` (C minus the donor's own context
  label = catalog backlog not yet in OT at the seed date), controls `catxfer_ridge_perm`, `catxfer_jac_perm`, `catxfer_ridge_new_perm`
  (gene axis of C permuted, seed 7). cs_donor lasso weights not implemented (time).
- Runtime: channel ~20 s per benchmark split (ridge inverse dominates); temporal stacker ~1.5 min per run, forward ~15-20 min.

## Channel alone (temporal; `results/catxfer_alone_temporal.tsv`)
| channel | split | AP | nuAP | AUROC | sa_AP | sa_nuAP | onnet nuAP |
|---|---|---|---|---|---|---|---|
| catxfer_ridge | dev | 0.023 | 0.023 | 0.687 | 0.024 | 0.026 | 0.042 |
| catxfer_ridge | confirm | 0.030 | 0.026 | 0.675 | **0.038** | 0.036 | 0.035 |
| catxfer_jac | confirm | 0.028 | 0.024 | 0.570 | 0.029 | 0.033 | 0.032 |
| catxfer_ridge_new | confirm | 0.014 | 0.014 | 0.645 | 0.014 | 0.017 | 0.021 |
| catxfer_jac_new | confirm | 0.014 | 0.015 | 0.548 | 0.013 | 0.017 | 0.022 |
| catxfer_ridge_perm | confirm | 0.002 | 0.003 | 0.502 | 0.001 | 0.002 | 0.004 |
| gwas_catlag (ref, 38 % covered, NaN elsewhere) | confirm | 0.047 | 0.049 | 0.529 | 0.024 | 0.027 | 0.057 |
| dc_ridge_pos (ref) | confirm | 0.113 | 0.106 | 0.715 | 0.047 | 0.046 | 0.126 |

The "not yet in OT" restriction (`_new`) halves the signal: most of what transfers is catalog genes the donor already has in OT.
Forward dev alone (842 tasks): catxfer_ridge AP 0.048, nuAP 0.056, AUROC 0.746 vs permuted 0.004 / 0.004 / 0.497.

## Increment in the stacker (temporal, cv training)
Commands: `python -m modeng.stack --bench temporal --channels <v3cv_ptr list>,catxfer_ridge,catxfer_ridge_new --tag v3ptr_ea4`;
same with `catxfer_ridge_perm,catxfer_ridge_new_perm` (`v3ptr_ea4perm`, same width); positive control `v3ptr_ea4ref` (no new channel)
reproduces `stack_v3cv_ptr` exactly (dev AP 0.1648 nuAP 0.1579, confirm 0.1434 / 0.1841). Logs `results/stack_v3ptr_ea4*.txt`.
Paired cluster bootstrap (`modeng.compare.paired_bootstrap`, 5,000 draws); strata = own GWAS (gwas_catlag non-NaN) / none.

| comparison | split | sa_nuAP | sa_AP | nuAP | AP | onnet nuAP | onnet AP |
|---|---|---|---|---|---|---|---|
| ea4 vs leader | dev (n 53, sa 28) | -0.0025 [-0.011,+0.005] | -0.0007 [-0.005,+0.003] | -0.0011 [-0.010,+0.006] | -0.0026 [-0.008,+0.003] | -0.0079 [-0.020,+0.002] | -0.0041 [-0.010,+0.001] |
| ea4 vs leader | confirm (n 138, sa 71) | +0.0054 [-0.008,+0.017] | **+0.0060 [-0.000,+0.012] P 0.03** | +0.0028 [-0.002,+0.009] | +0.0031 [-0.001,+0.008] P 0.08 | +0.0002 [-0.008,+0.008] | +0.0042 [-0.000,+0.009] P 0.03 |
| ea4 vs perm | dev | -0.0011 | -0.0004 | -0.0043 | -0.0042 | -0.0105 [-0.022,-0.001] | -0.0038 |
| ea4 vs perm | confirm | **+0.0163 [+0.002,+0.034]** | **+0.0102 [+0.003,+0.017]** | +0.0047 [+0.000,+0.010] | +0.0057 [+0.002,+0.010] | +0.0031 | +0.0060 [+0.002,+0.010] |
| perm vs leader | dev | -0.0014 | -0.0002 | +0.0032 | +0.0016 | +0.0026 | -0.0002 |
| perm vs leader | confirm | -0.0109 [-0.031,+0.003] | -0.0042 | -0.0019 | -0.0026 | -0.0030 | -0.0017 |

Means: confirm sa_nuAP 0.1054 (ea4) vs 0.1000 (leader) vs 0.0890 (perm); sa_AP 0.0649 / 0.0589 / 0.0547.
Strata (mean deltas ea4 vs leader): dev own-GWAS (24) AP +0.003, nuAP +0.007; dev no-GWAS (29) AP -0.007, nuAP -0.007;
confirm own-GWAS (52) AP +0.003, nuAP -0.002; confirm no-GWAS (86) AP +0.003, nuAP +0.006, sa_nuAP +0.011 (36 sa tasks).
Post hoc pooled dev+confirm (191; not pre-registered): sa_AP +0.0041 [-0.001, +0.008] P 0.05, AP +0.0016, nuAP +0.0017.
All four variants (`v3ptr_ea4all`, + jac and jac_new): weaker (dev AP -0.006 [-0.012, -0.000]; confirm AP +0.0025, sa_AP +0.0036).

### Over the new leader `stack_v3ptr_p0.35` (positives deg^-0.35, committed 06:28 while this ran)
`--pos-deg-power 0.35`, same channels + catxfer_ridge, catxfer_ridge_new (`v3ptr_p0.35_ea4`) and the permuted twin (`v3ptr_p0.35_ea4perm`):

| comparison | split | sa_nuAP | sa_AP | nuAP | AP | onnet nuAP | onnet AP |
|---|---|---|---|---|---|---|---|
| ea4 vs p0.35 leader | dev | -0.0037 | +0.0047 [-0.001,+0.010] | -0.0057 | -0.0035 | -0.0113 [-0.023,-0.001] | -0.0047 |
| ea4 vs p0.35 leader | confirm | +0.0015 [-0.011,+0.012] | +0.0022 [-0.004,+0.009] | -0.0018 | +0.0010 | +0.0031 | +0.0027 |
| ea4 vs perm | confirm | +0.0040 | +0.0037 | +0.0032 | +0.0043 [+0.000,+0.010] | +0.0049 | +0.0050 [+0.000,+0.011] |

Means confirm sa_nuAP 0.0920 vs 0.0905, sa_AP 0.0675 vs 0.0654. Over the stronger leader the strict-a gain mostly disappears.

Reading: the permuted control moves the stacker by up to +-0.003 AP / +-0.01 sa_nuAP (LightGBM noise from two extra columns), so the
dev result is within noise and the confirm gain is about 1-2x that noise. The confirm gain lands where pre-registered (no-own-GWAS
stratum), dev does not. Also noted: AUROC rises consistently (dev 0.842 -> 0.849, confirm 0.857 -> 0.864).

## Forward (fair-by-date for the catalog cutoff 2025-09)
`--bench forward`, cv, channels of `stack_v3cv` (v3cv list without pt_ridge*) + catxfer_ridge, catxfer_ridge_new (`v3cv_ea4`,
log `results/stack_forward_v3cv_ea4.txt`) vs the existing `stack_v3cv` (forward has no strict-a target set):

| split | metric | ea4 vs v3cv | delta [95 % CI] |
|---|---|---|---|
| dev (842) | nuAP | 0.2441 vs 0.2441 | +0.0000 [-0.0036, +0.0036] |
| dev | AP | 0.1520 vs 0.1538 | -0.0017 [-0.0044, +0.0009] |
| dev | onnet nuAP | 0.2671 vs 0.2690 | -0.0019 [-0.0062, +0.0022] |
| confirm (797) | nuAP | 0.2212 vs 0.2221 | -0.0009 [-0.0042, +0.0023] |
| confirm | AP | 0.1380 vs 0.1402 | -0.0022 [-0.0047, +0.0001] |
| confirm | onnet nuAP | 0.2508 vs 0.2507 | +0.0001 [-0.0037, +0.0039] |

No gain on forward (the large, date-fair benchmark): the temporal-confirm gain does not replicate. The forward permuted twin was
stopped for the deadline (not needed given a null real result).

## Leakage class
As `gwas_catlag`: catalog associations filtered by DATE ADDED <= seed cutoff on every benchmark (temporal-clean by date), residual
route = the 2026-10 release's trait mapping / MAPPED_GENE annotation and the modeng ontology releases used for descendants. Context
side = masked context only (same donor blocking as `dc_ridge`). On forward many PheCode studies were added 2026-09-08 and are
excluded by the date filter.

## What failed / caveats
- `_new` (backlog not yet in OT) is weaker than the full catalog row, against the E-A4 mechanism claim (the cross term beyond labels
  is small; most transferred signal duplicates label transfer, with catalog weighting).
- Jaccard top-50 weights are worse than ridge weights in the stacker.
- Temporal confirm has been viewed many times this night (selection); dev is small (28 tasks with strict-a targets).
- No cs_donor-weight variant; no `--train past` stacker run (past-trained stacks are below cv ones).

## New ideas
- Use C as a graded matrix (-log10 p, number of independent loci) and normalise rows by size; huge ancestor rows currently get
  binary rows covering thousands of genes.
- Trait-level donors instead of disease rows: ridge of the seed vector directly on the 2,363 catalog trait rows (seed-enrichment
  weights, no OT mapping needed), which also covers traits without an OT disease.
