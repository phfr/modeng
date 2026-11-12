# R1 implementer B: gene kernels that carry transfer to zero-support genes (EV-2 / EV-8 proxy / ML-1 / 15 view lifts)

Run on 2026-11-12, 04:13 to 06:30, on a machine shared with the other implementers (load average 130 to 165 on 32 cores, RAM 83/90 GB).
A stackP run that takes 24 s on an idle machine took 16 to 25 min here. Because of that, and because the lead asked at 05:20 for one
stacker process at a time, the **forward increment runs were cancelled**. The forward results below are single-channel only, plus
the stack_base reference.

## Verdict
**Discard as a stacker channel family. Keep coldstart_profile and lift_go_t0 / lift_depmap20q4 as candidates for the zero-support stratum only.**
- The lifts carry real signal for genes that transfer cannot reach. In the zero-transfer-support stratum (about 60 % of temporal targets),
  base+lifts raises AUROC by +0.055 [+0.041, +0.072] on temporal confirm and doubles nuAP (0.0041 -> 0.0074). The permuted-kernel control
  gives -0.003 [-0.007, +0.001], so the gain is not an artefact.
- The stratum is still nearly unrankable in absolute terms (zero-support AP 0.002 -> 0.003, R@100 0.004 -> 0.015). The global gains are
  AUROC only: AP and nuAP do not improve over stackP_base or stackP_bdv, so the change fails the F2-3 Pareto rule (AP and nuAP must both improve).
- In the zero stratum, stackP_bdv is already 5x better than stackP_base (z_nuAP 0.027 vs 0.004). On top of bdv the kernels add +0.010 zero-stratum AUROC and -0.003 z_nuAP.
- Over stackP_bdv the increment is +0.0055 AUROC [+0.003, +0.008], with AP -0.001 and nuAP -0.003 (both inside the CI). The bdv view
  similarity channels (vmax_*) already hold most of what a kNN lift adds.

## What was built
- `src/modeng/ext/kernels_b.py`: reviewed and fixed the untested partial file from the first attempt (review below).
  Channels (`fn(B, split_tag)`, sorted-ENSG order, work on `B.name=='past'`):
  - `lift_<view>` for the 15 views: `K_v @ T`, where K_v is the row-normalised cosine kNN graph (k=20, self excluded, negative cosines set to 0)
    and T is the exactly masked `transfer_jaccard`. Genes missing from the view get NaN.
  - `lift_paralog`: Compara 103 `within_species_paralog`, 13,319 pairs, weighted by identity, same-locus (< 1 Mb) pairs dropped.
    `lift_paralog_all` keeps the same-locus pairs.
  - **new** `lift_paralog_wide`: `within_species_paralog` + `other_paralog` with mean identity >= 30 %, same-locus pairs dropped.
    Covers 10,062 genes instead of 5,551.
  - `lift_afdb`: AFDB/Foldseek structural-cluster co-membership (EV-8 proxy; the GO column is never loaded), same-locus pairs dropped.
  - `coldstart_profile` / `coldstart_profile_t0` (ML-1, ridge form): rank-128 SVD of a safe context (context minus the union of all
    tasks' masked pairs; now asserted). A ridge maps the views to the gene factors and is cross-fitted over 5 gene folds; the transfer
    weights are then applied to the imputed columns. Held-out-gene |V| vs |Vhat| correlation is 0.37.
  - `lift_perm_*`: negative controls with the same kernel after a gene permutation (P K P^T).
- `scripts/r1_kernels_b.py`:
  - exactness test (`exact`);
  - stacker driver that also saves the score matrices to `data/cache/stackS/` (`stack`);
  - single-channel table (`alone`, written to `results/kernels_b_alone_*.tsv`);
  - zero-support stratum analysis (`zero`).

### Review of the partial file (masking exact? quantised? order?)
- **Masking: exact.** A lift is a fixed, label-free linear map applied to the cached masked transfer T.
  Brute-force check (`exact`): T recomputed from `B.context_for(task)` without the correction trick, on every 25th task.
  - max |T_cached - T_brute|: 1.2e-7 on temporal, 5.7e-6 on forward (this is the 1e-5 quantisation).
  - max |lift - K T_brute|: 9.5e-7 on temporal, 5.7e-6 on forward.
  - Targets with T > 0 in the cache but 0 in the brute-force T: **0**.
- **Quantised:** lift outputs are now rounded with np.round(x, 6). T was already quantised to 1e-5, and K holds no subtraction that
  depends on labels, so there is no residue path; the rounding is defensive.
- **Order:** views and donors assert `sorted_ensg`; kernels are built on `D.universe()` indices; the permutation uses a fixed rng(1).
- **Coldstart leakage:** an assert now checks that the safe context holds none of the masked pairs. Temporal: 206,018 -> 204,999 nnz.
  Past: 121,402 -> 121,070. The imputation is gene-cross-fitted, so a known gene is not encoded by its own column.
- Gap found: the paralog file from the first attempt held only `within_species_paralog` (13,319 rows). Compara 103 also lists 134,875
  `other_paralog` pairs. Hence `lift_paralog_wide`.

## Leakage class
- Temporal-clean on temporal: seq views (esmc, prott5, esm2_15b, phylo_npp), depmap20q4, go_t0, reactome_t0, Compara 103 (2021-02).
  These form the `kb_clean` set (paralog, paralog_wide, afdb, 7 T0 lifts, coldstart_t0).
- AFDB clusters (2023) and the ESM-2 / ProtT5 models postdate T0 but are label-free, sequence-only tools.
- Modern views (hpa_*, gtex, ps_*): transductive on temporal. `kb_all` contains them.
- On forward and holdout everything is fair, except ps_cd4 (2025), which is transductive on forward.

## Results

### Single channels, temporal confirm (138 tasks; z_* = zero-support targets as positives, supported targets removed; pool = genes with T == 0)
| channel | AP | nuAP | AUROC | R@100 | z_nuAP | z_AUROC | pool AUROC |
|---|---|---|---|---|---|---|---|
| transfer_jaccard | 0.1189 | 0.1042 | 0.705 | 0.283 | 0.0013 | 0.489 | 0.496 |
| popularity | 0.0192 | 0.0121 | 0.787 | 0.097 | 0.0048 | 0.675 | 0.694 |
| coldstart_profile | 0.0262 | 0.0138 | 0.708 | 0.116 | 0.0059 | 0.641 | 0.646 |
| coldstart_profile_t0 | 0.0259 | 0.0127 | 0.704 | 0.109 | 0.0056 | 0.635 | 0.641 |
| lift_prott5 | 0.0143 | 0.0101 | 0.670 | 0.062 | 0.0049 | 0.627 | 0.632 |
| lift_go_t0 | 0.0170 | 0.0149 | 0.669 | 0.076 | 0.0062 | 0.608 | 0.614 |
| lift_depmap20q4 | 0.0161 | 0.0164 | 0.615 | 0.052 | 0.0055 | 0.612 | 0.613 |
| lift_reactome_t0 | 0.0210 | 0.0194 | 0.641 | 0.083 | 0.0079 | 0.590 | 0.591 |
| lift_phylo_npp | 0.0122 | 0.0108 | 0.643 | 0.057 | 0.0100 | 0.595 | 0.598 |
| lift_paralog_wide | 0.0065 | 0.0056 | 0.524 | 0.044 | 0.0028 | 0.523 | 0.521 |
| lift_paralog | 0.0056 | 0.0075 | 0.472 | 0.027 | 0.0030 | 0.482 | 0.474 |
| lift_afdb | 0.0066 | 0.0069 | 0.515 | 0.034 | 0.0077 | 0.519 | 0.513 |
| lift_ps_* (5 views) | <= 0.005 | | 0.50-0.55 | | | 0.52-0.55 | |
| lift_perm_esmc / go_t0 / paralog_wide | 0.002-0.003 | | 0.495-0.505 | | | 0.50-0.51 | 0.49-0.51 |

Forward dev (842 tasks) has the same ordering (`results/kernels_b_alone_forward_dev.tsv`):
- coldstart_profile: AUROC 0.682, pool AUROC 0.615.
- lift_go_t0: AP 0.0168, pool AUROC 0.593.
- popularity: pool AUROC 0.700. transfer_onto: pool AUROC 0.664.
- perm controls: AUROC about 0.50.

**Homology (EV-2/EV-8) is weak.** Paralog and AFDB lifts cover few genes and are near chance in the zero stratum (pool AUROC 0.47-0.52).
A paralog's transfer score does not predict the zero-support targets. The useful kernels are the functional ones (GO, DepMap,
PLM, NPP) and the ML-1 imputation, and none of them beats popularity in the zero pool.

### Increment, temporal, trained on past (stackP), paired cluster bootstrap vs stackP_base (n=138 confirm / 53 dev)
| stack | split | AP | nuAP | AUROC | R@100 | dAP | dnuAP | dAUROC |
|---|---|---|---|---|---|---|---|---|
| stackP_base (reproduced: 0.0810 / 0.1341) | confirm | 0.0810 | 0.1341 | 0.7920 | 0.2358 | | | |
| + KHOM+15 lifts+coldstart (`kb_all`) | confirm | 0.0747 | 0.1255 | 0.8184 | 0.2450 | -0.0063 [-0.015,+0.001] | -0.0086 [-0.021,+0.004] | +0.0263 [+0.020,+0.034] |
| + temporal-clean set (`kb_clean`) | confirm | 0.0724 | 0.1232 | 0.8147 | 0.2446 | -0.0086 [-0.016,-0.002] | -0.0109 [-0.023,-0.000] | +0.0227 [+0.017,+0.029] |
| + 18 permuted lifts (`kb_perm`, NEG CONTROL) | confirm | 0.0774 | 0.1239 | 0.7912 | 0.2361 | -0.0036 [-0.008,+0.001] | -0.0102 [-0.018,-0.002] | -0.0009 [-0.004,+0.002] |
| kb_all | dev | 0.0772 | 0.1041 | 0.8023 | 0.2383 | -0.0005 | -0.0145 [-0.028,-0.002] | +0.0238 [+0.017,+0.031] |
| kb_clean | dev | 0.0737 | 0.1047 | 0.7969 | 0.2350 | -0.0041 | -0.0140 | +0.0184 |
| kb_perm | dev | 0.0749 | 0.1202 | 0.7771 | 0.2233 | -0.0028 | +0.0016 | -0.0014 |
| + compact 5 (`kb_top5`: coldstart, lift depmap20q4/go_t0/prott5/paralog_wide) | confirm | 0.0780 | 0.1312 | 0.8137 | 0.2450 | -0.0030 [-0.012,+0.005] | -0.0029 [-0.014,+0.009] | +0.0216 [+0.015,+0.029] |
| kb_top5 | dev | 0.0800 | 0.1150 | 0.7966 | 0.2322 | +0.0022 [-0.006,+0.011] | -0.0037 [-0.014,+0.008] | +0.0182 [+0.012,+0.026] |

kb_top5 picks the five channels with the highest gain importance in the past-trained kb_all model. The choice used only past
training, not temporal labels. Fewer channels mean less dilution: AP and nuAP are flat within the CI and AUROC is still +0.02, but the change is still not Pareto.
Zero stratum for kb_top5 on confirm: z_nuAP +0.0036 [+0.001,+0.008], z_AUROC +0.048 [+0.033,+0.066], z_R@100 0.004 -> 0.016.

The negative control matters for reading the AP and nuAP losses. Adding 18 pure-noise channels already costs -0.010 nuAP on confirm
(the LightGBM feature_fraction dilutes transfer_jaccard). Measured against the permuted control, kb_all is about AP -0.003, nuAP +0.002
and AUROC +0.027. So the real kernels add ranking information in the bulk of the list but none at the top.

### Increment over stackP_bdv (base+donor+view), temporal
| | split | AP | nuAP | AUROC | R@100 |
|---|---|---|---|---|---|
| stackP_bdv (post-fix reference) | confirm | 0.0913 | 0.1506 | 0.8252 | 0.2598 |
| stackP_bdv + kb_all | confirm | 0.0903 (-0.0010 [-0.007,+0.005]) | 0.1481 (-0.0026 [-0.011,+0.005]) | 0.8307 (+0.0055 [+0.003,+0.008]) | 0.2584 (-0.0014) |
| stackP_bdv | dev | 0.1018 | 0.1237 | 0.8103 | 0.2523 |
| stackP_bdv + kb_all | dev | 0.1041 (+0.0023 [-0.004,+0.009]) | 0.1270 (+0.0033 [-0.008,+0.014]) | 0.8141 (+0.0037 [+0.000,+0.008]) | 0.2504 |

### Zero-transfer-support stratum (targets with transfer_jaccard == 0), temporal
Stratum size:
- dev: 1,159 of 1,886 targets (61.5 %).
- confirm: 2,643 of 4,438 targets (59.6 %); every task has at least one zero-support target.

Metrics:
- z_* treats the zero-support targets as positives over the full ranking, with the supported targets removed.
- pool AUROC is computed only among genes with T == 0.
- Deltas vs stackP_base, from the paired cluster bootstrap.

| method | split | z_AP | z_nuAP | z_AUROC | z_R@100 | z_R@200 | pool AUROC | d z_nuAP | d z_AUROC |
|---|---|---|---|---|---|---|---|---|---|
| stackP_base | confirm | 0.0024 | 0.0041 | 0.635 | 0.004 | 0.012 | 0.655 | | |
| stackP_kb_all | confirm | 0.0034 | 0.0074 | 0.691 | 0.015 | 0.038 | 0.711 | +0.0033 [+0.001,+0.007] | +0.055 [+0.041,+0.072] |
| stackP_kb_clean | confirm | 0.0034 | 0.0071 | 0.683 | 0.019 | 0.038 | 0.703 | +0.0030 [+0.001,+0.005] | +0.048 [+0.034,+0.064] |
| stackP_kb_perm (control) | confirm | 0.0022 | 0.0041 | 0.632 | 0.003 | 0.012 | 0.652 | -0.0000 [-0.000,+0.000] | -0.003 [-0.007,+0.001] |
| stackP_kb_all | dev | 0.0041 | 0.0089 | 0.665 | 0.026 | 0.042 | 0.684 | +0.0041 [+0.001,+0.010] | +0.046 [+0.031,+0.063] |
| stackP_kb_clean | dev | 0.0036 | 0.0066 | 0.656 | 0.021 | 0.037 | 0.674 | +0.0018 [+0.000,+0.004] | +0.036 [+0.022,+0.052] |
| stackP_kb_perm (control) | dev | 0.0026 | 0.0047 | 0.617 | 0.007 | 0.012 | 0.636 | -0.0001 | -0.003 [-0.009,+0.003] |
| popularity alone | confirm | 0.0049 | 0.0048 | 0.675 | 0.038 | 0.059 | 0.694 | | |
| coldstart_profile alone | confirm | 0.0063 | 0.0059 | 0.641 | 0.051 | 0.074 | 0.646 | | |
| stackP_bdv | confirm | 0.0121 | 0.0270 | 0.709 | 0.046 | 0.066 | 0.728 | | |
| stackP_bdv + kb_all | confirm | 0.0113 | 0.0244 | 0.719 | 0.045 | 0.070 | 0.737 | -0.0026 [-0.011,+0.004] (vs bdv) | +0.010 [+0.005,+0.015] (vs bdv) |
| stackP_bdv | dev | 0.0146 | 0.0332 | 0.681 | 0.045 | 0.071 | 0.699 | | |
| stackP_bdv + kb_all | dev | 0.0119 | 0.0285 | 0.687 | 0.052 | 0.071 | 0.705 | -0.0047 [-0.014,+0.002] (vs bdv) | +0.006 [-0.002,+0.013] (vs bdv) |

**Key finding for the stratum:** the donor and view channels in stackP_bdv already lift the zero-support stratum about 5x over
stackP_base (z_AP 0.0024 -> 0.0121, z_nuAP 0.0041 -> 0.0270 on confirm). don_hpo2101, the second most important feature, is the likely
carrier: phenotype donors reach genes that the disease rows do not. On top of bdv the kernels add only +0.01 zero-stratum AUROC, and
zero-stratum AP and nuAP go slightly down. The zero-support problem is better addressed by donor rows (new label sources) than by
smoothing transfer over gene-similarity kernels.

Caveat: stackP_base (deg^-0.5 positive weights) ranks zero-support targets worse than popularity alone (pool AUROC 0.655 vs 0.694).
Part of the kb_all gain moves the stack back toward a popularity-like ordering in that pool. The z_nuAP gain (+0.003, which is
degree-weighted) is larger than popularity's (0.0048) and the permuted control is null, so part of the gain is genuine kernel
signal. In absolute terms, though, zero-support targets remain essentially unrankable (z_AP 0.003 against a random level of about 0.002).

### Forward
- `stack_base` (cv) reproduced on dev: AP 0.0809, nuAP 0.1447, AUROC 0.8231, R@100 0.2211. This equals `stack_base_w05` from another agent.
- No forward increment: `kb_all` was killed at 05:20 on the lead's RAM instruction; at the observed speed (base cv took 55 min) it
  would not have finished by the deadline.
- The single-channel forward table above is the forward evidence.

## Positive / negative controls
- Positive: stackP_base was reproduced exactly (0.0778 / 0.1187 dev, 0.0810 / 0.1341 confirm), and stackP_bdv confirm nuAP 0.1506
  matches the post-fix reference.
- Negative: permuted kernels give single-channel AUROC of about 0.50, ~0 in the zero stratum, and AUROC -0.001 in the stack. The nuAP
  loss of the permuted stack (-0.010) shows that the channel-count dilution cost is about as large as kb_all's own nuAP loss.

## Runtime
- Kernels: about 70 s total (15 views + 3 homology + permutations; cache `data/cache/kernels_b/`).
- Channels: 1-30 s each; coldstart about 30 s per benchmark.
- stackP: 16-25 min each under load (24 s idle).

## What failed / not done
- Forward increments, a holdout run, and EV-8 Pfam-domain donors (the AFDB proxy was null, so the Pfam parse was not worth it).
- Homology lifts are near chance. Coverage is low, and paralog transfer is not predictive for the new targets.

## New ideas while working
1. **Dilution is a real cost**: 18 noise channels cost -0.010 nuAP. Increment tests should be paired with a permuted-channel control
   of the same width, or the stacker should use feature_fraction=1 / fewer channels. Many R1 "no gain" verdicts may be dilution.
2. A single **stratum-gated channel**: the lift or coldstart score multiplied by 1[T == 0], or a separate stacker for the zero pool.
   The lifts help only where T == 0, and spreading them over all genes costs AP at the top.
3. The base stack ranks zero-support targets worse than popularity alone. The deg^-0.5 weighting trains the model to distrust
   non-transfer evidence, so the zero pool may need its own weighting.
