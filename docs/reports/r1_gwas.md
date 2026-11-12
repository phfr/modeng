# R1 implementer D: GWAS statistics before they cross the threshold (SG-7, CI-08 / WC-XPOWER, LT-9)

2026-11-12, implementer D. Code: `src/modeng/ext/gwas_d.py` (channels), `scripts/gwas_d_eval.py` (alone + paired comparison
with covered / GWAS-only strata), `scripts/gwas_d_stack.py` (runs `modeng.stack` unchanged, also saves score matrices).
Outputs: `results/gwas_d/alone.tsv`, `results/gwas_d/compare.tsv`, leaderboard rows `stackP_d_*` / `stack_d_*`.
No changes to `channels.py`, `stack.py`, `tasks.py`, `metrics.py`, `bench.py`.

## Verdict
**Keep `gwas_catlag` (SG-7, GWAS Catalog ingestion lag); it is the cluster's real signal. MAGMA own/proxy-trait z channels
(LT-9, WC-XPOWER) are weak alone; the seed-weighted cross-trait z (CI-08 linear arm) is anti-informative -> discard.**
On temporal (train on `past`), adding the five GWAS channels to `stackP_bdv` gives dev AP +0.017 [0.005, 0.034],
nuAP +0.022 [0.006, 0.042]; confirm AP +0.008 [-0.004, 0.021], nuAP +0.003 [-0.012, 0.018], AUROC +0.0075 [0.005, 0.011],
R@100 +0.013 [0.004, 0.024]. On the GWAS-only target stratum the gain is large and replicates: AP 0.019 -> 0.048 (dev),
0.017 -> 0.044 (confirm); R@100 +0.06 on both. `gwas_catlag` is the top feature by gain in the stacker.
Under the F2-3 rule (a gain must improve BOTH AP and nuAP): **dev passes clearly (both CIs > 0); confirm is directionally
positive on both but neither CI excludes 0** (covered confirm tasks: AP +0.016 [-0.004, 0.035], nuAP +0.011 [-0.017, 0.036]).
The gene-permuted negative control gives -0.003 to -0.005 everywhere (no gain). Status: **promising, keep `gwas_catlag`,
needs an archived-catalog rerun (leakage route below) and the forward increment**. **Forward (CV, fair-by-date inputs): very large gain, both AP and nuAP:
dev AP 0.081 -> 0.132 (+0.051 [0.045, 0.057]), nuAP 0.145 -> 0.213 (+0.068 [0.060, 0.077]); confirm AP 0.077 -> 0.121
(+0.044 [0.038, 0.050]), nuAP 0.137 -> 0.199 (+0.062 [0.053, 0.071]).** (CIs too narrow: forward tasks are hierarchically
correlated, bootstrap unit = task.) The forward gain must be read with the catalog-curation caveat below; it says that on
forward a large share of "new" 26.09 genes is GWAS Catalog content that was public before 25.09.

## What was built
Trait data (read-only from modet):
* 108 MAGMA studies <= T0 (`gwas_magma_t0/<GCST>_w10.parquet`) + 85 further post-T0 studies (`gwas_magma/<disease>.parquet`,
  92 unique GCST, 7 shared with the T0 set) -> Z [193 studies x 19,255 genes], raw MAGMA z; standardised version per study
  (median / MAD).
* Study dates, trait text and mapped trait ids from the GWAS Catalog studies table (release 2026-10-22).
* Availability rule per benchmark (publication DATE <= cutoff): past 2019-04-30 (88 studies), temporal 2021-04-30 (108),
  forward 2025-09-30 (182), holdout all. `B.name == 'past'` therefore uses only studies dated <= 2019-04 (NaN otherwise).

Disease -> trait mapping: catalog `MAPPED_TRAIT_URI` ids (+ modet's own study->disease assignment) matched against the
task disease and its ontology descendants (weight 1) and its specific ancestors (weight 0.7, ancestors with <= 100
descendants) using `modeng.data.descendants/relatives`; semantic fallback (LT-9) = TF-IDF char 3-5-gram cosine between the
disease name and the cleaned trait text (>= 0.55, top 5), weight = cosine.

Channels (`fn(B, split_tag)`, NaN = disease not covered):

| channel | idea | definition |
|---|---|---|
| `gwas_own` | CI-08 own trait | max over own/descendant/ancestor-mapped studies of w * MAGMA z |
| `gwas_sem` | LT-9 | weighted mean standardised z over ontology + semantic proxy studies |
| `gwas_xpow` | WC-XPOWER | per study 2-component normal mixture (EM), posterior noncentrality, P(z' > 4.56 at N' = 3N); max over the gwas_sem studies |
| `gwas_seedw` | CI-08 / WC-XPOWER linear arm | per study seed-enrichment z_e = sum_seeds z / sqrt(n); weights max(z_e - 2, 0); weighted mean z over all available studies (every disease) |
| `gwas_seedw_far` | diagnostic | gwas_seedw with genes within 1 Mb of a seed set to 0 |
| `gwas_catlag` | SG-7 | GWAS Catalog associations p <= 1e-6, DATE ADDED TO CATALOG <= cutoff, mapped to the disease (descendants w=1, specific ancestors w=0.7); gene = MAPPED_GENE symbols; score = max w * -log10 p (cap 50); 0 for other genes of a covered disease |
| `gwas_catlag12` | SG-7 sensitivity | same, associations added >= 12 months before the cutoff |
| `gwas_sem_perm`, `gwas_seedw_perm` | negative controls | gene axis of Z permuted (fixed seed) |

ML-4 (power extrapolation trained on release pairs) was not built (time; the WC-XPOWER transform is the untrained version).

## Coverage
| benchmark | own (ontology) | own + semantic | catalog (catlag) |
|---|---|---|---|
| temporal dev (53) | 16 | 18 (17 with >= 1 non-NaN gene) | 24 |
| temporal confirm (138) | 46 | 58 (53) | 52 |
| past train (1,786; 900 used by stackP) | 255 | 385 | 360 of the 900 |
| forward dev (842) | 148 | 191 (169) | 671 |
| forward confirm (797) | 123 | 164 (143) | 636 |

## Channels alone (covered tasks only; transfer_jaccard on the same tasks for reference)
All-gene universe; `go_*` = GWAS-only target stratum (temporal targets with `evidence_class == 'gwas_only'` in
`arrival_devall_t0_string115_phys_notm_400.parquet`; other targets removed from the ranking).

| bench/split | channel | n cov | AP | nuAP | AUROC | R@100 | transfer AP (same tasks) | GWAS-only AP / R@100 (transfer) |
|---|---|---|---|---|---|---|---|---|
| temporal dev | gwas_own | 16 | 0.021 | 0.024 | 0.589 | 0.058 | 0.078 | 0.018 / 0.071 (0.059 / 0.138) |
| temporal dev | gwas_sem | 17 | 0.019 | 0.023 | 0.578 | 0.050 | 0.076 | 0.015 / 0.061 |
| temporal dev | gwas_xpow | 17 | 0.021 | 0.023 | 0.594 | 0.056 | 0.076 | 0.017 / 0.061 |
| temporal dev | gwas_seedw | 53 | 0.002 | 0.002 | **0.480** | 0.000 | 0.123 | 0.002 / 0.000 |
| temporal dev | gwas_catlag | 24 | **0.066** | **0.066** | 0.550 | 0.125 | 0.071 | **0.085 / 0.173** (0.060 / 0.150) |
| temporal dev | gwas_catlag12 | 23 | 0.051 | 0.054 | 0.532 | 0.104 | | 0.061 / 0.137 |
| temporal dev | gwas_sem_perm | 17 | 0.003 | 0.003 | 0.496 | 0.005 | | |
| temporal confirm | gwas_own | 46 | 0.025 | 0.025 | 0.598 | 0.051 | 0.101 | 0.027 / 0.067 (0.035 / 0.121) |
| temporal confirm | gwas_xpow | 53 | 0.029 | 0.025 | 0.590 | 0.058 | 0.096 | 0.037 / 0.085 (0.033 / 0.149) |
| temporal confirm | gwas_catlag | 52 | 0.047 | 0.049 | 0.529 | 0.069 | 0.110 | **0.083 / 0.150** (0.027 / 0.106) |
| temporal confirm | gwas_catlag12 | 50 | 0.046 | 0.047 | 0.525 | 0.067 | | 0.084 / 0.148 |
| forward dev | gwas_own | 148 | 0.021 | 0.026 | 0.616 | 0.076 | 0.073 | |
| forward dev | gwas_xpow | 169 | 0.019 | 0.023 | 0.612 | 0.073 | 0.075 | |
| forward dev | gwas_seedw | 840 | 0.004 | 0.004 | 0.538 | 0.003 | 0.092 | |
| forward dev | gwas_catlag | 671 | **0.082** | **0.087** | 0.672 | **0.233** | 0.086 (nuAP 0.086, R@100 0.220) | |
| forward confirm | gwas_catlag | 636 | 0.078 | 0.089 | 0.665 | 0.219 | 0.082 (nuAP 0.081) | |
| forward dev | gwas_sem_perm | 169 | 0.003 | 0.003 | 0.498 | 0.003 | | |

Full table: `results/gwas_d/alone.tsv`.

Observations:
* **Catalog lag is large.** Of the catalog genes hit at p <= 1e-6 for a covered disease, only 14-17 % are already seeds.
  On forward, **38 % of all new 26.09 targets of covered diseases are catalog hits added <= 2025-09** (19,321 / 51,204 dev;
  18,804 / 49,230 confirm); precision of non-seed hits ~0.10. On temporal, 17.5 % (dev) / 7.5 % (confirm) of targets are
  hit, and 32-39 % of the non-seed hits are genes the temporal benchmark excludes as "unknown (weak evidence)" (SG-7
  predicted this removal by design). Example (temporal dev, refractive error): AP 0.39, R@100 0.77; top non-seed catalog
  genes RASGRF1, ABI3, RASGEF1B, TJP2, CFAP299, LRFN5, NOG, ZBTB38, DLG2 all arrive by 25.09.
* **Ingestion-lag sensitivity:** dropping associations added in the last 12 months before the cutoff keeps the confirm signal
  unchanged (GWAS-only AP 0.083 -> 0.084) and keeps ~75 % on dev: the effect is not driven by additions just before the seed
  release.
* MAGMA own-trait z is weak as a ranker over all genes (AUROC ~0.59-0.62, AP 0.02) but clearly above the gene-permuted control
  (AP 0.003, AUROC 0.50). The WC-XPOWER crossing transform is no better than linear z (temporal dev AP 0.021 vs 0.021,
  confirm 0.029 vs 0.025, forward 0.019 vs 0.021): the kill criterion "crossing beats linear by >= 30 % on GWAS-only recall"
  is not met (0.061 vs 0.071 dev, 0.085 vs 0.067 confirm).
* **`gwas_seedw` is anti-informative** (AUROC 0.48 temporal, below the permuted control 0.50; R@100 ~0), also after removing
  genes within 1 Mb of a seed (AUROC 0.473) and among genes with a value (0.463). Studies whose z is enriched at the seeds are
  mostly studies whose hits already ARE the seeds (OT took them in); their other high-z genes do not arrive. CI-08's
  "borrow from correlated traits" premise fails at the gene level with these 108-193 studies.

## Increment tests
Temporal, `--train past` (`modeng.stack.run_past`, 900 past episodes), reference rerun in the same session
(`stackP_d_bdv` = `base,donor,view`; reproduces the leaderboard `stackP_bdv` 7555a67 exactly on dev: AP 0.1018, nuAP 0.1237,
confirm AP 0.0913, nuAP 0.1506 = positive control). `gw` = gwas_own, gwas_sem, gwas_xpow, gwas_seedw, gwas_catlag.
Paired cluster bootstrap (5,000), `results/gwas_d/compare.tsv`.

| run | split | AP | nuAP | AUROC | R@100 | onnet AP | onnet nuAP |
|---|---|---|---|---|---|---|---|
| stackP_d_bdv (ref) | dev | 0.1018 | 0.1237 | 0.8103 | 0.2523 | 0.1210 | 0.1731 |
| stackP_d_bdv_gw | dev | **0.1189** | **0.1452** | 0.8210 | 0.2747 | 0.1385 | 0.1896 |
| stackP_d_bdv_perm (neg. control) | dev | 0.0993 | 0.1203 | 0.8082 | 0.2456 | 0.1196 | 0.1697 |
| stackP_d_bdv (ref) | confirm | 0.0913 | 0.1506 | 0.8252 | 0.2598 | 0.0987 | 0.1712 |
| stackP_d_bdv_gw | confirm | 0.0995 | 0.1537 | 0.8327 | 0.2727 | 0.1058 | 0.1725 |
| stackP_d_bdv_perm (neg. control) | confirm | 0.0883 | 0.1462 | 0.8234 | 0.2532 | 0.0977 | 0.1704 |

Deltas gw - ref, [95 % CI]:

| split | subset | n | AP | nuAP | AUROC | R@100 |
|---|---|---|---|---|---|---|
| dev | all | 53 | +0.017 [0.005, 0.034] | +0.022 [0.006, 0.042] | +0.011 [0.004, 0.018] | +0.023 [0.000, 0.052] |
| dev | covered (any GWAS channel) | 26 | +0.034 [0.012, 0.065] | +0.041 [0.014, 0.076] | +0.017 | +0.049 [0.007, 0.103] |
| dev | uncovered | 27 | +0.000 [-0.007, 0.009] | +0.003 [-0.011, 0.019] | +0.005 | -0.003 |
| dev | GWAS-only targets | 29 | +0.029 [0.010, 0.057] (0.019 -> 0.048) | | +0.023 | +0.061 [0.018, 0.124] |
| confirm | all | 138 | +0.008 [-0.004, 0.021] | +0.003 [-0.012, 0.018] | +0.0075 [0.005, 0.011] | +0.013 [0.004, 0.024] |
| confirm | covered | 71 | +0.016 [-0.004, 0.035] | +0.011 [-0.017, 0.036] | +0.009 [0.003, 0.016] | +0.021 [0.006, 0.039] |
| confirm | uncovered | 67 | +0.000 | -0.006 [-0.015, 0.005] | +0.006 | +0.004 |
| confirm | GWAS-only targets | 76 | +0.027 [0.013, 0.044] (0.017 -> 0.044) | | +0.025 [0.011, 0.054] | +0.057 [0.029, 0.090] |

Negative control (`gwas_sem_perm`, `gwas_seedw_perm` added to bdv): all-task deltas AP -0.003 / nuAP -0.003 (dev),
AP -0.003 / nuAP -0.004 (confirm); GWAS-only AP -0.001 / +0.001: no gain, as required.
Feature gain in `stackP_d_bdv_gw`: gwas_catlag is #1 (236k, ahead of don_hpo2101 149k and transfer_jaccard 144k),
r_gwas_catlag #5, gwas_seedw #15 (used, presumably as a negative cue); gwas_own/sem/xpow are not in the top 30.
Attribution runs (bdv + catlag only, bdv + catlag12, bdv + MAGMA only) were queued but dropped after the coordinator's RAM
limit (one stacker at a time); alone results and feature gains point to catlag as the source of the gain.

Forward, `--train cv` (`modeng.stack.run`, grouped 5-fold OOF on dev, dev-trained model on confirm), base + gw vs the
leaderboard `stack_base_w05` (same code and settings; another agent's run, deterministic; my own base run was killed to obey
the one-stacker RAM rule). No forward `bdv` reference exists, so the forward test is over `base`.

| run | split | AP | nuAP | AUROC | R@100 | onnet AP | onnet nuAP |
|---|---|---|---|---|---|---|---|
| stack_base_w05 (ref) | dev | 0.0809 | 0.1447 | 0.8231 | 0.2211 | 0.0856 | 0.1602 |
| stack_d_base_gw | dev | **0.1316** | **0.2131** | 0.8586 | 0.3178 | 0.1422 | 0.2404 |
| stack_base_w05 (ref) | confirm | 0.0770 | 0.1373 | 0.8117 | 0.2062 | 0.0819 | 0.1578 |
| stack_d_base_gw | confirm | **0.1209** | **0.1989** | 0.8491 | 0.3028 | 0.1311 | 0.2289 |

| split | subset | n | AP | nuAP | AUROC | R@100 |
|---|---|---|---|---|---|---|
| dev | all | 842 | +0.051 [0.045, 0.057] | +0.068 [0.060, 0.077] | +0.036 | +0.097 [0.085, 0.109] |
| dev | covered (own MAGMA or catlag) | 676 | +0.060 [0.053, 0.067] | +0.083 [0.073, 0.094] | +0.041 | +0.113 |
| dev | uncovered (only gwas_seedw) | 166 | +0.013 [0.003, 0.023] | +0.007 [-0.005, 0.021] | +0.012 | +0.031 |
| confirm | all | 797 | +0.044 [0.038, 0.050] | +0.062 [0.053, 0.071] | +0.037 | +0.097 [0.085, 0.109] |
| confirm | covered | 648 | +0.052 [0.045, 0.060] | +0.075 [0.064, 0.085] | +0.042 | +0.114 |
| confirm | uncovered | 149 | +0.008 [-0.001, 0.018] | +0.005 [-0.009, 0.022] | +0.019 | +0.021 |

Feature gain (forward confirm model): transfer_jaccard 948k, **gwas_catlag 835k (#2)**, transfer_onto 525k, ..., gwas_seedw
93k, gwas_xpow 26k, gwas_sem 11k, gwas_own 11k. The forward negative control (perm channels) was not run (RAM rule, time);
the temporal one is clean.


## Leakage class
* `gwas_catlag`: filter on DATE ADDED TO CATALOG <= seed date approximates an archived snapshot. Residual routes: (1) trait
  ids (MAPPED_TRAIT_URI) and MAPPED_GENE come from the 2026-10 release, so later re-mapping of an old association to the
  task's EFO term leaks; (2) the ontology used to map traits to tasks includes the 2509/2609 releases (as the harness does
  everywhere). Class: **temporal-clean up to curation back-fill** on temporal and forward; the 12-month-lag variant shows the
  signal does not depend on the final months. An archived 2021-04 / 2025-09 catalog release would close route (1).
* MAGMA channels: temporal-clean by publication date on temporal (92 of the 108 T0 studies are the 2018 SAIGE UKB PheCode
  GWAS whose catalog accessions were created 2024); on forward fair by publication date, but many post-T0 PheCode studies were
  added to the catalog on 2026-09-08 (summary statistics public since 2024-07). past uses only studies <= 2019-04.
* `gwas_seedw` uses only task seeds; same date rules.

## Runtime
Channel build: Z matrix 6 s, matching ~10-40 s per benchmark, catalog table ~60 s (once); all channels for temporal + forward
+ past < 10 min CPU. stackP runs took ~20-25 min each because the machine was at load 150 (other agents).

## What failed / caveats
* CI-08 seed-weighted proxy (anti-informative, see above); WC-XPOWER non-linearity (no gain over linear z).
* LT-9 semantic proxies add only 1-2 diseases on temporal dev (16 -> 17-18) and produce some wrong matches with char n-grams
  (e.g. "lung cancer" ~ "cancer of eye" at 0.54, removed by the 0.55 threshold); a sentence encoder would be better but the
  coverage ceiling is the trait list (193 studies), not the matcher.
* Only ~30 % of temporal dev diseases have any own GWAS; gains are confined to covered diseases (uncovered: delta ~0, as it
  should be).
* Overlap with other implementers: H's cross-trait "GWAS hit anywhere / L2G prior" and E's per-datasource backlog channels may
  carry part of the same information; the combination was not tested here.

## New ideas
1. **Archived catalog snapshots** (EBI FTP keeps dated releases) for 2019-04 / 2021-04 / 2025-09: removes the re-mapping route
   and probably makes `gwas_catlag` the cleanest strong channel we have.
2. **Catalog-lag with locus-to-gene instead of MAPPED_GENE:** OT later attaches genes by L2G, not by nearest gene; scoring all
   genes within +-500 kb of a lead SNP by distance / coding-variant / eQTL evidence should lift recall on the GWAS-only stratum.
3. **Seed-depleted loci:** the negative seedw result suggests a feature "locus already explained by a seed" (catalog lead
   SNP within 500 kb of a seed) as a down-weighting feature for catlag hits.
4. Extend catlag to **sub-threshold catalog rows** (the catalog file keeps p <= 1e-5) with a per-study power weight
   (N, year): a cheap version of ML-4.
