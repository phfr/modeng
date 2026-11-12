# r4: sub-threshold GWAS loci -> gene assignment for strict-a (F2-6 / SG-1 lite)

## 0. Pre-registration (written and committed before any result of these channels was seen)
- Reference: modeng-v1 `stack_v3ptr_p0.35` (temporal, cv-trained, power 0.35, v1 channel list of `docs/preregistration_modeng_v1.md`).
- Main variant `v1_subthr` = v1 channels + `subthr_near,subthr_cs2g,subthr_win` (3 channels). Control `v1_subthrperm` = v1 + the 3 gene-permuted
  channels (same width). Further variants only if chosen on dev (each counted below); every variant that reaches confirm is scored there once.
- Primary endpoint (strict-a): sa_nuAP and sa_AP of the stacker vs modeng-v1 (all genes) and sa_onnet_nuAP vs modet's headline
  (`scripts/leader_endpoints.py`), paired cluster bootstrap.
- Dev rule (53 dev diseases, 28 with strict-a targets): a variant goes forward as "candidate" if dev sa_nuAP delta vs v1 > 0, overall dev AP and nuAP
  deltas >= -0.003, and its sa_nuAP delta exceeds the permuted control's. Alone screen (channel alone on covered tasks, sa columns) decides which
  single-channel variants are worth a stack run.
- Claim on confirm (138 diseases, 71 with strict-a): only if the confirm sa_nuAP CI vs v1 excludes 0 and the permuted control shows no gain;
  otherwise "no strict-a gain".

## 1. Verdict (2026-11-12, ~1.5 h)
**No strict-a gain: discard as a modeng-v1 increment; keep the lead table as an input for later SNP-to-gene work.** Sub-threshold loci carry a real
but tiny signal (alone: 3-14x enrichment over base rate, clearly above the permuted control), yet they add only 7 (dev) / 24 (confirm) strict-a
targets that `gwas_catlag` does not already flag, over 17 dev / 53 confirm diseases with a mapped study. In the stacker the gain is within noise and
does not replicate: confirm sa_nuAP -0.006 [-0.017, +0.003] (3 channels) and -0.007 [-0.018, +0.002] (nearest gene only) vs modeng-v1;
vs modet's headline strict-a onnet nuAP 0.104 vs 0.109 (-0.005 [-0.025, +0.018]); v1 itself is 0.110 (+0.001). 2 variants, each scored on confirm once (plus 2 permuted controls).
Borzoi variant-effect scoring was not run: the assignment layer is not the bottleneck (too few new loci), so changing the gene pick per lead cannot move strict-a.

## 2. What was built
- `scripts/subthr_extract.py`: for modet's 108 harmonised GWAS summary statistics dated <= 2021-04-23 (GWAS Catalog DATE; 88 <= 2019-04 for `past`),
  SNPs with p <= 1e-5 (mawk filter, CRLF-safe; all rsID columns kept) -> `data/subthr/snps_v2/<gcst>.parquet`; build-37 positions from modet's 1000G
  EUR .bim (same as modet's MAGMA leads), MHC removed; greedy +-500 kb clumping by p over all p <= 1e-5 SNPs -> `data/subthr/leads.parquet`.
  Result: 105 studies with leads, **7,277 sub-threshold leads** (5e-8 < p <= 1e-5, > 500 kb from any stronger lead of the study) and 1,035 GWS leads
  (GWS leads reproduce modet's `<gcst>_leads.parquet`: 80/80 for GCST001255). GCST004461 has a broken p column (1.42 M rows with p = 0): p = 0 dropped there.
  Runtime ~15 min (7 jobs) + clumping 1 min.
- `src/modeng/ext/subthr.py` channels (disease -> study mapping = gwas_d ontology + semantic matches, studies dated <= cutoff; NaN if none, 0 for unhit genes; per-gene max over leads x studies):
  `subthr_near` (w * -log10 p to the nearest-TSS gene <= 500 kb; TSS from NCBI37.3 gene.loc by strand, gnomAD v2.1.1 start as fallback),
  `subthr_cs2g` (w * -log10 p * cS2G score; Gazal 2022 1000G EUR SNP-to-gene, hg19, Zenodo 6354007, built from pre-2021 resources),
  `subthr_win` (w * -log10 p * exp(-d/50 kb), d = lead-TSS distance <= 500 kb), `*_perm` (gene axis permuted), `subthr_gws_*` (diagnostic: GWS leads).
- `scripts/subthr_alone.py` (alone on covered tasks, `results/subthr/alone.tsv`, `hits.tsv`), `scripts/subthr_stack.sh` (stack runs, power 0.35, v1 channels + new),
  `scripts/subthr_compare.py` (paired cluster bootstrap 5,000 vs `stack_v3ptr_p0.35`, `results/subthr/compare.tsv`), endpoints via
  `scripts/leader_endpoints.py stack_v1_subthr` (`results/review/leader_endpoints_stack_v1_subthr.tsv`).

## 3. Alone (temporal, covered tasks: dev 17 (13 with sa), confirm 53 (35 with sa))
| split | channel | AP | nuAP | sa_AP | sa_nuAP | non-seed hits | target hits | sa hits |
|---|---|---|---|---|---|---|---|---|
| dev | subthr_near | 0.018 | 0.015 | 0.017 | 0.036 | 1,549 | 51 | 20 |
| dev | subthr_cs2g | 0.012 | 0.013 | 0.008 | 0.019 | 468 | 26 | 10 |
| dev | subthr_win | 0.012 | 0.015 | 0.007 | 0.022 | 13,250 | 95 | 37 |
| dev | perm controls | 0.003 | 0.003 | 0.001 | 0.001 | ~1,500 / 480 / 13,280 | 1 / 0 / 38 | 0 / 0 / 11 |
| dev | gwas_catlag (ref) | 0.082 | 0.079 | 0.032 | 0.040 | 2,148 | 187 | 60 |
| dev | transfer (ref) | 0.076 | 0.066 | 0.033 | 0.040 | | | |
| confirm | subthr_near | 0.006 | 0.007 | 0.009 | 0.014 | 3,353 | 46 | 27 |
| confirm | subthr_cs2g | 0.009 | 0.007 | 0.014 | 0.012 | 758 | 32 | 19 |
| confirm | subthr_win | 0.006 | 0.005 | 0.006 | 0.009 | 27,678 | 97 | 50 |
| confirm | perm controls | 0.002 | 0.002 | 0.001 | 0.002 | | 6 / 2 / 46 | 2 / 1 / 23 |
| confirm | gwas_catlag (ref) | 0.037 | 0.035 | 0.023 | 0.024 | 2,947 | 137 | 57 |
Precision of a sub-threshold nearest-gene hit: 3.3 % (dev) / 1.4 % (confirm) vs ~0.2 % base rate; cS2G picks fewer genes at similar precision.
Overlap on strict-a targets (near or cs2g hit vs catlag hit): dev 7 subthr-only / 13 both / 47 catlag-only; confirm 24 / 9 / 48.

## 4. Increment over modeng-v1 (temporal, cv-trained, deltas vs `stack_v3ptr_p0.35`, 95 % cluster-bootstrap CI)
| variant | split | AP | nuAP | sa_AP | sa_nuAP | R@100 |
|---|---|---|---|---|---|---|
| v1 + near,cs2g,win | dev | -0.002 [-0.009,+0.006] | +0.000 | -0.002 | +0.002 [-0.003,+0.008] | +0.006 |
| v1 + 3 perm (control) | dev | -0.002 | +0.002 | -0.007 | -0.006 [-0.015,+0.004] | +0.004 |
| v1 + near | dev | -0.002 | +0.003 | -0.000 | +0.001 [-0.007,+0.010] | +0.011 [+0.003,+0.020] |
| v1 + near_perm (control) | dev | -0.004 | -0.001 | -0.003 | -0.006 | +0.005 |
| v1 + near,cs2g,win | confirm | -0.001 [-0.005,+0.003] | -0.002 [-0.008,+0.004] | -0.001 | **-0.006 [-0.017,+0.003]** | |
| v1 + 3 perm (control) | confirm | -0.003 | -0.004 | +0.002 | +0.001 [-0.008,+0.012] | |
| v1 + near | confirm | +0.001 | -0.001 | -0.005 [-0.011,+0.001] | **-0.007 [-0.018,+0.002]** | |
| v1 + near_perm (control) | confirm | -0.000 | +0.001 | -0.003 | +0.002 [-0.007,+0.009] | |
Both variants passed the pre-registered dev rule (sa_nuAP > 0, AP/nuAP >= -0.003, above control), but by margins (+0.001 to +0.002) smaller
than the control-to-control spread (+-0.006): the dev rule is too weak at 28 sa tasks. Confirm: no CI excludes 0, point estimates negative, controls
at ~0 -> **"no strict-a gain"** (pre-registered outcome). vs modet headline (confirm, `leader_endpoints_stack_v1_subthr.tsv`): strict-a onnet nuAP
0.104 vs 0.109 (-0.005 [-0.025, +0.018]), onnet AP +0.023 [+0.004, +0.037]; all-gene sa_nuAP 0.085 vs 0.109 (-0.024 [-0.047, -0.002]); overall AP/nuAP unchanged.
Feature gain: no subthr channel in the top 25 of the 3-channel model; r_subthr_near ranks ~20 in the near-only model.
Variants counted: 2 stack variants (+2 controls), each confirm scored once; 8 channels looked at alone on dev and confirm.

## 5. Leakage
Temporal: clean by date (summary statistics published <= 2021-04-23, 1000G positions, NCBI37.3 gene positions, cS2G built from <= 2021 resources);
disease -> study mapping uses gwas_d's ontology releases (mild transductive route, as everywhere). `past`: only the 88 studies <= 2019-04
(NaN otherwise). forward/holdout: same 108 studies (no post-T0 summary statistics extracted).

## 6. Why it fails and what next
- Coverage ceiling: only 17/53 dev and 53/138 confirm diseases map to one of the 108 studies, and 92 of them are the 2018 SAIGE UKB PheCode GWAS
  (modest case counts); strict-a arrivals mostly come from new, larger GWAS on other traits, not from these studies' sub-threshold loci.
- What sub-threshold loci find is mostly already in `gwas_catlag` (catalog keeps p <= 1e-5 rows; 9-13 shared sa hits) or in L2G priors.
- Better routes for strict-a: summary statistics of many more <= T0 studies (GWAS Catalog full sumstats ~2021, FinnGen R5, Pan-UKBB 2020) to lift
  coverage; or catalog sub-threshold rows (1e-6 < p <= 1e-5) with cS2G on all diseases (gwas_d idea 4), which is cheap.
