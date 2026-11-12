# r6 (R4 plan P2): FinnGen R4 genetics channel + gnomAD constraint

Date: 2026-11-12. Data: FinnGen R4 public results (DF4, public 2020-11-30, 176,899 people, 2,444 endpoints), anonymous download from
`gs://finngen-public-data-r4` (user decision 2026-11-12: "use public bucket now"; the user handles the access form / terms; earlier stop: commit c2c3956).
Acknowledgement: "We want to acknowledge the participants and investigators of the FinnGen study." Cite Kurki et al., Nature 613, 508-518 (2023).
Never used: `coding_variants/` (2021-11, post-T0), any R5+ freeze.

## 0. Pre-registration (written and committed before any FinnGen channel was scored)
Witness definition (R4 strategy review B, section 1a): a target is *traced* by a witness if it is in the top 200 of that witness's per-task
ranking over the task's candidates (universe minus `task.exclude`; average ties; NaN last). Genetic witnesses: `gwas_catlag` (existing) and the
new FinnGen channels `fg_hit` (credible-set lead -> nearest-TSS gene) and `fg_hit_cs2g` (credible-set variants -> cS2G gene).
1. **Kill rule (decides the programme):** on temporal **dev** (53 diseases), share of strict-a targets (`task.alt['sa']`) traced by any genetic
   witness {gwas_catlag, fg_hit, fg_hit_cs2g}. Reference: gwas_catlag alone ~13 % (R4 B). **< 20 % -> stop**: no summary-statistics download,
   record "strict-a is not reachable from FinnGen R4 genome-wide hits". 20-25 % -> report as weak, sub-threshold stage optional. >= 25 % -> pass
   the coverage gate; `fg_sub` (p <= 1e-5 leads from streamed summary statistics of mapped endpoints) is then built.
   wide_dev (45 diseases) reported with the same rule as a secondary, non-deciding number. wide_confirm is never scored; temporal confirm is not
   scored in this round either (dev-only stack runs).
2. **Increment (only if 1 passes; dev OOF, cv, deg^-0.35, scores saved):** `v1 + fg channels` vs `stack_v3ptr_p0.35` and vs the same-width
   gene-permuted control `v1 + fg_*_perm`; keep only if sa_nuAP delta >= +0.010 (paired cluster bootstrap), the permuted control's delta within
   +-0.003, and overall AP and nuAP deltas >= -0.003. Same comparison on the v1clean channel list.
3. **Label-process share:** among strict-a targets newly traced by FinnGen (traced by fg_* but not gwas_catlag), the share whose 25.09 OT
   evidence for the disease (or a descendant) includes a FinnGen study (`gwas_credible_sets`, studyId FINNGEN_*). Reported, not a decision.
4. **Constraint `gnomad_loeuf`** (independent of the kill rule): alone (AP, nuAP, sa_AP, sa_nuAP) on temporal dev and its increment over v1 with
   a gene-permuted control on dev; drop if its deltas are within the control's.
5. Endpoint -> disease mapping precision: 30 mapped (endpoint, disease) pairs sampled at random (seed 0), judged by hand from names
   (correct / too broad / too narrow / wrong); reported as is.

## 1. Verdict (2026-11-12, after the pre-registration above)
**Kill rule triggered: stop.** On temporal dev, genetic witnesses now trace **16.3 % of strict-a targets (79 / 485), below 20 %**.
gwas_catlag alone traces 12.8 %, which matches R4 B's 13 %. FinnGen R4 genome-wide hits add 17 targets, +3.5 points.
By the pre-registered rule, no summary statistics were downloaded and `fg_sub` was not built.
Recorded: **"strict-a is not reachable from FinnGen R4 genome-wide hits"** (dev, 28 strict-a diseases).
The few targets FinnGen does trace are mostly label process: **15 of the 17** FinnGen-only traced targets reached OT 25.09 through a FinnGen R12 study.
wide_dev (secondary): 10 strict-a tasks, 106 targets, **0 traced** by any genetic witness, gwas_catlag included.
**Constraint `gnomad_loeuf`: drop.** Every increment is within or below its permuted control.

## 2. What was built
- Download: `data/raw/finngen_r4/` holds the manifest, pheno_n and 1,115 SuSiE `.cred.bgz` files.
  It also holds `susie_cs/`: SuSiE `.snp.bgz` files streamed (4.0 GB), keeping only credible-set rows (3.7 MB). The full files were not kept.
  Checksums are in `susie_cred.sha256` and `susie_snp_stream.sha256`; every download is logged in `downloads.tsv`.
- `scripts/finngen_build.py` writes `data/finngen/cs.parquet`: 67,398 credible-set variants in 4,400 sets from 1,035 endpoints.
  It lifts hg38 to hg19 with its own code over the UCSC chain (99.7 % land on the same chromosome; checked on rs7903146, hg19 10:114758349).
  87 % of variants get a 1000G rsID; 94 % of leads have p <= 5e-8.
- `src/modeng/ext/finngen.py`:
  - Endpoint map `data/finngen/endpoint_map.tsv`: 1,531 of 2,264 endpoints, 1,851 pairs. Routes: R12 code -> OT 25.09 study diseaseIds
    (post-T0 vintage) 1,625; normalised name or exact synonym from OT 21.04 575; both 349. 605 of the 1,035 endpoints with credible sets are mapped.
  - Channels `fg_hit` (lead -> nearest TSS within 500 kb, -log10 p) and `fg_hit_cs2g` (PIP x cS2G x -log10 p), plus `*_perm`.
    They return NaN outside temporal and temporal_wide.
- `src/modeng/ext/constraint.py`: `gnomad_loeuf` (= -oe_lof_upper, gnomAD v2.1.1) and `gnomad_loeuf_perm`.
- Scripts: `scripts/finngen_trace.py` (kill rule), `scripts/finngen_labelproc.py`, `scripts/r6_alone.py`,
  `scripts/r6_stack_dev.py` (dev-only OOF; no confirm scoring, no leaderboard row).
  Outputs are in `results/finngen/` and `results/temporal/dev/stack_r6_*`.

## 3. Mapping precision (30 random pairs, seed 0, judged by hand from names)
| judgement | count |
|---|---|
| correct | 17 |
| too broad (e.g. gestational oedema -> pregnancy disorder; benign colon -> colonic neoplasm) | 10 |
| too narrow (gastroduodenal ulcer -> duodenal ulcer; NSCLC -> NSCLC adenocarcinoma) | 2 |
| wrong / not a disease (invasive ventilation) | 1 |

Precision is 57 % strict and 90 % if "too broad" counts as usable. Too-broad pairs come mostly from the OT R12 study mapping.

## 4. Kill-rule numbers (temporal dev; 28 strict-a tasks, 485 targets; top 200 of the candidate ranking)
| witness | strict-a traced | share |
|---|---|---|
| gwas_catlag | 62 | 12.8 % |
| fg_hit | 16 | 3.3 % |
| fg_hit_cs2g | 16 | 3.3 % |
| fg_hit or fg_hit_cs2g | 18 | 3.7 % |
| **any genetic witness** | **79** | **16.3 % (< 20 %: stop)** |
| FinnGen only (not gwas_catlag) | 17 | 3.5 % |

FinnGen covers 21 of 28 strict-a dev tasks (25 of 53 dev tasks), but it traces few of their targets.
Over all targets of those tasks it traces 31 (fg_hit) and 34 (fg_hit_cs2g) of 1,403; gwas_catlag traces 165.

**Label-process share:** all 17 FinnGen-only traced targets have OT 25.09 credible-set evidence in the disease family, and **15 / 17 (88 %)**
include a FinnGen R12 study (`results/finngen/labelproc_temporal_dev.tsv`).
One example is 6 / 6 chronic venous insufficiency genes from I9_VEINSOTH / I9_DISVEINLYMPH. The FinnGen channel therefore mainly forecasts OT's later ingestion of FinnGen.

## 5. Alone on temporal dev (NaN ranked last)
Columns are AP, nuAP, sa_AP and sa_nuAP. "all 53" means all 53 dev tasks; "covered 25" means the 25 tasks that FinnGen covers.
| channel | AP, all 53 | AP, covered 25 | sa_AP, covered 25 | sa_nuAP, covered 25 |
|---|---|---|---|---|
| fg_hit | 0.013 | 0.025 | 0.039 | 0.037 |
| fg_hit_perm | 0.003 | 0.003 | 0.001 | 0.002 |
| fg_hit_cs2g | 0.012 | 0.024 | 0.034 | 0.031 |
| gwas_catlag | 0.031 | 0.055 | 0.021 | 0.026 |
| transfer_jaccard | 0.123 | 0.076 | 0.026 | 0.026 |
| gnomad_loeuf | 0.005 | 0.005 | 0.004 | 0.003 |
| gnomad_loeuf_perm | 0.003 | 0.004 | 0.004 | 0.004 |

The FinnGen signal is real where FinnGen has coverage: its strict-a AP is above gwas_catlag and transfer on the covered tasks.
But it reaches too few targets to move the coverage gate. LOEUF alone is near its permuted control on strict-a (AUROC 0.64 vs 0.49).

## 6. Stack increments (temporal dev OOF, v1 recipe, deg^-0.35; paired cluster bootstrap 5,000 vs `stack_v3ptr_p0.35`)
The v1 reproduction with `scripts/r6_stack_dev.py` is exact: AP 0.1756, nuAP 0.1502.
| variant | AP | nuAP | sa_AP | sa_nuAP |
|---|---|---|---|---|
| v1 + gnomad_loeuf | -0.004 [-0.008,+0.001] | -0.002 | +0.001 | -0.008 [-0.018,+0.003] |
| v1 + gnomad_loeuf_perm (control) | -0.005 | +0.001 | +0.005 | +0.006 [-0.006,+0.023] |
| v1 + fg_hit, fg_hit_cs2g (exploratory, not deciding) | +0.001 | +0.002 | +0.021 [+0.009,+0.035] | +0.021 [+0.007,+0.038] |
| v1 + fg perm (control) | -0.001 | +0.002 | +0.012 [+0.002,+0.023] | +0.018 [+0.002,+0.037] |
| fg vs its permuted control | +0.002 [-0.002,+0.007] | +0.000 | +0.008 [-0.001,+0.022] | +0.003 [-0.005,+0.013] |

- The fg increment was not deciding because the kill rule had already fired. It would also have failed rule 2: the permuted control gains +0.018 sa_nuAP,
  which is outside +-0.003. At 28 strict-a tasks, adding any two columns moves sa_nuAP by about 0.02.
- v1clean was not run, because the gate failed. Temporal confirm, wide_confirm and wide_dev were not stack-scored.

## 7. Leakage and caveats
- Temporal-clean by date: R4 objects are dated 2020-11-26 / 30; the chain file, the 1000G panel and cS2G are pre-2021.
- Mechanism: label process (88 % above).
- The `ot2509_code` mapping route is post-T0 vintage; `name21` is T0 vintage.
- gnomAD v2.1.1 is clean on temporal.
- wide_dev has only 10 strict-a tasks, so its zero is weak evidence; it agrees with dev.

## 8. What next
- Do not download FinnGen summary statistics for `fg_sub`.
  Sub-threshold loci from 108 GWAS-Catalog studies already failed in r4, and the genome-wide FinnGen tier traces only 3.5 %.
- The remaining strict-a arrivals mostly come from post-T0 large meta-analyses (MVP 2024 and others), which no T0 input can witness.
- If FinnGen is reused, use it only in the label-process arm (forecaster), next to gwas_catlag.

## 9. Runtime and resources
About 1.5 h in total. Downloads 4.3 GB streamed, 16 MB kept. Every run used 8 threads or fewer; no GPU.
