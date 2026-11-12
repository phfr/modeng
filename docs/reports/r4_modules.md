# r4: modeng-v1 disease modules and their submodules (temporal confirm, 2026-11-12)

**Question.** Turn modeng-v1's ranked lists (`stack_v3ptr_p0.35`) into disease modules (with a calibrated cutoff), split them into
functional submodules with subex's confirmed decomposition (`leiden_anno`, subex S-62/S-63), and check them against the future targets.
**Benchmark:** temporal confirm, 138 diseases. Seeds are OT 21.04 genes. Targets are the OT 25.09 additions, used only for evaluation.
**Status:** descriptive, not pre-registered. Confirm has already been viewed many times (FINAL report §1, caveat 5). Read the paired numbers
as estimates, not as a clean confirmation.

## Pipeline
1. `scripts/export_modules.py` writes `data/modules/` (gitignored, 143 MB). Per disease it writes:
   - a subex score artefact: TSV `gene_id, score, calib_p, is_seed` plus a sidecar `scores.json` with `task_id` and `namespace`. `subex module --component calib_p` reads it.
   - module lists at tau 0.2 and 0.3: `module_tau{T}.tsv`.
   - a contract-v1 `kind = "module"` npz on STRING-T0 (`t0_string115_phys_notm_400`).
   The calibration is an isotonic map from score to P, fitted on the dev split's out-of-fold scores of modeng-v1 and multiplied by the dev-only shrink 0.8.
   The rule is module(d, tau) = seeds + candidates with P >= tau.
   Comparators keep the same sizes: the same number of added genes per disease and tau, taken as the top-N of `transfer_jaccard` or of modet's headline (modet-U-cov-Lb).
2. `scripts/decompose_modules.py` decomposes the modules into submodules. It runs the same code path as `subex decompose --method leiden_anno --overlap none --graph t0_string115_phys_notm_400`,
   but loads the graph and the library once per worker: the CLI spends about 45 s per call on loading.
   - Check: on one module the real CLI gave identical labels.
   - Scope: 6 CPU workers. 552 modules were decomposed (modeng at both taus, both comparators at tau 0.2), with 0 failures.
3. `scripts/eval_modules.py` runs the evaluation and writes `results/modules/r4_*.tsv`. `scripts/module_examples.py` writes `results/modules/r4_examples.md`.

**Why not `subex module` for the cutoff:** subex's BH cutoff on the degree-stratified z of our score is not meaningful for an already calibrated probability.
On metabolic syndrome it selects 898 genes. We keep our calibrated tau and pass subex the gene set.

**Leakage:**
- Seeds, the STRING 11.5 graph and the T0 GO / Reactome library used by `leiden_anno` are dated <= 2021-04.
- The DepMap 20Q4 coherence view is dated 2020-11.
- The only post-T0 content is what modeng-v1 already uses: the PubTator / GWAS-Catalog mapping vintages and modern views, as listed in the FINAL report §1 caveat 5.
- No new input was added.

## 1. Module level (`r4_module_summary.tsv`, `r4_module_paired.tsv`)
| method | tau | diseases with additions | median added | pooled precision | mean precision | pooled recall of future targets | added genes off STRING-T0 |
|---|---|---|---|---|---|---|---|
| **modeng-v1** | 0.2 | 127 / 138 | 6 | **0.289** (predicted 0.310) | 0.371 | **0.081** | 21.6 % (74 of the 359 hits) |
| transfer_jaccard, same size | 0.2 | 127 | 6 | 0.209 | 0.249 | 0.058 | 14.3 % |
| modet headline, same size | 0.2 | 127 | 6 | 0.178 | 0.206 | 0.050 | 0 % (on-network ranker) |
| **modeng-v1** | 0.3 | 116 | 3 | **0.336** (predicted 0.353) | 0.412 | 0.057 | 20.9 % |
| transfer_jaccard | 0.3 | 116 | 3 | 0.256 | 0.293 | 0.044 | 13.6 % |
| modet headline | 0.3 | 116 | 3 | 0.213 | 0.251 | 0.037 | 0 % |

Paired hits, modeng-v1 minus the comparator, summed over diseases. The intervals are 95 % cluster-bootstrap intervals:
- tau 0.2, vs transfer: **+100 [+57, +155]** (68 wins / 22 losses)
- tau 0.2, vs headline: **+138 [+87, +200]** (76 / 14)
- tau 0.3, vs transfer: +61 [+25, +105]
- tau 0.3, vs headline: +93 [+50, +147]

**Calibration:** with the dev-only shrink 0.8, realised precision is close to predicted (0.29 vs 0.31, 0.34 vs 0.35). The ~1.4x optimism of the unshrunk map (FINAL §4b) is mostly removed.

## 2. Submodules (`r4_submodule_summary.tsv`, `r4_submodule_within.tsv`)
- **Shape:** the modules are dominated by seeds (median 27 seeds vs 6 added genes). `leiden_anno` gives a median of 2 submodules.
  Almost every submodule contains seeds: only 2 of 357 modeng submodules have none, and they hold 34 added genes with 2 hits.
  So the question "are new genes in seed-free submodules?" barely arises at these cutoffs. Seed-free submodules appear only in large-addition diseases (psoriasis example below).
- **Precision of added genes by the seed fraction of their submodule (modeng, tau 0.2):**

  | seed fraction | 0 | (0, 0.5] | (0.5, 0.8] | (0.8, 1) |
  |---|---|---|---|---|
  | precision | 0.06 | 0.26 | 0.29 | 0.36 |
  | n | 34 | 242 | 441 | 257 |

  New genes that land in seed-dense submodules come true more often. The submodule with the most seeds holds 53 % of a disease's added genes and 58 % of its hits.
- **Within-disease permutation:** hit labels were permuted among each disease's added genes, which removes between-disease differences.
  - tau 0.2: the concentration is weak (z 1.1, p 0.15).
  - tau 0.3: it is clear (z 3.0, p 0.001).
  - transfer at tau 0.2: none (z 0.0).
  - headline at tau 0.2: weak (z 1.4).
  This is post hoc, so treat it as a lead: seed density of the submodule could be a re-ranking feature, pre-registered on dev first.
- **Off-graph genes:** 21.6 % of modeng's added genes, and 74 of its 359 hits, are not on STRING-T0, so no submodule contains them.
  That is a large blind spot of PPI-based decomposition for a ranker that uses non-network channels.
  Fix options: decompose on the union network (`t0_string115+t0_biogrid4197+huri`), or attach off-graph genes by annotation similarity. leiden_anno already completes unannotated genes by proximity, but only on the graph.

## 3. Held-out functional coherence (DepMap 20Q4 co-essentiality, `r4_coherence_summary.tsv`)
subex has no direct CLI for submodule coherence on an arbitrary module, so I used its `Views` loader. The statistic is the mean correlation of
same-submodule pairs minus different-submodule pairs, z against 200 label permutations, on diseases with at least 6 covered genes and at least 2 submodules.

| method | tau | n | mean diff | median z | frac p <= 0.05 |
|---|---|---|---|---|---|
| modeng-v1 | 0.2 | 117 | 0.010 | 1.02 | 0.33 |
| transfer_jaccard | 0.2 | 123 | 0.008 | 0.86 | 0.29 |
| modet headline | 0.2 | 121 | 0.008 | 1.04 | 0.31 |

Submodules are modestly coherent on a view not used to build them, and about equally so for all three module sources.
This is expected: the seeds make up most of every module, so the decomposition mostly reflects the seeds.

## 4. Three examples (`results/modules/r4_examples.md`, full gene lists with P and channel flags)
- **How the examples were chosen:** the rule was fixed before reading the descriptions: at least 2 submodules, at most 150 genes, at least 3 hits, one disease per cluster, most added genes on the graph.
- **What "channels" means here:** the modeng-v1 channels in which a gene ranks in the disease's top 1 %. This shows where the gene stands out; it is not a SHAP attribution of the stacker.
- **Submodule names:** they are the top T0 GO / Reactome term, which is circular for `leiden_anno`.

- **Metabolic syndrome (EFO_0000195):** 23 seeds and 46 added genes, 37 of which became targets (precision 0.80; 109 new targets in total). Three submodules:
  - (a) *plasma lipoprotein assembly / clearance*: seeds APOB, CETP, LPL, ABCA1. New and confirmed: APOC1, SCARB1, LDLR, LPA, LIPG, ABCG8, ANGPTL4, TRIB1, PCSK9.
    Mostly flagged by pathway donors (MSigDB Reactome / WikiPathways), PubTator ridge and HPO donors.
  - (b) *regulation of insulin secretion*: seeds MTNR1B and others. New and confirmed: HNF4A, TCF7L2, IRS1, SLC30A8, PROX1, BDNF.
    Mostly flagged by the GWAS-side label-process channels: gwas_catlag, L2G priors (pri_l2g_*, hz_gwas).
  - (c) an unannotated-named adiposity / GWAS-locus group around FTO and GALNT2: COBLL1, FADS2, IGF2BP2, CDKAL1 (all confirmed).
  - CMIP and SLC39A8 (both confirmed, P 0.80) are off-graph.
- **Neuromuscular protein-defect diseases (MONDO_0016139):** 53 seeds and 35 added genes, 8 confirmed. Four submodules:
  - *myofibril*: MYPN, DES, LDB3 confirmed; also MYOT, CRYAB, DNAJB6.
  - *dystrophin-glycoprotein complex*: DAG1 confirmed; also FHL1, ITGA7, LTBP4.
  - *cation transport*: SCN4A, STIM1, ORAI1, TRPV4, none confirmed yet.
  - *N-glycan biosynthesis*: LARGE1 confirmed; also GNE, TRAPPC11, GFPT1.
  Driven by the HPO donor (don_hpo2101), disease-context ridge (dc_ridge) and PubTator ridge, i.e. phenotype similarity rather than GWAS.
  Many unconfirmed additions are well-known myopathy genes. They are probably OT-label gaps, not errors.
- **Psoriasis vulgaris (EFO_1001494):** 15 seeds and 63 added genes, only 6 confirmed. It is the over-calling case and has the one large seed-free submodule.
  - (a) an unannotated-named NF-kB / GWAS-locus group: seeds TNIP1, NFKBIA, REL. New: CARD14 and ETS1 confirmed; also KSR1, CYLD.
  - (b) *regulation of immune effector process*: seeds IL12B, TRAF3IP2, TNFAIP3. New: IL2RA and IRF1 confirmed; also IFNG, IL18R1, IL17F.
  - (c) seed-free *lymphocyte activation*: PTPN2 and ELMO1 confirmed; also IKZF1, JAK2, RORC, NFKB1, NR5A2. Many of these are immune-shared loci pulled in by transfer from related immune diseases.
  - 14 GWAS-locus genes are off-graph (RNF186, ZNF831, GCKR, ...).
  - Channels: gwas_catlag, transfer_jaccard and dc_ridge on almost every gene, i.e. cross-disease immune transfer plus GWAS-Catalog arrivals.

## Takeaways
1. At equal size, modeng-v1 modules hold about 40 % more future targets than transfer and about 60 % more than modet's headline, with CIs above 0. The shrunk calibration is roughly honest.
2. leiden_anno submodules are seed-dominated at these cutoffs and only modestly coherent on held-out co-essentiality, about the same for every module source.
3. New genes in seed-dense submodules are more often right (0.36 vs 0.26). Within diseases this holds clearly only at tau 0.3; it is post hoc.
4. 22 % of the called genes cannot be placed in submodules because they are off STRING-T0.

**Next:**
- Decompose on the union network, or attach off-graph genes.
- Pre-register "seed fraction of the submodule" as a dev re-ranking feature.
- Run subex's submodule suite (`--suite submodules`) if subex accepts these modules as a model (needs a subex-side store; not done here because subex must not be edited).
