# Learned combination of dated evidence witnesses for disease gene prioritisation: a temporally controlled study

*modeng write-up, 2026-11-12. Sources: every number below is taken from the project files named next to it (mostly
`docs/reports/`, `docs/reviews/`, `docs/insights.md`, `results/leaderboard.tsv`). Nothing new was computed for this text.*

**How to read the numbers.** Each result carries one of two labels.
- **[DEV]**: development number. It comes from a split that was looked at many times during model building, so selection
  effects are not corrected. Temporal confirm (138 diseases) belongs here: by the end of the project, 116 distinct methods,
  97 of them stacker variants, had a leaderboard row on it (`results/leaderboard.tsv`).
- **[CONF]**: confirmatory number. The analysis was pre-registered, committed before scoring, and the split was scored once.
  Only one experiment in this project is [CONF]: the scoring of the frozen `wide_confirm` fold (section 5.5).

Brackets give paired cluster-bootstrap 95 % intervals over diseases (clusters = ontology-connected groups;
`src/modeng/compare.py`). "AP +0.03 [+0.01, +0.05]" means a paired difference in mean per-disease AP.

---

## Abstract

Given the known genes of a disease (seeds), disease gene prioritisation ranks the other ~19,000 protein-coding genes by how
likely each is to become a disease gene. The predecessor project, modet, found that cross-disease transfer dominates this task:
genes of similar diseases are the strongest predictor, and the protein network adds little beyond node degree. modeng asked
whether many weak, dated, mostly non-network evidence sources ("witnesses"), joined by a learned combiner, can beat that
transfer baseline under strict temporal control. Seeds come from Open Targets (OT) 21.04 and targets are the genes first
labelled by 25.09.

During development, a LightGBM stacker over 65 witness channels (`modeng-v1`) beat raw Jaccard transfer on 138 temporal
diseases [DEV]: AP 0.153 vs 0.119 (+0.034 [+0.011, +0.059]) and node-uniform AP (nuAP) 0.177 vs 0.104 (+0.073 [+0.053, +0.099]).
On modet's candidate set it also beat modet's headline model: AP +0.063 [+0.049, +0.082], nuAP +0.031 [+0.012, +0.054]. It
tied modet on modet's primary endpoint, new genetic discoveries ("strict-a"; nuAP +0.001 [-0.016, +0.025]). About half of
the AP margin over transfer came from training on arrivals of the same time window. Outside that window, the gain depended
on channels that forecast the OT labelling process rather than biology.

One pre-registered, single-look test was run on 35 diseases that had never been used [CONF]. The model trained on 712
off-panel diseases, without label-process channels, beat transfer on both primary metrics: AP 0.118 vs 0.091
(+0.027 [+0.004, +0.049]) and nuAP 0.126 vs 0.084 (+0.042 [+0.006, +0.086]). It was one of four pre-registered arms and
the only one to pass; the pass does not survive a Bonferroni correction for four arms.

Two analyses explain why the gain is modest. Only 46 % of future targets show any strong pre-seed trace in any witness, and
95 % of strict-a targets have no genetic witness at all. The binding limit is evidence that did not exist at the seed date,
not the learner.

---

## 1. Background

### 1.1 What modet found
modet (`/home/ph/code/modet`) built a temporal benchmark for disease gene prioritisation. It takes OT genetic gene sets at
release 21.04 as seeds, uses genes newly labelled by 25.09 as targets, and runs on 191 non-sealed panel diseases. On this
benchmark, one feature, maximal transfer from related diseases (`tr_max`), reached about four times the node-uniform AUPRC
of the best pure network method (0.103 vs 0.027; modet `docs/how_modet_works/05_features.md`). In nuAP, nearly everything
the network added was log-degree: on STRING, BioGRID and HuRI, 89-112 % of the network's contribution was degree
(modet `paper/FINDINGS.md`, F-92). modet's headline model, `modet-U-cov-Lb`, ranks network genes only. modet's README sums
this up as "transfer between related diseases dominates, and the protein network adds little".

### 1.2 The question
If transfer dominates and the network contributes mostly degree, gains must come from somewhere else. modeng asked whether
the following, combined, produce a ranking that beats transfer and modet's headline under the same temporal rules:
- many independent evidence sources dated no later than the seed release: phenotypes, literature, pathway gene sets,
  GWAS Catalog entries, gene embeddings;
- learned joint regressions over disease rows;
- one learned combiner.

A second question ran alongside the first. When a model does gain, is it learning biology, or learning how the OT label
pipeline adds genes?

### 1.3 Setting
The project was an autonomous sprint by LLM agents. It started 2026-11-11 23:56; the fresh-fold confirmation was scored on
2026-11-12 at 15:16-15:30. Section 3 describes the process. All inputs, outputs and decisions are logged in the
repository (`CLAUDE.md`, `docs/journey.md`).

---

## 2. Benchmarks and leakage control

### 2.1 Benchmarks
All benchmarks are built by `src/modeng/tasks.py` (definitions in `docs/reports/benchmarks.md`).
- Universe: 19,255 HGNC protein-coding genes.
- Labels: OT genetic gene sets (GWAS, ClinVar, curated Mendelian, burden), taken from modet's ground truth.
- Leakage class: every benchmark states one.
- Sealed fold: modet's 97 sealed-fold diseases and their ontology relatives (6,182 ids) are never tasks.

| benchmark | seeds -> targets | splits (diseases) | leakage class | role |
|---|---|---|---|---|
| **temporal** | OT 21.04 -> new in 25.09 (modet panel, STRING-T0) | dev 53 / confirm 138 | temporal-clean | main development benchmark |
| **forward** | OT 25.09 -> new in 26.09; inputs <= 25.09 | dev 842 / confirm 797 | temporal-clean | replication in a later window |
| **holdout** | OT 26.09, 20 % of genes hidden | dev 303 / confirm 297 | holdout-current (transductive) | leak tests, sanity |
| **past** | OT 19.04 -> new in 21.04 | 1,786 episodes | temporal-clean | prospective training set |
| **temporal_wide** | as temporal, 792 off-panel diseases | train 712 / dev 45 / **confirm 35 (frozen)** | temporal-clean | training at scale and the one fresh confirmation |

**Temporal** reuses modet's panel rule exactly:
- Seeds: OT 21.04 strong genes, re-propagated through the 25.09 ontology.
- Targets: genes new by 25.09.
- Excluded from ranking: genes with weak 21.04 evidence ("unknown").
- Harness parity: on modet's candidate set the harness reproduces modet's headline exactly (nuAP 0.1775, maximum
  per-disease difference 4e-16; `docs/reports/baselines.md`).

**temporal_wide** (`docs/reports/r6_wide.md`, `docs/preregistration_wide.md`) applies the same rule to every other
eligible disease, giving 792. Splitting them was hard because the ontology entangles these diseases with the 191 panel
tasks: only 286 of the 792 are not direct ontology relatives of a panel task. Before any score existed on any split, the
split was fixed by hub peeling plus a hash over ontology components and frozen by tag `wide-confirm-frozen` (commit
`e6f00fa`) together with a sha256 of the 35 confirm ids. Code guards make scoring `wide_confirm` impossible without an
environment flag.

### 2.2 Metrics
Each metric is a per-disease value averaged over diseases (`src/modeng/metrics.py`).
- **AP**: average precision over the candidate set (universe minus seeds and unknowns).
- **nuAP**: AP with each positive weighted 1/max(degree, 1) in the benchmark network. This is modet's node-uniform AUPRC,
  which rewards finding low-degree genes.
- **Degree-matched AP**: AP within the target's degree bin (off-network + 10 deciles), averaged over positives.
  Re-weighting by degree cannot move it.
- **onnet**: AP and nuAP computed on network genes only, which is modet's candidate set. All comparisons with modet's
  headline use onnet.
- **strict-a** (`sa`): modet's primary endpoint. Targets are restricted to arrival class A (new genetic evidence first seen
  2022-2025 and still present in 26.09), on diseases with >= 5 such targets (71 of 138 temporal confirm diseases).

AP and nuAP can disagree. A model that down-weights hubs raises nuAP and lowers AP. During the project an evolved transfer
kernel won on nuAP alone by halving AP (section 6). After that, the project used a joint rule: **a claim needs a gain on AP
and on nuAP**, with degree-matched AP reported as well (`docs/insights.md`, 05:20 entry).

### 2.3 Masking
- Temporal and forward: the context is a past snapshot. Each task masks only its own disease row and its descendants
  (`Benchmark.context_for`).
- Holdout: also masks near-duplicate rows. Ancestors are re-propagated bottom-up so that no "missing from the ancestor" tell
  remains.
- Sensitivity check: with modet-style family masking, temporal transfer AP falls from 0.120 to 0.084
  (`docs/reports/benchmarks.md`). The narrower mask is deliberate, because related diseases carry legitimate past knowledge.
- Wide training: training on `temporal_wide` uses a **relative-target guard**, which removes the targets of any evaluation
  disease from the labels of its training-set ontology relatives. Without the guard, temporal dev AP rose by
  +0.011 [+0.003, +0.019] while wide_dev did not move, which identifies the rise as leakage (`docs/reports/r6_wide.md`).

### 2.4 Baselines (temporal confirm, 138 diseases; `docs/reports/baselines.md`) [DEV]
| method | AP | nuAP | AUROC | onnet AP | onnet nuAP |
|---|---|---|---|---|---|
| random | 0.002 | 0.003 | 0.51 | 0.003 | 0.003 |
| STRING degree | 0.004 | 0.001 | 0.61 | 0.004 | 0.001 |
| popularity (number of context diseases) | 0.019 | 0.012 | 0.79 | 0.022 | 0.014 |
| random walk with restart on STRING | 0.014 | 0.006 | 0.65 | 0.017 | 0.013 |
| transfer_onto (ontology-weighted) | 0.059 | 0.055 | 0.62 | 0.063 | 0.068 |
| **transfer_jaccard** (top-50 diseases by seed Jaccard) | **0.119** | **0.104** | 0.70 | 0.128 | 0.128 |
| modet headline `modet-U-cov-Lb` | (0.081) | (0.080) | 0.76 | 0.096 | **0.173** |

`transfer_jaccard` is the strongest simple baseline on every benchmark (AP: temporal 0.12, forward 0.085, holdout 0.22).
The headline's all-gene values are in parentheses because it ranks off-network genes last. About 18 % of temporal targets
are off-network, so only onnet columns are compared with the headline.

### 2.5 Leaks and bugs found and fixed
Each of these was found before the corresponding result was recorded as solid. Sources: `docs/invalidations.md`,
`docs/reviews/`, `docs/reports/r6_*`.
1. **Float-residue tie leak.** Subtracting masked pairs left residue of about 1e-7 on hidden targets, which broke ties in
   their favour. Holdout transfer_onto nuAP fell from 0.117 to 0.036 once scores were quantised. Found by an LLM critic.
2. **Gene-order bug.** Views were built in HGNC row order, but the harness uses ENSG order. The tell was every family scoring
   AUROC < 0.5.
3. **Holdout ancestor tell.** Blanket ancestor masking made transfer_onto anti-informative (AUROC 0.29). Fixed by
   bottom-up re-propagation, and a test now guards it.
4. **CV leak inside `past`.** Grouping by modet cluster leaked through propagated ancestors: held-out AP 0.31, against 0.04
   for transfer. Fixed with Louvain leak folds.
5. **Evaluation-universe artefact.** On all genes, transfer seemed to beat modet's headline. The cause was off-network
   targets, not a real reversal.
6. **Relative-target leak in wide training** (section 2.3). Fixed by the guard.
7. **Vintage leaks.**
   - An HPA expression file dated 2025-12 was in "modern" views. These were replaced by HPA v20.1 / GTEx v8 in a clean arm.
   - The GWAS-Catalog channel would have read the full 2026 catalog on `temporal_wide` without an explicit cutoff mapping.
     A bit-identity port check caught this (`docs/reports/r6_wide_full.md`).
   - Two vintages remain post-T0 and could not be removed: the PubTator3 and GWAS-Catalog mappings (2026 releases, applied
     to pre-T0 PMIDs and associations).

---

## 3. Process

### 3.1 Agents
The ideators, critics, implementers and reviewers were LLM agents: Claude models, called "Fable" in the review and
ideation files. A lead agent of the same kind coordinated them. The user set the goal ("be ambitious", modern tools,
multi-round ideation with reviews). The user also made two decisions recorded in the files: using the public FinnGen R4 bucket and
approving that the `wide_confirm` fold be spent.
The agents can be wrong in the same confident register whether they are right or not. The safeguards below exist for that
reason, and section 2.5 and the R2 review show they were needed.

### 3.2 Ideation protocol
Before the first idea, the protocol was based on a literature review (`docs/research/00_ideation_process.md`):
- **Independent generation before any interaction.** Interaction structure collapses diversity in multi-agent ideation.
- **Distinct lenses and "tail" ideas.**
- **A retrieval-grounded novelty check.**
- **A critic panel with three lenses**: statistics/leakage, biology/curation and engineering.
- **A closed loop against a frozen evaluator.**

LLM judges and tournaments were not used for selection. The review found LLM judges of idea quality near chance (45-53 %
balanced accuracy; Si, Yang, Hashimoto, https://arxiv.org/abs/2409.04109, accessed 2026-11-12, cited there).

### 3.3 Scale of the search
Counts are from `docs/reports/FINAL_overnight_2026-11-12.md` section 2 and `docs/ideas.md`.
- **Round 1:** 8 independent ideators (causal inference, evolution/structure, LLM/text, ML, recommender systems, single
  cell, statistical genetics, wildcard) proposed 73 ideas. Three critics and a novelty check followed.
- **Wave 1:** 8 implementer agents plus the lead tested about 30 ideas as channels in the stacker, each against a
  same-width permuted control. One evolutionary program search ran (40 candidate programs, fitness on 600 `past` episodes).
- **Round 2:** 2 fresh-island ideators (16 ideas) and 2 evolution ideators working from the measured results.
- **Afternoon cycle:** three literature/data sweeps (cell types and foundation models; genetics, structure and drugs;
  model families) and five pre-registered tests: KG embeddings, cell-type witness, FinnGen R4, wide training, and two-arm
  experts.
- **Idea register:** 109 ideas. 11 tested positive, 17 tested negative, 4 tested with partial or other outcomes, 76 never
  tested, 1 parked.

### 3.4 Reviews
| review | object | main outcome |
|---|---|---|
| R1 red team (`docs/reviews/R1_fable_redteam.md`) | round-1 past-trained stacker | no leak; the claimed nuAP gain was mostly degree tilt (degree-matched AP equal at powers 0 / 0.25 / 0.5); no gain on strict-a |
| R2 final-report review | overnight report | numbers correct, framing overclaimed in six places. Examples: "about 20 variants on confirm" was really 59 stackers / 78 methods; a table quoted the wrong model; the power grid was chosen after confirm results by power were known. 16 corrections applied |
| R3 red team of modeng-v1 | final overnight model | no input-date leak, no memorisation; out of window the gain is carried by label-process channels; 37 % of confirm targets had prior OT evidence in the family |
| R4 strategy A / B + synthesis | whole programme | stacker near its ceiling; scale and a fresh fold are the levers; strict-a needs data that did not exist at T0 |

Rules adopted along the way (`docs/PROCESS_v2.md`):
- same-width permuted controls for every increment test;
- three seeds, or no statement, for deltas below about 0.007 (the retrain noise);
- AP **and** nuAP;
- pre-registration (commit, hash, tag) before scoring a fresh split;
- one look per split.

The largest process error was selection on the temporal confirm split. It was used as a scoreboard overnight, which is why
none of its numbers are [CONF].

---

## 4. Model

### 4.1 Architecture
**Channels.** A witness is any function `(task, gene) -> score` computed from inputs dated no later than the seed release
(plug-in modules in `src/modeng/channels.py` and `src/modeng/ext/`). Each channel enters the model twice, as its raw value
and as its within-task percentile rank.

**Combiner.** A LightGBM binary classifier over (task, gene) rows. Positives are a disease's targets and negatives are
sampled candidates. Positives are weighted by degree^-p.
- The weighting trades hub recall (AP) for low-degree recall (nuAP).
- R1 showed that the weighting itself adds no information: degree-matched AP was flat across powers.
- p = 0.35 was chosen on the dev mean of AP and nuAP. The grid was set after confirm results by power had been seen.

**`modeng-v1`** (`stack_v3ptr_p0.35`, frozen in `docs/preregistration_modeng_v1.md`, tag `modeng-v1-prereg`):
- 65 channels plus log(number of seeds), which gives 131 features;
- 400 boosting rounds and 3,000 negatives per task;
- trained by disease-grouped cross-validation on the 53 temporal dev diseases.

**`v1clean`**: modeng-v1 with the 16 post-dated expression and Perturb-seq views and 4 k-means donor families replaced by
HPA v20.1 and GTEx v8 similarities. It has 51 channels and no transductive input.

**`W_v1clean`**: the v1clean channels trained on the 712 `wide_train` diseases with the relative-target guard. It uses
600 rounds, 1,500 negatives and at most 60 positives per task.

**`W_v1clean` without label-process channels** drops `gwas_catlag`, `pri_l2g_n`, `pri_l2g_max` and `hz_gwas`, leaving
47 channels. This is the arm that passed the fresh-fold test.

### 4.2 Witnesses
The table below follows the R5 audit (`docs/reports/r5_witness_audit.md`). Columns:
- **"alone"**: the channel used as a ranking by itself, as temporal confirm AP / nuAP. For comparison, transfer_jaccard
  scores 0.119 / 0.104.
- **"removal"**: 3-seed group ablation from modeng-v1 on temporal confirm (removed minus full; negative means the group
  helps).

All values in the table are [DEV] and descriptive. Leakage classes:
- *clean*: content and annotation dated <= T0.
- *vintage*: pre-T0 content with post-T0 annotation or model version.
- *transductive*: post-T0 data that contain no labels.
- *label-process*: dated <= T0, but forecasts how OT creates labels.

| witness (group) | what it computes | data, date | class | alone AP / nuAP | removal dAP / dnuAP |
|---|---|---|---|---|---|
| base: transfer_jaccard, transfer_onto, popularity, degree, RWR | disease-to-disease transfer, gene popularity, STRING degree / RWR | OT 21.04; STRING v11.5 (2021-08) | clean / vintage | 0.119 / 0.104 (transfer) | +0.002 / **-0.018 [-0.022, -0.013]** |
| `dc_ridge`, `dc_ridge_pos` | EASE/SLIM-style ridge regression of the seed vector on masked context disease rows | OT 21.04 | clean | 0.104 / 0.114; 0.113 / 0.106 | -0.004 / **-0.006 [-0.011, -0.001]** |
| `cs_donor` | non-negative lasso of the seeds over the top-300 Jaccard disease rows | OT 21.04 | clean | 0.076 / 0.072 | +0.002 / +0.001 |
| `pt_ridge`, `pt_ridge_pos` | ridge of the seeds on PubTator3 gene x MeSH-disease sets | PubTator3, PMID <= 2021-03 (2026 mapping) | vintage | 0.075 / 0.068 | **-0.004 [-0.007, -0.002] / -0.004 [-0.007, -0.002]** |
| `dA_pt` | top-50 hypergeometric transfer over the same literature sets | as above | vintage | 0.093 / 0.095 | +0.000 / +0.001 |
| `ph_mean`, `ph_top3` | IDF-weighted HPO-profile similarity of gene to seeds | HPO 2021-01-15 | clean | 0.076 / 0.069 | -0.002 / -0.004 |
| priors `pri_mend_n`, `pri_hpo_n`, `hz_mend` | Mendelian-evidence count in other diseases; HPO term count | OT 21.04, HPO 2021-01 | clean | 0.015-0.019 | priors group +0.002 / -0.001 |
| priors `pri_l2g_n`, `pri_l2g_max`, `hz_gwas` | gene is an L2G >= 0.5 winner for other traits | OT 21.04 genetics portal | label-process | 0.007-0.020 | (in priors group) |
| `gwas_catlag` | GWAS Catalog associations added <= 2021-04-30, mapped to the disease or relatives | GWAS Catalog (2026 trait mapping) | label-process + vintage | 0.047 / 0.049 | -0.004 / **-0.007 [-0.016, -0.000]**; dev AP -0.025 |
| 15 donor families `don_*` | Jaccard transfer over virtual donor gene sets: GO, Reactome, HPO, IMPC, complexes, paralogs, embedding k-means | MSigDB 7.4, HPO 2021-01 etc.; 4 transductive | 9 clean, 2 vintage, 4 transductive | best `don_hpo2101` 0.080 / 0.068; 11 families <= 0.010 | -0.002 / -0.003 |
| 30 view similarities `vmean_*`, `vmax_*` | cosine of gene to seeds in 15 gene embeddings (ESM C, ProtT5, DepMap, GO, Reactome, expression, Perturb-seq) | mixed | 8 clean, 6 vintage, 16 transductive | AP 0.003-0.018 | **+0.006 [+0.001, +0.012] / +0.008 [+0.003, +0.014]** (removal helps) |
| clean views `vc_*` (v1clean only) | the same similarity on HPA v20.1 and GTEx v8 | 2020-11, 2017-06 | clean | AP 0.005-0.006 | - |

Findings of the audit:
- **Two models live inside one.**
  - On GWAS-dominated diseases (39 of 138 confirm), only `gwas_catlag`, the L2G priors and the base gene prior matter,
    and the model is weak there (AP 0.092).
  - On Mendelian-dominated diseases (96), the disease-row ridge, literature and HPO phenotype witnesses matter
    (AP 0.174).
- **The transductive witnesses contribute nothing.** The clean arm equals modeng-v1: confirm AP 0.150 vs 0.153,
  -0.004 [-0.009, +0.001].
- **With 53 training diseases the 30 view similarities hurt.** With 712 training diseases they help: v1clean beat its
  same-width permuted control on temporal confirm by +0.010 AP, and the pruned v2 was 0.007 AP below v1clean
  (`docs/reports/r6_wide_full.md`). "Views hurt" was an overfitting effect.
- **Retrain noise:** changing only the LightGBM seed moves confirm AP by up to 0.006 (R5) or 0.007 (R3).

---

## 5. Results

### 5.1 Temporal benchmark: modeng-v1 vs transfer and vs modet's headline [DEV]
Source: `results/review/leader_endpoints_stack_v3ptr_p0.35.tsv`. Dev is 53 diseases with out-of-fold predictions; confirm
is 138 diseases, scored by a model trained on dev.

| endpoint | dev (53) | confirm (138) |
|---|---|---|
| AP, modeng-v1 vs transfer | 0.176 vs 0.122, +0.053 [+0.035, +0.076] | 0.153 vs 0.119, **+0.034 [+0.011, +0.059]** |
| nuAP vs transfer | 0.150 vs 0.096, +0.055 [+0.034, +0.079] | 0.177 vs 0.104, **+0.073 [+0.053, +0.099]** |
| degree-matched AP vs transfer | +0.067 [+0.050, +0.086] | +0.061 [+0.040, +0.083] |
| hub-gene AP vs transfer | +0.030 [+0.007, +0.054] | -0.006 [-0.029, +0.019] |
| recall@100 vs transfer | +0.072 [+0.052, +0.097] | +0.041 [+0.008, +0.070] |
| onnet AP vs modet headline | 0.197 vs 0.127, +0.070 [+0.044, +0.099] | 0.160 vs 0.096, **+0.063 [+0.049, +0.082]** |
| onnet nuAP vs modet headline | 0.217 vs 0.190, +0.027 [-0.006, +0.063] | 0.203 vs 0.173, **+0.031 [+0.012, +0.054]** |
| strict-a onnet nuAP vs modet headline | +0.019 [-0.015, +0.058] (n = 28) | **0.110 vs 0.109, tie** (+0.001 [-0.016, +0.025]; n = 71) |

These numbers are not a confirmation:
1. Up to the night's final model, 59 stacker variants (78 methods) had been scored on temporal confirm (R2). By the end,
   116 methods had been scored there.
2. The degree power was chosen after confirm results by power were known.
3. The stacker was trained on dev-disease arrivals from the same 21.04 -> 25.09 window.
4. The R3 red team ruled out several alternative explanations:
   - gene memorisation: carry-over of seen genes is worth +0.003 to +0.007 AP, at seed-noise level;
   - input-date leaks: every channel recomputes exactly from raw inputs;
   - cluster overlap between dev and confirm: the gain is larger in the 38 confirm diseases whose cluster holds no dev
     disease.

### 5.2 Prospective version [DEV]
Source: `docs/reports/r3_prospective.md`. To remove same-window training, the v1 recipe was trained only on 1,786 `past`
episodes (19.04 -> 21.04; `stackP_v1past`).

| temporal confirm (138) | AP | nuAP | onnet AP | onnet nuAP |
|---|---|---|---|---|
| transfer_jaccard | 0.119 | 0.104 | 0.128 | 0.128 |
| stackP_v1past (past-trained) | 0.137 | 0.186 | 0.142 | 0.205 |
| modeng-v1 (cv on dev, same window) | 0.153 | 0.177 | 0.160 | 0.203 |

Paired differences on confirm:
- Past-trained vs transfer: AP +0.018 [-0.005, +0.043] (not significant); nuAP +0.082 [+0.057, +0.110].
- Past-trained vs modet's headline: onnet AP +0.046 [+0.030, +0.066]; onnet nuAP +0.032 [+0.003, +0.066]; strict-a onnet
  nuAP -0.000 [-0.031, +0.031].
- Past-trained vs modeng-v1: AP -0.016 [-0.029, -0.005].

About half of modeng-v1's AP margin over transfer therefore comes from training on the test window. The nuAP gain survives
prospective training. The AP gain is small (about +0.02) and fragile.

### 5.3 Forward window (25.09 -> 26.09) [DEV]
Sources: FINAL report section 1c, `docs/insights.md` 07:10 and 08:05 entries. The v1 recipe was trained in-window by
cross-validation on 842 forward dev diseases and tested on 797 forward confirm diseases. No pt_ridge channel was used.

| forward confirm (797) | AP | nuAP |
|---|---|---|
| transfer_jaccard | 0.085 | 0.083 |
| v1 recipe | 0.151 | 0.214 |
| v1 recipe without label-process channels | 0.103 | 0.156 |

There are no valid CIs here: forward tasks are hierarchically correlated, and each task was its own bootstrap cluster.
`gwas_catlag` is the second most important feature of the forward model.

A model trained on temporal dev and applied unchanged to forward reached AP 0.124. Without label-process channels it
reached 0.082, equal to transfer (-0.003 [-0.009, +0.002]; R3).

Without label-process channels, then, transfer of the gain to another time window is weak. In-window training keeps
about a third of the AP gain.

### 5.4 Training at scale on temporal_wide [DEV]
Source: `docs/reports/r6_wide_full.md`, pre-registered at commit `768c27d`. W_v1clean was compared with the v1clean
recipe cross-validated on the 53 dev diseases.
- **Temporal dev:** nuAP +0.028 [+0.013, +0.046]; AP +0.012 [-0.003, +0.027].
- **wide_dev (45):** nuAP +0.027 [-0.003, +0.065]; AP +0.005 [-0.014, +0.025].
- **Decision:** the pre-registered rule required an AP gain on both dev sets, so W_v1clean was **not adopted** as the
  new candidate.
- **Against transfer on wide_dev:** AP +0.029 [+0.004, +0.058], nuAP +0.054 [+0.025, +0.092].

Scale buys nuAP more clearly than AP.

### 5.5 Primary confirmatory result: fresh fold of 35 diseases [CONF]
Source: `docs/reports/r7_wide_confirm.md`.

**What was frozen and when.**
- The fold was frozen at 14:30 (tag `wide-confirm-frozen`) before any model or baseline was scored on any temporal_wide
  split.
- Arms, endpoint, comparator and decision rule were fixed in Amendment 1, commit `2fb1b77`, at 15:16:30. The first scoring
  command started at 15:16:49.
- Each arm was run once with seed 0. Nothing was changed after looking.

**The fold.** 35 diseases in 28 ontology components with 297 targets; median 11 seeds and 7 targets. Leakage class:
temporal-clean. modet's headline cannot be compared here, because none of these diseases is in modet's run.

**Primary endpoint.** Paired AP and nuAP against transfer_jaccard, with 10,000 cluster-bootstrap resamples. An arm is
confirmed only if both 95 % CIs lie above 0. Four arms were tested with no multiplicity correction, as stated in the
amendment.

| arm | training | AP | ΔAP | nuAP | ΔnuAP | passes |
|---|---|---|---|---|---|---|
| transfer_jaccard | - | 0.091 | | 0.084 | | |
| A modeng-v1 | 53 temporal dev | 0.105 | +0.014 [-0.016, +0.043] | 0.147 | +0.063 [+0.005, +0.145] | no |
| B v1clean | 53 temporal dev | 0.106 | +0.015 [-0.017, +0.050] | 0.142 | +0.059 [-0.004, +0.149] | no |
| C W_v1clean | 712 wide_train | 0.115 | +0.024 [-0.001, +0.047] | 0.119 | +0.036 [+0.001, +0.073] | no |
| **D W_v1clean, no label-process channels** | 712 wide_train | **0.118** | **+0.027 [+0.004, +0.049]** | **0.126** | **+0.042 [+0.006, +0.086]** | **yes** |

**Reading arm D.**
- The pass is narrow. Bootstrap P(Δ <= 0) is 0.010 for AP and 0.009 for nuAP.
- With Bonferroni-width intervals (98.75 %), the lower bounds are -0.002 for both metrics, so D would not survive a
  correction for four arms.
- The arms cannot be told apart. The minimum detectable between-arm effect is about 0.03-0.06 AP, and D vs C is
  +0.003 [-0.007, +0.016] AP. "Label-process channels hurt" is therefore not a supported reading.

**Secondary endpoints (descriptive).** For arm D vs transfer:
- degree-matched AP +0.087 [+0.043, +0.133];
- precision@20 +0.023 [+0.008, +0.039];
- onnet AP +0.055 [+0.013, +0.103].

Strict-a could not be evaluated: only 4 tasks had at least 5 strict-a targets.

**Allowed claim** (fixed in advance): the wide-trained clean model without label-process channels ranks newly arriving
disease genes better than transfer_jaccard on 35 fresh, pre-registered off-panel diseases (AP +0.027 [+0.004, +0.049],
nuAP +0.042 [+0.006, +0.086]). It was one of four pre-registered arms, uncorrected, and does not survive a Bonferroni
correction for four arms.

The frozen 53-disease recipes (A, B) are **not confirmed** on AP.

### 5.6 Disease modules [DEV]
Source: `docs/reports/r4_modules.md`. Ranked lists were turned into modules: seeds plus every candidate with calibrated
probability >= 0.2. Calibration used an isotonic map fitted on dev out-of-fold scores, shrunk by a dev-only factor of 0.8.
The comparators added the same number of genes per disease. On temporal confirm:
- **Hits:** modeng-v1 modules held 359 future targets, against 259 for transfer and 221 for modet's headline. Paired
  differences: +100 [+57, +155] and +138 [+87, +200].
- **Pooled precision:** 0.289 (predicted 0.310), against 0.209 and 0.178.

Decomposition with subex `leiden_anno`:
- Modules split into a median of 2 submodules, nearly all containing seeds.
- 21.6 % of modeng-v1's added genes are off STRING-T0, including 74 of its 359 hits, so they fall into no submodule.

The module analysis was not pre-registered.

---

## 6. Negative results
Each idea below was tested on dev with a permuted or ablation control. Most had a decision rule written before the run.
Sources: `docs/invalidations.md`, `docs/insights.md`, per-idea reports.

| idea | test | result |
|---|---|---|
| Evolved transfer kernel (LLM as mutation operator, 40 programs; `r2_evolve.md`) | nuAP fitness on 600 past episodes; stacker increment | nuAP +0.035 alone, but 75-85 % of it is degree down-weighting and AP roughly halves; stacker increment -0.003 [-0.008, +0.002]. Showed that nuAP alone can be gamed |
| Seed-set transformer trained on arrivals (`r1_settx.md`) | alone + stacker increment | memorises arrivals; stacker nuAP -0.016 to -0.019. Self-supervised twin reaches transfer level, no increment |
| Gene-kernel and homology lifts (`r1_kernels.md`) | increment over donors + views | AUROC +0.0055 only; AP / nuAP slightly negative; homology lifts near chance |
| Perturb-seq regulators / responders; DepMap, IMPC, perturbation donor sets (`r1_perturb.md`, `r1_donors.md`) | alone + increment vs permuted | AUROC 0.49-0.50 for regulators (equal to permuted); no increment. The biology is real but is not what OT labels reward |
| Trending and directed transfer; learned disease encoder (`r1_disease.md`) | increment | trending worse; directed no confirm gain; encoder mixed |
| Own-trait and proxy-trait MAGMA z (`r1_gwas.md`) | increment | weak; seed-weighted cross-trait z worse than random |
| Catalog backlog transfer E-A4 (`r2_ea4.md`) | strict-a increment | sa_AP +0.002 on confirm; kill rule met |
| Strict-a gating, class-A head, area flags (`r3_strict_a.md`) | 8 dev variants, 5 scored on confirm | no strict-a gain. Context features +0.006 AP overall, not adopted into the frozen recipe |
| GWAS power forecaster ML-4 (`r4_powerfc.md`) | strict-a increment | dev gain did not replicate (confirm sa_AP -0.001) |
| Sub-threshold loci -> nearest gene (`r4_subthreshold.md`) | strict-a increment | confirm sa_nuAP -0.006 [-0.017, +0.003] |
| Cached KG embeddings, RotatE on PrimeKG-T0 (`r5_kge.md`) | increment vs degree-permuted KG | confirm -0.003 AP / -0.004 nuAP; strict-a -0.013 |
| Cell-type-of-action witness (`r5_celltype.md`) | increment, kill rule | real signal (about 3x its permuted control) but redundant; confirm AP -0.003 |
| FinnGen R4 credible sets (`r6_finngen.md`) | share of strict-a targets traced by genetics, kill at < 20 % | 12.8 % -> 16.3 %: killed. 15 of 17 added targets came into OT through later FinnGen freezes (label process) |
| gnomAD LOEUF constraint | increment vs permuted | worse than its control |
| Two-arm experts + gate (`r7_experts.md`, pre-registered) | gate vs W_v1clean on two dev sets | not adopted: the two dev sets disagree in sign, all CIs include 0 |
| "Disagreement" list (high biology / low label process) as candidate new biology | new-data precision of top 50 | 0.0006 / 0.0003, equal to popularity-matched random; it picks the least-studied genes |

**Pattern.** After round 1, every new biology-derived similarity witness carried signal on its own but added nothing to
the stacker. Every attack on strict-a that reused data dated at or before T0 failed.

---

## 7. What the gain is and is not

### 7.1 Label-process share
Arrivals were typed by `src/modeng/arrival_types.py` (`docs/reports/r7_experts.md`):

| arrival type | share of targets | what it is |
|---|---|---|
| new data | about a third | first evidence 2022-2025, still present in 26.09 |
| label process | 35-43 % | ontology event, GWAS-Catalog ingestion lag, or old evidence ingested late |
| non-persistent | 13-28 % | not present in 26.09 |

The benchmark's AP is dominated by label-process arrivals. On temporal dev, W_v1clean reaches AP 0.256 on label-process
targets but only 0.056 on new-data targets; the corresponding transfer values are 0.172 and 0.026 [DEV].

The model still finds new-data arrivals better than transfer: 0.038 vs 0.011 on wide_dev. In absolute terms it finds them
poorly.

R3 found that 37 % of temporal confirm targets already had OT 21.04 evidence for the disease family, mostly text mining.
After removing those targets [DEV]:
- the AP gain over transfer survives: +0.025 [+0.012, +0.040];
- the nuAP gain over modet's headline becomes a tie: +0.011 [-0.006, +0.031].

On forward, "backlog" channels raise the stacker from nuAP 0.137 to 0.307. These channels use sub-threshold evidence that
already existed at the seed release, and 47 % of next-year arrivals already had evidence for the disease family
(`r1_backlog.md`). That is a forecast of the OT pipeline, not biology, and it was kept out of every model above.

### 7.2 The strict-a ceiling
Source: `docs/reviews/R4_strategy_B.md`; descriptive, on modeng-v1's saved temporal scores. A target counts as "traced"
if it is in the top 200 (about 1 %) of at least one of nine seed-specific witnesses.

- **Coverage.** 46 % of confirm targets are traced. Of those, modeng-v1 puts 67 % in its top 200. Of the untraced 54 %,
  it puts only 3.8 % there (chance 1.1 %), where it behaves as a gene-popularity prior.
- **What the untraced targets are.** 71 % are GWAS-only arrivals, and 30 % are non-persistent.
- **Strict-a has no genetic witness.** Only 4.6 % of strict-a targets are traced by a genetic witness (`gwas_catlag`).
  The model has no genetic witness for 95 % of the genetic discoveries it is asked to predict.

Oracle bounds on confirm AP (modeng-v1 0.153):

| bound | AP |
|---|---|
| perfect per-disease choice among 16 single witnesses | 0.188 |
| max(stack, best witness) per disease, i.e. the bound for any gate or mixture of experts | 0.202 |
| all traced targets ranked first, the rest unchanged | 0.57 |

The reviewer estimated that the stacker family can reach confirm AP 0.17-0.18.

Strict-a is coverage-limited. Only 34 of the 71 confirm strict-a diseases have their own GWAS study dated <= T0
(`r4_powerfc.md`). Most strict-a GWAS genes come from studies that did not exist at T0, such as later FinnGen freezes
(`r6_finngen.md`).

### 7.3 What may be said
- **May be said:**
  - On a fresh, pre-registered fold, a learned combiner of dated witnesses beat raw transfer on AP and nuAP by modest
    margins, without label-process channels [CONF, narrow].
  - On the development benchmark it beat modet's headline on all arrivals [DEV].
- **May not be said:**
  - any gain on new genetic discoveries (strict-a);
  - any claim about cross-window generalisation without label-process channels;
  - any ranking among the four confirmed arms;
  - any modet comparison on the fresh fold.

The gain comes mostly from finding arrivals that already had a pre-seed trace in phenotype, literature or related-disease
knowledge. It does not come from predicting new genetics.

---

## 8. Limitations
1. **One small confirmation.** n = 35 diseases in 28 clusters, median 7 targets, one pass out of four uncorrected arms.
   The minimum detectable effect against transfer is about 0.04 AP, so the confirmation can detect only large effects.
   The fold is now spent.
2. **Development numbers carry unknown selection.** More than 100 methods were scored on temporal confirm. No
   winner's-curse correction was applied.
3. **Same-window training.** Every wide and cv model is trained on 21.04 -> 25.09 arrivals of other diseases. The
   prospective twin loses half of the AP margin (section 5.2), and the fresh fold confirms only the in-window claim.
4. **Annotation vintage.** The PubTator3 and GWAS-Catalog mappings are 2026 releases applied to pre-T0 content. Lag tests
   by R1 and R3 (2-year PubTator lag, 12-month catalog lag) changed nothing, but archived vintages were not available.
5. **Labels are OT's.** Targets mix new biology with curation and ingestion events (section 7.1), and 13-28 % of them do
   not persist. Gains on these targets are not gains on biology.
6. **Ontology entanglement.** Off-panel diseases are closely related to the panel, so the fresh fold could only be built
   from a few unrelated components, and wide-trained models needed a leak guard.
7. **Seed noise.** Retraining noise is 0.005-0.008 AP, the same size as most single-channel increments. Many increment
   statements rest on single seeds or on 3 seeds.
8. **Forward CIs** treat 797 hierarchically related tasks as independent, so no forward interval is reported.
9. **Agent-run research.** Agents designed, ran and reported the experiments. Reviews caught overclaims and a leak, but no
   human re-derived the results. This write-up was also written by an agent from the project files (Appendix B lists a
   number check).

---

## 9. Outlook
Two follow-up projects split the two skills that the present benchmark mixes.

1. **Forward / new-data confirmation.** Use the next OT release after 26.09 as the fresh fold, as listed in
   `docs/preregistration_modeng_v1.md`.
   - Freeze the recipe (W_v1clean without label-process channels as the primary arm).
   - Fix a minimum fold size in advance, for example >= 300 diseases with >= 3 new genes (R4-B).
   - Type every arrival.
   - Report a biology arm and a label-process forecaster as separate products.

   New data could reopen strict-a: full biobank summary statistics with a sample-size growth law, and borrowing across
   genetically correlated endpoints (R4-B D1). Hit lists will not.
2. **New-biology benchmark.** Make persistent new-data events the primary endpoint, with label-process channels
   forbidden. Use outcome sets that are not OT labels as secondary truths: FinnGen freeze-to-freeze loci, exome burden
   results published after T0, drug-programme outcomes (R4-B D2). On such a benchmark, a model that gains only on
   ingestion lag would be identified as a pipeline forecaster.

modet's sealed fold should stay sealed. Its primary endpoint is strict-a, where every modeng variant ties (R4 synthesis).

---

## Appendix A. Reproducibility
- **Repository:** `/home/ph/code/modeng`. The write-up is based on commit `c483fb2`.
- **Data root:** `/mnt/msadb/modeng/`; `data/<sub>` are symlinks to it.
- **Python:** `~/biotools/venvs/bio-net/bin/python`, with `PYTHONPATH=src`.
- **Tags and commits:**
  - `modeng-v1-prereg` -> `bd80b6a`: frozen modeng-v1 recipe.
  - `wide-confirm-frozen` -> `e6f00fa`: temporal_wide builder, split and frozen ids `data/wide_confirm_ids.txt` +
    `docs/preregistration_wide.sha256`.
  - `768c27d`: pre-registration of wide training.
  - `328dabf`: pre-registration of experts and gate.
  - `2fb1b77`: Amendment 1, the confirmatory scoring plan.
  - `4c936ff`: wide_confirm result.
- **Baselines:** `python -m modeng.bench --method all --bench all --split both --jobs 12`
- **modeng-v1 on temporal:** `python -m modeng.stack --bench temporal --channels base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,hz_gwas,hz_mend,ph_mean,ph_top3,pt_ridge,pt_ridge_pos --pos-deg-power 0.35`
- **Prospective twin:** the same with `--train past` and the `_pf` channels (`docs/reports/r3_prospective.md`, section 2).
- **Wide arms:** `python -m modeng.stack --train wide --pos-deg-power 0.35` with the channel list
  `results/v1clean_channels.txt`, minus the four label-process channels for arm D.
  - Scoring wide_confirm needed `--score-confirm` plus `MODENG_SCORE_WIDE_CONFIRM=1`. The fold is spent; do not rerun it
    as a test.
  - Driver: `results/p4/run.sh`. Analysis: `scripts/p4_analyze.py`.
- **Endpoints and bootstraps:** `scripts/leader_endpoints.py`, `src/modeng/compare.py`. Witness audit: `scripts/r5_audit_*.py`.
- **Modules:** `scripts/module_cutoff.py`, `scripts/export_modules.py`, `scripts/decompose_modules.py`,
  `scripts/eval_modules.py`.
- **Tests:** `tests/test_harness.py`, `tests/test_wide.py`.

## Appendix B. Number check
Fifteen numbers were drawn at random (Python `random.seed(15)`) from the numeric tokens of this document, excluding
section headings and the appendices. Each was compared with its source file after the text was written.

| # | number in text (where) | source | match |
|---|---|---|---|
| 1 | power 0.25 (section 3.4, R1 row) | `docs/reviews/R1_fable_redteam.md` table: "stack, power 0.25" | yes |
| 2 | 25.09 (abstract, target release) | `docs/reports/benchmarks.md`: `ot_current_sets` = 25.09 | yes |
| 3 | +0.003 to +0.007 AP memorisation carry-over (5.1) | `docs/reviews/R3_fable_redteam_v1.md` verdict | yes |
| 4 | P@20 +0.023 [+0.008, +0.039], arm D (5.5) | `docs/reports/r7_wide_confirm.md` secondary table, row D | yes |
| 5 | onnet AP vs headline +0.063 [+0.049, +0.082] (abstract) | FINAL report, corrected summary item 1 and section 1c | yes |
| 6 | transfer_onto AUROC 0.62 (2.4) | `results/leaderboard.tsv` temporal confirm transfer_onto auroc 0.6173 | yes (rounded) |
| 7 | dc_ridge input OT 21.04 (4.2) | `docs/reports/r5_witness_audit.md` inventory: "OT 21.04 masked context" | yes |
| 8 | modeng-v1 AP gain +0.034 [+0.011, +0.059] (abstract, 5.1) | FINAL report section 1c; `results/review/leader_endpoints_stack_v3ptr_p0.35.tsv` (0.0341, 0.0105, 0.0592) | yes |
| 9 | arm D CI lower bound +0.004 (abstract) | `docs/reports/r7_wide_confirm.md` primary table | yes |
| 10 | next OT release after 26.09 (section 9) | `docs/preregistration_modeng_v1.md` candidate fresh folds (1) | yes |
| 11 | cell-type witness confirm AP -0.003 (section 6) | `docs/insights.md` 14:30 entry ("confirm AP -0.003, nuAP -0.005") | yes |
| 12 | arm C ΔAP +0.024 (5.5) | `docs/reports/r7_wide_confirm.md` primary table | yes |
| 13 | RWR AUROC 0.65 (2.4) | `results/leaderboard.tsv` temporal confirm rwr_string auroc 0.6530 | yes |
| 14 | oracle per-disease witness choice AP 0.188 (7.2) | `docs/reviews/R4_strategy_B.md` section 1b | yes |
| 15 | arm D AP 0.118 (abstract, 5.5) | `docs/reports/r7_wide_confirm.md`; `results/leaderboard.tsv` stackW_P4D_v1clean_nolab wide_confirm 0.1181 | yes |

All 15 match. Beyond the sample, the baseline table (2.4) was checked cell by cell against `results/leaderboard.tsv`,
and the witness table (4.2) against the R5 audit tables.
