# R3 red-team review of `modeng-v1` (= `stack_v3ptr_p0.35`), the final model (Fable reviewer, 2026-11-12 07:25-08:10)

Scope: what R1 and R2 left unreviewed: input leakage of the v3 channels (OT 21.04 priors, HPO 2021-01 phenotype matching,
PubTator ridge, GWAS-Catalog lag), cv training on same-window arrivals, and near-deterministic channels.
Scripts: `scripts/review2_{common,retrain,leaks,window,strata,summary,crosswindow,crosswindow_past}.py`. Outputs and logs:
`results/review2/`. Nothing was written to the leaderboard or the channel cache. Intervals are 95 % paired cluster bootstrap
(5,000 draws, 41 confirm clusters) unless stated. Confirm = 138 diseases, 4,438 target pairs. The machine was shared
(load 55-75), so every retrain used 2-4 threads.

## Verdict: HOLDS WITH CAVEATS

- **No input-date leak, no masking error and no near-deterministic channel was found.** Every v3 channel recomputes exactly
  from raw inputs, the date filters do what they say, and the model reproduces (AP 0.1532, nuAP 0.1770).
- **The stacker did not mainly learn "which genes arrive in 2021-2025".** Gene-identity carry-over from the dev arrivals is
  worth about +0.003 to +0.007 AP of the +0.034 gain over transfer, no more than the seed-to-seed spread. The model trained only on the 53 temporal dev diseases
  also beats transfer in another window (forward 25.09 -> 26.09: AP 0.124 vs 0.085, nuAP 0.142 vs 0.083).
- **Caveat 1 (new, the main one): outside its training window the gain is carried by the label-process channels.** Without
  `gwas_catlag`, `pri_l2g_n`, `pri_l2g_max`, `hz_gwas` the same temporal-trained model equals transfer on forward
  (AP 0.082 vs 0.085, -0.003 [-0.009, +0.002]; nuAP +0.015 [+0.008, +0.023]). In the earlier window (19.04 -> 21.04, inputs dated <= 19.04
  only) the model without `gwas_catlag` keeps a small gain (AP 0.063 vs 0.050, +0.013 [+0.010, +0.017]; nuAP +0.033
  [+0.027, +0.038]); with `gwas_catlag` it reaches AP 0.110.
- **Caveat 2 (new): 37 % of the confirm targets already had Open Targets 21.04 evidence for the disease or a descendant**
  (mostly text mining; 4.6 % ClinVar / curated grade). The gain over transfer survives their removal; the nuAP gain over
  modet's headline does not (+0.011 [-0.006, +0.031]).
- **Caveat 3 (new): confirm is not cluster-disjoint from the training diseases** (100 of 138 confirm diseases sit in a
  cluster that also holds dev diseases). No inflation was found: the gain is larger in the 38 confirm-only-cluster diseases.
- Caveats from R1 / R2 stay: 59+ variants scored on confirm, power 0.35 chosen after confirm results by power were known,
  tie with modet's headline on strict-a, PubTator / GWAS-Catalog annotation vintage is 2026.

## Check 0: reproduction

| run | AP | nuAP | onnet AP | onnet nuAP | strict-a onnet nuAP |
|---|---|---|---|---|---|
| leaderboard `stack_v3ptr_p0.35` | 0.1532 | 0.1770 | 0.1599 | 0.2034 | 0.1101 |
| retrained here (`ref`, seed 0, 4 threads) | 0.1532 | 0.1770 | 0.1599 | 0.2034 | 0.1101 |
| other seed 1 (negative sample + LightGBM seed) | 0.1598 | 0.1825 | 0.1676 | 0.2123 | 0.1126 |
| other seed 2 | 0.1544 | 0.1769 | 0.1612 | 0.2017 | 0.1148 |

Seed 0 is not a lucky seed (it is the lowest of three). Seed-to-seed differences reach 0.007 AP / 0.006 nuAP (seed 1 minus
seed 0: AP +0.007 [+0.002, +0.011]), so deltas of that size below are within seed noise (single seed per variant).

## Check 1: input leakage of the v3 channels (brute-force recomputation, `results/review2/leaks.log`)

| channel | independent recomputation | date / vintage check | sensitivity retrain (delta vs modeng-v1 on confirm) |
|---|---|---|---|
| `pri_l2g_n`, `pri_l2g_max`, `pri_mend_n` | from the raw `evidence_2104` parquet with DuckDB, 6 confirm tasks: max abs diff 2e-7 / 0 / 1e-7 | `ot_genetics_portal` 21.04: latest publication year 2019, 0 of 388,324 rows after 2021; rows of the task disease and its descendants are excluded (verified by the recomputation) | all four label-process channels dropped: AP -0.007 [-0.020, +0.004], nuAP -0.008 [-0.022, +0.002], strict-a onnet nuAP -0.020 [-0.034, -0.008] |
| `ph_mean`, `ph_top3`, `pri_hpo_n`, `hz_mend` | dense recomputation, 4 tasks: max abs diff 3e-7, NaN pattern identical | the file is a real 2021-01-15 vintage: 4,241 genes and 7,729 terms vs 5,119 / 10,265 in HPO 2025-09; 924 genes got their first HPO profile later and are NaN here (8.9 % of confirm target pairs) | not run (covered by "no gene-level channels" below for `pri_hpo_n`, `hz_mend`) |
| `pt_ridge`, `pt_ridge_pos` | dense ridge from `a_pt21`, 4 tasks: max abs diff 5e-8 | PMID cut 33,750,000 (2021-03-22, checked by R1) | 2-year lag (`a_pt19`, PMID <= 2019-03) for `dA_pt`, `pt_ridge`, `pt_ridge_pos`: AP -0.002 [-0.004, +0.001], nuAP +0.000 [-0.006, +0.004] |
| `gwas_catlag` | from the association table, 5 covered tasks: max abs diff 0 | 252,119 associations added <= 2021-04-30; 0 of them from a study published after that date | 12-month lag (`gwas_catlag12`): AP -0.002 [-0.006, +0.002], nuAP -0.002 [-0.008, +0.004] |

**Do 21.04 evidence rows contain temporal targets?** Yes, and the `unknown` exclusion does not remove them
(`results/review2/ev2104_target_overlap.tsv`):

- 1,629 of 4,438 confirm target pairs (36.7 %) have some 21.04 evidence row for the task disease or one of its descendants
  (sampled non-target candidates: 12.1 %; `unknown` genes: 28.5 %). By source: europepmc 1,263, phenodigm 431, eva 187,
  phewas_catalog 156, cancer_gene_census 151, genomics_england 116, uniprot_variants 102.
- 205 pairs (4.6 %; non-targets 0.08 %) have ClinVar >= 0.5 or curated evidence, 8 pairs an L2G >= 0.5 winner. All 205 sit on
  descendant rows (none on the task disease itself); 148 of them are modet arrival class B (old evidence, ingested later).
- The priors cannot read these rows (the family is excluded), and the label context masks them. But the same knowledge
  reaches the model through the literature and HPO channels, which is allowed by date and is not a forecast of new evidence.

Endpoints of modeng-v1 after removing such targets from the ranking (`results/review2/strata.tsv`):

| confirm targets kept | pairs | AP vs transfer | nuAP vs transfer | onnet AP vs modet headline | onnet nuAP vs modet headline |
|---|---|---|---|---|---|
| all | 4,438 | +0.034 [+0.010, +0.059] | +0.073 [+0.053, +0.098] | +0.063 [+0.050, +0.081] | +0.031 [+0.011, +0.054] |
| no ClinVar / curated / L2G 21.04 evidence in the family | 4,227 | +0.030 [+0.009, +0.052] | +0.065 [+0.046, +0.089] | +0.048 [+0.034, +0.066] | +0.021 [+0.001, +0.045] |
| same, ancestors included | 3,747 | +0.031 [+0.015, +0.048] | +0.056 [+0.040, +0.076] | +0.032 [+0.019, +0.048] | +0.009 [-0.007, +0.029] |
| no 21.04 evidence of any source in the family | 2,809 | +0.025 [+0.012, +0.040] | +0.042 [+0.025, +0.065] | +0.027 [+0.015, +0.042] | +0.011 [-0.006, +0.031] |

Reading: the gain over transfer is not produced by already-evidenced targets. The nuAP advantage over modet's headline is:
it is a tie as soon as targets with prior genetic evidence in the family or an ancestor are removed (same message as the
strict-a tie).

Not testable here: PubTator3 annotations are the 2026 version applied to old abstracts; GWAS-Catalog `MAPPED_TRAIT_URI`
and `MAPPED_GENE` are the 2026-10 mapping (no archived vintages on disk). The date lags above bound the publication-date
route only.

## Check 2: cv training on same-window arrivals (`results/review2/window.log`, `summary.tsv`, `crosswindow*.log`)

Structure: 1,578 genes (8.2 %) are a target of at least one dev disease; 32.5 % of confirm target pairs are such genes
(non-target candidates 7.4 %). A scorer that only counts dev arrivals per gene (no disease information) reaches AP 0.018,
nuAP 0.013, AUROC 0.62 on confirm (transfer 0.119 / 0.104 / 0.70). 11 confirm diseases have an ontology relative among the
dev diseases; no confirm disease shares more than half of its targets with a dev disease (8 share > 20 %).

| test | result on confirm | reading |
|---|---|---|
| Half-gene training: two models, each trained on the rows of one random half of the genes; "seen" = gene scored by the model that had its rows, "unseen" = by the other (same training size) | split A: seen - unseen AP +0.007 [+0.002, +0.014], nuAP +0.006 [-0.002, +0.016]; split B: AP +0.003 [-0.004, +0.010], nuAP +0.004 [-0.005, +0.012] | gene identity carries over, but no more than seed noise |
| "Unseen" model vs transfer (genes never in a training row, half the training data) | split A: AP +0.018 [-0.002, +0.038], nuAP +0.043 [+0.025, +0.065]; split B: AP +0.022 [+0.003, +0.042], nuAP +0.047 [+0.029, +0.068] | most of the gain needs no gene memory |
| Gene-level channels (`degree`, `popularity`, `pri_l2g_n`, `pri_l2g_max`, `pri_mend_n`, `pri_hpo_n`, `hz_gwas`, `hz_mend`) dropped | AP 0.147 (-0.006 [-0.014, +0.001]), nuAP 0.163 (-0.015 [-0.022, -0.008]); still +0.028 [+0.009, +0.049] AP and +0.058 [+0.041, +0.080] nuAP over transfer | they add nuAP, not the bulk |
| Same channels with the gene axis permuted by one fixed permutation (a per-gene identity code without content) minus "dropped" | permutation 1: AP +0.003 [+0.000, +0.006], nuAP +0.002 [-0.005, +0.007]; permutation 2: AP +0.002 [-0.002, +0.007], nuAP +0.004 [-0.003, +0.010] | identity memorisation is within seed noise |
| Gain over transfer by target class | gene also a dev target: AP +0.029 [+0.003, +0.057], nuAP +0.071 [+0.047, +0.099]; gene not a dev target: AP +0.024 [+0.002, +0.047], nuAP +0.051 [+0.028, +0.081] | no concentration on dev-target genes |
| False positives: share of dev-target genes in the top 100 non-targets | modeng-v1 29.7 %, transfer 23.0 % (base rate 8.2 %) | a mild tilt towards "genes that arrive" |
| Cluster overlap with training diseases | 100 confirm diseases in clusters shared with dev: AP +0.024 [-0.004, +0.054], nuAP +0.059 [+0.040, +0.082]; 38 in confirm-only clusters: AP +0.060 [+0.029, +0.089], nuAP +0.109 [+0.062, +0.158] | no inflation by related training diseases |
| Train only on the 17 dev diseases with no shared cluster and no ontology relation to confirm | AP 0.123, nuAP 0.150; three random 17-disease subsets: AP 0.127 / 0.102 / 0.127, nuAP 0.140 / 0.126 / 0.143 | relatedness does not matter; training-set size does (with 17 diseases the AP gain over transfer is gone) |

**Cross-window transfer (the direct test).** The modeng-v1 recipe is trained on the 53 temporal dev diseases only
(arrivals 21.04 -> 25.09) and applied unchanged to another window. One run each, no tuning, not pre-registered;
`pri_mend_n` is dropped on forward (not available there).

| trained on temporal dev, applied to | model | transfer | delta AP | delta nuAP |
|---|---|---|---|---|
| forward confirm (25.09 -> 26.09, 797 diseases), all channels | AP 0.124, nuAP 0.142 | 0.085 / 0.083 | +0.039 [+0.028, +0.052] | +0.059 [+0.046, +0.072] |
| same, 721 diseases unrelated to the 53 training diseases | AP 0.128, nuAP 0.147 | 0.086 / 0.083 | +0.042 [+0.034, +0.050] | +0.064 [+0.054, +0.074] |
| forward confirm, without the four label-process channels | AP 0.082, nuAP 0.098 | 0.085 / 0.083 | -0.003 [-0.009, +0.002] | +0.015 [+0.008, +0.023] |
| past episodes (seeds 19.04, targets new by 21.04, 900 diseases), the 12 channels whose inputs are dated <= 19.04 (base, `cs_donor`, `dA_pt`, `dc_ridge(_pos)`, `pt_ridge(_pos)`, `gwas_catlag`) | AP 0.110, nuAP 0.129 | 0.050 / 0.045 | +0.060 [+0.052, +0.068] | +0.084 [+0.075, +0.094] |
| past episodes, the same without `gwas_catlag` (11 channels) | AP 0.063, nuAP 0.078 | 0.050 / 0.045 | +0.013 [+0.010, +0.017] | +0.033 [+0.027, +0.038] |
| same, 747 diseases unrelated to the 53 training diseases | AP 0.062, nuAP 0.081 | 0.050 / 0.049 | +0.012 [+0.008, +0.015] | +0.032 [+0.026, +0.039] |

(Forward intervals: bootstrap over 425 ontology-connected components for the first row, over single tasks for the others.
Past intervals: modet clusters where defined, else single diseases. The priors and HPO channels do not exist for 19.04; donor
families and views are left out on past because several are dated after 19.04, e.g. HPO 2021-01. With them: AP 0.110, and
0.068 without `gwas_catlag`. `gwas_catlag` alone has AP 0.133 on the 43 % of past episodes it covers.)
For comparison the forward cv-trained model has AP 0.151, nuAP 0.214: same-window training on 842 diseases adds
+0.027 AP [+0.021, +0.032] and +0.072 nuAP [+0.065, +0.080] over the model trained in the other window.

Reading:
- The learned function is not tied to the 2021-2025 window: it transfers to 2025-2026 arrivals with a gain as large as on
  temporal confirm. "It only learned which genes arrive in this window" is refuted.
- What transfers to forward is the label-process part (GWAS-Catalog hits and L2G winners that Open Targets ingests later).
  The other channels (ridge / lasso donors, PubTator, HPO, views) add nothing in AP over transfer on forward when trained in
  the other window. The forward leaderboard says the same for cv training (`stack_base` AP 0.077, `stack_dc_ridge` 0.087,
  transfer 0.085), so this is at least partly the nature of one-year arrivals, not only a training-window effect.
- On temporal confirm the non-label-process part does hold with same-window training (AP 0.146, +0.027 [+0.012, +0.043]
  over transfer; nuAP +0.065 [+0.049, +0.086]). In the earlier window they transfer with a small gain (AP +0.013, about a quarter above
  transfer; nuAP +0.033), so they are not purely a same-window artefact, but four fifths of the out-of-window AP gain there
  is `gwas_catlag`.

## Check 3: near-deterministic channels (`results/review2/channel_auroc.tsv`)

- Mean per-task AUROC on confirm targets, best single channels: `popularity` 0.787, `dA_pt` 0.773, `pt_ridge_pos` 0.771,
  `ph_mean` 0.723, `pri_mend_n` 0.720, `dc_ridge_pos` 0.715, `transfer_jaccard` 0.705. No channel is above 0.95 (threshold
  for suspicion); the permuted control `pri_l2g_n_perm` is at 0.503.
- Best single-channel AP: `transfer_jaccard` 0.119, `dc_ridge_pos` 0.113, `dc_ridge` 0.104, `dA_pt` 0.093. No new channel
  beats transfer alone.
- Per task: at most 7 of 138 diseases have AUROC > 0.95 for any one channel (`ph_mean`, `ph_top3`, `pt_ridge_pos`);
  `transfer_jaccard` itself has 3. modeng-v1: mean AUROC 0.864, 17 diseases > 0.95, highest per-disease AP 0.81.
- Feature gain (`results/stack_v3ptr_p0.35.txt`): `dc_ridge_pos` 13.2k, `hz_gwas` 11.5k,
  `r_dc_ridge_pos` 10.7k, `gwas_catlag` 10.1k, `transfer_jaccard` 9.5k, `log_nseeds` 5.7k. No single feature dominates;
  two of the top four are label-process channels (`hz_gwas`, `gwas_catlag`).

## Not tested / residual

- Archived PubTator and GWAS-Catalog mapping vintages (not on disk).
- STRING 11.5 (August 2021) as in R1.
- Selection: all of this is on temporal confirm, which was scored 59+ times; the forward and past transfer runs above are
  single untuned runs but on benchmarks that were also viewed during the night.
- Dev OOF numbers were not re-derived; the half-gene and permutation tests were run on confirm only.

## What may be claimed (exact sentences)

1. "On the temporal benchmark (seeds Open Targets 21.04, targets new by 25.09, 138 confirm diseases) modeng-v1 ranks future
   disease genes better than raw disease-to-disease transfer: AP 0.153 vs 0.119 (+0.034 [+0.011, +0.059]), nuAP 0.177 vs
   0.104 (+0.073 [+0.053, +0.099]). A red-team review found no input-date leak, no masking error and no channel that nearly
   determines the targets; the gain persists for the 63 % of targets without any Open Targets 21.04 evidence in the disease
   family (AP +0.025 [+0.012, +0.040], nuAP +0.042 [+0.025, +0.065])."
2. "The model is trained on arrivals of the same time window in other diseases. Memorised gene identity explains at most
   about +0.007 AP of the gain, and the model trained on 53 diseases of the 2021-2025 window also beats transfer on
   2025-2026 arrivals (AP 0.124 vs 0.085, nuAP 0.142 vs 0.083, 797 diseases)."
3. "That out-of-window gain comes from channels that forecast the Open Targets label pipeline (GWAS-Catalog hits and L2G
   winners not yet ingested); without them the out-of-window model equals transfer in AP on 2025-2026 arrivals (0.082 vs
   0.085) and is slightly better on 2019-2021 arrivals (AP 0.063 vs 0.050, +0.013 [+0.010, +0.017]). The larger gain of the
   other channels is shown only with same-window training (temporal confirm without the label-process channels: AP +0.027
   [+0.012, +0.043], nuAP +0.065 [+0.049, +0.086] over transfer)."
4. "Against modet's headline on modet's candidate set modeng-v1 is better for all arrivals (AP +0.063 [+0.049, +0.082], nuAP
   +0.031 [+0.012, +0.054]) but ties on strict-a (nuAP +0.001 [-0.016, +0.025]) and on targets without prior genetic
   evidence in the disease family or an ancestor (nuAP +0.009 [-0.007, +0.029])."

Not claimable: "predicts new biology better than transfer" without the same-window qualifier; "beats modet's headline"
without "all arrivals"; any confirm number as a confirmation (development numbers; the pre-registered fresh-fold run in
`docs/preregistration_modeng_v1.md` is still the test). The pre-registered arm without label-process channels should be
promoted from "leakage class to report" to a co-primary comparison, because this review shows it is where the claim can fail.
