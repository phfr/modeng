# R1 implementer H: channel priors, channel-typed transfer, phenotype matching (2026-11-12)

Ideas: critic_bio CI-06 / SG-8 (channel hazards), SG-4 -> B1 (HPO phenotype matching), "channel-typed transfer".
Code: `src/modeng/ext/priors_h.py` (channels), `scripts/priors_h_strata.py` (stacker runner = protocol of `modeng.stack.run`,
plus GWAS / non-GWAS target strata), `scripts/priors_h_one.sh` / `scripts/priors_h_stack.sh` (configs), `scripts/priors_h_alone.py`,
`scripts/priors_h_alone_strata.py`, `scripts/priors_h_compare.py` (paired cluster bootstrap via `modeng.compare.paired_bootstrap`).
Per-task results: `results/<bench>/<split>/hS_<tag>.parquet`; leaderboard rows `hS_<tag>`; channels alone:
`results/priors_h_alone_{temporal,forward}_h.tsv`, `results/priors_h_alone_strata_temporal.tsv`.

## 0. Critic facts re-verified (modet `arrival_devall_t0_string115_phys_notm_400`, universe = 19,255 sorted ENSG)
| fact | critic | here |
|---|---|---|
| gwas_only arrivals (n = 2,863) that are a T0 L2G >= 0.5 winner for some trait (`evidence_2104` ot_genetics_portal) | 86.4 % vs base 41.5 % | **86.4 %** vs base **40.6 %** (7,823 universe genes); any L2G row 98.8 % vs 86.0 % |
| ClinVar / curated arrivals with an HPO 2021-01-15 gene profile | 70.7 / 70.9 % vs base 21.7 % | clinvar_only **74.1 %**, curated_only **71.1 %**, mixed_non_gwas 78.1 %, burden_only 67.2 %, gwas_only 39.2 %; base **22.0 %** (4,241 genes, Entrez -> HGNC map) |
Small differences = gene-id mapping (universe-only genes, Entrez via modet's HGNC table). Both facts hold.

## 1. Channels (all exact w.r.t. task masks; `past` -> all NaN since there is no 19.04 evidence)
- `pri_l2g_n` log1p(# evidence diseases where gene is a GWAS winner: 21.04 ot_genetics_portal L2G >= 0.5 / 25.09 gwas_credible_sets >= 0.5),
  `pri_l2g_max`, `pri_mend_n` (eva >= 0.5 + gene2phenotype / genomics_england / clingen / uniprot_*; temporal only),
  `pri_hpo_n` (# HPO terms), `pop_<src>` per-datasource popularity (temporal: europepmc, phenodigm, chembl, expression_atlas, eva,
  ot_genetics_portal; forward: gwas_credible_sets). All evidence counts EXCLUDE the task family (own disease + all ontology
  descendants), uniformly for all genes (no target-specific subtraction).
- `hz_gwas` = pri_l2g_n x mix_g, `hz_mend` = pri_hpo_n x (1 - mix_g); mix_g = share of seeds with GWAS-winner evidence in the task family.
- `tr_gwas` / `tr_nongwas`: transfer_jaccard donors (top-50 Jaccard on the full masked rows, own row excluded) but each donor row
  keeps only its GWAS-supported genes (E_g = GWAS-winner evidence of the row's disease or a descendant) / only the rest.
  Masked pairs subtracted exactly from the part they belong to, then `np.round(.,5)` (leakage rule 1). Check: tr_gwas + tr_nongwas =
  transfer_jaccard to 1e-5 (rounding) on temporal dev.
- `ph_mean`: gene x HPO-term matrix, IDF-weighted, L2-normalised rows; disease profile = normalised sum of seed rows; score = cosine.
  `ph_top3`: mean of the 3 largest gene-seed cosines. NaN = gene without HPO (78 % of genes on temporal, 73 % forward).
- Negative controls: `ph_mean_perm` (HPO matrix rows permuted over genes), `pri_l2g_n_perm` (gene identities permuted).

Leakage class: temporal = OT evidence 21.04 + HPO 2021-01-15 (Wayback): **temporal-clean**. forward = evidence 25.09 (only
gwas_credible_sets is on disk) + HPO 2025-09-01 (`data/raw/hpo_2025-09-01`, hp.obo data-version 2025-09-01; downloaded by another
wave-1 agent): **temporal-clean** (inputs <= seed release). No 26.09 file used. Holdout not run (current HPO would be transductive).

## 2. Channels alone (nuAP; NaN filled with the row minimum)
Temporal (dev 53 / confirm 138):
| channel | dev nuAP | conf nuAP | dev AUROC | conf AUROC |
|---|---|---|---|---|
| transfer_jaccard (ref) | 0.0961 | 0.1042 | 0.682 | 0.705 |
| popularity (ref) | 0.0111 | 0.0121 | 0.777 | 0.787 |
| ph_mean | **0.0695** | **0.0691** | 0.697 | 0.723 |
| ph_top3 | 0.0619 | 0.0592 | 0.696 | 0.722 |
| tr_nongwas | 0.0732 | 0.0920 | 0.632 | 0.657 |
| tr_gwas | 0.0367 | 0.0267 | 0.541 | 0.552 |
| pri_l2g_n | 0.0157 | 0.0196 | 0.662 | 0.656 |
| pri_mend_n | 0.0068 | 0.0088 | 0.693 | 0.720 |
| ph_mean_perm (neg) | 0.0066 | 0.0031 | 0.487 | 0.488 |
| pri_l2g_n_perm (neg) | 0.0027 | 0.0025 | 0.515 | 0.503 |

Temporal strata (alone; gw = gwas_only targets, ng = clinvar/curated/mixed non-GWAS targets, other targets removed; tasks with >= 3):
| channel | dev gw | dev ng | conf gw | conf ng | conf gw AUROC | conf ng AUROC |
|---|---|---|---|---|---|---|
| transfer_jaccard | 0.065 | 0.123 | 0.034 | 0.174 | 0.644 | 0.765 |
| tr_gwas | **0.086** | 0.003 | **0.052** | 0.009 | 0.634 | 0.523 |
| tr_nongwas | 0.007 | **0.125** | 0.003 | **0.176** | 0.530 | 0.756 |
| ph_mean | 0.009 | 0.099 | 0.012 | 0.125 | 0.609 | **0.825** |
| pri_l2g_n | 0.030 | 0.006 | 0.030 | 0.006 | **0.826** | 0.556 |
| popularity | 0.012 | 0.018 | 0.012 | 0.010 | 0.802 | 0.805 |
(n tasks: dev gw 27 / ng 41; confirm gw 69 / ng 103.) The two channels separate cleanly: typed transfer beats full transfer within
its own stratum (GWAS: +0.02 dev / +0.018 confirm nuAP), ph_mean ranks Mendelian arrivals at AUROC 0.83, pri_l2g_n ranks GWAS arrivals
at AUROC 0.83-0.85 (better than popularity's 0.80-0.82).

Forward (dev 842 / confirm 797):
| channel | dev nuAP | conf nuAP | dev AUROC | conf AUROC |
|---|---|---|---|---|
| transfer_jaccard (ref) | 0.0964 | 0.0831 | 0.732 | 0.726 |
| popularity (ref) | 0.0262 | 0.0256 | 0.785 | 0.775 |
| tr_gwas | **0.0641** | **0.0601** | 0.649 | 0.646 |
| tr_nongwas | 0.0472 | 0.0352 | 0.601 | 0.600 |
| pri_l2g_n | 0.0388 | 0.0372 | 0.747 | 0.743 |
| hz_gwas | 0.0355 | 0.0371 | 0.734 | 0.733 |
| ph_mean | 0.0344 | 0.0283 | 0.652 | 0.651 |
| pop_gwas_credible_sets | 0.0272 | 0.0252 | 0.735 | 0.733 |
| ph_mean_perm (neg) | 0.0044 | 0.0048 | 0.500 | 0.498 |
| pri_l2g_n_perm (neg) | 0.0086 | 0.0067 | 0.503 | 0.502 |
On forward the GWAS channel dominates (tr_gwas > tr_nongwas; pri_l2g_n alone beats popularity), the reverse of temporal, where
the Mendelian part of the donor rows carries more (tr_nongwas 0.073-0.092) and HPO matching is strong.

## 3. Increment tests (stacker, `--train cv`: dev = 5-fold cluster-grouped OOF, confirm = model trained on all dev)
Protocol identical to `modeng.stack.run` (same features, neg = 3000, 400 rounds, deg^-0.5 positive weights) but run through
`scripts/priors_h_strata.py` with LightGBM `num_threads` 4 (h_base) / 2-3 (others) because the machine was at load 120-150.
My own `hS_h_base` reference therefore replaces the pre-fix `stack_base_w05` (dev 0.1056 / conf 0.1319; mine 0.1026 / 0.1331).
The thread-count difference is covered by the negative control (same base, permuted channels).

Temporal (pre-chosen candidate = all three families on base; choice by dev):
| config | dev nuAP | conf nuAP | dev AP | conf AP | conf AUROC | conf R@100 | conf onnet nuAP | conf gw nuAP | conf ng nuAP |
|---|---|---|---|---|---|---|---|---|---|
| h_base | 0.1026 | 0.1331 | 0.0811 | 0.0921 | 0.798 | 0.256 | 0.168 | 0.050 | 0.186 |
| h_base_all (+pri,tr,ph) | **0.1222** | **0.1532** | **0.1054** | **0.1138** | 0.832 | 0.277 | 0.188 | 0.062 | 0.208 |
| h_base_ph | 0.1142 | 0.1526 | 0.0936 | 0.1091 | 0.809 | | | 0.056 | 0.213 |
| h_base_tr | 0.1011 | 0.1328 | 0.0819 | 0.0944 | 0.798 | | | 0.052 | 0.187 |
| h_base_pri | 0.1088 | 0.1287 | 0.0867 | 0.0955 | 0.826 | | | 0.059 | 0.173 |
| h_base_neg (perm controls) | 0.1043 | 0.1287 | 0.0826 | 0.0929 | 0.798 | 0.260 | 0.166 | 0.050 | 0.189 |
| h_bdv (base,donor,view) | 0.1046 | 0.1396 | 0.0940 | 0.1075 | 0.830 | | | 0.053 | 0.206 |
| h_bdv_all | 0.1106 | 0.1516 | 0.1020 | 0.1151 | 0.847 | | | 0.068 | 0.201 |
| h_base_ph_gw (exploratory, chosen after seeing the ablations: base + ph + pri_l2g_n, pri_l2g_max, hz_gwas) | 0.1144 | 0.1548 | 0.1023 | 0.1126 | | | | 0.067 | 0.216 |
Reference (different protocol, trained on `past`): stackP_bdv dev 0.1269 / conf 0.1506.

Paired cluster bootstrap (5,000; delta [95 % CI], P(delta <= 0)):
- h_base_all vs h_base: nuAP dev **+0.0196** [-0.0017, +0.0429] P 0.039; confirm **+0.0201** [+0.0048, +0.0383] P 0.002.
  AP dev +0.0243 [+0.0068, +0.0440]; confirm +0.0217 [+0.0096, +0.0364]. AUROC confirm +0.034 [+0.029, +0.040].
  Strata confirm: gw +0.0117 [+0.0001, +0.0251]; ng +0.0216 [-0.0042, +0.0543]. (Dev gw -0.010, n = 27.)
- h_base_ph vs h_base: dev +0.0116 [-0.0040, +0.0288]; confirm **+0.0196** [+0.0034, +0.0382] P 0.008; ng confirm +0.027 [+0.002, +0.059].
- h_base_tr vs h_base: dev -0.0015, confirm -0.0002 (nothing: the stacker already has transfer_jaccard; the split adds no new ranking).
- h_base_pri vs h_base: dev +0.0062 [-0.0056, +0.0200]; confirm -0.0044 [-0.011, +0.004]; AUROC +0.027-0.030 both splits; gw confirm
  +0.0085 [+0.0003, +0.018] but ng confirm -0.013 (priors are AUROC / bulk-ranking features, not top-of-list).
- h_bdv_all vs h_bdv: dev +0.0060 [-0.0086, +0.0216]; confirm **+0.0120** [+0.0033, +0.0204] P 0.004; gw confirm +0.0142 [+0.0049, +0.0243].
  (bdv already contains `don_hpo2101`, HPO-term donor rows, which carries part of the phenotype signal.)
- Negative control h_base_neg vs h_base: dev +0.0017 [-0.0042, +0.0077]; confirm -0.0044 [-0.0083, +0.0003]; AP +0.0016 / +0.0008. ~0, as required.

Forward (dev 842 / confirm 797; `--train cv`):
| config | dev nuAP | conf nuAP | dev AP | conf AP | dev AUROC | conf AUROC | conf R@100 | conf onnet nuAP | LightGBM threads |
|---|---|---|---|---|---|---|---|---|---|
| h_base | 0.1447 | 0.1373 | 0.0809 | 0.0770 | 0.823 | 0.812 | 0.206 | 0.158 | 4 |
| h_base_neg (perm controls) | 0.1502 | 0.1401 | 0.0843 | 0.0785 | 0.825 | 0.814 | 0.207 | 0.160 | 3 |
| h_base_all (+pri,tr,ph) | **0.1680** | **0.1513** | **0.0969** | **0.0873** | 0.852 | 0.844 | 0.230 | 0.170 | 3 |
Paired bootstrap (task = cluster on forward):
- h_base_all vs h_base: nuAP dev +0.0233 [+0.0180, +0.0286], confirm +0.0140 [+0.0097, +0.0185]; AP dev +0.0160 [+0.0123, +0.0196],
  confirm +0.0103 [+0.0076, +0.0131]; AUROC +0.029 / +0.032; R@100 +0.024 / +0.024 (all P < 0.001).
- **Negative control is NOT zero on forward**: h_base_neg vs h_base nuAP dev +0.0054 [+0.0034, +0.0075], confirm +0.0028 [+0.0011, +0.0046];
  AP +0.0034 / +0.0015. Cause not resolved: h_base ran with 4 LightGBM threads, neg/all with 3 (machine RAM / load), and two extra
  noise columns change feature_fraction sampling. Either way it is a protocol-noise floor of ~0.003-0.005, not a leak of my channels
  (the permuted channels carry no gene identity).
- The clean comparison (same thread count, same number of added columns) is **h_base_all vs h_base_neg**: nuAP dev +0.0178
  [+0.0127, +0.0230], confirm +0.0111 [+0.0068, +0.0158]; AP dev +0.0125 [+0.0089, +0.0161], confirm +0.0088 [+0.0061, +0.0117].
  Both AP and nuAP improve on both splits (lead rule F2-3 satisfied).
- Forward ablations (3 threads, vs h_base_neg; run one at a time after the RAM warning):
  | config | dev nuAP | conf nuAP | dev AP | conf AP | d nuAP dev / conf vs neg | d AP dev / conf vs neg |
  |---|---|---|---|---|---|---|
  | h_base_ph | 0.1528 | 0.1395 | 0.0864 | 0.0789 | +0.0026 [-0.0016,+0.0071] / -0.0007 | +0.0021 / +0.0004 |
  | h_base_pri | 0.1620 | 0.1495 | 0.0922 | 0.0849 | **+0.0118** [+0.0073,+0.0167] / **+0.0094** [+0.0057,+0.0133] | +0.0078 [+0.0049,+0.0109] / +0.0064 [+0.0039,+0.0089] |
  | h_base_tr | 0.1471 | 0.1370 | 0.0844 | 0.0789 | -0.0031 / -0.0032 | +0.0000 / +0.0004 |
  On forward the gain comes from the channel priors (GWAS-winner counts / gwas_credible_sets popularity), not from HPO matching;
  on temporal it is the reverse (HPO matching carries it, priors fail the AP-and-nuAP rule). Typed transfer adds nothing on either.
- Not run: bdv + mine on forward.

## 4. Verdict
**Keep** `ph_*` (temporal, Mendelian channel) and `pri_*`/`hz_*`/`pop_*` (forward, GWAS channel); **discard** `tr_*` (no increment anywhere). The two benchmarks reward different channels, so keep both families and let the stacker weight them.
- Temporal (temporal-clean): base + mine improves nuAP AND AP on dev (+0.020 / +0.024) and confirm (+0.020 / +0.022, CIs exclude 0 on confirm);
  over base,donor,view +0.006 / +0.008 dev, +0.012 / +0.008 confirm (confirm CIs exclude 0). Negative control ~0 (dev +0.002, confirm -0.004).
  h_base_all confirm nuAP 0.153 / AP 0.114 vs stackP_bdv 0.151 / 0.091 (different training protocol).
- Ablation (temporal): the gain is HPO phenotype matching (h_base_ph: confirm +0.020 nuAP / +0.017 AP, Mendelian stratum +0.027);
  channel priors improve AUROC (+0.03) and the GWAS stratum (+0.009 confirm) but cost nuAP on the Mendelian stratum, so they fail the
  AP-and-nuAP rule alone; typed transfer adds nothing on top of transfer_jaccard (although alone it ranks each stratum better than full transfer).
- Forward (temporal-clean): +0.018 / +0.011 nuAP and +0.013 / +0.009 AP over the matched noise control (dev / confirm), carried by the
  priors family (h_base_pri +0.012 / +0.009 nuAP, +0.008 / +0.006 AP); the
  forward negative control itself is +0.003-0.005 over h_base (thread-count / extra-column noise floor; must be checked by the lead
  before any claim < 0.01).
- Caveats: dev = 53 tasks on temporal (dev CIs touch 0 for nuAP); temporal channels are NaN on `past`, so no `--train past` result yet.

## 5. Runtime, what failed, notes
- Channel computation: temporal ~1-3 s per channel (evidence aggregation by duckdb 13 s once, cached in `data/cache/priors_h/`),
  forward 3-25 s per channel. Stacker runs 10-20 min each on temporal and 40-60+ min on forward at machine load 120-150.
- `results/priors_h_alone_temporal.tsv` was overwritten by another agent running my alone-script with its own channels; my numbers are in
  `results/priors_h_alone_temporal_h.tsv` (the script now writes `_h.tsv`).
- `tr_*` give no increment: splitting the donor rows does not change donor choice and LightGBM already sees transfer + priors.
  A version with channel-specific donor selection (Jaccard on the GWAS part only) was not tried.
- Not done: `--train past` (my channels are NaN on 19.04 episodes; HPO 2019 and 19.04 evidence would be needed), holdout.
- Not changed: no edits to channels.py / stack.py / tasks / metrics; the runner sets `ST.LGB["num_threads"]` at runtime only.

## 6. New ideas
1. Phenotype matching is the strongest new temporal-clean signal so far for the Mendelian channel (alone nuAP 0.069, close to transfer).
   Next: HPO ontology propagation (hp.obo 2021) + Resnik / best-match-average similarity instead of flat IDF cosine; add disease-side
   phenotypes directly (OT disease -> OMIM / Orphanet xref -> phenotype.hpoa) instead of deriving the profile from seeds; that would
   also work for diseases whose seeds lack HPO (22 % coverage limit).
2. Gate by channel: the stacker cannot exploit the seed channel mix well with 53 dev tasks; train on `past` with 19.04-era HPO
   (2019 HPO archive) and evidence 19.04 (not on disk) to get 1,786 episodes.
3. Mouse phenotypes (IMPC/MGI MP) matched through the same IDF profile machinery (critic EV-1), in the same channel.
4. pri_l2g_n lifts GWAS-stratum AUROC strongly but not nuAP: combine with locus-level information (a gene that is an L2G winner
   elsewhere AND sits near a sub-threshold signal for the disease) rather than as a global prior.
