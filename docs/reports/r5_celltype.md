# r5: cell-type witnesses CT-ACT (`ct_act`) and SC-3 cell-address donors (`ct_donor`)

## 0. Pre-registration (written and committed before any result of these channels was seen, 2026-11-12)
- Code: `src/modeng/ext/celltype.py` (docstring = exact definitions). Inputs: HPA v20.1 single-cell types (2020-11) and PoPS
  pre-2020 single-cell cluster DE t-statistics (2,583 columns); expression matching by HPA v20.1 tissue consensus. Temporal-clean on
  past, temporal and forward.
- Reference: modeng-v1 `stack_v3ptr_p0.35` (temporal, cv-trained, power 0.35, v1 channel list of `docs/preregistration_modeng_v1.md`).
- Main variant `v1_ct` = v1 channels + `ct_act,ct_donor`. Control `v1_ctperm` = v1 + `ct_act_perm,ct_donor_perm` (gene axis of the
  substrates permuted; same width). One variant only; diagnostics `ct_act_hpa`, `ct_act_pops` are looked at alone, not stacked
  unless counted here as an extra variant.
- Kill rule (pre-declared by the lead): if the increment of `v1_ct` over `v1_ctperm` is < +0.003 on BOTH AP and nuAP on dev AND on
  confirm -> discard. Keep only if AP or nuAP gains >= +0.003 over the control on dev and confirm, with no loss < -0.003 on the other,
  and degree-matched AP (`scripts/leader_endpoints.py`) not below v1. Paired cluster bootstrap (`modeng.compare`) on ap, nuap, sa_nuap.
- Alone: each channel on dev/confirm (AP, nuAP, AUROC) and in the zero-transfer-support stratum (`scripts/r1_kernels_b.py zero`),
  vs `vmean_hpa_sc` and the permuted controls. Secondary sweep-03 kill: alone not better than `vmean_hpa_sc` on zero-support AUROC.

## 1. Verdict (2026-11-12, ~1 h): DISCARD (pre-declared kill rule met)
`v1_ct` over the same-width gene-permuted control `v1_ctperm`: dev AP -0.008 [-0.015, -0.002], nuAP -0.006 [-0.014, +0.000];
confirm AP +0.001 [-0.002, +0.005], nuAP +0.002 [-0.002, +0.005]. Both < +0.003 on dev and confirm -> discard. Over modeng-v1 the
variant is flat-to-negative (confirm AP -0.003, nuAP -0.005; degree-matched AP 0.249 vs 0.252). Alone, `ct_act` also fails the
secondary sweep-03 kill (zero-support AUROC below `vmean_hpa_sc` on dev and confirm). One stack variant (+1 control) scored on confirm once.

## 2. What was built
- `src/modeng/ext/celltype.py`: substrates HPA v20.1 single-cell types (51 types, NX, file 2020-11-16; Z = log1p, column-centred,
  per-gene z) and PoPS pathway-naive `*_diffexprs_tstat_clusters*` columns (2,577 non-constant cluster DE t-stats from pre-2020
  human + mouse atlases, column z; 17,605 genes covered). Analytic expression-matched competitive null (10 bins of HPA v20.1 tissue
  mean log1p NX + missing bin). Channels `ct_act`, `ct_donor`, diagnostics `ct_act_hpa`, `ct_act_pops`, controls `ct_act_perm`,
  `ct_donor_perm` (gene axis of substrate + bins permuted, seed 7). Masking in `ct_donor`: addresses of rows with masked pairs are
  recomputed from the masked rows; own row and fam_rows excluded; masked pairs subtracted exactly, scores quantised to 1e-5.
- Sanity: enrichment is biologically sensible (cholelithiasis -> hepatocytes/cholangiocytes; refractive error -> Muller glia/bipolar;
  intestinal motility -> smooth muscle; mitochondrial complex deficiency -> cardiomyocytes/proximal tubule). HPA: 52/53 dev tasks with
  >= 1 type at e > 2 (median 5 types; permuted median 1); PoPS median 152 of 2,577 clusters (permuted 42).
- Scripts: `scripts/r5_celltype_alone.py` (-> `results/r5ct/alone.tsv`, `zero.tsv`), `scripts/r5_celltype_compare.py`
  (-> `results/r5ct/compare.tsv`), `results/r5ct/run.sh` (stack commands, logs `results/r5ct/*.out`),
  `scripts/leader_endpoints.py stack_v1_ct` (-> `results/review/leader_endpoints_stack_v1_ct.tsv`). Scores saved (`MODENG_SAVE_SCORES=1`).
- Works on past / temporal / forward (smoke-tested: forward dev 842 tasks, past 100 tasks). Runtime: channels ~2 min, each stack ~6 min.

## 3. Alone (temporal; all targets | zero-transfer-support targets, full ranking with supported targets removed)
| split | channel | AP | nuAP | AUROC | R@100 | z_AP | z_AUROC | z_R@100 |
|---|---|---|---|---|---|---|---|---|
| dev | transfer_jaccard | 0.123 | 0.096 | 0.682 | 0.257 | 0.001 | 0.474 | 0.000 |
| dev | vmean_hpa_sc | 0.008 | 0.006 | 0.631 | 0.034 | 0.003 | **0.602** | 0.012 |
| dev | ct_act | 0.007 | 0.007 | 0.622 | 0.038 | 0.004 | 0.580 | 0.023 |
| dev | ct_act_hpa / _pops | 0.005 / 0.009 | 0.005 / 0.008 | 0.604 / 0.598 | 0.017 / 0.046 | 0.002 / 0.005 | 0.569 / 0.574 | 0.009 / 0.033 |
| dev | ct_act_perm | 0.003 | 0.003 | 0.505 | 0.007 | 0.002 | 0.523 | 0.003 |
| dev | ct_donor | 0.062 | 0.044 | 0.697 | 0.176 | 0.004 | 0.550 | 0.026 |
| dev | ct_donor_perm | 0.066 | 0.054 | 0.668 | 0.180 | 0.004 | 0.527 | 0.021 |
| confirm | transfer_jaccard | 0.119 | 0.104 | 0.705 | 0.283 | 0.001 | 0.489 | 0.000 |
| confirm | vmean_hpa_sc | 0.009 | 0.008 | 0.636 | 0.036 | 0.004 | **0.612** | 0.021 |
| confirm | ct_act | 0.008 | 0.009 | 0.623 | 0.039 | 0.003 | 0.592 | 0.028 |
| confirm | ct_act_perm | 0.002 | 0.003 | 0.500 | 0.007 | 0.002 | 0.507 | 0.008 |
| confirm | ct_donor | 0.064 | 0.056 | 0.717 | 0.181 | 0.005 | 0.577 | 0.023 |
| confirm | ct_donor_perm | 0.055 | 0.049 | 0.676 | 0.172 | 0.002 | 0.539 | 0.011 |
Reading: `ct_act` carries real signal (2.5-3x its permuted control) but no more than plain seed similarity in the (modern) HPA view, and
less on zero-support AUROC. `ct_donor` is a weaker transfer (half of transfer_jaccard's AP); much of it is generic donor/popularity
signal (the permuted-address donor reaches AP 0.066 on dev). Address donors double zero-support z_AP over the control on confirm but at
an absolute level of 0.005.

## 4. Increment (temporal, cv-trained, power 0.35; paired cluster bootstrap 10,000, 95 % CI)
| comparison | split | AP | nuAP | sa_nuAP | AUROC | R@100 |
|---|---|---|---|---|---|---|
| v1_ct vs v1 | dev | -0.002 [-0.008,+0.003] | +0.000 [-0.006,+0.007] | +0.005 [-0.004,+0.015] | -0.002 | +0.005 |
| v1_ctperm vs v1 | dev | +0.005 [-0.002,+0.014] | +0.007 [-0.002,+0.016] | +0.004 | -0.001 | +0.002 |
| **v1_ct vs v1_ctperm** | dev | **-0.008 [-0.015,-0.002]** | **-0.006 [-0.014,+0.000]** | +0.000 | -0.000 | +0.003 |
| v1_ct vs v1 | confirm | -0.003 [-0.008,+0.002] | -0.005 [-0.011,+0.001] | -0.005 [-0.014,+0.003] | +0.003 [+0.001,+0.005] | +0.004 |
| v1_ctperm vs v1 | confirm | -0.005 | -0.007 [-0.013,-0.000] | -0.006 | +0.001 | +0.005 |
| **v1_ct vs v1_ctperm** | confirm | **+0.001 [-0.002,+0.005]** | **+0.002 [-0.002,+0.005]** | +0.001 | +0.001 | -0.001 |
Degree-matched AP (`leader_endpoints`): v1_ct 0.255 / 0.249 (dev / confirm) vs v1 0.256 / 0.252. Feature gain: `r_ct_act` rank 26, `ct_donor` not in the top 25.
Control-to-control spread (+-0.005 to 0.007 vs v1) is as large as any effect: adding two columns of any kind moves the stacker by this much.

## 5. Leakage
Temporal-clean on past, temporal and forward: HPA v20.1 (2020-11), PoPS single-cell features from <= 2020 atlases (matrix release
2023-24, no labels), only task seeds and the masked context. No label-process channel added.

## 6. Why it fails / what next
- The cell-type-of-action signal exists (sensible enrichments, 3x over permuted) but the genes it nominates are cell-type markers, and
  OT arrivals are not concentrated among markers of the disease's cell type beyond what seed similarity in expression space already gives
  (`vmean_*`). The stacker already has expression views; CT-ACT reorders that information without new information.
- Address donors reach zero-overlap diseases, but cell-type address is a coarse disease similarity (many diseases share "hepatocyte" or
  "neuron"), so the donors' genes are mostly generic; HPO/ontology donors already cover this stratum better.
- Not tried (would need a new pre-registration): strict-a-only use with MAGMA gene z instead of seeds (seismic-style, sweep 3.5);
  finer forward-only atlases (TS v2). Given this result, low priority.
