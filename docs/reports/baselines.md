# Baselines (harness v1, 2026-11-12)

Command: `PYTHONPATH=src python -m modeng.bench --method all --bench all --split both --jobs 12` (all runs < 15 s each, CPU).
Per-task metrics: `results/<bench>/<split>/<method>.parquet`; summary rows: `results/leaderboard.tsv`.
Metrics are task means. Main columns use the all-gene universe (19,255 genes minus seeds/unknown); `onnet_*` = network genes only;
`sa_ap` = temporal strict-a endpoint.

| bench    | split   | method           |   n |     ap |   nuap |   auroc |   r100 |   onnet_ap |   onnet_nuap |    sa_ap |
|:---------|:--------|:-----------------|----:|-------:|-------:|--------:|-------:|-----------:|-------------:|---------:|
| forward  | confirm | random           | 797 | 0.0039 | 0.0043 |  0.4929 | 0.0056 |     0.0045 |       0.0046 | nan      |
| forward  | confirm | degree           | 797 | 0.0046 | 0.0022 |  0.5688 | 0.0097 |     0.0047 |       0.0023 | nan      |
| forward  | confirm | popularity       | 797 | 0.0285 | 0.0256 |  0.7745 | 0.0769 |     0.0301 |       0.0327 | nan      |
| forward  | confirm | rwr_string       | 797 | 0.0127 | 0.0068 |  0.6092 | 0.0587 |     0.0144 |       0.0116 | nan      |
| forward  | confirm | rwr_union        | 797 | 0.0117 | 0.0059 |  0.6061 | 0.0542 |     0.0133 |       0.0095 | nan      |
| forward  | confirm | transfer_onto    | 797 | 0.0366 | 0.0373 |  0.7550 | 0.1011 |     0.0409 |       0.0453 | nan      |
| forward  | confirm | transfer_jaccard | 797 | 0.0849 | 0.0831 |  0.7257 | 0.2342 |     0.0908 |       0.0984 | nan      |
| forward  | dev     | random           | 842 | 0.0039 | 0.0044 |  0.5020 | 0.0065 |     0.0045 |       0.0053 | nan      |
| forward  | dev     | degree           | 842 | 0.0048 | 0.0023 |  0.5675 | 0.0097 |     0.0049 |       0.0024 | nan      |
| forward  | dev     | popularity       | 842 | 0.0301 | 0.0262 |  0.7849 | 0.0773 |     0.0320 |       0.0324 | nan      |
| forward  | dev     | rwr_string       | 842 | 0.0137 | 0.0076 |  0.6096 | 0.0584 |     0.0155 |       0.0114 | nan      |
| forward  | dev     | rwr_union        | 842 | 0.0114 | 0.0054 |  0.6038 | 0.0515 |     0.0130 |       0.0081 | nan      |
| forward  | dev     | transfer_onto    | 842 | 0.0386 | 0.0419 |  0.7618 | 0.1119 |     0.0409 |       0.0455 | nan      |
| forward  | dev     | transfer_jaccard | 842 | 0.0936 | 0.0964 |  0.7323 | 0.2471 |     0.0975 |       0.1052 | nan      |
| holdout  | confirm | random           | 297 | 0.0015 | 0.0018 |  0.5035 | 0.0040 |     0.0018 |       0.0024 | nan      |
| holdout  | confirm | degree           | 297 | 0.0026 | 0.0009 |  0.5878 | 0.0129 |     0.0028 |       0.0009 | nan      |
| holdout  | confirm | popularity       | 297 | 0.0316 | 0.0286 |  0.8454 | 0.1096 |     0.0340 |       0.0376 | nan      |
| holdout  | confirm | rwr_string       | 297 | 0.0221 | 0.0132 |  0.6560 | 0.1284 |     0.0252 |       0.0199 | nan      |
| holdout  | confirm | rwr_union        | 297 | 0.0171 | 0.0076 |  0.6520 | 0.1059 |     0.0208 |       0.0132 | nan      |
| holdout  | confirm | transfer_onto    | 297 | 0.1080 | 0.1083 |  0.8636 | 0.2247 |     0.1117 |       0.1158 | nan      |
| holdout  | confirm | transfer_jaccard | 297 | 0.2180 | 0.1919 |  0.8988 | 0.5186 |     0.2297 |       0.2144 | nan      |
| holdout  | dev     | random           | 303 | 0.0016 | 0.0027 |  0.4968 | 0.0045 |     0.0020 |       0.0034 | nan      |
| holdout  | dev     | degree           | 303 | 0.0024 | 0.0009 |  0.5915 | 0.0113 |     0.0026 |       0.0011 | nan      |
| holdout  | dev     | popularity       | 303 | 0.0307 | 0.0309 |  0.8451 | 0.1128 |     0.0347 |       0.0387 | nan      |
| holdout  | dev     | rwr_string       | 303 | 0.0244 | 0.0151 |  0.6566 | 0.1253 |     0.0280 |       0.0244 | nan      |
| holdout  | dev     | rwr_union        | 303 | 0.0182 | 0.0109 |  0.6534 | 0.1032 |     0.0218 |       0.0178 | nan      |
| holdout  | dev     | transfer_onto    | 303 | 0.1113 | 0.1170 |  0.8789 | 0.2314 |     0.1158 |       0.1200 | nan      |
| holdout  | dev     | transfer_jaccard | 303 | 0.2059 | 0.1981 |  0.9049 | 0.5080 |     0.2160 |       0.2103 | nan      |
| temporal | confirm | random           | 138 | 0.0022 | 0.0026 |  0.5090 | 0.0044 |     0.0027 |       0.0031 |   0.0012 |
| temporal | confirm | degree           | 138 | 0.0035 | 0.0013 |  0.6107 | 0.0119 |     0.0037 |       0.0014 |   0.0023 |
| temporal | confirm | popularity       | 138 | 0.0192 | 0.0121 |  0.7873 | 0.0969 |     0.0216 |       0.0136 |   0.0215 |
| temporal | confirm | rwr_string       | 138 | 0.0139 | 0.0061 |  0.6530 | 0.0722 |     0.0165 |       0.0126 |   0.0066 |
| temporal | confirm | rwr_union        | 138 | 0.0079 | 0.0032 |  0.6527 | 0.0411 |     0.0092 |       0.0049 |   0.0054 |
| temporal | confirm | transfer_onto    | 138 | 0.0587 | 0.0553 |  0.6173 | 0.1704 |     0.0631 |       0.0678 |   0.0200 |
| temporal | confirm | transfer_jaccard | 138 | 0.1189 | 0.1042 |  0.7049 | 0.2826 |     0.1277 |       0.1276 |   0.0579 |
| temporal | confirm | modet_headline   | 138 | 0.0812 | 0.0797 |  0.7565 | 0.2256 |     0.0964 |       0.1727 |   0.0580 |
| temporal | dev     | random           |  53 | 0.0023 | 0.0031 |  0.5015 | 0.0066 |     0.0029 |       0.0048 |   0.0013 |
| temporal | dev     | degree           |  53 | 0.0056 | 0.0014 |  0.6183 | 0.0139 |     0.0060 |       0.0016 |   0.0021 |
| temporal | dev     | popularity       |  53 | 0.0202 | 0.0111 |  0.7766 | 0.0902 |     0.0245 |       0.0177 |   0.0213 |
| temporal | dev     | rwr_string       |  53 | 0.0140 | 0.0050 |  0.6551 | 0.0649 |     0.0167 |       0.0106 |   0.0087 |
| temporal | dev     | rwr_union        |  53 | 0.0103 | 0.0030 |  0.6476 | 0.0400 |     0.0119 |       0.0051 |   0.0039 |
| temporal | dev     | transfer_onto    |  53 | 0.0503 | 0.0366 |  0.5805 | 0.1505 |     0.0529 |       0.0528 |   0.0125 |
| temporal | dev     | transfer_jaccard |  53 | 0.1229 | 0.0961 |  0.6825 | 0.2567 |     0.1343 |       0.1329 |   0.0307 |
| temporal | dev     | modet_headline   |  53 | 0.1067 | 0.0819 |  0.7544 | 0.2298 |     0.1271 |       0.1901 |   0.0373 |

## Notes
1. **Parity with modet (verified exactly).** `scripts/check_modet_parity.py` re-evaluates on modet's own covered panel
   (`panel_t0_string115_phys_notm_400`, ranking universe = T0 STRING nodes, nuAP degree = T0 STRING degree):
   modet_headline AP 0.1049 / nuAP 0.1775 / R@100 0.2723 on devall-191, per-disease identical to modet's
   `ucov_parity_string_s0` metrics (max |diff| 4e-16). transfer_jaccard there: AP 0.124 / nuAP 0.121 (~ modet's raw tr_max 0.122).
   The headline's scores are out-of-sample: modet trains on past transitions (OT 19.04 -> 21.04, `temporal.fit_core`
   training pool), scores panel rows only, sealed fold never scored.
2. **Why the headline looks weaker on the main (all-gene) metric:** ~18 % of temporal targets are off the T0 STRING network;
   the headline (network-only model) gives them the lowest score, and in nuAP they carry weight 1 (degree 0), i.e. the
   largest weight - so nuAP drops from 0.18 to 0.08. On `onnet` the headline is best on nuAP (0.19 dev / 0.17 confirm vs
   transfer 0.13) while transfer_jaccard is best on AP (0.134 / 0.128 vs 0.127 / 0.096). Compare with modet on `onnet_*`;
   use the all-gene columns for claims about the full gene universe. Paired bootstrap (cluster = modet fold cluster, temporal
   confirm, all-gene): transfer_jaccard - headline AP +0.038 [0.020, 0.058], nuAP +0.025 [0.012, 0.040].
3. **Transfer is the strongest simple rung everywhere** (temporal 0.12, forward 0.09, holdout 0.21 AP); popularity is a
   real baseline (AUROC 0.78-0.85) and RWR on PPI is weak (AP 0.01-0.02; union network worse than STRING).
4. **Leakage signals checked:** holdout transfer AP 0.21 (not near 1) after masking; an earlier blanket ancestor mask produced
   an anti-informative transfer_onto (AUROC 0.29) = a mask tell, fixed by bottom-up ancestor re-propagation (see benchmarks.md).
   Remaining known caveats: (a) holdout is transductive by nature: other diseases mapped from the same GWAS study (not
   ontology-related, Jaccard < 0.5) still carry the hidden gene; (b) forward: 71 % of new targets were already in some
   ancestor's 25.09 set via a sibling - legitimate past knowledge, but it makes forward partly a "sibling copy" task;
   (c) forward/holdout tasks are hierarchically correlated (ancestors/descendants are separate tasks; bootstrap cluster =
   task, so CIs there are too narrow); (d) temporal context masks only own row + descendants (sensitivity in benchmarks.md).
