# R1 red-team / pre-mortem of the round-1 headline claim (Fable reviewer, 2026-11-12 06:04-06:25)

Claim under review: "A LightGBM stacker trained on past OT episodes over transfer + virtual-donor + view channels plus
cs_donor, dA_pt and dc_ridge beats raw transfer and matches/beats modet's headline on the temporal benchmark: confirm nuAP
~0.17-0.18 vs 0.104 for transfer, AP ~0.11 vs 0.119."

## Verdict: HOLDS WITH CAVEATS (two parts of the sentence must be changed)

- **No leak, no bug and no selection effect was found.** The numbers reproduce exactly and survive every input-date ablation.
- **"0.175 vs 0.104 nuAP" overstates the model gain about twofold.** Most of that gap comes from the deg^-0.5 weighting of the
  training positives, which any scorer can copy. The degree-neutral gain over transfer is real but modest.
- **"Matches/beats modet's headline" is not supported on modet's own primary endpoint** (strict-a, network candidates, nuAP):
  0.093 vs 0.109. The advantage over the headline sits in arrivals with old evidence and in off-network genes.

All checks are on the model `stackP_bdv_cs_pt` (the 0.175 row). `dc_ridge` was never combined with cs_donor + dA_pt, so
"~0.17-0.18" rests on one run of one model with one seed.

Scripts: `scripts/review_{scores,pt19,bruteforce,analyze}.py`. Outputs: `results/review/` (score matrices `S_*.npy`,
`endpoints_*.parquet`, `endpoints_summary.tsv`, logs). Nothing was written to the leaderboard or the channel cache.
All intervals are 95 % paired cluster bootstrap (10,000 draws, 41 confirm clusters). Confirm = 138 diseases, dev = 53.

## Explanation 1 (most likely): the nuAP gap is mostly degree tilt, and the top of the list is not better

Check: retrain the same model with positive-weight power 0.5 / 0.25 / 0 (same channels, same past episodes), and score a
degree-matched endpoint (`degbin_ap` = AP within the target's degree bin: off-network + 10 deciles, averaged over positives).

| confirm | AP | nuAP | degree-matched AP | R@50 | R@100 | hub AP | onnet nuAP |
|---|---|---|---|---|---|---|---|
| transfer_jaccard | 0.119 | 0.104 | 0.191 | 0.222 | 0.283 | 0.102 | 0.128 |
| stack, power 0.5 (the claim) | 0.111 | 0.175 | 0.224 | 0.219 | 0.292 | 0.051 | 0.195 |
| stack, power 0.25 | 0.127 | 0.151 | 0.223 | 0.246 | 0.317 | 0.088 | 0.164 |
| stack, power 0 | 0.129 | 0.114 | 0.225 | 0.246 | 0.323 | 0.117 | 0.125 |

- The degree-matched AP is the same for all three powers (0.223-0.225). The weighting adds no information; it only moves
  rank mass from hubs to low-degree genes. Of the +0.071 nuAP over transfer, +0.062 disappears at power 0
  (+0.009 [-0.013, +0.035]).
- **Degree-matched endpoint survives:** +0.033 [+0.012, +0.053] on confirm, +0.025 [+0.004, +0.044] on dev (about +17 % relative).
- Top of the list at power 0.5 (the claimed model) vs transfer, confirm: AP -0.008 [-0.025, +0.009], R@50 -0.002, R@100 +0.009
  [-0.012, +0.031], P@20 -0.007, nDCG@100 -0.002, median first hit 3 vs 3.5. Hub targets (above-median degree) lose half:
  hub AP -0.052 [-0.070, -0.034], hub R@100 -0.068 [-0.106, -0.032]. Non-hub R@100 +0.036 [+0.018, +0.058].
  Diseases with < 20 seeds (n = 40): AP -0.047.
- A user who reads the top 50-100 gets nothing extra from the claimed model. At power 0.25 the same channels give
  R@100 +0.034 [+0.014, +0.055], AP +0.008 [-0.007, +0.026] and nuAP +0.047 [+0.028, +0.073]: this variant, not power 0.5,
  is the one that is better than transfer on both metrics. (It was chosen here after seeing confirm: confirm it on a fresh fold.)
- Fair nuAP baseline: transfer with a degree tilt (`evo_c7_jacdeg`, `evo_transfer`) reaches nuAP 0.134-0.140. Against these the
  stack is +0.036 to +0.042 nuAP (CIs exclude 0) and +0.044 AP [+0.030, +0.061]. That is the honest size of the nuAP gain.

## Explanation 2: the comparison with modet's headline fails on modet's primary endpoint

Check: per-disease paired comparison against `modet_headline` on the network candidate set, split by arrival class
(strict-a = evidence first dated 2022 or later and persistent; 71 confirm / 28 dev diseases with >= 5 such targets).
Confirm arrivals: A 39 %, B (evidence <= 2020, ingested later) 34 %, X (re-scored) 21 %, U 6 %.

| confirm, stack minus modet headline | stack | headline | delta [95 % CI] | win rate |
|---|---|---|---|---|
| onnet nuAP, all arrivals | 0.195 | 0.173 | +0.023 [+0.002, +0.048] | 0.54 |
| onnet AP, all arrivals | 0.120 | 0.096 | +0.023 [+0.008, +0.041] | 0.53 |
| **strict-a, onnet nuAP (modet's primary cell)** | 0.093 | 0.109 | **-0.016 [-0.038, +0.004]** | 0.37 |
| strict-a, onnet AP | 0.051 | 0.058 | -0.007 [-0.018, +0.002] | 0.32 |
| strict-a, all-gene nuAP | 0.069 | 0.109 | -0.040 [-0.066, -0.018] | 0.23 |
| non-A arrivals, onnet nuAP | 0.126 | 0.111 | +0.015 [-0.014, +0.041] | 0.63 |
| non-A arrivals, all-gene nuAP | 0.100 | 0.028 | +0.072 [+0.051, +0.092] | 0.92 |

- Dev: onnet nuAP +0.003 [-0.033, +0.038]; strict-a onnet nuAP -0.005 [-0.039, +0.029]. No advantage on dev at all.
- The only significant win over the headline is all-arrival onnet nuAP on confirm, and its lower bound is +0.002.
- The large all-gene wins are off-network targets that the headline cannot rank (already recorded in `invalidations.md`).
- Strict-a vs transfer: nuAP +0.017 [+0.007, +0.029] (onnet +0.037), but AP -0.019 [-0.034, -0.005]. The claimed model is
  worse than raw transfer in AP on genuinely new evidence.
- Reading: the stacker is trained on 19.04 -> 21.04 arrivals, which are mostly ingestion of existing knowledge, and its best
  new channels (HPO 2021-01, PubTator <= 2021-03) carry exactly that knowledge. It forecasts what OT attaches next better than
  it forecasts new evidence. This matches the round-1 backlog finding.

## Explanation 3 (least likely, checked and not found): leakage, masking error, selection on confirm

| suspected route | check | result |
|---|---|---|
| stale or wrong channel cache | brute-force recompute with `B.context_for` on 10 confirm and 4 past tasks | transfer_jaccard max diff 5e-6 (quantisation); cs_donor, dA_pt, don_hpo2101, vmax_esmc identical |
| wrong PubTator cut | recompute with the neighbouring cut | temporal cache = a_pt21, past cache = a_pt19 (other cuts differ, corr 0.87-0.89) |
| PMID date vs publication date | 2-year lag: temporal scored with the 2019-03 cut | confirm AP 0.110, nuAP 0.174, onnet nuAP 0.195 (unchanged from 0.111 / 0.175 / 0.195) |
| modern inputs (HPA, GTEx, Perturb-seq views and their k-means donors, 20 channels) | drop and retrain | confirm AP 0.109, nuAP 0.173, onnet nuAP 0.196 (unchanged) |
| ontology-rewiring targets | 326 of 6,324 targets carry a masked pair, and all 326 are still in an unmasked ancestor row; remove them from the ranking | nuAP gain over transfer +0.065 [+0.046, +0.092], AP -0.005 (unchanged pattern) |
| sealed fold | ids in tasks and context rows | 0 in temporal and past tasks, 0 in both contexts |
| selection on confirm | leaderboard count; null spread | 69 methods scored on temporal confirm (59 stackers, 16 `stackP_bdv_*`). Six null-like variants: SD of the nuAP increment 0.0017. The cs_donor + dA_pt increment is +0.025 on confirm and +0.026 on dev; dev and confirm increments correlate 0.97 over the 15 variants. Winner's curse is negligible for the increment |
| concentration | per-disease deltas | nuAP gain positive in 86 % of diseases and 40 of 41 clusters; top 5 diseases carry 31 %; without the top 10 still +0.042 |

PMID cut check (PubMed, accessed 2026-11-12): PMID 33750000 is dated 2021-03-22 ([DOI](https://doi.org/10.1002/ajmg.a.62165));
30900000 is dated 2019-03-21 ([DOI](https://doi.org/10.1007/s00221-019-05520-3)); the neighbour 33749990 carries an issue date
of 2021-03-31 ([DOI](https://doi.org/10.1111/apha.13650)), so records near the cut can have print dates a few days to weeks
later. The 2-year-lag run makes this irrelevant.

Residual items, not tested:
- STRING 11.5 (degree, RWR, the nuAP weights) was released in August 2021, after the 21.04 seeds. Inherited from modet.
- 58 of the 900 training episodes are the evaluation diseases two years earlier. Legitimate (all <= T0), but worth stating.
- `dc_ridge` drops donor rows that carry a masked pair, which is a target-dependent selection. It can only hurt targets, so it is not a leak.
- The summary table lists the headline only as "onnet ~0.17"; the strict-a endpoint is missing from every round-1 report.
- Doc timestamps in `insights.md` / `invalidations.md` (01:20-02:40) do not match git and file times (fixes committed 00:15 and 00:26).

## What may be claimed (exact sentences)

1. "On the temporal benchmark (confirm, 138 diseases, all-gene universe) the past-trained stacker with cs_donor and dated
   literature donors raises nuAP from 0.104 (raw transfer) to 0.175 and from 0.134-0.140 (degree-tilted transfer) by +0.04
   [+0.02, +0.06], at unchanged AP against raw transfer (0.111 vs 0.119, -0.008 [-0.025, +0.009]); a degree-matched AP rises
   from 0.191 to 0.224 (+0.033 [+0.012, +0.053])."
2. "The gain comes from low-degree targets; recall of hub targets halves at this setting, and recall@50 / P@20 are unchanged.
   With positive-weight power 0.25 the same channels give AP 0.127, nuAP 0.151 and recall@100 0.317 vs 0.283 (to be confirmed
   on a fresh fold)."
3. "On modet's candidate set the stacker is comparable to modet's headline: all-arrival nuAP 0.195 vs 0.173 on confirm
   (+0.023 [+0.002, +0.048]; dev +0.003), but on modet's primary strict-a endpoint it is not better (nuAP 0.093 vs 0.109,
   -0.016 [-0.038, +0.004])."
4. "No input-date leak was found: a 2-year literature lag and removal of all post-2021 views leave the result unchanged."

Not claimable: "beats modet's headline"; "beats transfer" without naming nuAP or the degree-matched endpoint; any statement
that the claimed model is better for the top of the list; "~0.18".

## Recommended before round 2 builds on this
- Report strict-a (onnet) and the degree-matched AP for every stacker row, next to AP and nuAP.
- Use degree-tilted transfer as the nuAP baseline.
- Fix the positive-weight power by pre-registration (0.25 looks better than 0.5 on both metrics) and confirm on a fresh fold.
- Run cs_donor + dA_pt + dc_ridge together with 3 seeds before quoting a range.
