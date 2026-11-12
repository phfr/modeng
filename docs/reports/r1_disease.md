# R1 implementer C: better disease-to-disease transfer (IR-6, CI-04, ML-8, EASE/SLIM ridge)

## Pre-registration (written and committed 2026-11-12 ~04:30, before any result of these channels was seen)
Fixed parameters (no tuning on dev or confirm):
- `dc_trend` (IR-6): donor matrix `L + beta * Delta`, `Delta` = context pairs absent from the previous release
  (temporal 21.04 vs 19.04; forward 25.09 vs 21.04), **beta = 1**, Jaccard similarity on the context, top-50, exact
  masking correction + quantisation. `dc_delta` = transfer with `Delta` alone (stacker column). NaN on `past`/holdout.
- `dc_directed` (CI-04): `w(a->d) = jaccard(a, d) * exp(theta * asym)`, `asym = log((n_{a->d}+1)/(n_{d->a}+1))`,
  `n_{a->d}` = genes of a in the previous release that are new in d in the current context (masked), **theta = 1**,
  clipped to [-3, 3], no branch shrinkage. NaN on `past`/holdout.
- `dc_learned` (ML-8 without text): LightGBM regression of donor precision (share of donor's non-seed genes that
  arrive) on pair features (Jaccard, intersection, sizes, overlap ratios, ontology relation, Jaccard rank), trained
  on `past` episodes (19.04 -> 21.04), 2-fold cluster cross-fit on `past` itself; transfer over top-50 by prediction.
- `dc_ridge` / `dc_ridge_pos` (EASE/SLIM-style disease regression): seed vector regressed on unit-norm donor rows
  with **lambda = 1**, all masked rows (fam rows + any row with a masked pair) removed as donors (exact masking);
  score = donors^T w (`dc_ridge`), or with w clipped >= 0 and top-50 (`dc_ridge_pos`).
Claim-deciding comparisons: (1) each channel alone vs `transfer_jaccard`, paired cluster bootstrap (`modeng.compare`),
nuAP and AP, temporal dev+confirm and forward dev+confirm. (2) increment `base + channel` vs `base`: temporal
`--train past` (stackP; only for channels defined on past: dc_learned, dc_ridge, dc_ridge_pos) and `--train cv` on
forward (all channels) and temporal cv (IR-6/CI-04 only, since they are NaN on past).
Keep rule: increment >= +0.005 nuAP on dev AND confirm with the same sign, P(delta<=0) < 0.1 on at least one.
Negative control: best channel with genes permuted by a fixed random permutation -> increment ~0.

## Results
(pending)
