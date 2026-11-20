# modeng in one page (closed 2026-11-12)

## How much better
| comparison | gain | evidence |
|---|---|---|
| vs raw transfer (strongest simple baseline), 35 untouched diseases (`docs/reports/r7_wide_confirm.md`) | AP 0.091 -> 0.118 (+30 %), nuAP 0.084 -> 0.126 (+50 %) | **confirmatory** (pre-registered, one look); narrow, not under Bonferroni over 4 arms |
| vs modet headline `modet-U-cov-Lb`, modet's candidate set, all arrivals, 138 temporal confirm diseases | AP 0.096 -> 0.160 (+65 %), nuAP 0.173 -> 0.203 (+18 %), degree-matched AP +0.08 | development (confirm looked at repeatedly; optimistic); prospective twin keeps ~70 % |
| vs modet headline, strict-a (new genetic discoveries 2022+) | tie | all variants |
| modules at matched size (`r4_modules.md`) | ~60 % more later-confirmed genes than modet, ~40 % more than transfer | descriptive |
Realistic summary: roughly 30-60 % better at ranking future disease genes overall; no better on genuinely new genetic discoveries.

## How
One small LightGBM combiner (the standard part) over evidence modet did not have: learned disease-disease explanation (ridge, sparse
joint fit), PMID-date-cut literature sets (PubTator3), HPO phenotype matching, gene track-record / GWAS-catalog-lag priors (label-process,
left out of the confirmed model), many pre-2021 gene-set families and embeddings, training on 712 diseases instead of 53, mild
down-weighting of hub genes (deg^-0.35). Deep models (seed-set transformer, KG embeddings) added nothing: the gain is the evidence.

## The second result (the insight)
About 40 % of what Open Targets calls "new" disease genes is old evidence attached later, another share is ontology re-mapping; only about a
third is genuinely new data. Models can win benchmarks by forecasting the database (weak evidence one year earlier more than doubles forward
nuAP, 0.14 -> 0.31). On genuinely new biology every method is weak (modeng AP 0.04-0.06), and 54 % of future genes have no pre-seed trace
at all. Many reported gains in the field may therefore measure bookkeeping, not discovery -> successor project `/home/ph/code/nebb`.

## Use it
`python -m modeng.predict --disease <id> --out ranked.tsv [--module m.tsv]` (`PYTHONPATH=src`); model + card `/mnt/msadb/modeng/models/modeng_v2_2104/`;
config `configs/modeng_v2.yaml`; calibrated P uses shrink 0.8 (carried over from modeng-v1); this model's dev cross-fit shrink is ~0.89, so P is conservative and approximate (model card).

## Where to read more
`docs/WRITEUP.md` (paper draft), `docs/reports/FINAL_overnight_2026-11-12.md` (full log of results, sections 1-11), `docs/PROCESS_v2.md` (lessons),
`docs/insights.md`, `docs/invalidations.md`, `docs/ideas.md` (~110 ideas with status), `docs/reviews/` (Fable reviews), `docs/reports/` (all experiments).
