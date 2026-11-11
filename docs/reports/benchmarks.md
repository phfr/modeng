# modeng benchmarks (harness v1, 2026-11-12)

Built by `src/modeng/tasks.py` (definitions and leakage policy in its module docstring; cached to `data/cache/bench_*.pkl`).
Gene universe: 19,255 HGNC protein-coding genes. Labels: Open Targets genetic gene sets (modet `groundtruth/ot_*_sets.parquet`;
`ot_t0_sets` = 21.04, `ot_current_sets` = 25.09, `ot_current_sets_2609` = 26.09, `ot_t19_sets` = 19.04; verified in
`modet_data/groundtruth.py` / `production.py`). Ontology: OT `disease_<release>` tables, relatives = union over 21.04/25.09/26.09.

| bench    | split   |   n_tasks |   median_seeds |   median_targets |   median_fam_rows |   n_context_rows | network                    | leakage                                               |
|:---------|:--------|----------:|---------------:|-----------------:|------------------:|-----------------:|:---------------------------|:------------------------------------------------------|
| temporal | dev     |        53 |           38   |             23   |                 3 |             3700 | t0_string115_phys_notm_400 | temporal-clean                                        |
| temporal | confirm |       138 |           27   |             21   |                 3 |             3700 | t0_string115_phys_notm_400 | temporal-clean                                        |
| forward  | dev     |       842 |           39.5 |             14.5 |                 2 |             6164 | string_phys_notm_400       | temporal-clean (labels 25.09->26.09; inputs <= 25.09) |
| forward  | confirm |       797 |           37   |             13   |                 2 |             6164 | string_phys_notm_400       | temporal-clean (labels 25.09->26.09; inputs <= 25.09) |
| holdout  | dev     |       303 |           37   |              9   |                 5 |             7799 | string_phys_notm_400       | holdout-current                                       |
| holdout  | confirm |       297 |           38   |              9   |                 4 |             7799 | string_phys_notm_400       | holdout-current                                       |

`median_fam_rows` = context rows whose target pairs are fully masked for the task (own row + descendants [+ near-duplicates on holdout]).

## Definitions
- **temporal** [temporal-clean]: modet panel `panel_allgene_t0_string115_phys_notm_400` (seeds = 21.04 genes, targets = 21.04->25.09
  arrivals, unknown = weak-evidence genes, excluded from ranking). dev = modet's old dev split (53), confirm = other non-sealed (138).
  Context = OT 21.04 sets with sealed ids (+ relatives) removed. Network T0 STRING 11.5 phys no-TM >= 0.4 (RWR, degree, nuAP weights).
  Secondary targets: `sa` = modet strict-a (arrival A & persistent; other targets removed from ranking; >= 5 kept), `onnet`.
- **forward** [temporal-clean labels; inputs <= 25.09]: OT 25.09 -> 26.09. Seeds = 25.09 genes of d and of its 26.09-ontology
  descendants (re-propagation; 12 % of raw "new" genes only arrive through new descendant links and are seeds, not targets),
  5..1000 seeds, >= 3 targets. Context = OT 25.09. Split = sha1(id) parity.
- **holdout** [holdout-current]: OT 26.09 diseases with 15..1000 genes, 600 sampled (seed 0), 20 % of genes hidden per disease
  (seeded). Context = OT 26.09 with leakage masking (below). Split = sha1(id) parity.
- **onnet** (all benchmarks, secondary): only genes with degree > 0 in the benchmark network are ranked/counted (modet's
  candidate set). 15-19 % of targets are off-network on every benchmark.

## Sealed policy
modet's 97 sealed fold diseases (`folds_v1.json`) and all their ontology ancestors/descendants that are not non-sealed modet
panel diseases (6,182 ids in total) are never tasks. Temporal: they are also removed from the context matrix. Forward/holdout:
they remain as context rows (other diseases' knowledge), never as tasks.

## Leakage masking (per task, `Benchmark.context_for(task)`)
- holdout: hidden pairs zeroed in the disease, all its descendants and near-duplicate rows (Jaccard >= 0.5); ancestors are
  re-propagated bottom-up (ancestor keeps g iff a non-masked descendant still has g). Blanket-masking ancestors (first version)
  created a tell: "gene in a sibling row but missing from the common ancestor" = hidden; with it `transfer_onto` scored
  AUROC 0.29 (anti-informative) - the test `test_holdout_no_mask_tell` now guards this.
- temporal / forward: context is a past snapshot, so it cannot contain the future; only the own row and descendants are masked
  (temporal: 287 / 6,324 targets were in the disease's own 21.04 row, because modet propagates T0 evidence with the 25.09 ontology).
  Sensitivity with the holdout-style family mask: temporal transfer_jaccard AP 0.120 -> 0.084, forward 0.089 -> 0.073
  (near-duplicate / ancestor rows carry legitimate past knowledge; modet's `tr_max` excludes such families by design).
