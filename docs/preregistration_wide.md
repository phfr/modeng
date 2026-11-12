# Pre-registration: `temporal_wide` benchmark and the frozen `wide_confirm` fold
Written and frozen 2026-11-12 14:30 CET, **before any model or baseline has been scored on any `temporal_wide` split**
(only counts, set sizes and arrival-class tallies were computed). Builder: `src/modeng/tasks.py`
(`wide_candidates`, `wide_split`, `wide_arrival`, `build_temporal_wide`), tests `tests/test_wide.py`. Git tag: `wide-confirm-frozen`.

## Construction (leakage class: temporal-clean, same as `temporal`)
- Source: modet's `data/processed/groundtruth/ot_temporal.parquet` (`modet_data.groundtruth.build`), i.e. exactly the rule of the
  temporal panel: OT genetic evidence (GWAS / ClinVar score >= 0.5, curated Mendelian / burden any score), 25.09 ontology;
  seeds = OT 21.04 strong genes of the disease and its 25.09 descendants (re-propagated), targets = 25.09 genes not in the seeds and
  without weak 21.04 evidence, `unknown` = weak 21.04/25.09 evidence, strong direct 25.09 genes of ancestors, new genes with weak 21.04
  evidence, 21.04 genes lost in 25.09. exclude = seeds + unknown. Only diseases with >= 10 T1 genes exist in that file (modet's rule).
  Parity: the rule reproduces seeds / targets / unknown of all 260 `panel_allgene_t0_string115_phys_notm_400` diseases exactly.
- Filters: modet panel rule 1 (disease branch: a therapeutic area outside modet's BAD_TA, no HP / GO / OBA ids); 5 <= seeds <= 1000,
  >= 3 targets on the 19,255-gene universe; not one of the 191 `temporal` tasks; not in `D.excluded_task_ids()` (sealed + relatives).
  -> **792 diseases**.
- Context: OT 21.04 matrix without the sealed ids (identical to `temporal`); masking scope `self+desc` (as `temporal`).
- Secondary target set `sa` (strict-a): modet's `modet_data.arrival.build_arrival` (read-only import) on all 792 x targets
  (first evidence year from the OT 26.09 cumulative timeseries, persistence in 26.09): kept = arrival A & (persistent | margin_ok),
  task gets `sa` if >= 5 kept. Arrival classes over 24,024 pairs: B 9,440, A 8,168, X 4,878, U 1,538 (185 pairs of unmapped ids dated
  through descendants). Cached at `data/cache/arrival_wide.parquet`.

## Split rule (uses ids, ontology and 25.09 set overlaps only)
Graph nodes = 792 wide diseases + 191 temporal tasks; edges = ontology relatives (ancestor / descendant in 21.04 / 25.09 / 26.09)
or near-duplicate 25.09 sets (Jaccard >= 0.5). The raw graph has one component with 620 of the 792 diseases (broad groupers chain
everything), so **hub peeling**: while the largest component has > 10 nodes, its wide member with most live neighbours is removed and
assigned to wide_train. Then: component containing a temporal task -> wide_train; a wide disease that is another modet fold disease
(devall of the BioGRID / HuRI panels) -> wide_train; every other component -> sha1(smallest id of the component) % 100:
< 60 wide_train, < 80 wide_dev, else wide_confirm.

| split | diseases | components | median seeds | median targets | targets | `sa` tasks (pairs) |
|---|---|---|---|---|---|---|
| wide_train | 712 (hub 202, touches temporal 402, modet fold 12, hash 96) | 391 | 25 | 12.5 | 23,281 | 297 (7,387) |
| wide_dev | 45 | 33 | 9 | 6 | 446 | 10 (106) |
| **wide_confirm** | **35** | 28 | 11 | 7 | 297 | 4 (50) |

Guarantees (tested): no wide_dev / wide_confirm disease is an ontology relative of a temporal task; relatives across splits are only
peeled hubs (wide_train); hashed components never mix splits.

## Freeze
- `data/wide_confirm_ids.txt` (35 ids, sorted), sha256 in `docs/preregistration_wide.sha256`.
- `Benchmark.split('wide_confirm')` raises unless `MODENG_SCORE_WIDE_CONFIRM=1`; `split('all')` omits wide_confirm without it;
  `python -m modeng.stack --train wide` scores wide_confirm only with `--score-confirm` plus the env flag.
- **Use:** wide_confirm is scored once, by a separate pre-registration (synthesis plan P4) that names arms, endpoint, comparator and
  decision rule, committed before the flag is set. Development and selection use wide_train / wide_dev only.

## Known limitations (stated before looking)
- Small: 35 confirm diseases, median 7 targets, 4 strict-a tasks: power for AP differences of ~0.01 is low; strict-a is not
  evaluable on wide_confirm alone. The ontology is so entangled with the 191 burnt tasks that only 286 of 792 diseases are not
  direct relatives of one; the 200 / 300 unrelated diseases foreseen by R4 strategy A do not exist under this rule.
- Wide tasks are smaller and more often rare-disease / Orphanet groupers than the panel; results are reported by target-count stratum.
- Training on wide_train is same-window (21.04 -> 25.09 labels) and wide_train contains ancestors of evaluation diseases: `--train wide`
  drops, per training task, the targets of evaluation tasks that are its ontology relatives (`stack.relative_target_guard`).
