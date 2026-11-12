# r6: `temporal_wide` benchmark (R4 plan P1), 2026-11-12 14:00-15:10
Commit `e6f00fa` (builder, tests, stack `--train wide`) tagged **`wide-confirm-frozen`** before any scoring; this report: next commit.
Pre-registration of construction and split: `docs/preregistration_wide.md` (+ `.sha256`, ids `data/wide_confirm_ids.txt`).
Leakage class: temporal-clean (seeds / context OT 21.04, targets 21.04 -> 25.09); `--train wide` is **same-window training**.
**wide_confirm has not been scored by anything.**

## Construction (details in the pre-registration)
- Rule = modet's `ot_temporal.parquet` (`modet_data.groundtruth.build`): seeds 21.04 (re-propagated over 25.09 descendants), targets
  = new 25.09 genes without weak 21.04 evidence, exclude = seeds + modet `unknown` (weak evidence, ancestors' direct genes, ...).
  Parity check: the rule reproduces seeds / targets / unknown of all 260 modet all-gene panel diseases exactly.
- Filters: disease branch (modet panel rule 1), 5..1000 seeds, >= 3 targets, not temporal (191), not sealed-related -> **792 diseases**.
  Implicit modet filter: >= 10 T1 genes (only such diseases are in `ot_temporal`).
- Context = temporal's 21.04 matrix (sealed rows removed, identical matrix, tested); masking `self+desc`.
- Strict-a `sa`: modet's `build_arrival` (read-only import; first-evidence year from the 26.09 timeseries, persistence in 26.09) run on
  all 24,024 (disease, target) pairs in ~90 s: A 8,168 / B 9,440 / X 4,878 / U 1,538. `sa` on 311 tasks (>= 5 kept A pairs).

## Split
| split | diseases | components | median seeds / targets | targets | `sa` tasks |
|---|---|---|---|---|---|
| wide_train | 712 (hub 202, touches temporal 402, modet fold 12, hash 96) | 391 | 25 / 12.5 | 23,281 | 297 |
| wide_dev | 45 | 33 | 9 / 6 | 446 | 10 |
| wide_confirm (frozen) | 35 | 28 | 11 / 7 | 297 | 4 |

- **Main finding of the build: the off-panel diseases are ontologically entangled with the 191 temporal tasks.** Without hub handling one
  component holds 620 / 792 diseases; only 286 / 792 are not direct ontology relatives of a temporal task. The 200 dev + 300 confirm
  unrelated diseases foreseen by R4-A do not exist under a relatedness rule; the free pool is ~180, i.e. 45 dev / 35 confirm at 60/20/20.
- Hub peeling (largest component <= 10; peeled hubs -> train) was chosen on counts only (threshold 3 / 5 / 10 / 20 give 37 / 37 / 35 / 29
  confirm diseases).
- Guard: `split('wide_confirm')` raises without `MODENG_SCORE_WIDE_CONFIRM=1`; `stack --train wide --score-confirm` needs the flag too.

## Sanity checks (`tests/test_wide.py`, 7 pass; old `tests/test_harness.py` 17 pass, 2 skip)
frozen-fold guard; frozen ids == rebuilt ids and sha256; no overlap with temporal / sealed ids; identical context; task rules
(disjoint seeds / targets / exclude, sa subset); hidden targets never in the context view of the disease or its descendants (60 tasks);
wide_dev / confirm diseases are never relatives of temporal tasks and relatives across splits are only peeled hubs; hashed components
never mix splits; relative-target guard covers all relatives' targets.

**Relative-target guard (new, `stack.relative_target_guard`).** wide_train holds ancestors of the evaluation diseases, whose 25.09 labels
contain the evaluation targets in the same window. The guard removes, per train task, the targets of evaluation tasks that are its ontology
relatives (12,202 positive pairs in 506 train tasks). It matters: without it, temporal dev rises AP +0.011 [+0.003, +0.019], nuAP +0.026
[+0.009, +0.046] while wide_dev (few train relatives) does not move (AP -0.004, nuAP -0.005) -> the unguarded temporal gain is leakage.
All stackW numbers below are guarded.

## Baselines on wide_dev (45 tasks; all-gene AP / nuAP / AUROC / R@100 | onnet AP / nuAP / saAP)
| method | AP | nuAP | AUROC | R@100 | onnet AP | onnet nuAP | sa AP |
|---|---|---|---|---|---|---|---|
| random | 0.001 | 0.001 | 0.50 | 0.006 | 0.001 | 0.002 | 0.001 |
| degree | 0.003 | 0.001 | 0.62 | 0.023 | 0.003 | 0.001 | 0.002 |
| popularity | 0.014 | 0.004 | 0.76 | 0.100 | 0.016 | 0.006 | 0.048 |
| rwr_string | 0.013 | 0.006 | 0.67 | 0.072 | 0.015 | 0.007 | 0.003 |
| **transfer_jaccard** | **0.100** | **0.129** | 0.65 | 0.247 | 0.110 | 0.155 | 0.026 |
| transfer_onto | 0.025 | 0.041 | 0.58 | 0.101 | 0.036 | 0.073 | 0.023 |
(wide_train, descriptive: transfer_jaccard AP 0.111 nuAP 0.101, i.e. wide tasks behave like the panel: temporal dev transfer 0.122 / 0.096.)

## Stacks (modet-free channel sets: `base` = 5 baselines; `wclean` = base + 11 clean donor families + 14 clean view similarities + 6 HPA20/GTEx8
views = the first 36 channels of `results/v1clean_channels.txt`, no ext channels that rely on modet panel artefacts)
stackW = trained on wide_train (guarded; 600 rounds, 1,500 negatives, <= 60 positives per task, as `--train past`); stackP = past twin (19.04 -> 21.04).

| eval | model | AP | nuAP | AUROC | R@100 | onnet AP | onnet nuAP |
|---|---|---|---|---|---|---|---|
| temporal dev | stackP_base | 0.078 | 0.119 | 0.78 | 0.223 | 0.096 | 0.186 |
| temporal dev | **stackW_base** | 0.097 | 0.125 | 0.79 | 0.251 | 0.116 | 0.190 |
| temporal dev | stackP_wclean | 0.100 | 0.122 | 0.81 | 0.245 | 0.119 | 0.172 |
| temporal dev | **stackW_wclean** | 0.119 | 0.129 | 0.82 | 0.276 | 0.134 | 0.177 |
| temporal confirm (spent, descriptive) | stackW_base / stackW_wclean | 0.100 / 0.122 | 0.142 / 0.164 | 0.80 / 0.83 | 0.270 / 0.293 | | |
| wide_dev | stackP_base | 0.066 | 0.129 | 0.77 | 0.203 | 0.081 | 0.168 |
| wide_dev | **stackW_base** | 0.093 | 0.153 | 0.78 | 0.223 | 0.103 | 0.182 |
| wide_dev | stackP_wclean | 0.066 | 0.098 | 0.82 | 0.296 | 0.070 | 0.113 |
| wide_dev | **stackW_wclean** | 0.094 | 0.131 | 0.83 | 0.281 | 0.095 | 0.149 |

Paired cluster bootstrap (95 % CI):
- wide- vs past-trained, temporal dev: base AP +0.019 [+0.008, +0.032], nuAP +0.006 [-0.006, +0.020]; wclean AP +0.019 [+0.005, +0.032], nuAP +0.007 [-0.005, +0.019].
- wide- vs past-trained, wide_dev: base AP +0.027 [+0.005, +0.055], nuAP +0.024 [-0.005, +0.063]; wclean AP +0.028 [-0.000, +0.060], nuAP +0.033 [-0.001, +0.067].
- vs transfer_jaccard, wide_dev: stackW_base AP -0.007 [-0.033, +0.019], nuAP +0.024 [+0.002, +0.050]; stackW_wclean AP -0.006 [-0.043, +0.029], nuAP +0.002 [-0.063, +0.056].
- vs transfer_jaccard, temporal dev: stackW_wclean AP -0.004 [-0.030, +0.020], nuAP +0.033 [+0.016, +0.051].

Reading: same-window wide training beats the past-trained twin by ~+0.02 AP on both evaluation sets (scale / same-window helps, as R4-A
predicted), but with the modet-free channels the stack does not beat transfer_jaccard on AP anywhere, and wide_dev is too small (45 tasks,
median 6 targets) to separate anything below ~0.03 AP. For reference, the full v1clean recipe (51 channels incl. ext witnesses, cv on 53 dev)
has temporal dev AP 0.177; those ext channels were not run on temporal_wide (several key on `B.name` / modet panel artefacts).

## Caveats / next
- Power: wide_confirm (35 diseases, 297 targets, 4 sa tasks) can confirm only large effects; strict-a is not evaluable there. If the P4
  claims need more, a second fresh fold must come from the next OT release, not from re-splitting (the freeze is done).
- `docs/preregistration_modeng_v1.sha256` lists `src/modeng/stack.py`; the backward-compatible edit (new optional `drop` argument,
  `run_wide`) changes that hash: verify v1 against tag `modeng-v1-prereg`.
- Next (P1 cont.): port the v2 / v1clean ext channels to `temporal_wide` (each must handle `B.name == 'temporal_wide'` with the 21.04
  cutoff), then run R4-A's decisive test on wide_dev (v2 wide-trained vs v1clean trained on the 53 old dev), pre-registered first.
- Past twin on wide_dev: `scripts/past_on_wide.py <channels> <tag>`. Files: `src/modeng/tasks.py` (`wide_candidates`, `wide_split`, `wide_arrival`, `build_temporal_wide`, split guard), `src/modeng/stack.py`
  (`--train wide`, `--score-confirm`, `--no-guard`, `relative_target_guard`), `tests/test_wide.py`; caches `data/cache/bench_temporal_wide.pkl`,
  `data/cache/arrival_wide.parquet`; results `results/temporal_wide/wide_dev/*.parquet`, `results/temporal/{dev,confirm}/stackW_*.parquet`.
