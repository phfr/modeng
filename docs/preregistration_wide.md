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

## Amendment 1 — confirmatory scoring plan (P4), written before wide_confirm is scored
Written 2026-11-12, committed before `MODENG_SCORE_WIDE_CONFIRM=1` is set for any command. The user approved spending the fold.
Pre-check: `sha256sum -c docs/preregistration_wide.sha256` passed for `data/wide_confirm_ids.txt` (and the other four frozen files)
before this amendment was appended; from now on `docs/preregistration_wide.md` is verified against tag `wide-confirm-frozen` (its
.sha256 line refers to the frozen original). No model, baseline or feature has been computed on wide_confirm so far (no channel cache
`temporal_wide_wide_confirm_*` exists). Driver code committed with this amendment: `scripts/p4_score.py` (arms A, B,
transfer_jaccard), `scripts/p4_analyze.py` (endpoints and bootstraps); both were dry-run on wide_dev only (no writes; `results/p4/dryrun_wide_dev.log`: A and B reproduce their temporal-confirm rows exactly, 0.1532 / 0.1497; B on wide_dev 0.1236 = r6's dev-trained v1clean). Python env `~/biotools/venvs/bio-net/bin/python`.
Working-tree note: an uncommitted, concurrent edit to `src/modeng/tasks.py` (another agent's P3 work) makes `tasks.load` add alt target sets `newdata` / `labelproc` (`src/modeng/arrival_types.py`); it is additive (targets / exclude / sa / onnet unchanged; wide_dev transfer_jaccard reproduces its leaderboard row 0.0996 / 0.1294) and is not part of this amendment; any newdata / labelproc metric on wide_confirm is exploratory.

**Arms** (one run each, LightGBM seed 0, no re-tuning, nothing changed after looking; positives weighted deg^-0.35 in every arm):
- **(A) modeng-v1**: the frozen recipe of `docs/preregistration_modeng_v1.md` (= `stack_v3ptr_p0.35`): channels
  `base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,hz_gwas,hz_mend,ph_mean,ph_top3,pt_ridge,pt_ridge_pos`,
  400 rounds, 3000 negatives per task, trained on all 53 temporal dev diseases (the model `stack.run` uses for temporal confirm), applied
  once to wide_confirm (`scripts/p4_score.py`). Check (not a gate): the same model reproduces the recipe's temporal-confirm row (AP 0.1532).
- **(B) v1clean**: same, channel list `results/v1clean_channels.txt` (51 channels; amendment 1 of the v1 pre-registration). Check: confirm AP 0.1497.
- **(C) W_v1clean**: v1clean channels trained on wide_train (`python -m modeng.stack --train wide --score-confirm --pos-deg-power 0.35`,
  `run_wide` defaults: 600 rounds, 1500 negatives, <= 60 positives, relative-target guard ON — the guard now also covers wide_confirm targets),
  seed 0, tag `P4C_v1clean`.
- **(D) W_v1clean without label-process channels**: (C) minus `gwas_catlag, pri_l2g_n, pri_l2g_max, hz_gwas` (47 channels), tag `P4D_v1clean_nolab`.
- **Comparators:** `transfer_jaccard` (primary comparator); `modet_headline`: **unavailable** — it exists only for the 191 temporal panel
  diseases and 0 / 35 wide_confirm ids are in modet's headline run (checked on ids only), so no modet comparison is made.

**Primary endpoint** (per arm, vs transfer_jaccard, all-gene evaluation): paired per-task difference in AP **and** in nuAP; cluster bootstrap
10,000 resamples (seed 0) over diseases with clusters = the ontology components of the split rule (task `cluster` field, 28 components;
`modeng.compare.paired_bootstrap`). An arm is **confirmed better than transfer_jaccard on wide_confirm only if both 95 % CIs lie above 0.**
Four arms are tested; no multiplicity correction is applied, and this is stated with every claim (arms are highly correlated; a claim
needing a correction would be one whose CI excludes 0 only barely, which is flagged).

**Secondary** (descriptive, same bootstrap, no claims): degree-matched AP (within degree-decile bins, `scripts/review_analyze.py`), P@20,
onnet AP / nuAP, and strict-a AP / nuAP only on tasks with >= 5 `sa` targets (4 tasks expected: means only, no CI). Results also by
target-count stratum (3-5, 6-10, > 10 targets).

**Between-arm comparisons** (C vs B, B vs A, D vs C, C vs A): descriptive only. From wide_dev's bootstrap spread scaled to 28 clusters,
the minimum detectable effect (80 % power, two-sided 5 %, MDE = 2.8 SE) is ~0.03-0.06 AP and ~0.04-0.08 nuAP for between-arm
differences, and ~0.04 AP / ~0.045-0.05 nuAP for arm vs transfer_jaccard; observed between-arm differences on temporal / wide_dev were
<= 0.02 AP, so wide_confirm cannot separate the arms and no between-arm sentence will be claimed.

**Failures:** any crash or anomaly is reported as is; no arm is re-run with changes. A re-run is allowed only for a crash unrelated to the
model (e.g. out of memory), with identical code and settings, and is reported.

**Allowed outcome sentences:** per arm X with both CIs > 0: "X ranks newly arriving disease genes better than transfer_jaccard on 35 fresh,
pre-registered off-panel diseases (wide_confirm; temporal-clean, OT 21.04 -> 25.09): AP +a [lo, hi], nuAP +b [lo, hi]." Otherwise: "X is not
confirmed better than transfer_jaccard on wide_confirm (AP ..., nuAP ...)." If C passes but D does not: the confirmation depends on
label-process channels and is stated as such.
