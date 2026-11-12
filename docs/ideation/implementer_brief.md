# Implementer brief (round 1 tests; RESTARTED 2026-11-12 04:15 after a usage-limit stop; hard deadline for your report: 06:40)

**Restart note:** wave 1 was stopped mid-work at 02:50. Check for partial files of your cluster (src/modeng/ext/*.py, data/donors, data/raw, docs/reports) and continue from them; verify them first (they are untested). Machine is now mostly idle: up to 6 threads per agent. Temporal references (post-fix, train on `past`): `stackP_base`, `stackP_bdv` (pre-fix values: confirm nuAP 0.134 / 0.151; transfer_jaccard 0.104). For temporal, prefer `--train past` (1,786 episodes) over CV on 53 dev tasks; your channel must then work on B.name=='past' (seed release 19.04) or return NaN. Forward/holdout: `--train cv` (default).

Read: `/home/ph/code/modeng/CLAUDE.md`, `docs/ideation/brief.md`, your assigned ideas in `docs/ideation/r1_*.md`, the critiques
`docs/ideation/r1_critic_*.md` (if present) and novelty `docs/ideation/r1_novelty.md`, `docs/reports/baselines.md`, `docs/insights.md`.

## Harness
- Python: `~/biotools/venvs/bio-net/bin/python`, `PYTHONPATH=/home/ph/code/modeng/src`. Benchmarks: `modeng.tasks.load('temporal'|'forward'|'holdout').split('dev'|'confirm')`.
  Task fields: seeds, targets, exclude (genes not ranked), mask_r/mask_c (masked context pairs), cluster. `B.context_for(task)` = the label
  matrix a task may see; `B.context_ids` = disease ids of context rows. Gene order = `modeng.data.universe()` (sorted ENSG). NEVER assume another order.
- **Write your channel** as `src/modeng/ext/<yourname>.py` with `CHANNELS = {"<channel>": fn}`, `fn(B, split_tag) -> float32 [n_tasks, n_genes]`
  (NaN = missing). It is auto-discovered by `modeng.channels.compute` (cached in data/cache/channels; delete your cache files if you change code).
- Evaluate a channel alone: `modeng.bench.evaluate(S, B, jobs=4)` (per-task metrics df; columns ap, nuap, auroc, recall@100, onnet_ap, onnet_nuap, sa_ap).
- **Increment test (the decisive one):** `python -m modeng.stack --bench <b> --channels base,<your_channel>[,...] --tag <tag>` vs
  `--channels base` (LightGBM over channels, 5-fold disease-grouped OOF on dev, trained-on-dev model on confirm; positives weighted deg^-0.5).
  Also run with `base,donor,view` + yours when the reference `stack_bdv_w05` exists in `results/leaderboard.tsv`. Paired comparison:
  `modeng.compare` (read it) on the per-task parquet files in `results/<bench>/<split>/`.
- Ready inputs: gene views `data/views/*.npz` (X [19255,d], mask; manifest `docs/reports/views_manifest.tsv`), virtual donor gene-set families
  `data/donors/*.npz` (load via `modeng.channels.load_family`), `docs/research/02_local_data_inventory.md` (all data on disk), `docs/research/01_sota_and_resources.md`.
- Leakage: each benchmark has a class (temporal = seeds OT 21.04, targets new by 25.09; forward = 25.09 -> 26.09; holdout = 26.09 CV).
  Inputs dated after the seed release make a result *transductive* on that benchmark: allowed, but say so in the report and prefer the
  benchmark where your input is fair (e.g. 2022-2025 data on `forward`). Never use OT 26.09 files as inputs on forward; never use modet's sealed diseases.
- Compute: the machine is shared and loaded: **at most 4 threads** (`OMP_NUM_THREADS=4`, LightGBM `num_threads` via env is set to 8 in stack.py: pass
  `--` nothing, just keep your own jobs at 4). GPU only via `~/biotools/bin/gpu-run --mem <GB> -- <cmd>`. Downloads allowed (put under
  `/home/ph/code/modeng/data/raw/<name>/`, note URL, date, size, sha256 in `docs/reports/downloads.tsv`). No personal/project names in web queries.
- Do NOT edit other agents' ext modules, `tasks.py`, `metrics.py`, `bench.py` or label files. Small fixes to `channels.py`/`stack.py` only if
  essential and backward compatible; say so in your report.

## Deliverable
`docs/reports/r1_<yourname>.md`: what was built (files, commands, commit), each channel alone and its increment over `stack_base`
(dev OOF and confirm; AP, nuAP, AUROC, R@100, onnet), a positive control (current best reproduced) and a negative control (e.g. your channel with
gene labels permuted, or donors shuffled: must give ~0 increment), leakage class per benchmark, runtime, what failed, new ideas you got while working.
Commit your files with explicit paths (never push). Reply with ≤200 words: numbers + verdict (keep / discard / promising-needs-work).
