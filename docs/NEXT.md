# modeng: RESUME HERE (written 2026-11-12 ~02:50 when the usage limit stopped the overnight sprint)
Read `CLAUDE.md`, then `docs/journey.md`, `docs/insights.md`, `docs/invalidations.md`, `docs/ideas.md` (73 R1 ideas).

## Done
- Ideation process researched (`docs/research/00_ideation_process.md`), SOTA + local resources (`01_`, `02_`).
- Round 1: 8 independent Fable ideators (`docs/ideation/r1_*.md`), 3 critics (stats, bio, eng) + novelty check.
- Harness `src/modeng` (temporal / forward / holdout + `past` training episodes 19.04->21.04), parity with modet's headline verified.
- 15 gene views (`data/views`), 15 virtual-donor families (`data/donors`), channels + LightGBM stacker (`channels.py`, `stack.py`, ext plugin hook `src/modeng/ext/`).
- Best so far (temporal confirm, all-gene): `stack_bdv_w05` nuAP 0.138 vs transfer_jaccard 0.104 (computed BEFORE the tie-leak fix: rerun).
- Leak fixed: float residue in the transfer masking correction (holdout transfer_onto 0.117 -> 0.036).

## Interrupted (usage limit) — check what each left on disk before restarting
- Wave-1 implementers A-H were STOPPED mid-work (brief: `docs/ideation/implementer_brief.md`; clusters in `docs/journey.md`). Look for partial files in `src/modeng/ext/`, `src/modeng/settx/`, `docs/reports/r1_*.md`, `data/donors`, `data/raw`; uncommitted ones are untested.
- A background `stackP` run (train on past) may have written `results/stackP_log.txt`.

## Next
1. Rerun stack_base / stack_bdv (cv and --train past) after the leak fix; record in insights.
2. Restart implementers (priority by critics: backlog exact-match, channel-typed priors + HPO phenotype matching, GWAS z on forward, kernels/cold-start, set transformer).
3. Round 2 (mutate/crossover/repair from measured results + 2 fresh islands), Fable review of the leading model, one confirm run.
