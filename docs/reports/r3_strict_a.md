# r3: attacking strict-a with a disease-type-gated stacker (2026-11-12 07:25-09:00)

**Goal.** modeng-v1 (`stack_v3ptr_p0.35`) only ties modet's headline on strict-a (genetic arrivals first seen 2022+, `sa` target set;
onnet sa_nuAP 0.110 vs 0.109) and is below it on all genes. Test the cheapest versions of E-B2(c) (task-context features so the
stacker can gate channel groups by disease type) and E-B3 (arrival-class-aware learning).

## What was built
- `src/modeng/stack_gate.py` (new; `stack.py` untouched, its features / folds / LGB params / negatives reused; power 0.35; v1 channels):
  - **ctx features** (9 per task, appended to every row; temporal-clean: OT 21.04 evidence, HPO 2021-01, GWAS Catalog <= 2021-04-30, seeds):
    share of seeds that are GWAS winners / Mendelian-type / GWAS-only in the task family, # family genes with GWAS / Mendelian evidence,
    own GWAS-Catalog hits yes/no + count, share of seeds with HPO annotation, mean seed log-degree (n_seeds is already `log_nseeds`).
    Therapeutic-area flags implemented (`--area`) but not run (time).
  - **class-A head** (`--head`): second LightGBM, same features, positives = class-A targets only (modet `arrival_devall`, arrival == 'A';
    dev labels only, same disease-grouped folds; other targets dropped, not negatives); blended as (1-a) log p_main + a log p_head.
  - **class-A upweighting** (`--a-weight 3`): one model, class-A positives of training tasks get 3x weight.
  - `--ctx-perm`: ctx rows permuted across tasks (same-width negative control).
- `scripts/r3_strict_a.py`: endpoints (identical `endpoints()` as `scripts/leader_endpoints.py` / `review_analyze.py`), dev-only
  selection rule (in the docstring, written before confirm was scored), paired cluster bootstrap vs modeng-v1 and vs modet headline.
- Runs: `scripts/r3_strict_a_runs.sh` (only gate_ctx --head completed; the rest of that queue was cancelled for time: machine load 60-75),
  `scripts/r3_strict_a_runs2.sh`. Logs + `endpoints_vs_ref.tsv` + `analysis.txt` in `results/r3_strict_a/`; scores saved as
  `results/temporal/<split>/stack_gate_*.scores.npy`; feature importance `results/importance_stack_gate_*.tsv`.

## Variants (8 looked at on dev; 5 scored on confirm, each once; reference = modeng-v1 `stack_v3ptr_p0.35`, same folds/channels/power)
gate_ctx (+ctx), gate_ctx_headA (head alone), 3 blends gate_ctx+headA a = 0.25 / 0.5 / 0.75 (dev only), gate_ctx_aw3, gate_ctxperm (control),
gate_ctxta (ctx + area flags). Blend rule (dev, written before confirm): a* = 0 (every blend loses > 0.003 dev AP/nuAP), so no blend reached confirm.
Order caveat: gate_ctx's confirm line (overall AP/nuAP/onnet, not sa) was printed by the run before batch 2 was decided; aw3, ctxperm, ctxta were added afterwards.

## Results (delta vs modeng-v1, paired cluster bootstrap 95 % CI; `results/r3_strict_a/analysis.txt`, `endpoints_vs_ref.tsv`)
Dev OOF (53 diseases; sa n = 28):
| variant | AP | nuAP | sa_AP | sa_nuAP | sa_onnet_AP | sa_onnet_nuAP |
|---|---|---|---|---|---|---|
| gate_ctx | +0.002 [-0.006,+0.008] | +0.004 [-0.003,+0.010] | +0.002 | +0.007 [-0.004,+0.023] | +0.005 | +0.007 [-0.007,+0.024] |
| gate_ctxperm (control) | -0.000 | +0.000 | +0.004 | +0.001 | +0.003 | -0.001 |
| gate_ctxta | +0.003 | +0.002 | +0.003 | +0.006 | +0.008 [+0.002,+0.015] | +0.007 |
| gate_ctx_aw3 | -0.003 | -0.006 | +0.001 | -0.004 | -0.001 | -0.008 |
| gate_ctx+headA 0.25 | -0.015 | -0.047 | +0.010 | +0.013 [-0.000,+0.028] | +0.000 | +0.001 |
| gate_ctx_headA (alone) | -0.088 | -0.111 | -0.006 | -0.020 | -0.016 | -0.034 |

Temporal confirm (138 diseases; sa n = 71):
| variant | AP | nuAP | sa_AP | sa_nuAP | sa_onnet_AP | sa_onnet_nuAP |
|---|---|---|---|---|---|---|
| gate_ctx | **+0.006 [+0.002,+0.010]** | +0.003 [-0.002,+0.008] | +0.001 [-0.004,+0.006] | -0.003 [-0.011,+0.004] | +0.002 [-0.003,+0.009] | -0.003 [-0.012,+0.006] |
| gate_ctxperm (control) | -0.001 | -0.002 | -0.002 | -0.006 | -0.001 | -0.007 |
| gate_ctxta | -0.001 | -0.002 | -0.003 | -0.005 | -0.006 [-0.013,-0.000] | -0.007 |
| gate_ctx_aw3 | +0.005 [+0.000,+0.009] | +0.000 | +0.006 [-0.003,+0.015] | +0.004 [-0.011,+0.020] | +0.003 | -0.003 |
| gate_ctx_headA (alone) | -0.065 | -0.134 | +0.013 [-0.011,+0.033] | -0.011 | -0.002 | -0.030 |
Absolute confirm, gate_ctx: AP 0.159, nuAP 0.180, onnet AP 0.167 / nuAP 0.211, degree-matched AP +0.007 [+0.003,+0.011] vs v1.
**vs modet headline (confirm, `results/review/leader_endpoints_stack_gate_ctx.tsv`): strict-a onnet nuAP 0.108 vs 0.109 (-0.002 [-0.022,+0.024]),
onnet AP 0.083 vs 0.058 (+0.025 [+0.009,+0.041]); all-gene sa_nuAP 0.087 vs 0.109 (-0.022 [-0.045,+0.004]), sa_AP +0.008 n.s.** = still a tie / below.

## Verdict
- **Strict-a not moved.** No variant improves strict-a nuAP on confirm (best: aw3 sa_nuAP +0.004, CI spans 0; it failed the dev rule).
  Dev strict-a gains of gate_ctx (+0.007) did not replicate (confirm -0.003); with 28 dev sa tasks dev cannot select for strict-a.
- **ctx features: small overall gain, keep as optional.** gate_ctx confirm AP +0.006 [+0.002,+0.010], onnet AP +0.007, degree-matched AP +0.007,
  nuAP +0.003 n.s., above the permuted control on both splits (control ~0). Gain sits in non-class-A arrivals (nonA_AP +0.004), not strict-a.
  The trees use `ctx_hpo_share`, `ctx_fam_mend_n`, `ctx_mix_gwas` (ranks 11/13/20 by gain). Not a frozen-recipe change (seen on confirm; needs the fresh fold).
- **Area flags overfit** (best on dev onnet, worse than v1 on confirm): 53 dev diseases are too few for 4 area categories.
- **Class-A head: discard.** Trained on ~51 dev tasks' class-A genes only, it is weak even on strict-a (dev sa_onnet_nuAP -0.034); any blend
  costs > 0.015 AP / 0.047 nuAP on dev. Class-A x3 weighting trades nuAP for nothing on dev. The arrival-class signal is too thin in 53
  diseases; E-B3 as designed (heads trained on past episodes with 21.04-derived classes) is the version that has enough rows.
- Leakage: ctx = temporal-clean; class-A labels = dev labels only (confirm never used for fitting).

## Next
E-B3 on `past` episodes (1,786 tasks, classes from 21.04 evidence) is the only arrival-class route with enough data; strict-a probably
needs a new input (e.g. archived GWAS-Catalog/summary-statistics vintages, rare-variant burden), not a new learner gate.
Runtime: ~45 min of 8-thread compute (machine load 50-75); variants that were planned but not run: noctx head.
