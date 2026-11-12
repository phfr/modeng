# R5 witness audit of modeng-v1 (`stack_v3ptr_p0.35`), 2026-11-12

Scope: every input channel ("witness") of the frozen modeng-v1 recipe (`docs/preregistration_modeng_v1.md`): what it
computes, where its data come from and how they are dated, how much it contributes, where it matters, and which channels
repeat each other. **All numbers here are descriptive.** Temporal confirm (138 diseases) has been looked at many times
overnight; no claim is decided here, and none of the "drop / merge" suggestions below may be confirmed on temporal confirm.
They need the fresh fold of the pre-registration (section 5 there).

Scripts (all `scripts/r5_audit_*.py`, outputs in `results/r5/`):
`r5_audit_ablate.py` (frozen recipe retrained per arm, cv on 53 dev diseases, deg^-0.35, 400 rounds, 3000 negatives,
LightGBM seed 0 = recipe; arms `_seed1`, `_seed2` = same arm with LightGBM seeds 1 and 2), `r5_audit_shap.py` (gain and
`pred_contrib` SHAP of the dev->confirm model), `r5_audit_channels.py` (coverage, alone metrics, redundancy),
`r5_audit_cost.py` (timings), `r5_audit_summarize.py` (bootstrap, disease types). The seed-0 full arm reproduces the
leaderboard row of modeng-v1 exactly (dev AP 0.1756 / nuAP 0.1502, confirm 0.1532 / 0.1770).

__SECTIONS__
