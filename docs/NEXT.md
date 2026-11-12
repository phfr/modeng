# modeng: RESUME HERE (updated 2026-11-12 ~06:30)
Read `CLAUDE.md`, then **`docs/reports/FINAL_overnight_2026-11-12.md`** (result, claimable sentences), `docs/insights.md`, `docs/invalidations.md`,
`docs/ideas.md` (~110 ideas with status), `docs/reviews/R1_fable_redteam.md`.

## State
- Final model **modeng-v1** = `stack_v3ptr_p0.35` (frozen, `docs/preregistration_modeng_v1.md`, tag `modeng-v1-prereg`): LightGBM stacker, cv on 53 temporal
  dev diseases, positives deg^-0.35, channels base, donor, view, cs_donor, dA_pt, dc_ridge(_pos), gwas_catlag, pri_*/hz_*, ph_*, pt_ridge(_pos).
  Prospective twin `stackP_v1past` (trained on 19.04->21.04; `src/modeng/ext/past_fill.py`). Command: `PYTHONPATH=src python -m modeng.stack ...`.
- Robust: nuAP and onnet AP over transfer and over modet's headline, also prospectively. Not robust: AP over transfer prospectively (n.s.),
  strict-a (tie with modet's headline in every variant), cross-window gains without label-process channels.
- Module calls: `scripts/module_cutoff.py stack_v3ptr_p0.35` (dev-fitted isotonic; shrink ~0.8).
- Sealed fold: NOT recommended now (its primary is strict-a); use the next OT release, no-label-process arm co-primary.

## Afternoon 2026-11-12: research cycle + fresh confirmation done (FINAL report section 11). Recommended model: W_v1clean without label-process channels (wide-trained). wide_confirm is SPENT. Next: next OT release as confirmation; consider merging the two dev sets for future selection.

## Evening 2026-11-12: modeng-v2 packaged (= r7 arm D). Config `configs/modeng_v2.yaml`, `scripts/train_v2.py` (model + card on `/mnt/msadb/modeng/models/modeng_v2_2104/`, reproduces arm D exactly), `python -m modeng.predict` (README 'Using modeng-v2'), `tests/test_predict.py`.

## Morning follow-up done (07:20-08:30): R3 red team, prospective version, forward no-label-process arm, strict-a attack (negative), calibration refit — see FINAL report section 9.

## Next (in order)
1. Fresh confirmation: everything was viewed on temporal confirm; build an untouched fold (e.g. next OT release, or diseases outside the panel)
   and pre-register the leader recipe + power before looking.
2. Strict-a: genetic arrivals remain the hard stratum (E-A4, ML-4 power extrapolation, archived GWAS-catalog vintages).
3. Vintage parity for past training (E-B5: HPO 2021-01 contains 19.04->21.04 arrivals), archived PubTator / GWAS-Catalog mappings.
4. Module structure: decompose called modules with subex `leiden_anno`; Perturb-seq co-response for interpretation.
5. Rules learned: quantise any masked-pair subtraction; same-width permuted controls; claims need AP AND nuAP (and degree-matched AP);
   never background `cd x && cmd`; one stacker per agent when RAM < 20 GB.
