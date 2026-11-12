# modeng: RESUME HERE (updated 2026-11-12 ~06:30)
Read `CLAUDE.md`, then **`docs/reports/FINAL_overnight_2026-11-12.md`** (result, claimable sentences), `docs/insights.md`, `docs/invalidations.md`,
`docs/ideas.md` (~110 ideas with status), `docs/reviews/R1_fable_redteam.md`.

## State
- Leader: `stack_v3cv_ptr` = LightGBM stacker (`src/modeng/stack.py`, cv on 53 temporal dev diseases, positives deg^-0.5) over channels
  base, donor, view, cs_donor, dA_pt, dc_ridge(_pos), gwas_catlag, pri_*/hz_* priors, ph_* HPO phenotype matching, pt_ridge(_pos).
  Command: `PYTHONPATH=src python -m modeng.stack --bench temporal --channels <list> --tag <tag>` (`MODENG_SAVE_SCORES=1` saves score matrices).
- Beats modet's headline on all arrivals (onnet AP +0.050, nuAP +0.039, degree-matched AP +0.077); TIES on strict-a (modet's primary endpoint).
- Loses hub-gene AP vs raw transfer (deg^-0.5 weighting); power sweep (0.25 / 0.35) running at 06:30 -> `results/stack_v3ptr_p*.txt` (choose on DEV only).
- Module calls: `scripts/module_cutoff.py` (isotonic, dev-fitted; shrink predictions by 0.7).
- Running at 06:30: E-A4 catalog backlog transfer for strict-a (`docs/reports/r2_ea4.md` when done).

## Morning follow-up done (07:20-08:30): R3 red team, prospective version, forward no-label-process arm, strict-a attack (negative), calibration refit — see FINAL report section 9.

## Next (in order)
1. Fresh confirmation: everything was viewed on temporal confirm; build an untouched fold (e.g. next OT release, or diseases outside the panel)
   and pre-register the leader recipe + power before looking.
2. Strict-a: genetic arrivals remain the hard stratum (E-A4, ML-4 power extrapolation, archived GWAS-catalog vintages).
3. Vintage parity for past training (E-B5: HPO 2021-01 contains 19.04->21.04 arrivals), archived PubTator / GWAS-Catalog mappings.
4. Module structure: decompose called modules with subex `leiden_anno`; Perturb-seq co-response for interpretation.
5. Rules learned: quantise any masked-pair subtraction; same-width permuted controls; claims need AP AND nuAP (and degree-matched AP);
   never background `cd x && cmd`; one stacker per agent when RAM < 20 GB.
