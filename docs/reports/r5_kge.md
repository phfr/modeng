# R5: cached RotatE (PrimeKG-T0) as a channel: F1 cheapest decisive test (2026-11-12)

## Pre-registered decision rule (written and committed before any run)
- Runs (temporal, cv on dev, dev-trained model on confirm, `--pos-deg-power 0.35`):
  - `v1_kge` = v1 channel list + `kge_rotate`;
  - `v1_kgeperm` = v1 channel list + `kge_rotate_perm` (same-width control: RotatE retrained on the degree-permuted PrimeKG-T0);
  - reference `stack_v3ptr_p0.35` (modeng-v1, saved scores).
- Endpoints: AP, nuAP, sa_nuAP (paired cluster bootstrap, `modeng.compare`, 10,000) and degree-matched AP (`scripts/leader_endpoints.py`).
- **Keep `kge_rotate` only if** `v1_kge` beats the control `v1_kgeperm` on BOTH AP and nuAP on BOTH temporal dev AND temporal
  confirm (mean delta > 0 in all four; CIs reported). Otherwise discard. A gain over the reference that the permuted control
  also shows counts as no gain. Caveat fixed in advance: modet picked the RotatE hyper-parameters and checkpoint on the
  53 dev diseases, so dev is optimistic; confirm is decisive for the sign.
- Secondary (reported, not deciding): the same comparisons with the 586 KG-leaky target pairs (see leakage audit) removed.
- Alone: `kge_rotate`, `kge_rotate_perm` vs `transfer_jaccard` on dev and confirm.
