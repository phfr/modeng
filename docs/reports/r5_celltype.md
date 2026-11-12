# r5: cell-type witnesses CT-ACT (`ct_act`) and SC-3 cell-address donors (`ct_donor`)

## 0. Pre-registration (written and committed before any result of these channels was seen, 2026-11-12)
- Code: `src/modeng/ext/celltype.py` (docstring = exact definitions). Inputs: HPA v20.1 single-cell types (2020-11) and PoPS
  pre-2020 single-cell cluster DE t-statistics (2,583 columns); expression matching by HPA v20.1 tissue consensus. Temporal-clean on
  past, temporal and forward.
- Reference: modeng-v1 `stack_v3ptr_p0.35` (temporal, cv-trained, power 0.35, v1 channel list of `docs/preregistration_modeng_v1.md`).
- Main variant `v1_ct` = v1 channels + `ct_act,ct_donor`. Control `v1_ctperm` = v1 + `ct_act_perm,ct_donor_perm` (gene axis of the
  substrates permuted; same width). One variant only; diagnostics `ct_act_hpa`, `ct_act_pops` are looked at alone, not stacked
  unless counted here as an extra variant.
- Kill rule (pre-declared by the lead): if the increment of `v1_ct` over `v1_ctperm` is < +0.003 on BOTH AP and nuAP on dev AND on
  confirm -> discard. Keep only if AP or nuAP gains >= +0.003 over the control on dev and confirm, with no loss < -0.003 on the other,
  and degree-matched AP (`scripts/leader_endpoints.py`) not below v1. Paired cluster bootstrap (`modeng.compare`) on ap, nuap, sa_nuap.
- Alone: each channel on dev/confirm (AP, nuAP, AUROC) and in the zero-transfer-support stratum (`scripts/r1_kernels_b.py zero`),
  vs `vmean_hpa_sc` and the permuted controls. Secondary sweep-03 kill: alone not better than `vmean_hpa_sc` on zero-support AUROC.
