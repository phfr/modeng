# r4: sub-threshold GWAS loci -> gene assignment for strict-a (F2-6 / SG-1 lite)

## 0. Pre-registration (written and committed before any result of these channels was seen)
- Reference: modeng-v1 `stack_v3ptr_p0.35` (temporal, cv-trained, power 0.35, v1 channel list of `docs/preregistration_modeng_v1.md`).
- Main variant `v1_subthr` = v1 channels + `subthr_near,subthr_cs2g,subthr_win` (3 channels). Control `v1_subthrperm` = v1 + the 3 gene-permuted
  channels (same width). Further variants only if chosen on dev (each counted below); every variant that reaches confirm is scored there once.
- Primary endpoint (strict-a): sa_nuAP and sa_AP of the stacker vs modeng-v1 (all genes) and sa_onnet_nuAP vs modet's headline
  (`scripts/leader_endpoints.py`), paired cluster bootstrap.
- Dev rule (53 dev diseases, 28 with strict-a targets): a variant goes forward as "candidate" if dev sa_nuAP delta vs v1 > 0, overall dev AP and nuAP
  deltas >= -0.003, and its sa_nuAP delta exceeds the permuted control's. Alone screen (channel alone on covered tasks, sa columns) decides which
  single-channel variants are worth a stack run.
- Claim on confirm (138 diseases, 71 with strict-a): only if the confirm sa_nuAP CI vs v1 excludes 0 and the permuted control shows no gain;
  otherwise "no strict-a gain".
