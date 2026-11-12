#!/bin/bash
# r4 ML-4 power forecaster: increment runs over modeng-v1 (stack_v3ptr_p0.35), cv on temporal dev, power 0.35, scores saved.
# Pre-registered (written and committed before any run): variants V1 = v1+pfc_own, V2 = v1+pfc_own+pfc_sem; controls:
# C1 = v1+pfc_own_perm (same-width gene-permuted), C2 = v1+gwas_own (plain T0 MAGMA z, what the forecast must beat).
# Selection (dev only, scripts/powerfc_analyze.py): among V1, V2 pick the higher dev mean delta vs v1 of (sa_ap, sa_nuap, sa_onnet_ap,
# sa_onnet_nuap), requiring dev ap and nuap >= v1 - 0.003; if neither qualifies or neither has a positive SA4 delta: no variant chosen.
# Every run's confirm is scored once (stack.py prints it); 4 confirm scorings are counted.
set -e
PY=$HOME/biotools/venvs/bio-net/bin/python
export PYTHONPATH=/home/ph/code/modeng/src OMP_NUM_THREADS=8 MODENG_SAVE_SCORES=1
V1=base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,hz_gwas,hz_mend,ph_mean,ph_top3,pt_ridge,pt_ridge_pos
for x in "pfc_own v1_powerfc" "pfc_own,pfc_sem v1_powerfc2" "pfc_own_perm v1_powerfc_perm" "gwas_own v1_gwasown"; do
  set -- $x
  $PY -m modeng.stack --bench temporal --channels $V1,$1 --pos-deg-power 0.35 --tag $2 > /home/ph/code/modeng/results/powerfc/run_$2.log 2>&1
done
