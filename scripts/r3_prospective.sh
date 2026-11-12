#!/bin/bash
# r3_prospective: modeng-v1 channel recipe trained on `past` episodes (19.04 -> 21.04) with past-fill (_pf) priors.
set -e
export PYTHONPATH=/home/ph/code/modeng/src OMP_NUM_THREADS=8 MODENG_SAVE_SCORES=1
PY=~/biotools/venvs/bio-net/bin/python
V1PF=base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n_pf,pri_l2g_max_pf,pri_mend_n_pf,pri_hpo_n_pf,hz_gwas_pf,hz_mend_pf,ph_mean_pf,ph_top3_pf,pt_ridge,pt_ridge_pos
NOPRI=base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pt_ridge,pt_ridge_pos
cd /home/ph/code/modeng
$PY -m modeng.stack --bench temporal --train past --pos-deg-power 0.35 --channels $V1PF --tag v1past
# ablation: same, without the past-filled prior/HPO channels (only channels natively defined on past)
$PY -m modeng.stack --bench temporal --train past --pos-deg-power 0.35 --channels $NOPRI --tag v1past_nopri
