#!/bin/bash
cd /home/ph/code/modeng
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 MODENG_SAVE_SCORES=1 PYTHONPATH=/home/ph/code/modeng/src
V1=base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,hz_gwas,hz_mend,ph_mean,ph_top3,pt_ridge,pt_ridge_pos
PY=/home/ph/biotools/venvs/bio-net/bin/python
$PY -W ignore -m modeng.stack --bench temporal --channels $V1,ct_act,ct_donor --pos-deg-power 0.35 --tag v1_ct > results/r5ct/stack_v1_ct.out 2>&1
$PY -W ignore -m modeng.stack --bench temporal --channels $V1,ct_act_perm,ct_donor_perm --pos-deg-power 0.35 --tag v1_ctperm > results/r5ct/stack_v1_ctperm.out 2>&1
