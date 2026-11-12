#!/bin/bash
# r3 strict-a, second batch (after gate_ctx --head): permuted-context control, then ctx + class-A upweight x3.
export OMP_NUM_THREADS=8 PYTHONPATH=/home/ph/code/modeng/src
PY=~/biotools/venvs/bio-net/bin/python
CH=base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,hz_gwas,hz_mend,ph_mean,ph_top3,pt_ridge,pt_ridge_pos
O=/home/ph/code/modeng/results/r3_strict_a
$PY -m modeng.stack_gate --channels $CH --tag gate_ctxperm --ctx-perm > $O/gate_ctxperm.txt 2>&1
$PY -m modeng.stack_gate --channels $CH --tag gate_ctx_aw3 --a-weight 3 > $O/gate_ctx_aw3.txt 2>&1
