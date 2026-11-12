#!/bin/bash
# r3 strict-a: gated stacker variants (src/modeng/stack_gate.py), v1 channels, power 0.35. Sequential, 8 threads.
export OMP_NUM_THREADS=8 PYTHONPATH=/home/ph/code/modeng/src
PY=~/biotools/venvs/bio-net/bin/python
CH=base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,hz_gwas,hz_mend,ph_mean,ph_top3,pt_ridge,pt_ridge_pos
O=/home/ph/code/modeng/results/r3_strict_a
cd /home/ph/code/modeng
$PY -m modeng.stack_gate --channels $CH --tag gate_ctx --head > $O/gate_ctx.txt 2>&1
$PY -m modeng.stack_gate --channels $CH --tag gate_noctx --no-ctx --head > $O/gate_noctx.txt 2>&1
$PY -m modeng.stack_gate --channels $CH --tag gate_ctxperm --ctx-perm > $O/gate_ctxperm.txt 2>&1
$PY -m modeng.stack_gate --channels $CH --tag gate_ctxta --area > $O/gate_ctxta.txt 2>&1
echo DONE > $O/runs.done
