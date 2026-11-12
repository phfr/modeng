#!/bin/bash
# One stacker at a time (lead rule: RAM); start only with >= 15 GB available.
cd /home/ph/code/modeng; export OMP_NUM_THREADS=4 PYTHONPATH=/home/ph/code/modeng/src; P=~/biotools/venvs/bio-net/bin/python; L=results/donors_a
waitfree() { while pgrep -f "modeng.stack.*--tag (bdv_dA_pt|bdv_permA|bdv_dAps|base_dA_pt)" >/dev/null || [ $(free -g | awk '/Mem/{print $7}') -lt 15 ]; do sleep 20; done; }
waitfree; $P -m modeng.stack --bench temporal --train past --channels base,donor,view,dA_perm_a_impc_orth,dA_perm_hpo2101 --tag bdv_permA > $L/tP_bdv_permA.log 2>&1
waitfree; $P -m modeng.stack --bench temporal --train past --channels base,donor,view,dA_pt,dA_a_impc_orth,dA_a_dm_codep --tag bdv_dApre21 > $L/tP_bdv_dApre21.log 2>&1
waitfree; $P -m modeng.stack --bench forward --channels base,dA_pt --tag base_dA_pt > $L/fwd_base_dA_pt.log 2>&1
