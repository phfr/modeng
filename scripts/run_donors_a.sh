#!/bin/bash
# Increment tests for implementer A donor channels. Usage: run_donors_a.sh temporal|forward
cd /home/ph/code/modeng
export OMP_NUM_THREADS=4 PYTHONPATH=/home/ph/code/modeng/src
P=~/biotools/venvs/bio-net/bin/python
L=results/donors_a
NEW="dA_a_impc_orth dA_a_dm_codep dA_a_ps_k562_resp dA_a_ps_k562_reg dA_a_ps_rpe1_resp dA_a_ps_rpe1_reg dA_a_ps_cd4_resp dA_a_ps_cd4_reg dA_pt"
OLD=$(for f in $(cut -f1 data/donors/MANIFEST.tsv); do printf "dA_%s," $f; done); OLD=${OLD%,}
if [ "$1" = temporal ]; then
  $P -m modeng.stack --bench temporal --train past --channels base,donor,view --tag bdv_refA > $L/tP_bdv_refA.log 2>&1
  for c in $NEW; do $P -m modeng.stack --bench temporal --train past --channels base,donor,view,$c --tag bdv_$c > $L/tP_bdv_$c.log 2>&1; done
  $P -m modeng.stack --bench temporal --train past --channels base,donor,view,dA_perm_a_impc_orth,dA_perm_hpo2101 --tag bdv_permA > $L/tP_bdv_permA.log 2>&1
  $P -m modeng.stack --bench temporal --train past --channels base,donor,view,$OLD --tag bdv_dAold > $L/tP_bdv_dAold.log 2>&1
fi
if [ "$1" = forward ]; then
  ALLN=$(echo $NEW | tr ' ' ',' | sed 's/,dA_a_ps_cd4_resp,dA_a_ps_cd4_reg//')
  $P -m modeng.stack --bench forward --channels base,dA_pt --tag base_dA_pt > $L/fwd_base_dA_pt.log 2>&1 &
  $P -m modeng.stack --bench forward --channels base,$ALLN --tag base_dAnew > $L/fwd_base_dAnew.log 2>&1 &
  # base_permA launched separately
  wait
fi
if [ "$1" = forward_bdv ]; then
  ALLN=$(echo $NEW | tr ' ' ',' | sed 's/,dA_a_ps_cd4_resp,dA_a_ps_cd4_reg//')
  $P -m modeng.stack --bench forward --channels base,donor,view,$ALLN --tag bdv_dAnew > $L/fwd_bdv_dAnew.log 2>&1
fi
