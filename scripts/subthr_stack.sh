#!/bin/bash
# r4: increment of the sub-threshold channels over modeng-v1 (stack_v3ptr_p0.35), plus same-width permuted control. Sequential, 8 threads.
V1=base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,hz_gwas,hz_mend,ph_mean,ph_top3,pt_ridge,pt_ridge_pos
PY=/home/ph/biotools/venvs/bio-net/bin/python
export PYTHONPATH=/home/ph/code/modeng/src OMP_NUM_THREADS=8 MODENG_SAVE_SCORES=1
cd /home/ph/code/modeng
for spec in "$@"; do
  tag=${spec%%:*}; ch=${spec#*:}
  echo "=== $tag $ch $(date)"
  $PY -m modeng.stack --bench temporal --channels $V1,$ch --pos-deg-power 0.35 --tag $tag > results/subthr/$tag.log 2>&1
  tail -40 results/subthr/$tag.log | grep -E "temporal|subthr"
done
