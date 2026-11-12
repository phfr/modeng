#!/bin/bash
# P4: the one confirmatory run on wide_confirm (pre-registration: docs/preregistration_wide.md Amendment 1, commit 2fb1b77)
PY=/home/ph/biotools/venvs/bio-net/bin/python
R=/home/ph/code/modeng
V1C=$(cat $R/results/v1clean_channels.txt)
NOLAB=$(echo "$V1C" | tr ',' '\n' | grep -vxE 'gwas_catlag|pri_l2g_n|pri_l2g_max|hz_gwas' | paste -sd,)
echo "=== A+B $(date '+%F %T') free $(free -g | awk '/Mem/{print $7}') GB"
OMP_NUM_THREADS=8 MODENG_SCORE_WIDE_CONFIRM=1 $PY $R/scripts/p4_score.py wide_confirm
echo "=== C $(date '+%F %T') free $(free -g | awk '/Mem/{print $7}') GB"
(cd $R && OMP_NUM_THREADS=8 MODENG_SAVE_SCORES=1 MODENG_SCORE_WIDE_CONFIRM=1 PYTHONPATH=$R/src $PY -m modeng.stack --train wide --score-confirm --pos-deg-power 0.35 --channels "$V1C" --tag P4C_v1clean)
echo "=== D ($(echo $NOLAB | tr ',' '\n' | wc -l) channels) $(date '+%F %T') free $(free -g | awk '/Mem/{print $7}') GB"
(cd $R && OMP_NUM_THREADS=8 MODENG_SAVE_SCORES=1 MODENG_SCORE_WIDE_CONFIRM=1 PYTHONPATH=$R/src $PY -m modeng.stack --train wide --score-confirm --pos-deg-power 0.35 --channels "$NOLAB" --tag P4D_v1clean_nolab)
echo "=== done $(date '+%F %T')"
