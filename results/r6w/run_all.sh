#!/bin/bash
cd /home/ph/code/modeng
PY=~/biotools/venvs/bio-net/bin/python
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
for a in "wide v1clean 0" "wide v2 0" "wide v1clean 1" "wide v2 1" "wide v1perm 0" "cv2wide v1clean" "past v1clean" "past v2"; do
  echo "=== $a $(date +%T) free: $(free -g | awk '/Mem/{print $7}') GB"
  /usr/bin/time -f "TIME %e s MAXRSS %M KB" $PY scripts/r6_wide_full.py $a 2>&1 | grep -v -i warning
done
echo DONE
