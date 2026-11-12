#!/bin/bash
# Implementer C: sequential stacker queue (one stacker at a time, RAM guard >= 15 GB available)
cd /home/ph/code/modeng; export PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=~/biotools/venvs/bio-net/bin/python
wait_ok() { while pgrep -f "modeng.stack --bench forward --channels base,dc_ridge" >/dev/null 2>&1 || [ $(free -g | awk '/Mem/{print $7}') -lt 15 ]; do sleep 15; done; }
run() { wait_ok; echo "=== $*"; $PY -m modeng.stack "$@" 2>&1 | grep -v Warning | head -12; }
run --bench temporal --train past --channels base,donor,view,dc_ridge,dc_ridge_pos --tag bdv_dc_ridge
run --bench temporal --channels base --tag dcref_base
run --bench temporal --channels base,dc_trend,dc_delta,dc_directed --tag dc_prev
