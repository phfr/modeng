#!/bin/bash
# Implementer C increment tests on temporal (--train past for past-defined channels, cv for previous-release channels)
cd /home/ph/code/modeng; export PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=~/biotools/venvs/bio-net/bin/python
run() { echo "=== $*"; $PY -m modeng.stack "$@" 2>&1 | grep -v Warning | head -12; }
while pgrep -f "train past --channels base --tag dcref_base" >/dev/null 2>&1; do sleep 10; done   # first run (started earlier)
run --bench temporal --train past --channels base,dc_ridge,dc_ridge_pos --tag dc_ridge
run --bench temporal --train past --channels base,dc_learned --tag dc_learned
run --bench temporal --channels base --tag dcref_base
run --bench temporal --channels base,dc_trend,dc_delta,dc_directed --tag dc_prev
run --bench temporal --train past --channels base,dc_negperm_dc_ridge,dc_negperm_dc_ridge_pos --tag dc_negperm
run --bench temporal --train past --channels base,donor,view,dc_ridge,dc_ridge_pos --tag bdv_dc_ridge
