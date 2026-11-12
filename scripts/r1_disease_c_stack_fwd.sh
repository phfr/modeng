#!/bin/bash
# Implementer C increment tests on forward (cv on dev, dev-trained model on confirm)
cd /home/ph/code/modeng; export PYTHONPATH=src OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=~/biotools/venvs/bio-net/bin/python
run() { echo "=== $*"; $PY -m modeng.stack "$@" 2>&1 | grep -v Warning | head -12; }
run --bench forward --channels base --tag dcref_base
run --bench forward --channels base,dc_ridge,dc_ridge_pos --tag dc_ridge
run --bench forward --channels base,dc_trend,dc_delta,dc_directed --tag dc_prev
run --bench forward --channels base,dc_learned --tag dc_learned
run --bench forward --channels base,dc_negperm_dc_ridge,dc_negperm_dc_ridge_pos --tag dc_negperm
