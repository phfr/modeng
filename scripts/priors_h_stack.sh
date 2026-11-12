#!/bin/bash
# implementer H: increment tests (stacker protocol of modeng.stack.run, --train cv, LightGBM 4 threads) for src/modeng/ext/priors_h.py
B=$1; shift
PY=~/biotools/venvs/bio-net/bin/python
export PYTHONPATH=/home/ph/code/modeng/src OMP_NUM_THREADS=4
if [ "$B" = temporal ]; then
  PRI=pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,hz_gwas,hz_mend,pop_europepmc,pop_phenodigm,pop_chembl,pop_expression_atlas,pop_eva,pop_ot_genetics_portal
else
  PRI=pri_l2g_n,pri_l2g_max,pri_hpo_n,hz_gwas,hz_mend,pop_gwas_credible_sets
fi
TR=tr_gwas,tr_nongwas
PH=ph_mean,ph_top3
NEG=ph_mean_perm,pri_l2g_n_perm
declare -A C=([h_base]=base [h_base_all]=base,$PRI,$TR,$PH [h_base_ph]=base,$PH [h_base_tr]=base,$TR [h_base_pri]=base,$PRI
              [h_base_neg]=base,$NEG [h_bdv]=base,donor,view [h_bdv_all]=base,donor,view,$PRI,$TR,$PH)
for tag in "$@"; do
  echo "== $tag ${C[$tag]}"
  $PY scripts/priors_h_strata.py $tag ${C[$tag]} $B 2>&1 | grep -v -i warning
done
