# Pre-registration: modeng-v1 frozen recipe (2026-11-12 07:25, code at commit 9e0c20e)
Purpose: one confirmatory run on diseases/time windows NOT used overnight (all temporal dev/confirm and forward dev/confirm results are development numbers).
- **Model:** `python -m modeng.stack --bench <bench> --channels base,donor,view,cs_donor,dA_pt,dc_ridge,dc_ridge_pos,gwas_catlag,pri_l2g_n,pri_l2g_max,pri_mend_n,pri_hpo_n,hz_gwas,hz_mend,ph_mean,ph_top3,pt_ridge,pt_ridge_pos --pos-deg-power 0.35`
  (LightGBM params as in src/modeng/stack.py LGB, 400 rounds, 3000 negatives per task, trained on all available training tasks of the fresh benchmark's training split; no tuning on the fresh fold).
- **Comparators:** transfer_jaccard; modet headline where available (onnet).
- **Primary endpoint:** paired difference in AP AND nuAP vs transfer_jaccard, cluster bootstrap (10,000), both must exclude 0 for a claim; secondary: degree-matched AP, P@20, strict-a (if defined), onnet vs modet headline.
- **Candidate fresh folds (in order):** (1) the next Open Targets release after 26.09 (seeds 26.09, targets new), (2) diseases outside every panel used here with >= 5 seeds and >= 3 new genes in 25.09 -> 26.09 excluded from forward dev/confirm (none exist by construction: needs a new release), (3) modet's sealed fold only with the user's authorisation via modet's ledger.
- **Leakage classes to report:** label-process channels (gwas_catlag, pri_l2g_*) and an arm without them; PubTator/GWAS-Catalog mapping vintage.
- **Outcome sentences:** claim only if both CIs exclude 0; otherwise "not confirmed on a fresh fold".

## Amendment 1 (2026-11-12 14:00, before any fresh-fold data exists)
Add a co-primary **clean arm** `v1clean`: the v1 recipe with the post-dated views (hpa_sc, hpa_tissue, gtex [subex expr_*], ps_*) and their k-means donor families
(km_hpa_sc, km_gtex, km_ps_k562, km_ps_cd4) removed and replaced by vc_mean_/vc_max_ of HPA v20.1 single-cell, HPA v20.1 tissue and GTEx v8
(channel list: results/v1clean_channels.txt). On temporal confirm it is equivalent to v1 (all deltas within noise). Also report an arm without the
label-process channels (gwas_catlag, pri_l2g_n, pri_l2g_max, hz_gwas), as recommended by the R3 red team.
