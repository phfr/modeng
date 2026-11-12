# Round-1 measured results (for round-2 ideators and reviewers; 2026-11-12 ~06:05)
Primary: temporal benchmark (seeds OT 21.04, targets new by 25.09), all-gene universe, stacker trained on 1,786 `past` episodes (19.04->21.04),
dev = 53 diseases (model choices), confirm = 138 diseases. Claims need BOTH AP and nuAP (nuAP alone is gameable by degree down-weighting).
| model / channel | temporal confirm AP | nuAP | note |
|---|---|---|---|
| transfer_jaccard (raw) | 0.119 | 0.104 | strongest simple baseline |
| modet headline (onnet only comparable) | - | onnet ~0.17 | modet-U-cov-Lb |
| stackP_base (transfer, onto, popularity, degree, RWR) | 0.081 | 0.134 | deg^-0.5 weighted positives |
| stackP_bdv (+15 donor families, +30 view similarities) | 0.091 | 0.151 | |
| + cs_donor (joint non-negative lasso over disease rows) | 0.096 | 0.158 | Pareto vs bdv; perm control ~0 |
| + dA_pt (PubTator3 MeSH-disease literature donors, PMID date cut) | 0.102 | 0.166 | Pareto vs bdv (CI excl. 0) |
| + cs_donor + dA_pt | 0.111 | 0.175 | |
| + dc_ridge (EASE-style disease ridge) | 0.103 | 0.169 | Pareto vs bdv, also forward |
| + gwas_catlag (GWAS Catalog ingestion lag) | +0.008 | +0.003 | dev significant; GWAS-only AP x2.5 |
Negative / null: evolved transfer kernel (nuAP via degree only), seed-set transformer (memorises arrivals; aux-only = transfer level), gene-kernel/homology lifts
(AUROC only), Perturb-seq regulators/responders (chance), perturbation/DepMap/IMPC donor sets (random), trending/directed transfer, learned disease encoder,
own/proxy-trait MAGMA z, cs_universal (no gain beyond cs_donor).
Label-process (forward 25.09->26.09): backlog channels (sub-threshold evidence at seed release) stack nuAP 0.137 -> 0.307, AP 0.077 -> 0.196; catalog lag 0.137 -> 0.199.
Leaks caught: float-residue tie leak in masked transfer; past-cluster CV leak; universe-order bug; holdout ancestor tell.
Reports: docs/reports/r1_*.md, r2_evolve.md; facts: docs/insights.md; dead ends: docs/invalidations.md.
