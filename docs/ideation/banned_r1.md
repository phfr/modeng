# Already explored (do not re-propose unless you name a NEW reason it would work now)
From modet (measured, mostly no gain beyond transfer + covariates + degree): network propagation (RWR, dc-RWR, DIAMOnD, heat), GNN over PPI (= MLP),
GenePlexus, denser STRING with text mining, KG embeddings (PrimeKG/OptimusKG RotatE/ComplEx) as features, HumanNet / functional / tissue networks as pooled features,
PoPS-like ridge head on 57k features, GTEx/HPA tissue-context weighting, EASE set completion, GO/Reactome "function users", CatBoost/XGBoost swaps,
locus prior (genes near seeds), T0 UKB MAGMA studies as donors, provenance-restricted transfer, tier-0 gene-propensity prior, reserving top-100 slots for propagation,
calibration (isotonic) of ranks, degree weighting tricks (deg^-0.5 positive weights).
From modetor planning (ideas only, not run): identifier-repair label audit, arrival-class decomposition, thinning GWAS to simulate arrivals, short-list endpoints,
bundle/locus units, transposed task (gene -> next disease), rolling-origin windows, ancestry holdout, mouse KO oracle, smoothing over network.
From subex: annotation (GO/Reactome) Leiden clustering for module decomposition (confirmed best decomposition).
