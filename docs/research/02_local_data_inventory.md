# Local data inventory (already downloaded by modet and subex; reuse before downloading) — 2026-11-12
Read-only for modeng; copy or derive into `modeng/data/` when needed. Provenance/sha256: subex `manifests/data_manifest.tsv` (750 rows) and `docs/data.md`;
modet `data/processed/features/MANIFEST.tsv`, `docs/raw_data_downloads_2026-11.tsv`.

## /mnt/msadb/modet/downloads (229 GB)  (also symlinked under modet/data/raw)
GWAS: `gwas_sumstats` (72 GB, post-T0), `gwas_sumstats_t0` (90 GB), `magma` (3.4 GB), `neale_rg(_table)` (UKB genetic correlations), `cs2g`, `pops_features` (25 GB).
Open Targets: `ot2104`, `ot_evidence/{evidence_2104,evidence_2509,study_2509}` (4.2 GB, per-evidence rows incl. source/PMID), `v2/{ot2306_assoc,ot2306_disease,ot2609_assoc,ot2609_disease}`,
`ot_chembl_2104/2509`, `ot_known_drug_2509`. Networks/KG: `humanbase` (13 GB tissue networks), `humannet_v2`, `string_v11.0`, `string_v11.5_links_full`, `bioplex3`, `corum_v3_0`,
`complex_portal_2021-03-10`, `primekg`, `obnb` (8 GB), `dream` (DREAM module challenge networks). Other: `depmap_21q1`, `gnomad_v2.1.1`, `hpa_v20.1`, `impc_release-13.0`,
`msigdb_v7.4`, `hpo_annotations_2021-01_wayback`, `ensembl103_compara`, `uniprot`, `orgHsEgDb`, `minikel2024`, `gps_duffy2024`.
## modet/data/processed
`groundtruth/` (OT 19.04/21.04/25.09/26.09 sets, panels, arrival classes), `networks/` (STRING/BioGRID/HuRI/union), `features/` (ESM C 300M, GO T0, GTEx v8, Reactome T0),
`covariates/`, `gwas_magma/` (38 GB post-T0 per-trait MAGMA gene z), `gwas_magma_t0/` (4.4 GB), `obnb/`.
## /mnt/msadb/subex (152 GB)
raw: `tabula_sapiens_v2` (54 GB), `cd4_perturbseq` (58 GB, 2025 genome-scale CD4 T-cell Perturb-seq), `nadig2025` (14 GB, HepG2/Jurkat Perturb-seq), `perturbseq` (Replogle K562/RPE1),
`depmap_20q4`, `depmap_current(_portal)`, `afdb_v4_human` + `afdb_clusters_v3` (AlphaFold), `uce_esm2_protein_embeddings`, `uniprot_prott5_embeddings`, `esmc_sprot2021_02`,
`meltome`, `proteomehd`, `cptac_pre2021`, `human_cell_map`, `cladeoscope` (phylogenetic profiles), `human_gem`, `intact`, `biogrid*` (3.4.157, 4.4.233), `string_v12`,
`zhang2025_humanppi`, `burke2023_huintaf2` (AF2 human interactome), `wright2025`, `complexes`, `go_2018_02/2024_04_24/archive/current`, `reactome_v88`, `hpa*`, `gtex*`, `ensembl104`, `ncbi`.
derived (ready matrices, gene lists alongside): DepMap 20Q4 (raw/GLS/PCA/robust-PCA), expression (GTEx, HPA tissue, HPA single cell), Perturb-seq signatures
(K562 GWPS + essential, RPE1, HepG2, Jurkat, CD4 2025 rest/stim; pc10 / systema variants), OpenCell pairs, temporal GO pairs, AFDB pLDDT, IntAct 2021/current.
## /mnt/msadb/databases: `uniref30_2302` (455 GB, for MSA)
