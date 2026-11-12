# r1 implementer G: perturbation biology (regulators of the disease program)

Ideas: CI-01 (upstream-of-core), SC-1 (virtual Perturb-seq screen, seed module as readout), SC-3 / SC-9 lite (cell-context routing
by perturbational coherence). Date: 2026-11-12, round 1 (restarted 04:15).

## What was built
`src/modeng/ext/perturb_g.py` (partial, untested file from the stopped attempt; reviewed, extended, tested). Atlases from
`/mnt/msadb/subex/derived/ps_*.npy` (rows = CRISPRi target, columns = measured readout gene; 5 PCs projected out, rows standardised):
K562 GWPS (9,820 x 7,941 after mapping to the universe), RPE1 essential, HepG2 / Jurkat essential (Nadig 2025), primary CD4 T 2025 Stim8hr
and Rest. Only task seeds are used; no labels, no context.

Review of the partial file (all checked):
- **Axis.** Row ids = `ps_<x>_genes.txt` (targets, consistent with the row order of both build paths in subex `build_views.py` /
  `build_t1_views.py`); readout ids = `var.gene_id` of the Replogle h5ad (plain, not categorical; `__categories` only holds gene_name)
  or `genes` of `ps_<x>_bulk.npz` (non-ENSG readouts such as `CUSTOM001_PuroR` drop out). Shapes asserted against the matrix.
- **Gene order.** Both axes mapped to `modeng.data.universe()` (sorted ENSG) with `get_indexer`; first occurrence kept.
- **Normalisation.** Regulator z: signed mean (and |.| mean) of row g over the seed readouts, null mean = per-row mean over random
  readout sets matched on responsiveness decile (analytic), null sd = 48 random readout sets per size on a log grid (captures readout
  correlation). Responder z: mean of column g over the seed perturbations vs column mean, null sd from random perturbation sets.
- **Sanity check (positive biology control).** Seeds = 14 cholesterol-synthesis genes: K562 regulators rank MBTPS2, SREBF2, SCAP, INSIG1
  in the top 12 (|z| 14-17); responders rank ACAT2, TMEM97, ERG28, MVD, FDPS, HMGCS1 on top. The statistics do what they claim.
- Added: **readout co-response** (`pg_rsm`, `pg_rsx`, SC-1 lite "seed module as readout"): cosine of gene g's response profile across all
  perturbations to the seeds' profiles (mean, top-3 mean), max over atlases. This is the column view; the existing `ps_*` views in
  `data/views` are the row (perturbation-profile) view.

Channels (each `[n_tasks, n_genes]`, NaN = gene not covered): `pg_up` (max |regulator z|), `pg_upmag`, `pg_dn` (max |responder z|),
`pg_upr` / `pg_dnr` (routed to the atlas with the largest seed-program coherence = SC-3/SC-9 lite), `pg_ups` / `pg_dns` (per-gene
standardised across tasks: removes "regulates everything"), `pg_coh` (task-level coherence), `pg_rsm`, `pg_rsx`.
Variants: `pgnc_*` = without the CD4 2025 atlases (date-fair on forward), `pgneg_*` = negative control (row ids and readout ids
permuted within each atlas: identical coverage set, gene-content link broken; this is also the critic's "perturbed-indicator control").

RESULTS_PLACEHOLDER
