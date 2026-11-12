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

Memory fix: the partial file kept every loaded atlas in a module cache (about 8 GB per process with the cosine view); two runs were
OOM-killed on the shared machine. Atlases are now freed after use; only the result matrices are memoised.

## Commands
```
PYTHONPATH=src python -m modeng.stack ...   # via a wrapper that also saves the score matrices (scratchpad runstack_g.py, = modeng.stack.run_past/run + np.save)
temporal --train past: base,donor,view                        tag G_bdv        (positive control = stackP_bdv)
temporal --train past: base,donor,view,pg_{up,upmag,dn,upr,dnr,ups,dns,coh,rsm,rsx}   tag G_bdv_pg
temporal --train past: base,donor,view,pg_rsm,pg_rsx           tag G_bdv_pgrs
temporal --train past: base,donor,view,pgneg_rsm,pgneg_rsx     tag G_bdv_pgnegrs  (negative control)
python -m modeng.compare --bench temporal --split <dev|confirm> --a stackP_G_bdv_pg --b stackP_G_bdv --metric <m>
```
Zero-transfer-support stratum: positives whose `transfer_jaccard` score in their task is 0 (61 % of temporal dev positives,
60 % of confirm); pooled recall@100 and per-task AP of these positives with the other positives removed (scratchpad `stratum.py`).

## Results

### Each channel alone (temporal, uncovered genes put at the bottom; base rate AP about 0.0025)
| channel | coverage | dev AP | dev AUROC | dev R@100 | confirm AP | confirm AUROC | confirm R@100 |
|---|---|---|---|---|---|---|---|
| pg_up (regulator) | 0.68 | 0.0030 | 0.539 | 0.012 | 0.0028 | 0.549 | 0.012 |
| pg_upmag | 0.68 | 0.0030 | 0.546 | 0.009 | 0.0032 | 0.551 | 0.011 |
| pg_dn (responder) | 0.58 | 0.0060 | 0.540 | 0.013 | 0.0033 | 0.551 | 0.008 |
| pg_upr (routed) | 0.34 | 0.0032 | 0.506 | 0.012 | 0.0029 | 0.512 | 0.009 |
| pg_dnr (routed) | 0.45 | 0.0047 | 0.514 | 0.011 | 0.0031 | 0.531 | 0.007 |
| pg_ups / pg_dns | 0.68 / 0.58 | 0.0027 / 0.0031 | 0.538 / 0.543 | | 0.0023 / 0.0029 | 0.547 / 0.550 | |
| **pg_rsm (co-response)** | 0.61 | **0.0076** | **0.579** | **0.026** | **0.0058** | **0.592** | **0.023** |
| pg_rsx | 0.61 | 0.0070 | 0.579 | 0.030 | 0.0062 | 0.585 | 0.023 |
| pgneg_up / pgneg_dn / pgneg_rsm (dev) | same | 0.0025 / 0.0027 / 0.0034 | 0.534 / 0.539 / 0.535 | 0.005 / 0.007 / 0.005 | | | |
| transfer_jaccard (reference) | 1.0 | 0.122 | 0.682 | 0.257 | 0.119 | 0.705 | 0.283 |

Critic's kill test (r1_critic_stats item 8: AUROC of targets among **covered** genes, forward dev, 842 tasks; zero-support = targets with
transfer_jaccard 0):

| channel | covered AUROC, all targets | covered AUROC, zero-support targets |
|---|---|---|
| pg_up / pg_upmag / pg_ups | 0.494 / 0.495 / 0.501 | 0.503 / 0.500 / 0.511 |
| pg_dn / pg_dns | 0.491 / 0.504 | 0.497 / 0.509 |
| pgnc_up / pgnc_dn (no CD4) | 0.477 / 0.496 | 0.482 / 0.500 |
| pgneg_up / pgneg_dn | 0.489 / 0.479 | 0.493 / 0.485 |
| **pg_rsm / pg_rsx** | **0.583 / 0.587** | **0.551 / 0.556** |
| pgnc_rsm | 0.575 | 0.543 |
| pgneg_rsm | 0.480 | 0.493 |

The regulator and responder statistics (CI-01 / SC-1 as specified) are **at chance among covered genes**: their small alone-AUROC
above 0.5 is the coverage effect (the permuted control reproduces it). Kill criterion met (< 0.55). The readout co-response
(gene g responds to the same knock-downs as the seeds) passes it (0.58, zero-support 0.55), with or without CD4.

### Increment over stackP_bdv (temporal, train on `past`, 900 episodes; paired bootstrap over modet clusters)
Positive control: G_bdv reproduces stackP_bdv exactly (dev AP 0.1018 / nuAP 0.1237; confirm 0.0913 / 0.1506).

| run | split | AP | nuAP | AUROC | R@100 | onnet AP | dAP [95 % CI] | dnuAP [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| G_bdv (ref) | dev | 0.1018 | 0.1237 | 0.8103 | 0.2523 | 0.1210 | | |
| G_bdv_pg (all 10) | dev | 0.0961 | 0.1143 | 0.8082 | 0.2472 | 0.1168 | -0.0057 [-0.0100, -0.0018] | -0.0094 [-0.0180, -0.0020] |
| G_bdv (ref) | confirm | 0.0913 | 0.1506 | 0.8252 | 0.2598 | 0.0987 | | |
| G_bdv_pg (all 10) | confirm | 0.0920 | 0.1466 | 0.8248 | 0.2607 | 0.1018 | +0.0007 [-0.0037, +0.0046] | -0.0040 [-0.0116, +0.0017] |
| G_bdv_pgrs (co-response only) | dev | 0.0988 | 0.1213 | 0.8085 | 0.2480 | 0.1153 | -0.0030 [-0.0087, +0.0025] | -0.0023 [-0.0103, +0.0050] |
| G_bdv_pgrs (co-response only) | confirm | 0.0901 | 0.1474 | 0.8229 | 0.2558 | 0.0983 | -0.0011 [-0.0049, +0.0026] | -0.0032 [-0.0102, +0.0026] |
| G_bdv_pgnegrs (neg. control: permuted ids) | dev | 0.0979 | 0.1177 | 0.8097 | 0.2528 | 0.1149 | -0.0040 [-0.0079, -0.0005] | -0.0060 [-0.0114, -0.0013] |
| G_bdv_pgnegrs (neg. control: permuted ids) | confirm | 0.0898 | 0.1497 | 0.8239 | 0.2524 | 0.0970 | -0.0015 [-0.0051, +0.0020] | -0.0009 [-0.0082, +0.0047] |

Real vs negative control (pgrs - pgnegrs): dev AP +0.0010 [-0.0034, +0.0059], nuAP +0.0037 [-0.0034, +0.0122]; confirm AP +0.0003
[-0.0021, +0.0028], nuAP -0.0023 [-0.0053, +0.0013]. Adding two pure-noise channels costs the past-trained stacker about 0.004 AP /
0.006 nuAP on dev, so the stacker noise floor for a 2-channel add is of the order of the effects seen; the real co-response channel
is indistinguishable from its permuted control.

Zero-transfer-support stratum (G_bdv_pg vs G_bdv): dev R@100 0.0259 vs 0.0242 (+0.0017 [-0.0036, +0.0069]), AP 0.0121 vs 0.0146
(-0.0025 [-0.0081, +0.0013]); confirm R@100 0.0280 vs 0.0295 (-0.0015 [-0.0046, +0.0019]), AP 0.0111 vs 0.0121 (-0.0010 [-0.0024, -0.0000]).
Zero-support stratum (G_bdv_pgrs vs G_bdv): dev R@100 0.0276 vs 0.0242 (+0.0035 [-0.0010, +0.0075]), AP -0.0005 [-0.0024, +0.0008];
confirm R@100 0.0288 vs 0.0295 (-0.0008 [-0.0039, +0.0021]), AP -0.0012 [-0.0033, +0.0002]. No stratum gain.

Under the F2-3 rule (a gain needs both AP and nuAP up) neither set is a gain on either split: the full set is significantly worse on
dev and neutral on confirm; the co-response-only set is neutral-to-slightly-negative everywhere. The co-response signal that exists alone
(AUROC 0.58) is already carried by bdv (expression views hpa_sc / gtex and the row-wise ps_* perturbation views).

The largest perturbation feature by gain in G_bdv_pg was `pg_coh` (a task-level constant, the seed-program coherence), i.e. the
stacker used the routing/coherence statistic as a task descriptor, not the gene scores; that is the likely source of the dev loss
(distribution shift between 900 past episodes and 53 dev tasks).

### Forward (cv)
Not run to completion: the forward CV stack with the date-fair `pgnc_*` set was OOM-killed once (machine at 83-90 GB) and the
lead's one-stacker rule plus the 06:40 deadline left no slot. The forward evidence is the channel-alone kill test above
(forward dev, 842 tasks). Not claimed.

## Verdict
**Discard** the regulator / responder / routing channels (CI-01, SC-1 as specified, SC-3/SC-9 lite): at chance among covered genes on
842 forward tasks, no increment on temporal (dev significantly negative with all 10 channels). The co-response view has a real alone
signal (covered AUROC 0.58, zero-support 0.55, permuted 0.48) but no increment over bdv: **promising only as a view** (see New ideas),
not as a stacker channel. Forward (cv) increment not run (memory / deadline).

## Leakage class
No labels and no context are read; seeds only. Every atlas is dated after OT 21.04 (Replogle 2022; Nadig GEO 2024-05; CD4 bioRxiv
2025-12), so on **temporal** all channels are **transductive** (label-free experimental data). On **forward** (seeds 25.09) the
K562 / RPE1 / HepG2 / Jurkat atlases are fair; CD4 2025 is after the seed date (use `pgnc_*`). Holdout: holdout-current.
The per-gene standardisation across tasks (`pg_ups`, `pg_dns`) uses the split's own seed sets (label-free, transductive across tasks).

## Runtime
Channel computation: about 20 s per atlas per split for the nulls and scores (6 atlases), 1-2 min per split for 53-842 tasks;
past train900 about 3 min. Stacker runs 15-25 min each on the overloaded machine (load 140-160).

## What failed / caveats
- The core CI-01 / SC-1 hypothesis (knock-downs that move the seed program are disease genes; genes that respond to seed knock-downs are
  disease genes) gives no signal among covered genes on 842 forward tasks, although the statistics recover textbook regulators
  (SREBF2 / SCAP / MBTPS2 for cholesterol synthesis). Reason: most disease seed sets are not a transcriptional program in K562 /
  RPE1 / HepG2 / Jurkat / CD4 T cells (median excess coherence 0.07-0.29 vs 0.01-0.07 for permuted ids; real but small).
- Routing by coherence (SC-3 / SC-9 lite: `pg_upr`, `pg_dnr`) is worse than the max over atlases.
- CRISPRi cis effects (bidirectional promoters) make genomic neighbours of seeds look like regulators/responders; not corrected.
- Two runs lost to OOM (my module's atlas cache; fixed) and one to a bug introduced by the fix (fixed, rerun).

## New ideas
- The co-response view (columns of Perturb-seq) is the only part with signal; it is a co-regulation network that covers non-perturbed
  genes. Build it once as a gene view in `data/views` (genes x perturbations, after PC removal) so `vmean/vmax` and the `km_` donors
  get it for free, instead of a bespoke channel.
- Per-seed (not per-set) regulator scores: max over seeds of |z| for perturbation g, to catch one-seed regulators in heterogeneous sets.
- Use coherence only as a reliability weight for the co-response view, not as a stacker feature.
