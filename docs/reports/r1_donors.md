# r1 donors (implementer A): virtual donors / universal donor transfer (SC-2, LT-2, EV-1; IR-10 not done)

Date 2026-11-12, 04:13-06:40 (machine load 120-170, RAM 80+/90 GB for most of the window; three of my stacker runs were
OOM-killed or stopped under the lead's one-stacker RAM rule, see "What failed").

## Verdict
- **KEEP `dA_pt`** (PubTator3 gene x MeSH-disease co-mention donors, PMID-dated per benchmark). It is the only new family with a
  gain. On temporal (`--train past`) it raises AP **and** nuAP over `stackP_bdv`, and the gain holds on confirm:
  confirm nuAP +0.0148 [+0.0055, +0.0238], AP +0.0106 [+0.0026, +0.0186].
- **DISCARD as donors** the experimental families: Perturb-seq responder / regulator sets (K562, RPE1, CD4), DepMap selective
  co-dependency sets and IMPC-with-orthologs. Used alone, each scores at the level of random or the permuted control
  (nuAP 0.002-0.009; random 0.003). The combined increment run for these families was killed (see "What failed").
- **KEEP the hypergeometric + IDF scorer** (`dA_<fam>`) as a small upgrade of the stock Jaccard donor transfer. It was tuned on
  temporal dev only. On confirm, which was not used for tuning, HPO-2101 goes from nuAP 0.068 to 0.077 and AP from 0.080 to 0.086.

## What was built
| file | content |
|---|---|
| `scripts/build_donors_a.py` | builds the new families into `data/donors/a_*.npz`. They are listed in a **separate** `data/donors/MANIFEST_A.tsv`, so the default `donor` group (and every `stack_bdv` reference) stays unchanged |
| `src/modeng/ext/donors_a.py` | channels: `dA_<fam>` (hyper scorer, any family), `dA_pt` (PubTator family picked by benchmark), `dJ_<fam>` (stock Jaccard), negative controls `dA_perm_<fam>` / `dJ_perm_<fam>` (gene identities permuted) |
| `scripts/tune_donors_a.py` | scorer grid on temporal dev only -> `results/donors_a/tune.tsv` |
| `scripts/alone_donors_a.py` | channel-alone metrics -> `results/donors_a/alone.tsv` |
| `scripts/run_donors_a.sh`, `scripts/queue_donors_a.sh` | increment runs (the queue runs one stacker at a time and starts a run only when >= 15 GB RAM is free) |
| `scripts/summ_donors_a.py` | paired cluster bootstrap of every run against its reference -> `results/donors_a/increments.tsv` |
| `data/raw/pubtator3/gene2pubtator3.gz` | downloaded, 758 MB, sha256 in `docs/reports/downloads.tsv` |

### New donor families (data/donors/MANIFEST_A.tsv)
| family | sets | median size | genes | date class | content |
|---|---|---|---|---|---|
| a_pt19 / a_pt21 / a_pt25 | 9,408 / 9,687 / 9,997 | 300 (cap) | 12.6k-13.1k | PMID cut | genes co-mentioned with a MeSH disease in >= 2 PMIDs (top 300 by PMID count). Cuts: PMID <= 30,900,000 (entry 2019-03-21) / <= 33,750,000 (2021-03-22) / <= 40,850,000 (2025-08-20). Cut dates were checked by fetching the PubMed record of each cut PMID |
| a_ps_{k562,rpe1,cd4}_resp | 2,384 / 1,991 / 2,871 | 16-27 | | 2022 / 2022 / 2025 | perturbed gene + readouts with \|z\| >= 4 (<= 200; perturbations with >= 10 hits). A first build at \|z\| >= 3 was mostly noise (about 22 false hits per perturbation are expected among 8k readouts) |
| a_ps_{k562,rpe1,cd4}_reg | 1,132 / 972 / 2,345 | 16-26 | | same | readout gene + perturbations with \|z\| >= 4 on it |
| a_dm_codep | 1,968 | 20 | 7,237 | pre21 | DepMap 20Q4 Chronos: selective genes (SD in the top 25 %, not common-essential), each with its top-25 co-dependent genes (r >= 0.2) |
| a_impc_orth | 1,045 | 46 | 6,155 | pre21 | IMPC 13.0 MP assertions mapped to human through MGI HOM orthology classes (instead of symbol upper-casing), with MP terms propagated to their is_a ancestors |

### Scorer (`dA_<fam>`)
Setup: seeds restricted to the genes the family covers; N = number of covered genes. Each donor set gets the weight
w = -log10 hypergeometric P(overlap >= k). The top-K donors are kept, and gene g scores
idf(g) * sum_d w_d * 1[g in d] / |d|^ALPHA, where idf(g) = log((1+n_sets)/(1+df_g)). Genes outside the family get NaN.
The grid was searched on **temporal dev only** (`results/donors_a/tune.tsv`, 5 families x 36 settings). Chosen setting:
K=50, ALPHA=0.5, IDF on, KMIN=1. Mean nuAP over the 5 families: 0.0211 vs 0.0190 for stock Jaccard.

## Results

### Channel alone (`results/donors_a/alone.tsv`)
| channel | temporal dev AP / nuAP / AUROC / R@100 | temporal confirm AP / nuAP / AUROC / R@100 | forward dev AP / nuAP |
|---|---|---|---|
| dA_pt | 0.089 / 0.062 / 0.755 / 0.232 | 0.093 / 0.095 / 0.773 / 0.258 | 0.047 / 0.050 |
| dA_hpo2101 (new scorer) | 0.092 / 0.071 / 0.693 / 0.185 | 0.086 / 0.077 / 0.720 / 0.211 | 0.034 / 0.039 |
| dJ_hpo2101 (stock Jaccard) | 0.089 / 0.063 / 0.631 / 0.178 | 0.080 / 0.068 / 0.648 / 0.205 | |
| dA_a_dm_codep | 0.014 / 0.009 | 0.008 / 0.010 | 0.006 / 0.005 |
| dA_a_impc_orth | 0.004 / 0.005 | 0.004 / 0.006 | 0.005 / 0.006 |
| dA_a_ps_* (6 families) | 0.004-0.006 / 0.002-0.007 | 0.004-0.006 / 0.002-0.005 | 0.004-0.007 / 0.004-0.006 |
| neg. control dA_perm_hpo2101 / dA_perm_a_impc_orth | 0.002 / 0.003 | 0.002 / 0.003 | 0.004 / 0.004 |
| (reference) transfer_jaccard | 0.123 / 0.096 | 0.119 / 0.104 | 0.094 / 0.096 |
| (reference) random | 0.002 / 0.003 | 0.002 / 0.003 | 0.004 / 0.004 |

### Increment tests (stacker, `modeng.stack`; paired cluster bootstrap, `results/donors_a/increments.tsv`)
Temporal is trained on `past` episodes (`--train past`); the reference is `stackP_bdv` (the parquet on disk; dev nuAP 0.1237, confirm 0.1506).
| run | split | AP | nuAP | AUROC | R@100 | onnet nuAP | delta AP [95 % CI] | delta nuAP [95 % CI] |
|---|---|---|---|---|---|---|---|---|
| stackP_bdv (ref) | dev | 0.1018 | 0.1237 | 0.810 | 0.252 | 0.173 | | |
| **stackP_bdv_dA_pt** | dev | 0.1076 | 0.1320 | 0.817 | 0.261 | 0.175 | +0.0058 [-0.0001, +0.0121], P(<=0) 0.03 | +0.0083 [-0.0028, +0.0223], P 0.08 |
| stackP_bdv (ref) | confirm | 0.0913 | 0.1506 | 0.824 | 0.254 | 0.169 | | |
| **stackP_bdv_dA_pt** | confirm | 0.1019 | 0.1655 | 0.835 | 0.282 | 0.182 | **+0.0106 [+0.0026, +0.0186]** | **+0.0148 [+0.0055, +0.0238]** |
| neg. control stackP_bdv_permA (+ dA_perm_hpo2101, dA_perm_a_impc_orth) | dev | 0.0967 | 0.1202 | 0.810 | 0.253 | 0.165 | -0.0051 [-0.0087, -0.0018] | -0.0035 [-0.0100, +0.0024] |
| neg. control stackP_bdv_permA | confirm | 0.0899 | 0.1485 | 0.824 | 0.256 | 0.170 | -0.0014 [-0.0066, +0.0024] | -0.0021 [-0.0102, +0.0039] |

The negative control (two donor channels with gene identities permuted) gives about 0 or slightly negative change (extra noise
features cost a little AP). So the dA_pt gain is not an artefact of adding two more channels.

Run-to-run noise of the reference: the earlier `stackP_bdv` row (4b8527e) had dev nuAP 0.1269. My rerun `stackP_bdv_refA` gave 0.1237
(its confirm half was OOM-killed). The dev differences are therefore within about ±0.003 of noise; the confirm gain of dA_pt is
about 5x that.

Forward: the reference `stack_base` was rerun (dev AP 0.0809 / nuAP 0.1447; confirm 0.0770 / 0.1373). The forward increment run
`stack_base_dA_pt` is queued last in `scripts/queue_donors_a.sh`. FORWARD_STATUS `stack_bdv_w05` on forward was stopped under the RAM
rule; another agent is running `base_donor_view_w05`.

## Leakage class
- `dA_pt`: temporal-clean on `temporal`. The `past` training episodes see only PMIDs <= 2019-03, and the temporal dev/confirm tasks
  see only PMIDs <= 2021-03 (seeds are from 2021-04). On `forward`, PMIDs <= 2025-08 are used, which is clean w.r.t. 25.09 seeds. On
  `holdout` the channel returns NaN. Caveats:
  (1) PMIDs are only approximately chronological. Every PMID below the cut was created before the cut, so the cut is conservative,
  but some records created before the cut are missing.
  (2) The PubTator3 NER/normalisation model is from 2023-24 (a modern tool applied to dated text).
  (3) The family is **disease-derived**. MeSH rows include the task's own disease, so part of the gain is "literature ahead of OT
  curation". This is legitimate on a temporal benchmark, and it overlaps OT's own europepmc text-mining datasource (the gain may
  emulate the label generator; see next steps).
- a_dm_codep, a_impc_orth (IMPC 13.0 2020-07; MGI orthology classes and MP ontology structure are 2025 files but carry no labels):
  temporal-clean.
- a_ps_k562 / rpe1 (2022): transductive on temporal, fair on forward. a_ps_cd4 (2025): transductive on temporal and on forward.

## What failed / not done
- Machine load and RAM. My temporal `bdv_dAps` (Perturb-seq bundle) and the confirm half of `bdv_refA` disappeared during the
  RAM crisis (OOM-killer events at 04:38-05:20). I stopped my forward `bdv_w05` and forward `base_permA` to comply with the
  one-stacker rule. The first DuckDB PubTator join was killed as well. It was replaced by a streaming pandas + scipy-sparse join
  that finished in 15 min.
- IR-10 (burden PheWAS) was not attempted: Genebass is not on disk (critic_eng: Hail tables, tens of GB).
- No IMPC 24.0 / modern MGI variant: their dates are unverified, possibly after 2025-09 (critic_eng F: MGI is dated 2026).

## Runtime
Donor build: PS/DepMap/IMPC 3 min, PubTator 15 min (38.6 M gene and 72.6 M disease mentions -> 11,140 MeSH). Tuning: 10 min.
dA_pt for 842 forward tasks: about 15 min (scipy hypergeom logsf is the bottleneck, see the speed-up below). One stackP run took 15-25 min under load 150.

## New ideas
1. **Per-arrival-class check of dA_pt** (clinvar / gwas / curated / literature arrivals). If the gain sits only in literature-borne
   arrivals, it is label-generator emulation (europepmc). Still useful for prediction, but worth stating.
2. **PubTator momentum** (LT-9 / SG-7 style): count only PMIDs from the 2 years before the cut (recent co-mention growth), as an
   extra family `a_pt21_recent`. It is cheap: `data/raw/pubtator3/mesh_gene_counts` can be rebuilt with cut bins.
3. **MeSH -> disease mapping instead of seed overlap** (LT-2's second route): pick the task's own MeSH row(s) via xrefs and use them
   as a direct per-gene feature (co-mention count, PMIDs), not only via donor overlap.
4. Experimental gene sets do not work as donors (seed overlap with 16-50-gene sets is 0-1 gene). If used at all, they belong in
   gene kernels / views (implementer B), not as donor rows.
5. Speed-up: vectorise the hypergeometric weights per task (log-binomial via gammaln over all sets at once) instead of scipy logsf.
