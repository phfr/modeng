# 04 Data-source sweep for new witnesses: human genetics, protein structure, drugs (2026-11-12)

Scope: sources that could give modeng new, non-transfer evidence ("witnesses"), above all for **strict-a** (modet class A: genetic arrivals whose
primary source first reaches threshold in 2022..2025). Dates are checked against **T0 = 2021-04-30** (temporal seeds, OT 21.04) and
**25.09** (forward seeds). Leakage classes as in `CLAUDE.md`: `temporal-clean` (content and public release <= seed release),
`transductive` (modern input, labels may leak through it), `label-process` (the input is, or feeds, the label pipeline: it forecasts
Open Targets, not new biology). No downloads were made. Cheap on-disk checks were run (scripts in the session scratchpad, numbers below).
Tags: [v] read on a primary page, [s] search snippet / secondary page only (unverified), [m] from memory, not checked (unverified).

## 0. Three facts measured first (they decide the ranking)
1. **What strict-a is made of** (modet `arrival_devall_t0_string115_phys_notm_400.parquet`, A & persistent, all 175 devall diseases, 1,894 pairs):
   gwas_only 1,534 (81 %), clinvar_only 339 (18 %), gwas+other 13, curated_only 8. Sources: gwas_credible_sets 1,547, eva 340,
   **gene_burden 10**, genomics_england 9. So exome burden is almost irrelevant to strict-a as a label, and GWAS credible sets dominate.
2. **Which cohorts produced the strict-a GWAS arrivals** (OT 25.09 `gwas_credible_sets` evidence, score >= 0.5, exact disease-id match;
   1,042 of 1,547 pairs matched; study via 26.09 credible-set table, 77 % of rows mapped): **52 % of pairs involve a FinnGen study,
   42 % a UKB-containing study, 66 % FinnGen or UKB**; the largest single source is the 2024 MVP multi-ancestry paper (112 studies; not
   public at T0). Publication years of the earliest supporting study: 2022 131, 2023 146, **2024 408**, 2025 79 pairs.
   On modeng's temporal benchmark: dev 312 direct pairs (FinnGen 65 %, FinnGen|UKB 71 %, 14/20 diseases), confirm 730 pairs (FinnGen 47 %,
   FinnGen|UKB 63 %, 33/53 diseases). In the **37 confirm strict-a diseases without an own T0 MAGMA study**, 19 have direct-matched pairs
   (341), of which **196 involve FinnGen and 217 FinnGen or UKB** (dev: 7 diseases, 135/157 FinnGen).
3. **What OT 21.04 already knew**: `ot_genetics_portal` 21.04 evidence = GCST 262k rows, **NEALE2 121k rows, SAIGE 5k rows, FinnGen 0**.
   So genome-wide-significant Neale UKB round 2 / SAIGE hits are already in the T0 seeds; FinnGen was not ingested at T0 at all.
   Consequence: the "coverage" lever for strict-a is **FinnGen R4 (public 2020-11-30)**, not UKB; UKB sources can add only sub-threshold signal,
   which r4 showed to be tiny (`docs/reports/r4_subthreshold.md`).

Caveat from r4 (`docs/reports/r4_powerfc.md`): even on the 34 confirm strict-a diseases that do have an own T0 study, own-study MAGMA z alone
has sa_AP 0.019 vs transfer 0.056 and adds nothing in the stacker. More coverage of the same kind is therefore not guaranteed to help; the
FinnGen case differs only because the later labels come from the **same cohort** (an earlier freeze of the same endpoint), which is a much
tighter witness than a different study of a related trait.

## 1. Ranked table
| # | source | mechanism (witness) | on disk / download | dates vs T0 (2021-04-30) and 25.09 | leakage class (temporal / forward) | measured on disk now | cheapest decisive test | kill rule |
|---|---|---|---|---|---|---|---|---|
| 1 | **FinnGen R4 public results** (176,899 people, 2,444 endpoints) | same-cohort earlier freeze: R4 lead variants and gene-level z for the endpoint matched to the disease forecast the R6-R12 credible sets that make up half of strict-a; gives an own study to diseases without one | download; GWS hit tables small, full sumstats ~0.5-1 GB per endpoint [m]; terms: request form, free, cite FinnGen (no explicit licence on the page) [v] | **DF4 public 2020-11-30** [v] (clean); DF5 public 2021-05-11 [v] = 11 days after T0 (sensitivity only); later DF6..DF12 2022-2024 [v] | temporal-clean by date; mechanism partly label-process (OT started ingesting FinnGen after T0, so part of the gain is ingestion lag, like `gwas_catlag`) / forward: R12 already in OT 25.09 labels, not useful | 47-65 % of strict-a GWAS pairs involve FinnGen; 196/341 pairs in confirm diseases lacking an own study (section 0) | P0 (no download): map R4 endpoint list to the 191 task diseases (ontology + name match as in `gwas_d`), count strict-a targets in newly covered diseases. P1: R4 GWS leads -> nearest-gene / cS2G channel `fg4_lead` (reuse `subthr.py` code). P2: MAGMA on matched endpoints -> `gwas_own`/`pfc_own`. Pre-register increment over modeng-v1 + permuted control | P0: < 100 strict-a targets (dev+confirm) in diseases R4 newly covers -> stop. P1/P2: confirm sa_nuAP CI vs v1 includes 0 or permuted control equal -> close the strict-a programme for GWAS inputs |
| 2 | **Gene-level constraint**: gnomAD v2.1.1 LOEUF/pLI (on disk), gnomAD v4.1 constraint (on disk), **AlphaMissense** gene-mean pathogenicity | constrained genes are over-represented among new genetic arrivals; constraint is not a stacker channel today (only a modet covariate and a powerfc feature) | v2.1.1 + v4.1 on disk (`~/biotools/data/evidence/raw/gnomad`); AlphaMissense gene-level file on Zenodo 10813168, CC BY 4.0 since 2024-03 [s], small (MB) | v2.1.1 2019/2020 (clean); v4.1 2024; AlphaMissense 2023-09 (trained on population frequency, ClinVar used for calibration/evaluation [m]) | v2.1.1: temporal-clean. v4.1, AlphaMissense: transductive on temporal (AlphaMissense leans towards ClinVar-type genes = mild label-process for clinvar arrivals); fine on forward | LOEUF bottom decile holds **17.3 % of confirm strict-a targets** (1,234) and 12.4 % of other targets vs 7.1 % of all genes (dev 16.9 / 12.6 %) | add static channels `gc_loeuf`, `gc_pli` (v2.1.1) to modeng-v1; second arm + AlphaMissense mean (forward only, or temporal flagged transductive); permuted control | confirm AP, nuAP and sa_nuAP deltas all within the permuted control (the HPO / Mendelian priors already carry it) -> drop |
| 3 | **GenCC submissions dated <= T0 from submitters that are NOT OT label sources** (Invitae/Labcorp, Ambry, Illumina, LMM, Myriad) | gene-disease validity calls that precede OT's curated sources for the same family | on disk (`gencc/gencc-submissions.tsv`, fetched 2026-10-07, 30,345 rows); GenCC terms: free, CC0 [m] | `submitted_as_date`: 8,446 rows <= T0 (all submitters); date semantics unverified (re-submissions may keep old dates; check with a Wayback snapshot of the 2021 download) | label-process on both benchmarks (curated arrivals forecast); also date-trust risk | exact MONDO family match: 105/138 confirm tasks covered, 327 non-seed genes flagged, **146 are targets (precision 0.45 vs base 0.0016)**, dev 80/127 (0.63); **strict-a hits 1 / 0** (they are mixed_non_gwas / curated arrivals) | channel `gencc_pre` (family max classification) in the label-process arm next to backlog channels; check overlap with `ph_*` HPO and `pri_mend` first (alone + increment) | increment over v1 + backlog channels < +0.005 AP on dev and confirm, or Wayback check shows dates are not original -> label-process only, never in the biology model |
| 4 | **Exome burden: Genebass** (394,841 UKB exomes, 4,529 phenotypes, gene burden + SKAT-O) and **AZ PheWAS portal** (269k/281k, later ~470k exomes) | allelic series: sub-threshold burden p for the matched phenotype marks the effector gene of GWAS loci; cross-trait burden profile as gene view (IR-10, SG-5, CI-09) | download; Genebass Hail tables, tens of GB [m] (app.genebass.org/downloads [s]); AZ portal per-gene/per-phenotype queries, bulk export unclear [m] | Genebass preprint 2021-06 [m], paper 2022; AZ paper received 2020-11-03, published 2021-08-10 [v]; portal public date unverified; 470k versions 2023+ | temporal: transductive (post-T0) and **label source** (OT `gene_burden` ingests Genebass/AZ/Regeneron [s]); forward: inputs <= 25.09 fine, but label-process for burden arrivals -> exclude `gene_burden` arrivals from targets | not on disk; only 10 strict-a and 201 burden-only arrivals overall, so the label route is small; the value would be the GWAS-arrival route | P0: download Genebass gene-level burden for phenotypes matched to dev diseases only; alone sa_AP on covered dev tasks vs `gwas_own` and permuted; then forward increment with burden arrivals removed | alone sa_AP on covered tasks not above `gwas_own` (0.019 confirm) -> drop; never use as input and label together |
| 5 | **Pan-UKBB 2020** / **Neale UKB round 2 (2018)** full sumstats | sub-threshold gene z for UKB traits (GWS part already in T0 seeds via NEALE2/SAIGE) | download; per-phenotype flat files on AWS [v], ~1-3 GB each [m]; Neale: free download, cite page [s] | Pan-UKBB first release 2020-06-15 [v]; Neale round 2 2018-08-01 [s] (both clean; later Pan-UKBB schema/h2 fixes 2022-2023 [s]) | temporal-clean | 42 % of strict-a GWAS pairs involve UKB-containing studies, but these are mostly large meta-analyses (MVP+UKB+FinnGen) | only as part of #1 P2 pipeline (same MAGMA code), for diseases still uncovered after FinnGen | same as #1 |
| 6 | **Zhang et al. 2025 RF2-PPI/AF2 human PPI screen** (17,849 PPIs at 90 % precision; DCA and RF2 scores for ~47M pairs) | structure-predicted edges reach dark genes; DCA scores = co-evolution kernel (EV-3) | **on disk now** (`/mnt/msadb/subex/raw/zhang2025_humanppi/`, downloaded 2026-11-12; 02 inventory is outdated), Dryad CC0 [v in README] | Dryad 2025-09-16; models trained on PDB + AFDB; `Disease1/2` columns (UniProt) must be dropped | label-free but post-T0: transductive (structure, not labels); database-selected pair subsets (S, P) are post-T0 knowledge, use de novo subsets (D, U) | seed-neighbour flag on temporal: all 17,077 mapped edges -> confirm 121 target hits in 10,541 flagged genes (precision 1.15 % vs 0.18 %), **only 16 off the STRING-T0 neighbourhood**, 15 strict-a; de novo subset: 42 hits / 2,195 (1.9 %), 5 off STRING, 5 strict-a | none needed at 90 % set (too few new hits); if anything, RWR on DCA/RF2 dense scores as an EV-3 kernel channel | already met for the 90 % edge set (adds ~16 targets beyond STRING-T0 on confirm): drop as a network; dense-score kernel only if a kernel slot is open |
| 7 | **Burke 2023 AF2 models of HuRI pairs** (pDockQ) | re-weight HuRI edges by structural plausibility | on disk (`table_s1_huri.tsv`; HuRI rows only) | NSMB 2023; AF2 weights PDB <= 2018-04 | temporal-clean as an edge re-score of a T0 network (subex provenance note) | 6,478 HuRI pairs with pDockQ > 0.23 | not worth a run: smaller than #6, which already fails | n/a (park) |
| 8 | **eQTL / pQTL colocalisation**: GTEx v8 (2019-2020), eQTLGen phase 1 (preprint 2018, paper 2021 [s]), INTERVAL pQTL (Sun 2018) [m], plasma-proteome MR + coloc (Zheng 2020, 1,002 proteins x 225 traits) [s]; UKB-PPP (2023) | which gene at a GWAS locus; cis-MR protein -> trait as gene-first evidence (CI-03) | download; eQTLGen cis files ~GB [m]; Zheng 2020 results via EpiGraphDB [s]; UKB-PPP via Synapse (post-T0) [m] | GTEx v8, eQTLGen, INTERVAL, Zheng 2020: pre-T0 (clean); UKB-PPP 2023: forward only | QTL coloc is already inside OT's L2G (the label generator): label-process flavour for gene choice | r4: the locus -> gene assignment layer is not the bottleneck (too few new loci) | Zheng 2020 MR+coloc pairs as a disease-matched gene flag: count how many temporal targets it touches (expect tens) | < 30 target hits on dev+confirm -> drop |
| 9 | **ChEMBL / OT chembl 21.04** (drug-target-indication, phases) | targets trialled for the disease family become genetic arrivals | on disk (`ot_chembl_2104`, CC0 / CC BY-SA) | release 2021-04-28 (clean) | temporal-clean; already used in `pri_*` popularity and `bl_drug` | family drug targets: confirm 73 target hits in 8,461 flagged genes (precision 0.86 % vs 0.18 %), 23 strict-a; dev 35 / 3,746 | none (weak; 4.7x enrichment, already partly in priors) | park |
| 10 | **ClinVar archived monthly release 2021-04** (variant_summary archive) | P/LP or VUS submissions <= T0 for gene-condition pairs OT 21.04 dropped (unmapped condition, conflicting, below 0.5) forecast eva arrivals (18 % of strict-a) | current file on disk (LastEvaluated filled for 94 % of GRCh38 rows; 10.8 % <= T0); archive at NCBI FTP `tab_delimited/archive/` [m], ~0.3 GB | 2021-04 archive is clean by date | **label-process** (eva = ClinVar is the label source); `bl_genweak` already uses OT eva at any score | not measured (MedGen -> EFO mapping needed) | count strict-a eva arrivals whose gene has a <= T0 ClinVar P/LP/VUS record under any condition in the disease family but no OT 21.04 eva row | < 50 such arrivals on dev+confirm -> drop |
| 11 | **Mouse beyond IMPC** (MGI HMD current), **zebrafish (ZFIN)** | phenolog donors (EV-1) | MGI on disk (current 2026); ZFIN downloads small, CC BY 4.0 [m]; ZFIN publication ids encode a date (ZDB-PUB-YYMMDD) [m] -> temporal cut possible | MGI current = post-T0; OT 21.04 `phenodigm` already holds mouse at T0 | transductive (MGI current) / clean with ZFIN pub-date cut | IMPC donors gave no increment (EV-1 negative) | only if a donor-family slot is free: ZFIN-pre-T0 donor family through orthologs | same rule as EV-1 (no increment over v1) |
| 12 | **Minikel 2024 / Pharmaprojects** | evaluation axis (drug success), not a witness | on disk (`minikel2024`, CC BY 4.0) | year_launch to 2022 | label-adjacent (used as outcome) | - | keep as secondary evaluation of modules | n/a |

## 2. Notes per source

**1. FinnGen R4.** Release table on finngen.fi: DF2 2020-01-14 (96,499 / 1,122 endpoints), DF3 2020-06-16 (135,638 / 1,801), **DF4 2020-11-30
(176,899 / 2,444)**, DF5 2021-05-11 (218,792 / 2,803), DF6 2022-01-24 ... DF12 2024-11-04 (500,348 / 2,502), DF13 2026-06-02 [v]. R3 is a
second clean freeze for the `past` benchmark (cutoff 2019-04: none; DF2 is 2020-01, so the past twin has no FinnGen input: report the
channel NaN on `past` and train cv on temporal dev, as for other new channels). Matching FinnGen endpoint codes (ICD-based) to EFO/MONDO
disease ids is the main engineering cost; OT 25.09 `study` table already maps later FinnGen endpoints to disease ids (`traitFromSourceMappedIds`,
`diseaseIds`) and endpoint codes are stable across freezes for most endpoints [m], so the mapping can be borrowed by endpoint code (mapping vintage is
post-T0: note it, as for the GWAS-Catalog trait mapping in `gwas_catlag`). Why this is not ML-4 again: ML-4 used 20 small GWAS-Catalog
studies and catalog-hit labels; here the witness is the same endpoint in the same cohort, and FinnGen GWS hits at R4 were never in OT 21.04
(section 0.3), so even P1 (GWS leads only, no sumstats) is a new, `gwas_catlag`-like channel. Honest framing: part of the strict-a gain would be
OT's later ingestion of FinnGen (a pipeline event), so report P1 also on the "no label-process" arm with FinnGen-only arrivals as a stratum.

**2. Constraint.** The 2.4x enrichment of strict-a targets in the LOEUF bottom decile (vs 1.75x for other targets) is the largest gene-level
signal for strict-a measured so far that is not yet a channel. The stacker may already reach it through `pri_*` (L2G-winner and Mendelian
priors) and popularity; the increment test answers that in ~30 min with on-disk data. AlphaMissense gene mean adds missense-specific
constraint; v4.1 constraint adds precision for short genes. Published support: constraint and missense pathogenicity predict
disease genes and drug-target safety (AlphaMissense paper, Science 2023 [s]; gnomAD v2 paper 2020 [m]).

**3. GenCC.** The precision (0.45) is far above any channel we have, but it is curation backlog: 0-1 strict-a hits; the arrivals it flags
are `mixed_non_gwas` / `curated_only`. It belongs to the label-process product (section 6 of the final report), never to biology claims.
Two checks before use: (a) date trust (GenCC launched its public download in 2021 [m]; whether `submitted_as_date` of commercial labs is the
original assertion date must be checked against an archived 2021 download); (b) overlap with HPO-2021 matching (74 % of ClinVar/curated
arrivals already have an HPO profile, `docs/reports/r1_priors.md`).

**4. Exome burden.** OT ingests gene-burden evidence from AZ PheWAS, Genebass and Regeneron (OT blog [s]), so burden is a label source.
For strict-a it hardly matters as a label (10 arrivals), but as an **input** for GWAS arrivals it is the cleanest gene-first evidence
(rare-variant burden and common-variant loci converge on the same genes in UKB exomes, e.g. Weiner et al. 2023 Nature [m]). Temporal is
transductive (all large burden results appeared after T0: AZ paper received 2020-11-03 but published 2021-08-10 [v]; Genebass 2021-06 [m]).
The fair benchmark is forward with `gene_burden` arrivals removed from targets. Small pre-T0 burden results (UKB 50k exomes, 2020 [m])
have too few traits.

**5. UKB summary statistics.** Because NEALE2 and SAIGE GWS hits are already T0 seeds, a UKB channel can only add sub-threshold
signal; r4 measured sub-threshold loci at 1.4-3.3 % precision with no stacker gain. Use only inside the #1 MAGMA pipeline for diseases
that remain without a study.

**6-7. Structure-predicted interactomes.** The Zhang 2025 data are on disk now (downloaded 2026-11-12 by the user). The measured overlap
says the 90 % edge set mostly duplicates STRING-T0 neighbourhoods for our targets (16 of 121 confirm hits are new). Together with
the negative `lift_afdb` kernel, structure-predicted edges are not a strict-a lever. Park; a dense DCA/RF2 kernel (EV-3) is the
only untested variant.

**8. QTL colocalisation.** Clean pre-T0 resources exist (GTEx v8, eQTLGen, INTERVAL, Zheng 2020 MR), but L2G already uses QTL evidence
to pick the gene, and r4 showed gene assignment within loci is not the bottleneck. Lowest-cost use: Zheng 2020's 111 MR+coloc protein-trait
effects (65 proteins, 52 phenotypes) as a direct flag [s]; expect very few hits.

**9-12.** ChEMBL drug targets are weakly enriched (4.7x) and already partly in priors. ClinVar archive and GenCC are label-process.
Mouse/zebrafish donors failed in IMPC form; ZFIN is cheap but low prior.

## 3. Recommended order (each pre-registered before running)
1. FinnGen P0 (endpoint mapping + coverage count; needs only the R4 endpoint manifest) -> P1 (R4 GWS leads channel) -> P2 (MAGMA on matched endpoints).
2. Constraint channels (on disk, ~30 min): `gc_loeuf`, `gc_pli` increment over modeng-v1 with permuted control; AlphaMissense on forward.
3. GenCC pre-T0 non-OT submitters in the label-process arm, after the date-trust check.
Everything else: parked with the kill rules above.

## Sources (accessed 2026-11-12)
- FinnGen data freezes and public release dates: https://www.finngen.fi/en/access_results [v]; DF5 announcement https://www.finngen.fi/en/public-release-finngen-data-freeze-5-results-and-summary-statistics [s]
- Pan-UKBB release posts: https://pan.ukbb.broadinstitute.org/blog/tags/release [v]; https://pan.ukbb.broadinstitute.org/blog/2020/06/15/first-release [s]
- Neale UKB round 2 announcement: https://nealelab.is/uk-biobank/ukbround2announcement [s]
- AZ PheWAS (Wang et al. Nature 2021, received 2020-11-03, published 2021-08-10, portal http://azphewas.com/): https://pmc.ncbi.nlm.nih.gov/articles/PMC8458098/ [v]
- Genebass downloads: https://app.genebass.org/downloads (via https://labs.icahn.mssm.edu/minervalab/resources/data-ark/genebass/) [s]
- OT gene-burden integration: https://blog.opentargets.org/how-open-targets-integrates-gene-burden/ [s] (cited in 01)
- AlphaMissense predictions, CC BY 4.0: https://zenodo.org/records/10813168 [s]; https://deepmind.google/research/projects/alphamissense/ [s]
- Zhang et al. 2025 Science, Dryad: https://datadryad.org/dataset/doi:10.5061/dryad.15dv41p84 [s]; local README [v]
- Burke et al. 2023 NSMB: local README `/mnt/msadb/subex/raw/burke2023_huintaf2/README.txt` [v]
- eQTLGen (Võsa et al. Nat Genet 2021): https://pmc.ncbi.nlm.nih.gov/articles/PMC8432599 [s]
- Zheng et al. 2020 Nat Genet, plasma-proteome MR: https://www.bristol.ac.uk/news/2020/september/genetic-study-of-proteins.html [s]
- ZFIN download files: https://dipper.readthedocs.io/en/latest/dipper.sources.ZFINSlim.html [s]
- Local: OT 21.04 / 25.09 evidence (`/mnt/msadb/modet/downloads/ot_evidence`), OT 26.09 credible sets, modet arrival table, GenCC (2026-10-07), gnomAD v2.1.1 / v4.1, ot_chembl_2104 README.
