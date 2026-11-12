# r6 (R4 plan P2): FinnGen R4 genetics channel + constraint: STOPPED at acquisition (access form)

Date: 2026-11-12. Status: **stopped before any download, mapping, channel code or scoring** (no result seen, nothing to pre-register yet).

## Why stopped
The task rule was: "if a registration form is required, STOP and report; do not register on the user's behalf".
FinnGen's results page (https://www.finngen.fi/en/access_results, accessed 2026-11-12) says: "Fill [this online form] to gain access to
FinnGen summary statistics"; download instructions are then sent by e-mail. No licence text is given; requested acknowledgement:
"We want to acknowledge the participants and investigators of the FinnGen study", cite Kurki et al., Nature 613 (2023), keyword "FinnGen".
So the official route is a registration form. Decision for the user: fill the form (it is free) or authorise use of the bucket as is.

## What was checked (metadata only, no file content downloaded)
The bucket `gs://finngen-public-data-r4` is anonymously listable (`https://storage.googleapis.com/storage/v1/b/finngen-public-data-r4/o`).
Object timestamps 2020-11-26 .. 2020-11-30 (pre-T0), except `coding_variants/` (2021-11-03, post-T0: never use on temporal).

| path | files | total size | use for P2 |
|---|---|---|---|
| summary_stats/R4_manifest.tsv | 1 | 0.56 MB | endpoint list (2,444) -> mapping |
| summary_stats/finngen_R4_pheno_n.tsv | 1 | < 1 MB | case/control counts |
| finemapping/*.SUSIE.cred.bgz | 1,115 endpoints | < 1 MB in total | **GWS loci + credible sets = `fg_hit`** (no separate "top hits" file exists in R4) |
| finemapping/*.SUSIE.snp.bgz | 1,115 | 4.0 GB | per-SNP PIP in fine-mapped regions (optional) |
| summary_stats/finngen_R4_<endpoint>.gz | 2,444 | 980 GB (~400 MB each) | sub-threshold leads (`fg_sub`) and MAGMA z; stream and keep p <= 1e-5 for mapped endpoints only (~100-300 endpoints -> ~40-120 GB transient) |
| annotations/finngen_R4_annotated_variants.gz | 1 | 8.9 GB | not needed |

## Plan once access is settled (unchanged from the task; kill rule to be committed before scoring)
1. Manifest + SuSiE cred files -> endpoint -> EFO/MONDO mapping (ICD xrefs + name match), precision on 30 hand-checked endpoints.
2. `src/modeng/ext/finngen.py`: `fg_hit` (credible-set lead -> nearest gene / cS2G, as `subthr.py`), `fg_sub` (p <= 1e-5 leads from streamed
   sumstats of mapped endpoints), optional MAGMA z; `*_perm` controls; NaN on `past`. `src/modeng/ext/constraint.py`: `gnomad_loeuf` (v2.1.1, on disk).
3. Pre-registered kill rule: dev strict-a targets traced by any genetic witness (top 200 of gwas_catlag or fg_*) < 20 % -> stop.
   Then alone (sa_ap, sa_nuap), increment over stack_v3ptr_p0.35 / v1clean with same-width permuted control, label-process share; temporal dev
   (and wide_dev if present) only, never wide_confirm.
Note: the constraint channel (`gnomad_loeuf`) does not depend on FinnGen and can be run independently if wanted.
