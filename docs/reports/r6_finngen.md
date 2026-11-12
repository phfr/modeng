# r6 (R4 plan P2): FinnGen R4 genetics channel + gnomAD constraint

Date: 2026-11-12. Data: FinnGen R4 public results (DF4, public 2020-11-30, 176,899 people, 2,444 endpoints), anonymous download from
`gs://finngen-public-data-r4` (user decision 2026-11-12: "use public bucket now"; the user handles the access form / terms; earlier stop: commit c2c3956).
Acknowledgement: "We want to acknowledge the participants and investigators of the FinnGen study." Cite Kurki et al., Nature 613, 508-518 (2023).
Never used: `coding_variants/` (2021-11, post-T0), any R5+ freeze.

## 0. Pre-registration (written and committed before any FinnGen channel was scored)
Witness definition (R4 strategy review B, section 1a): a target is *traced* by a witness if it is in the top 200 of that witness's per-task
ranking over the task's candidates (universe minus `task.exclude`; average ties; NaN last). Genetic witnesses: `gwas_catlag` (existing) and the
new FinnGen channels `fg_hit` (credible-set lead -> nearest-TSS gene) and `fg_hit_cs2g` (credible-set variants -> cS2G gene).
1. **Kill rule (decides the programme):** on temporal **dev** (53 diseases), share of strict-a targets (`task.alt['sa']`) traced by any genetic
   witness {gwas_catlag, fg_hit, fg_hit_cs2g}. Reference: gwas_catlag alone ~13 % (R4 B). **< 20 % -> stop**: no summary-statistics download,
   record "strict-a is not reachable from FinnGen R4 genome-wide hits". 20-25 % -> report as weak, sub-threshold stage optional. >= 25 % -> pass
   the coverage gate; `fg_sub` (p <= 1e-5 leads from streamed summary statistics of mapped endpoints) is then built.
   wide_dev (45 diseases) reported with the same rule as a secondary, non-deciding number. wide_confirm is never scored; temporal confirm is not
   scored in this round either (dev-only stack runs).
2. **Increment (only if 1 passes; dev OOF, cv, deg^-0.35, scores saved):** `v1 + fg channels` vs `stack_v3ptr_p0.35` and vs the same-width
   gene-permuted control `v1 + fg_*_perm`; keep only if sa_nuAP delta >= +0.010 (paired cluster bootstrap), the permuted control's delta within
   +-0.003, and overall AP and nuAP deltas >= -0.003. Same comparison on the v1clean channel list.
3. **Label-process share:** among strict-a targets newly traced by FinnGen (traced by fg_* but not gwas_catlag), the share whose 25.09 OT
   evidence for the disease (or a descendant) includes a FinnGen study (`gwas_credible_sets`, studyId FINNGEN_*). Reported, not a decision.
4. **Constraint `gnomad_loeuf`** (independent of the kill rule): alone (AP, nuAP, sa_AP, sa_nuAP) on temporal dev and its increment over v1 with
   a gene-permuted control on dev; drop if its deltas are within the control's.
5. Endpoint -> disease mapping precision: 30 mapped (endpoint, disease) pairs sampled at random (seed 0), judged by hand from names
   (correct / too broad / too narrow / wrong); reported as is.
