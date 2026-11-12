# Research process v2 (lessons from modeng, 2026-11-11/12) — reuse in fwdbench and nebb

## What worked
1. **Research the process first** (`docs/research/00_ideation_process.md`): independent ideators before any interaction, diversity prompting
   (personas/lenses + low-probability "tail" ideas), retrieval-grounded novelty check, critic panel with distinct lenses (statistics/leakage,
   domain biology/curation, engineering), and above all a **closed loop against a frozen evaluator**. LLM judges/tournaments are weak (skip them).
2. **Build the harness before the ideas run** (benchmarks, metrics, masking tests, baselines, parity with the predecessor). Every later result
   depended on it; a parity check against the predecessor's own numbers caught an evaluation-universe artefact on day 1.
3. **Plug-in channel API + one learned combiner** (`src/modeng/ext/*.py` + `stack.py`): any agent can add a witness in one file and test its
   increment in one command. **Same-width permuted controls** are mandatory (noise channels alone moved results by ±0.01).
4. **Pre-register before scoring** (commit + hash + tag), one confirm look per variant, numeric kill rules written before running.
5. **Fable red teams / reviews at each milestone** caught: a float-residue tie leak, overclaims (59 looks, not 20), the wrong model in a table,
   cv-vs-prospective training, label-process share. Independent strategy reviewers (one adversarial) converged on the right next step.
6. **Living docs** (`ideas.md` with status, `insights.md`, `invalidations.md`, `journey.md`, `NEXT.md`) let work resume after two usage-limit stops.

## What went wrong (avoid)
1. **Selection on the confirm split:** 59 stacker variants were viewed on the same 138 confirm diseases. → Hold a **frozen fresh fold from the
   start**, select on dev (pool dev sets; 53 tasks was too few: increments were below retrain noise ±0.005-0.008 AP), and look at confirm once.
2. **Training-set size was the biggest lever and was discovered last** (53 dev → 712 wide_train). → Size the training pool on day 1.
3. **Metric gaming:** nuAP alone rewards degree down-weighting (an "evolved" formula won by halving AP). → Claims need AP AND nuAP AND
   degree-matched AP; report hub/non-hub.
4. **Label process vs biology were mixed:** backlog / catalog-lag / L2G priors forecast what the database adds next (huge on forward: nuAP
   0.14 → 0.31). → Type every target (new data / ingestion lag / ontology event) from the start; keep a biology arm without label-process channels.
5. **Leaks found late:** float residue in masked subtraction, CV leak via ancestor/descendant propagation, gene-order mismatch, ontology tell,
   post-dated annotation vintages (2025-12 HPA file, 2026 GWAS-catalog mapping), relatives' targets in wide training. → Leak tests in the harness
   (brute-force recompute, permuted controls, vintage table per input) BEFORE the first idea is scored.
6. **Process mistakes:** `cd x && cmd &` ran commands in the wrong repo; RAM exhaustion with 8 parallel stackers; disk on `/` filled.
   → absolute paths only; one heavy job per agent; data root on `/mnt/msadb/<project>` from day 1; check `free -g` / `df -h`.
7. **Redundant witnesses:** most new similarity witnesses carried signal but added nothing. → Before building a witness, measure how many
   targets it can reach that existing witnesses cannot ("traced-target coverage"); kill if < a preset share.

## Protocol v2 (per round, ≤ 10 parallel agents)
0. Lead (≤ 1 h): harness + baselines + parity + leak tests + frozen fresh fold (hash + tag) + data root + `docs/NEXT.md`, brief, banned list.
1. Ideation: 6-8 independent Fable ideators (distinct lenses, ≥ 3 tail ideas each, every idea with data, leakage class, 2-h test, kill rule,
   **traced-target coverage estimate**). Novelty check agent in parallel.
2. Critics: 3 Fable critics (stats/leakage, domain, engineering) with light read-only checks; synthesis by the lead (no tournament).
3. Implementation wave: 6-8 implementers, one cluster each, shared brief (`docs/ideation/implementer_brief.md` template in modeng),
   permuted controls, dev-only selection, reports in `docs/reports/`.
4. Red team (Fable) of the leader + a final-report reviewer; then round 2 = mutate / crossover / repair from measured results + 2 fresh islands.
5. Strategy review (2 independent Fable reviewers, one adversarial: "is this paradigm at its ceiling?") before any big new build.
6. One pre-registered confirmation look.

## Reusable assets in modeng
Harness `src/modeng/` (tasks: temporal / forward / holdout / past / temporal_wide; metrics AP / nuAP / onnet / sa; stack.py with --train cv/past/wide;
compare.py cluster bootstrap; channels + ext plugin API), gene views and donor families (`/mnt/msadb/modeng/data/`), witnesses in `src/modeng/ext/`,
arrival typing (`src/modeng/arrival_types.py`), module calls (`scripts/module_cutoff.py`, `scripts/export_modules.py`, `decompose_modules.py`),
leakage rules (`docs/invalidations.md`), all reports (`docs/reports/`), reviews (`docs/reviews/`).
