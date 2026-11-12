# R4 synthesis: what to build next (lead, 2026-11-12, from two independent Fable strategy reviews A and B)
Both reviewers agree:
1. **The stacker is near its ceiling on the current benchmark; the limit is missing evidence and too few training diseases, not the learner.**
   B: 46 % of confirm targets have a seed-specific pre-seed trace and v1 already ranks 67 % of those in its top 200; the untraced 54 % are mostly
   GWAS-only/non-persistent arrivals where v1 is a gene prior; strict-a traced by a genetic witness 4.6 %; perfect per-disease gate <= +0.05 AP.
   A: training size is the biggest lever (17 dev diseases: AP = transfer; 53: 0.153; forward 842 in-window: +0.027 AP); recent increments < retrain noise.
2. **~700-1,170 untouched temporal diseases exist** (21.04 -> 25.09, never tasks, not sealed-related; raw counts before modet-style filtering).
   -> build `temporal-wide`: wide-train (scale), wide-dev, and a **fresh wide-confirm fold, frozen before anyone looks** (also solves the 59-looks problem).
3. **Strict-a needs new evidence: FinnGen R4 (pre-T0) / Pan-UKBB as a next-GWAS forecaster**, judged on wide strict-a tasks with its own permuted control
   (A: ~1/3 odds; B: kill if genetics-traced strict-a < 20 % on dev, pass >= 25 %). Label-process share must be reported.
4. **Separate biology from label process:** type arrivals (new data / ingestion lag / ontology event); persistent new-data arrivals = biology endpoint;
   two-arm experts (biology vs label-process) with a gate trained on wide-train; expert disagreement = candidate new biology (product, not accuracy).
5. **Not now:** deeper KG models, more similarity witnesses, LLM judge on existing benchmarks (memorisation); LLM extractor only as a later high-risk arm.
6. **Sealed fold: no** — confirm on wide-confirm first; reconsider only if a strict-a gain replicates there. Keep the next-OT-release pre-registration.

## Plan
- P1 (now): build `temporal-wide` with modet-style filters (weak-evidence 'unknown' exclusion, re-propagation, family masking), ontology-unrelated split,
  freeze + hash wide-confirm ids BEFORE any model is scored on it; retrain pruned v2 (audit: drop view similarities, removable groups) at scale on wide-train.
- P2 (parallel): FinnGen R4 acquisition + endpoint->disease mapping + lead/MAGMA channel; constraint channel (gnomAD v2.1.1).
- P3: arrival typing + two-arm experts with gate on wide-train.
- P4: one pre-registered scoring on wide-confirm of v1, v1clean, v2(-wide) and the arms; then decide on sealed fold.
