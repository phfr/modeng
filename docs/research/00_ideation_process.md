# AI-assisted ideation: what has evidence, and the protocol we use (researched 2026-11-12)

Scope: which parts of LLM/multi-agent scientific ideation are supported by evidence (not vendor claims), and a concrete
multi-round protocol for next-generation disease gene / disease module prediction, with a closed loop of cheap benchmark
tests. Builds on and replaces the snippet-level protocol in `/home/ph/code/modetor/docs/research/08_ideation_protocol.md`
(its section-5 weaknesses: same-model blind spots, snippet-level support, persona prompting without evidence, single-pass
tournaments, ideas never executed).

All sources accessed 2026-11-12. **Verification levels:** `[abs]` = abstract page read; `[full]` = relevant part of the
paper read (HTML); `[snippet]` = search-result or secondary summary only, **unverified**.

---

## 1. Evidence table

| # | Method / system | What was shown | Strength of evidence | Source |
|---|---|---|---|---|
| 1 | Si, Yang, Hashimoto: LLM vs expert ideas (NLP) | 100+ NLP researchers, blind review: LLM ideas judged **more novel** (p<0.05), slightly less feasible. Over-generation: 4,000 seed ideas/topic, only **~200 (5%) non-duplicate** (cosine 0.8), unique count plateaus. LLM judges at **45-53% balanced accuracy** (pairwise Claude 53.3%, AI-Scientist reviewer 43.3%) vs human-human 56.1% (random 50%). Ranking: zero-shot pairwise **Swiss tournament, 5 rounds**. | Strong (large blinded human study), but NLP prompting ideas only | https://arxiv.org/abs/2409.04109 `[abs]`, https://arxiv.org/html/2409.04109 `[full]` |
| 2 | Si, Hashimoto, Yang: ideation-execution gap | 43 experts executed ideas (>100 h each), blind review of results: LLM ideas lost significantly more than human ideas on all metrics; ranking flip (n.s.). Execution-stage reviewers weigh empirical results and rigour, which idea-stage reviewers ignored. | Strong design, small n (43), NLP only | https://arxiv.org/abs/2506.20803 `[snippet]` (consistent across several summaries) |
| 3 | HindSight (time-split idea evaluation) | Ideas generated with literature before cutoff T, matched against papers published in the next 30 months. LLM judge saw **no difference** between retrieval-augmented and vanilla generation (p=0.58); hindsight metric: retrieval-augmented **2.5x better** (p<0.001). Hindsight score **negatively correlated with LLM-judged novelty** (rho=-0.29). | Single preprint, AI/ML topics, but the design (temporal ground truth) is the right one | https://arxiv.org/abs/2603.15164 `[abs]` |
| 4 | Diversity collapse in multi-agent ideation (Chen et al., ACL 2026 Findings) | Interaction structure, not model weakness, collapses diversity: authority-led groups < junior-led; dense topologies converge early; group size has diminishing returns; stronger aligned models add less diversity per sample. Recommends preserving independence and disagreement. NGT (independent writing first) reported as most diverse (Vendi 8.08 vs 4.65). | Controlled experiments; one paper; NGT number `[snippet]` | https://arxiv.org/abs/2604.18005 `[abs]` |
| 5 | Barriers to diversity in LLM ideas (Deng, Brucks, Toubia) | Two mechanisms: **fixation** (early outputs narrow later ones) and **lack of knowledge partitioning** (one blended distribution). Personas restore partitioning but increase fixation; CoT reduces fixation; **both combined** gave the highest diversity, above independent humans. | Two controlled experiments (product ideation); one preprint | https://arxiv.org/abs/2602.20408 `[abs]` |
| 6 | Multi-agent ideation dialogue design (SIGDIAL 2025) | More agents, deeper ideation-critique-revision, more varied personas raise diversity; **more diverse critics raise feasibility** of final proposals. | One paper, LLM-judged outcomes | https://arxiv.org/abs/2507.08350 `[abs]` |
| 7 | Persona prompting, other studies | Diversity gain is **between** personas, not within one (story plots); sequential multi-persona prompting more diverse than parallel/collective (design concepts). | Small studies, non-science tasks | https://arxiv.org/abs/2504.13868, https://www.cambridge.org/core/product/3B346E253508337A4EE899499BE49D9B/core-reader `[snippet]` |
| 8 | Verbalized sampling | Asking for k answers with probabilities: diversity 1.6-2.1x in creative writing, no accuracy/safety loss; stronger models benefit more. Mode collapse attributed to typicality bias in preference data. | ICML 2026; many tasks; not tested on research ideation | https://arxiv.org/abs/2510.01171 `[abs]` |
| 9 | Google AI co-scientist | Generate / reflect / rank (Elo tournament via simulated debate) / evolve / proximity (clustering) / meta-review agents. Elo rises with test-time compute; Elo concordant with GPQA accuracy. Wet-lab: AML repurposing candidates (5 tested by experts' pick, 3 reported active `[snippet]`); cf-PICI mechanism matched an unpublished finding. **No component ablation found.** Elo is self-evaluation. | Vendor paper, real wet-lab hits, but candidates pre-filtered by experts and no ablation | https://arxiv.org/abs/2502.18864 `[abs]`; details `[snippet]` |
| 10 | Sakana AI Scientist v1, independent evaluation | 42% of experiments failed on coding errors; others flawed/misleading; literature novelty check labelled known ideas (micro-batching SGD) as novel; median 5 citations; "rushed undergraduate" quality; USD 6-15/paper. | Independent evaluation (preprint) | https://arxiv.org/abs/2502.14297 `[abs]` |
| 11 | AI Scientist v2 (agentic tree search) | 1 of 3 generated papers passed an ICLR 2025 *workshop* review (~6.33 avg), withdrawn by design. | Vendor, n=3, workshop bar | https://arxiv.org/abs/2504.08066 `[snippet]` |
| 12 | Virtual Lab (Zou lab, Nature 2025) | LLM "PI" + specialist agents + **critic agent**, human ~1% of words; designed 92 nanobodies with ESM/AlphaFold-Multimer/Rosetta; some with improved binding to newer variants, validated experimentally. Senior author: critic "quite essential", reduced hallucinations (interview, no ablation). | Real wet-lab validation; component claims anecdotal | https://www.eurekalert.org/news-releases/1092265, https://github.com/zou-group/virtual-lab `[snippet]` |
| 13 | Robin (FutureHouse) | Literature agent + data-analysis agent in a lab loop; proposed ripasudil for dry AMD, then an RNA-seq follow-up whose analysis fed the next hypothesis. Effect sizes reported inconsistently across summaries. | Single case study; wet-lab executed by humans | https://futurehouse.org/research-announcements/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system `[snippet]` |
| 14 | Kosmos (Edison/FutureHouse) | 12-h cycles of data analysis + literature; ~79% of statements judged accurate by experts; **interpretation statements ~58%** (weakest). | Vendor eval with external experts; numbers `[snippet]` | https://arxiv.org/abs/2511.02824 `[snippet]` |
| 15 | Agent Laboratory | Human feedback at each stage (co-pilot) improved quality; autonomous papers ~3.8/10 by humans. | Single paper | https://arxiv.org/abs/2501.04227 `[snippet]` |
| 16 | ResearchAgent | Literature graph + entity store + multiple LLM reviewers with human-derived criteria; iterative refinement beat baselines; **gains plateau around iteration 3**. | NAACL 2025; LLM+human evals | https://aclanthology.org/2025.naacl-long.342 `[snippet]` |
| 17 | Chain of Ideas | Ordering literature into citation chains before ideation; "Idea Arena" pairwise Elo judging, claimed high agreement with humans; +56 Elo over next baseline. | Authors' own eval; human eval thinly described | https://arxiv.org/abs/2410.13185 `[snippet]` |
| 18 | SciMON | Retrieve inspirations, compare to nearest prior work, iterate for novelty: novelty maintained or raised, but ideas still far behind papers in depth/utility. | ACL 2024, modest effect | https://aclanthology.org/2024.acl-long.18 `[snippet]` |
| 19 | Retrieval-grounded novelty checking | Retrieve-then-rerank checker ~13% higher agreement with experts than earlier checkers; retrieval quality is the bottleneck; well-known ideas mis-flagged as novel by naive checkers. Human novelty agreement itself low. | Several 2025 preprints | https://arxiv.org/abs/2506.22026 `[snippet]` |
| 20 | FunSearch (Nature 2023) | Frozen LLM + **automatic evaluator** in an evolutionary loop; island populations for diversity (worst islands reset from best); new cap-set constructions and bin-packing heuristics. | Peer-reviewed, verifiable results | https://www.nature.com/articles/s41586-023-06924-6 `[snippet]`, https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models `[snippet]` |
| 21 | AlphaEvolve | LLM edits code, evaluators score, evolutionary database keeps improvements; 4x4 complex matrix mult with 48 multiplications; production gains. **Ablations:** removing evolution, rich context, meta-prompt evolution, full-file evolution or the strong LLM each hurts. | Vendor, but results verifiable; ablations `[snippet]` | https://arxiv.org/abs/2506.13131 `[abs]` |
| 22 | AI research agents on MLE-bench (NeurIPS 2025) | Agents = search policy x operators. **Operator set mattered more than policy** (greedy with better operators 45.5% vs 39.8%). Memory operator no effect. **Systematic overfitting to validation**: selecting final by test would add 9-17 points; long runs: validation keeps improving while test plateaus or drops; keeping top-3 by validation recovers ~10%. | Peer-reviewed, large compute, ground-truth test | https://arxiv.org/abs/2507.02554 `[abs]`, https://arxiv.org/html/2507.02554 `[full]` |
| 23 | Keep-or-discard loop on a fixed time budget ("autoresearch") | Agent edits training code, 5-min fixed-budget run, keeps commit only if the frozen validation metric improves; ~12 experiments/h; evaluation code is off-limits to the agent. | Practitioner open-source, no formal evaluation | https://kingy.ai/news/autoresearch-karpathys-minimal-agent-loop-for-autonomous-llm-experimentation/ `[snippet]` |
| 24 | Biomni / TxAgent | Tool-rich biomedical agents beat bare LLMs on QA benchmarks (+6-12% for several LLMs on HLE-bio); independent CURE-Bench analysis: **tool retrieval quality drives TxAgent's performance**. Ideation not tested. | Benchmarks; relevant for tooling, not ideation | https://www.biorxiv.org/content/10.1101/2025.05.30.656746 , https://arxiv.org/abs/2512.11682 `[snippet]` |
| 25 | LLM judges: pairwise reliability | LLM judges are non-transitive and position-biased; round-robin + Bradley-Terry more robust than fixed-baseline; Swiss pairing approximates round-robin cheaply; position swapping helps but does not remove bias. | Several 2024-2026 papers | https://arxiv.org/abs/2502.14074, https://arxiv.org/abs/2406.07791 `[snippet]` |
| 26 | Self-critique | LLM self-confidence miscalibrated; models catch others' errors more often than their own identical errors; longer reasoning can increase overconfidence. | Preprints/survey | https://arxiv.org/abs/2511.07448 `[snippet]` |

---

## 2. What has evidence, and what is folklore

**Supported (by at least one controlled or ground-truth study):**
- **Independent generation before any interaction** (NGT): rows 4, 5. Interaction, shared context, and an authority agent
  reduce diversity.
- **Massive over-generation has sharply diminishing returns**: ~5% unique after dedup, plateau (row 1). More agents also
  give diminishing returns (row 4). So: many *different* prompts/contexts, not many samples of one prompt.
- **Diversity prompting**: verbalized sampling (row 8) and CoT + knowledge partitioning (row 5). Personas help **only as
  partitioning across agents**, and they increase fixation within one agent (rows 5, 7).
- **Diverse critics raise feasibility** of final ideas (row 6); a separate critic agent is used in all systems with real
  validations (rows 9, 12), though nobody ablated it.
- **Idea-stage LLM scores are weak predictors.** LLM judges are near chance on human-rated top vs bottom ideas (row 1),
  executed LLM ideas fall further than human ideas (row 2), LLM-judged novelty is *negatively* correlated with later
  real-world uptake (row 3). LLM "novelty" checks miss well-known ideas (rows 10, 19).
- **Literature retrieval helps**, but LLM judges cannot see that it helps (row 3); retrieval quality is the bottleneck for
  novelty checks and tool agents (rows 19, 24).
- **Closed loops with an automatic evaluator work** where an evaluator exists (rows 20-23): this is the strongest
  evidence in the whole field, because the evaluator is ground truth, not opinion. In these loops, **operators (what kind
  of edit/idea is allowed and how it is prompted) matter more than the search policy** (row 22), and the full context +
  strong model + evolution each matter (row 21).
- **Closed loops overfit the evaluator** (row 22): the more iterations against one validation set, the bigger the
  validation-test gap. This is the central risk for our setting with ~35-100 evaluable diseases.
- **Iterative refinement plateaus at ~3 rounds** (row 16); 2-3 rounds is enough.
- **Pairwise judging needs both orders and many comparisons**; Swiss pairing approximates round-robin (rows 1, 25).

**Folklore / unproven:**
- **Elo tournaments as a quality signal** (co-scientist, Chain of Ideas): self-evaluated, no independent check that Elo
  predicts ground truth beyond the GPQA proxy; row 1 suggests LLM pairwise judgments of research ideas are near chance.
  Useful as a cheap *filter for obvious losers*, not as a ranking of winners.
- **"Debate" between agents** improves hypotheses: no ablation found for ideation; row 4 shows interaction can harm
  diversity.
- **Named expert personas improve quality**: no evidence; only partitioning/diversity evidence.
- **The PI/lead-agent hierarchy** (Virtual Lab): works as a project organiser, but authority-led groups lose diversity
  (row 4). Keep the lead out of generation.
- **Self-reflection / self-critique raises quality**: weak (row 26); use fresh-context critics.
- **Meta-review agents, proximity agents**: plausible engineering (dedup is clearly needed, row 1), no ablation.
- **Red-teaming / pre-mortem for research ideas**: no study found that measures its effect on idea outcomes (search
  2026-11-12). Kept as cheap insurance against known failure modes (leakage), not as an evidence-based step.
- **Vendor "end-to-end discovery"** claims (rows 9, 11, 13, 14): real hits exist, but humans selected or executed the
  key steps; none isolates which agent component mattered.

**Implication for design:** spend little on idea-stage scoring sophistication and a lot on (a) diversity at generation,
(b) fast, honest execution against a frozen evaluator, and (c) protection against overfitting that evaluator.

---

## 3. Protocol for our setting

Setting: up to 10 parallel subagents per round (Fable), 2-3 rounds, one lead (the orchestrating agent) who does **not**
generate ideas. Benchmark: disease gene / module prediction with fixed seeds, stated leakage class (`temporal-clean`,
`transductive`, `holdout-current`), modet sealed diseases excluded always.

### 3.0 Before round 1 (lead, ~30 min; committed to git before any ideator runs)
1. **Evaluator freeze.** One script `eval` (read-only for all subagents) that takes a score file
   `(disease, gene, score)` and returns the primary metric + paired bootstrap CI vs the current best and vs 2-3 simple
   baselines (e.g. transfer from related diseases, degree, network propagation). Fixed disease list.
2. **Three disease splits:** `dev-A` (loop evaluation, used every round), `dev-B` (touched only at the end of each round
   for the top 2-3 survivors), `confirm` (touched once at the end of the sprint; never by ideators). This is the MLE-bench
   lesson (row 22): pick finalists by a split the loop never saw.
3. **Fact sheet** (≤1 page, numbers with file paths): data sizes, what the baselines score, known dead ends
   (`docs/invalidations.md`), leakage rules, compute limits (GPU via `gpu-run`, 1-2 h per test).
4. **Banned/explored list** (titles + one line each), from `docs/ideas.md`.
5. **Literature pack**: 20-40 retrieved abstracts (PubMed/bioRxiv/arXiv) on recent disease-gene/module methods, given as
   a file. Retrieval is grounded; ideators may search more.

### 3.1 Round structure (each round ~4-6 h wall time)
| Step | Agents | What | Evidence basis |
|---|---|---|---|
| G. Independent generation | 6-8 ideators in parallel, no shared context | Each gets brief + fact sheet + banned list + a **different knowledge slice** (one literature sub-pack and one "lens"). Must flip assumptions first (CoT), then produce 8-10 ideas with verbalized probabilities, ≥3 from the tail (p<0.1). | rows 4, 5, 8 |
| D. Dedup + cluster | lead (embedding + reading) | Embed titles+mechanisms, merge at high similarity, keep one representative per cluster, note cluster sizes (big cluster = mode, not evidence of quality). | rows 1, 9 |
| N. Novelty + grounding check | 1-2 agents | For each surviving idea: retrieve the 5 nearest papers (PubMed/arXiv/bioRxiv, not memory), say "done before / partial / new" with citations; check every data claim against the fact sheet or by a 1-command look at the file. | rows 3, 10, 19 |
| C. Critic panel | 3 critics, fresh context, different lenses (statistician/leakage; biologist/curator; engineer/compute) | Independent written critiques per idea: fatal flaw? leakage class? can a ≤2 h test be written? Critics do not see each other. | rows 6, 26 |
| T. Pairwise tournament | 2 judges x both orders | Swiss, 4-5 rounds over the ≤20 survivors, criterion below; Bradley-Terry aggregate. Output = top 6-8 + "probably worthless". | rows 1, 25 |
| R. Red team | 1-2 agents | Pre-mortem on the top 6-8: "it is a month later and this result was retracted; why?" Must produce at least one concrete leakage or confound test per idea, added to the test plan. | folklore, cheap |
| X. Closed-loop execution | 4-6 implementers in parallel (worktrees) | Each top idea: implement the **minimum viable test** in ≤1-2 h, run `eval` on `dev-A`, write a 1-page report with numbers, CI, the red-team check and runtime. | rows 20-23 |
| F. Feedback | lead | Results table (all, including failures) + critics' unresolved weaknesses + unexplored clusters go to next round. Rankings of *ideas* are not passed on; *measured results* are. | rows 2, 4, 22 |

**Tournament criterion (single question, asked in both orders):**
*"Which idea, if its 2-hour test were run on dev-A, has the higher expected improvement in the primary metric over the
current best, counting a leakage-tainted or unrunnable test as zero?"* Tie allowed. The judge must name the decisive
fact. Novelty, elegance and excitement are deliberately excluded (rows 2, 3: they mislead). A separate, smaller
"high-ceiling" bucket keeps the 1-2 best ideas that lose the tournament only on feasibility (avoid collapse to easy
ideas).

**Round 2 = recombination / evolution round** (FunSearch/AlphaEvolve analogue):
- 3-4 ideators get the *measured* results table and the code of the 2-3 best tests; operators are explicit:
  `mutate` (change one component of a winner), `crossover` (combine two winners whose gains look additive), `repair`
  (fix the named failure of a near-miss), `fresh` (2 ideators get no results at all, only the extended banned list: the
  "island" that prevents collapse onto round-1 winners).
- Ideas that failed in round 1 are not re-proposed without a stated new reason.
- Same D-N-C-T-R-X steps, shorter.

**Round 3 (optional):** only if round 2 produced a gain beyond the round-1 CI. Mostly `mutate`/`repair` on the 1-2 best,
plus one `fresh` island. Then stop (row 16: plateau ~3 iterations).

### 3.2 Closed-loop rules (the part with the strongest evidence, and the biggest risk)
- **Frozen evaluator**: implementers cannot edit `eval`, the split files, or the label files (checksum before/after).
- **Fixed budget per test** (≤2 h wall, GPU via `gpu-run --mem <GB>`); a test that does not finish is "not run", not
  "failed".
- **Every attempt is logged** (idea id, commit, metric, CI, runtime, status keep/discard/crash) in one table: the number
  of attempts is the multiplicity to correct for.
- **Keep rule:** a variant is kept for the next round only if it beats the current best on dev-A by more than the paired
  bootstrap CI half-width **and** passes its red-team check.
- **Selection rule:** finalists are chosen by dev-B, not dev-A; `confirm` is used once; report dev-A, dev-B and confirm
  numbers together. A large dev-A to dev-B drop is itself recorded in `docs/insights.md`.
- **Positive control** in each test (the current best rerun through the same code path) to catch pipeline bugs (row 10:
  42% coding failures).

### 3.3 Budget
Per round: 8 ideators + 2 novelty + 3 critics + 2 judges + 1 red team + ≤6 implementers ≈ 22 agent runs (never more than
10 concurrent). 2-3 rounds ≈ 50-65 agent runs. Wall time ~5 h/round; most of it is execution.

---

## 4. Prompt templates (paste; replace `{...}`)

### 4.1 Ideator (round 1)
```
You are one of several independent researchers. You will not see the others' ideas; do not try to guess them.
Task: propose methods that would improve {primary metric} for ranking disease genes / predicting disease modules on
the benchmark described in {fact_sheet_path}. Your knowledge slice: {literature_subpack_path}; your lens: {lens, e.g.
"population genetics", "single-cell biology", "graph ML", "causal inference", "information retrieval/ranking",
"biocuration/ontologies", "evolution/comparative genomics", "forecasting"}.
Already explored or banned (do not re-propose; you may contradict one with a stated reason): {banned_list_path}.
Rules: every claim about our data must cite the fact sheet or a file you inspected; leakage class must be stated
(temporal-clean / transductive / holdout-current); no modet sealed-fold disease may be used.

Step 1 (think first, write it down): list 8 assumptions the current best method and the baselines make, and for each,
what would follow if it were false.
Step 2: propose 8-10 ideas. For each, give the probability (0-1) that a typical strong LLM given this brief would propose
it. At least 3 ideas must have probability < 0.10. Do not pad: a vague idea is worse than none.
For each idea output exactly:
- id, title (≤10 words), mechanism (≤3 sentences: why it should add signal the baselines lack)
- p_typical (0-1)
- data needed (source, size, date, licence), leakage class and the main leakage route
- minimal test runnable in ≤2 h on dev-A with existing files (command-level sketch, inputs, expected runtime)
- expected effect on {primary metric} vs current best (number + why), and the kill rule (number)
- most likely reason it fails
Finish with: which of your ideas you would bet on, and which you think is most underrated.
```

### 4.2 Ideator (round 2, evolution)
```
Same rules as before. You now see measured results (not opinions): {results_table_path} and the code of the best tests
{paths}. Operators you must use: give 2 MUTATE (change one component of a winner), 2 CROSSOVER (combine two results whose
gains plausibly add; say why they are not redundant), 2 REPAIR (fix the specific, named failure of a near-miss), and
2 FRESH ideas unrelated to anything in the table (p_typical < 0.10). Same output fields as round 1, plus "builds on: {ids}".
Do not re-propose a failed idea unless you name the new reason it would now work.
```
(For the 2 `fresh` island ideators use 4.1 with the extended banned list and **no** results table.)

### 4.3 Critic (one per lens; fresh context)
```
You are a {statistician focusing on leakage and evaluation | disease biologist / biocurator | ML engineer focusing on
compute and implementation} reviewing research ideas you did not write. Be specific, not polite; you are scored on
finding real flaws, not on number of flaws. Inputs: {ideas_path}, {fact_sheet_path}, {novelty_report_path}.
You may inspect files read-only (one-line commands) to check data claims.
For each idea output:
- fatal flaw (yes/no; if yes, the flaw in one sentence with the fact that proves it)
- leakage: the concrete route by which post-seed disease evidence could enter, or "none found" + what you checked
- is the 2-hour test well-posed? what it would and would not show; the cheapest fix if not
- best realistic effect on {primary metric} (number) and your confidence (low/med/high)
- one change that would make it substantially better
Then list: ideas that are the same idea in different words; and important directions nobody proposed.
```

### 4.4 Tournament judge (one pair per call; each pair is judged twice with A/B swapped)
```
Compare two candidate ideas for the benchmark in {fact_sheet_path}. Critiques: {critiques_for_A_and_B}.
Question: which idea, if its 2-hour test were run on dev-A, has the higher expected improvement in {primary metric} over
the current best ({current_best_value})? A test that is leakage-tainted, unrunnable in 2 hours, or not testing the idea
counts as zero improvement. Ignore novelty, elegance and how exciting it sounds. Ignore length and presentation.
Idea A: {idea_A}
Idea B: {idea_B}
Answer in this format only:
decisive_fact: <one sentence, a specific fact about data, leakage or mechanism>
expected_gain_A: <number>  expected_gain_B: <number>
winner: A | B | tie
```
Aggregate: Swiss pairing on cumulative score, 4-5 rounds; each pair both orders (inconsistent verdict = tie); final
order by Bradley-Terry; report the share of order-inconsistent verdicts (if >30%, the judge is not discriminating: fall
back to critic consensus and test more ideas).

### 4.5 Red team / pre-mortem
```
It is one month from now. The result of idea {id} ({title}) looked like a clear win on dev-A, but it was shown to be
wrong and withdrawn. Write the three most likely explanations, most likely first. Consider: label leakage via inputs
dated after the seeds; information shared between seed and target genes through the network or features; gene
popularity / study bias (does it beat a degree or publication-count baseline?); disease overlap between train and test;
selection of the best of many attempts on the same split; pipeline bugs (does the current best reproduce?).
For each explanation give one check runnable in ≤20 minutes with existing files, with the number that would confirm it.
Do not propose fixes to the idea; only checks.
```

### 4.6 Implementer (closed-loop test)
```
Implement the minimal test of idea {id} as specified in {spec_path} within 2 hours wall time. GPU only via
~/biotools/bin/gpu-run --mem <GB> -- <cmd>. You may not modify {eval_script}, split files or label files (checksums are
verified). Run: (1) positive control = current best through your code path, must reproduce {value}±{tol};
(2) the idea on dev-A; (3) the red-team checks {checks}. Write {report_path}: what was run (commands, commit), numbers
with paired bootstrap CI vs current best and baselines, runtime, every deviation from the spec, and status
keep/discard/crash per the keep rule. Do not tune on anything except dev-A; do not touch dev-B or confirm.
```

---

## 5. Failure modes and safeguards

| Failure mode | Evidence | Safeguard |
|---|---|---|
| Diversity collapse (all agents propose the same 5 ideas) | rows 1, 4, 5 | independent context, knowledge slices, verbalized tail ideas, dedup with cluster sizes, `fresh` island in each later round, lead does not generate |
| Novel-sounding but unworkable ideas win | rows 2, 3 | tournament criterion excludes novelty/excitement; promotion only by measured result |
| "Novel" ideas that are already published | rows 10, 19 | retrieval-based novelty check with citations; humans/lead spot-check 3 |
| Hallucinated data facts (column exists, N genes) | rows 10, 12 | fact sheet with paths; critics and novelty agent verify with one-line file checks |
| Judge noise, position bias, non-transitivity | rows 1, 25 | both orders, Swiss + Bradley-Terry, decisive-fact field, report inconsistency rate; tournament only filters |
| Same-model blind spots (all agents are Fable) | row 26 | fresh-context critics with different lenses; ground truth from the evaluator; human review of the final shortlist; optionally one critic from a different model family if available |
| Overfitting the dev split by many attempts | row 22 | three splits, keep rule with CI, selection on dev-B, attempt log as multiplicity, `confirm` once |
| Leakage making a test look good | domain (temporal benchmarks) | leakage class per idea, red-team checks, positive control, popularity baseline |
| Pipeline bugs producing fake wins or fake losses | row 10 | positive control reproduces current best; implementers cannot touch eval; "not run" ≠ "failed" |
| Collapse to easy ideas | Si et al. 2026 (row in modetor protocol) `[snippet]` | high-ceiling bucket in each round |
| Anchoring of later rounds on earlier rankings | row 4 | pass measured results and weaknesses, not idea rankings |
| Endless iteration | row 16 | stop after round 3 or when a round adds nothing beyond the CI |
| Lost provenance | general | all ideas, critiques, tournaments, results to `docs/ideation/`, `docs/ideas.md`, `docs/invalidations.md`; commit before each next step |

**Residual limits (honest):** no published study ablates the whole pipeline for computational-biology method ideation;
the evidence that the *closed loop* works comes from domains with exact evaluators (math, code, Kaggle), while ours is
noisy (~35-100 diseases, paired SD ~0.03). The protocol therefore treats measured dev-A gains as hypotheses until dev-B
and confirm agree.
