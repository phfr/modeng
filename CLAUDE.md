# modeng: start here (read `docs/NEXT.md` first)
Next-generation **disease module prediction** (which genes belong to a disease, ranked; and the module they form).
Started 2026-11-11 23:56 as an autonomous overnight research sprint (user: be ambitious, no T0 limitation, modern tools, download what is needed,
multi-round AI ideation with Fable reviews, implement and benchmark the best ideas, keep all ideas / invalidations / insights in docs).
- Sibling repos (read freely, NEVER edit): `/home/ph/code/modet` (ranking benchmark, OT 21.04->25.09), `/home/ph/code/subex` (modules), `/home/ph/code/modetor` (planning).
- modet's **sealed-fold diseases** (`folds_v1.json`, `sealed: true`) are never used for development or evaluation here (keeps modet's ledger valid).
- No T0 restriction on inputs, BUT every benchmark states its leakage class: `temporal-clean` (inputs dated <= seed release), `transductive` (modern inputs, labels may leak through inputs), `holdout-current` (CV on current labels). A claim names its benchmark.
- GPU only via `~/biotools/bin/gpu-run --mem <GB> -- <cmd>`; tell subagents. Python env: `~/biotools/venvs/bio-net/bin/python` (see `~/biotools/README.md` for others).
- Docs are living: `docs/ideas.md` (every idea + status), `docs/insights.md` (facts learned, with evidence), `docs/invalidations.md` (what proved wrong / dead ends),
  `docs/journey.md` (dated log), `docs/reports/` (per-experiment reports). Update them as you go; verify subagent numbers before recording them as solid.
- **Data root: `/mnt/msadb/modeng/`** (like subex). `data/<sub>` are symlinks to `/mnt/msadb/modeng/data/<sub>` (cache, raw, views, donors, ...); put every new large file or download there, never on `/` (it filled up on 2026-11-12). Only small tracked files (e.g. `data/wide_confirm_ids.txt`) live in the repo.
- Commit with explicit paths; never push.
