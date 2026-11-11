# Invalidations and dead ends (assumptions refuted, ideas killed, mistakes caught; newest first)
- 2026-11-12 01:20 Apparent "transfer beats modet headline" on all-gene temporal was an evaluation-universe artefact (off-network targets), not a real reversal (harness agent; verified per disease).
- 2026-11-12 01:00 Holdout v1 masking leaked a tell (sibling-present / ancestor-missing): transfer_onto AUROC 0.29 exposed it; fixed.
- 2026-11-12 01:35 **Bug caught:** views/donors were built in HGNC-row order while the harness universe is sorted ENSG; first UDT run showed every family at AUROC < 0.5 (the tell). Fixed by `scripts/reorder_universe.py` (files carry `order=sorted_ensg`); build scripts now sort. Any consumer must use `modeng.data.universe()` order.
