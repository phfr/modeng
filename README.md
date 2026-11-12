# modeng: next-generation disease module prediction
Autonomous research sprint (2026-11-11/12). See `CLAUDE.md`, `docs/NEXT.md`, `docs/ideas.md`, `docs/insights.md`, `docs/invalidations.md`, `docs/journey.md`.

## Using modeng-v2
modeng-v2 is the recommended model of the sprint: r7 arm D, "W_v1clean without label-process channels". It is a LightGBM stacker
over 47 temporal-clean witness channels, trained on 712 `temporal_wide` wide_train diseases (OT 21.04 seeds, 25.09 arrivals) with
the relative-target guard on and positive weight deg^-0.35. The recipe, channel list, vintages and leakage class are in
`configs/modeng_v2.yaml`. The model files live on `/mnt/msadb/modeng/models/modeng_v2_2104/`, not in git. `MODEL_CARD.md` there
gives data, dates, metrics with CIs and known limits.

```bash
PY=/home/ph/biotools/venvs/bio-net/bin/python
# train + save (about 10 min, about 9 GB RAM; reproduces r7 arm D exactly; writes the model card)
OMP_NUM_THREADS=8 PYTHONPATH=/home/ph/code/modeng/src $PY /home/ph/code/modeng/scripts/train_v2.py
# rank genes for one disease (seeds = its OT 21.04 gene set) and call a module at calibrated P >= 0.2
PYTHONPATH=/home/ph/code/modeng/src $PY -m modeng.predict --disease EFO_0000270 --out /path/asthma.tsv \
    --module /path/asthma_module.tsv --tau 0.2
# own seeds (Ensembl ids or symbols), current OT 26.09 context (production only)
PYTHONPATH=/home/ph/code/modeng/src $PY -m modeng.predict --disease EFO_0000270 --seeds my_genes.txt --vintage 2609 --out /path/x.tsv
# tests (2 diseases, CLI, sealed refusal, benchmark reproducibility within 1e-6)
PYTHONPATH=/home/ph/code/modeng/src $PY -m pytest -q /home/ph/code/modeng/tests/test_predict.py
```
Outputs:
- `--out` TSV: candidates (seeds are not listed), ranked by score. Columns: `gene_id` (ENSG), `symbol`, `score`, `calib_p`,
  `on_network`, `top_witnesses` and `c_<group>`. `calib_p` = 0.8 x isotonic(score), fitted on dev: the approximate probability
  that the gene joins the disease's OT genetic set within about 4 years. `top_witnesses` and `c_<group>` are LightGBM pred_contrib
  log-odds summed per witness group (disease_transfer, gene_prior, network, pathway_complex, phenotype, literature,
  sequence_paralogy, expression_dependency, seed_count). The sidecar `<out>.json` records the seeds, the masking, the vintage and the config hash.
- `--module` TSV: the seeds plus candidates with `calib_p >= tau`. It also writes `<stem>.module.npz` (a subex artefacts_v1 module on
  STRING 11.5 T0). If subex is importable, it adds subex `leiden_anno` submodules (`<stem>.sub.npz/.tsv`). subex is only imported,
  never edited. `--no-subex` skips this step.

Semantics. predict builds one ad-hoc task. Its seeds come from `--seeds`, or by default from the disease's gene set in the vintage
release. Its context is that release's disease x gene matrix without modet's sealed ids. Masking is "self+desc": the disease's own row
and its descendants' rows keep only the seeds, and nothing else is masked. Channels are computed by the existing channel code, without
the disk cache. Side inputs are always 21.04-vintage (PubTator <= 2021-03, HPO 2021-01, OT 21.04 evidence, STRING 11.5).
- `--vintage 2104` (default) is the reproducible research mode. It uses the same inputs as the benchmarks. With a benchmark task's
  seeds and masked genes, predict reproduces the benchmark score within 1e-6 (`tests/test_predict.py`).
- `--vintage 2609` is the "current" mode: OT 26.09 seeds and context. Its labels contain every arrival up to 2026-09, so it is for
  producing candidate lists only and must never be used to evaluate anything. It is also a distribution shift: a larger context
  combined with 21.04 side inputs, and no retraining.
- modet sealed-fold diseases and their relatives are refused (`--allow-sealed` for production use only).

Limits (details in the model card):
- The fresh confirmation is narrow. On r7 wide_confirm (35 diseases), AP +0.027 [+0.004, +0.049] and nuAP +0.042 [+0.006, +0.086]
  over transfer. This was one of four arms with no correction, so the result is not Bonferroni-robust.
- Strict-a (new genetic discoveries) ties with modet's headline.
- Targets are Open Targets label arrivals, which is a label-process caveat.
- Calibration is approximate. Shrink 0.8 was carried over from modeng-v1. For v2, the dev cross-fit gives about 0.89, and on the
  spent temporal confirm set the shrunk P is conservative (at P >= 0.2, predicted 0.35 vs realised 0.45). See the model card.
