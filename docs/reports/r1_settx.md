# R1 settx: identity-blind episodic set-completion transformer (idea ML-5, implementer F)

Date 2026-11-12 04:13-06:40. Benchmark: **temporal** only (seeds OT 21.04, targets 21.04 -> 25.09). Training never
touches a 21.04 -> 25.09 arrival. Forward / holdout not run (time).

## Verdict (short)
- **Trained on `past` arrival episodes (the ML-5 recipe as briefed): discard.** The model memorises which genes arrived
  19.04 -> 21.04 (held-out-past AP 0.31 with naive cluster folds, 0.02 with leak-aware folds vs transfer 0.06) and that
  does not transfer to 21.04 -> 25.09: alone nuAP 0.063 / 0.072 (dev / confirm) < transfer_jaccard 0.096 / 0.104, and as a
  stacker channel it **hurts**: stackP_bdv + settx nuAP 0.104 / 0.135 vs stackP_bdv 0.124 / 0.151 (delta -0.019 [-0.031,
  -0.009] dev, -0.016 [-0.026, -0.008] confirm, cluster bootstrap).
- **Self-supervised only (`settx_auxonly`: never sees an arrival; trained to complete masked 19.04 disease rows and
  pre-2021 donor gene sets): promising.** Alone nuAP **0.124 / 0.130** (transfer 0.096 / 0.104; stackP_base 0.119 / 0.134),
  onnet nuAP 0.175 / 0.151. Stacker increment: see table below (FILL).

## What was built
| file | content |
|---|---|
| `src/modeng/settx/data.py` | gene inputs (15 views standardised, missing rows 0, view-mask; view sets all / clean / seq; optional shared row permutation = negative control); per-episode context features cache (transfer_jaccard, popularity from the harness on the masked context); `aux_episodes()` self-supervised episodes |
| `src/modeng/settx/model.py` | `SetTX`: per-view Linear(d_v, 32) x 15 + mask -> MLP -> 128-d gene embedding (no gene-id embedding, view dropout 0.15); seed self-attention layer + attention pooling; candidate -> seed cross-attention; head on [h_c, attn, z, h_c*z, log n_seeds, 5 context features]; optional additive context-only path |
| `src/modeng/settx/groups.py` | leak-aware fold groups for past episodes (Louvain on ontology ancestor/descendant + target-overlap graph) |
| `src/modeng/settx/train.py` | training (listwise sampled-softmax loss, positives weighted deg^-0.5, 64 positives + 384 uniform + 128 hard (transfer top-2000) negatives per episode, batch 16 episodes, 1,500 steps AdamW/OneCycle lr 1e-3, bf16); 3-fold cross-fitting on past + final model on all; `--only tune` (held-out past) |
| `src/modeng/settx/evaluate.py` | channel alone on temporal dev/confirm + zero-transfer-support stratum |
| `src/modeng/ext/settx_f.py` | channels `settx` (clean views), `settx_all` (all views), `settx_perm` (permuted views), `settx_auxonly`, `settx_auxperm` (load saved predictions; past = cross-fitted OOF, temporal = final model; NaN elsewhere) |

Commands (GPU via `~/biotools/bin/gpu-run --mem 8 --`; each training run 6-10 min wall for 3 folds + final, under the 25 min cap):
```
python -m modeng.settx.train --aux --steps 1500 --folds 3 --variant clean --views clean      # settx
python -m modeng.settx.train --aux --steps 1500 --folds 3 --variant all   --views all        # settx_all
python -m modeng.settx.train --aux --steps 1500 --folds 3 --variant perm  --views all --permute 1
python -m modeng.settx.train --aux --aux-frac 1.0 --steps 1500 --folds 3 --variant auxonly --views clean
python -m modeng.settx.train --aux --aux-frac 1.0 --steps 1500 --folds 3 --variant auxperm --views clean --permute 1
python -m modeng.settx.evaluate --variant clean all perm auxonly auxperm --write
python -m modeng.stack --bench temporal --channels base,donor,view,<channel> --train past --tag <tag>
```

## Episodes
- `past`: 1,786 tasks (seeds OT 19.04, targets 19.04 -> 21.04 arrivals, context 19.04 with harness masking).
- aux (self-supervised, hide 30 % of a set): 1,402 OT 19.04 context rows (10-500 genes; harness "family" masking:
  descendants + near-duplicates masked, ancestors re-propagated), 2,000 HPO 2021-01, 2,000 MSigDB GO, 1,255 Reactome,
  530 WikiPathways, 393 IMPC 13, 73 Complex Portal sets (donor families: own row masked). Context features of an aux
  episode are transfer / popularity computed within its own family. 7,653 episodes.
- default mix: 50 % past / 50 % aux per batch; `auxonly`: 100 % aux.

## Findings in order
1. **Cluster folds leak on past.** modet clusters cover only panel diseases; 1,616 clusters for 1,786 past tasks, so an
   ancestor and its descendants (whose propagated arrivals are subsets of each other) land in different folds.
   Held-out-past AP 0.31 (transfer 0.044) with cluster folds -> 0.022 with Louvain groups over the ontology +
   target-overlap graph (70 communities, 10-12 % of edge weight cut; transfer 0.064 on that split). Any future learned
   method that cross-fits on `past` needs these groups (`settx.groups.leak_folds`), not `task.cluster`.
2. **Arrival training = memorisation.** Without aux episodes the model is far below transfer on held-out past
   (AP 0.022 vs 0.064); context-only sanity model (gene embedding zeroed) 0.028 / wAP 0.030 (transfer wAP 0.035).
   Adding aux episodes: AP 0.058, wAP 0.046 (> transfer wAP 0.035). 3,000 steps / aux 0.75 / clean views / more
   regularisation all within noise (AP 0.049-0.055).
3. **Temporal alone (FILL table).**
4. **Past-vs-temporal reliability shift explains the stacker damage.** On past OOF the arrival-trained settx beats
   transfer (nuAP 0.074 vs 0.046) but on temporal it is below it (0.063-0.076 vs 0.096-0.104); the past-trained
   LightGBM therefore ranks `r_settx` as its top feature and degrades. Probable causes: arrival memorisation that is
   release-specific; and inputs that postdate the past seeds (HPO 2021-01, MSigDB 2021-03, GO/Reactome/DepMap views are
   <= 21.04 but > 19.04), which are clean for temporal but give the past OOF scores look-ahead.
5. **Negative control (perm, arrival-trained).** Permuted views give *higher* alone nuAP than real views (0.089 / 0.085)
   but lower zero-support AUROC (0.59 vs 0.67-0.68): with meaningless gene features the model leans on the transfer
   context features; with real features it trades head accuracy for a long-tail signal that is no better than popularity
   (zero-support AUROC popularity 0.68-0.69).

## Results (temporal; dev = 53, confirm = 138 tasks)
FILL

## Leakage class
- temporal: `settx`, `settx_auxonly`: temporal-clean (views seq / pre21 / annot <= 21.04; aux sets <= 2021-03; context
  features from the harness; training labels 19.04 -> 21.04 only). `settx_all`: transductive (HPA, GTEx, Perturb-seq).
- the past OOF scores used to train the stacker are NOT past-clean (HPO 2021-01, MSigDB 2021-03, GO/Reactome T0 views
  postdate 19.04); this can only inflate the channel's apparent reliability on past, i.e. biases the increment test
  against settx, not in its favour.
- No 21.04 context rows used as aux (would need excluding the 191 temporal diseases + relatives); no forward / holdout.

## Runtime
Feature caches 8 s (benchmark features), 60 s (aux). Training: 1,500 steps ~ 90-140 s per model on the RTX 5090
(~8 GB), 3 folds + final + prediction 6-10 min per variant. Stacker runs 15-20 min each (machine load ~157).

## What failed / caveats
- CUDA "device busy or unavailable" when several processes start at once under WSL (retry loop added in the shell).
- I looked at temporal dev numbers of a 200-step smoke model before the leak-aware folds existed; all later model choices
  (aux episodes, 1,500 steps) were made on held-out past, not on temporal. `auxonly` was a post-hoc variant chosen after
  seeing that arrival-trained variants failed on temporal dev: its confirm numbers are the confirmation.
- `settx_strict` (sequence views + 19.04 aux only, to make past OOF past-clean) was started and killed for time.

## New ideas
- Self-supervised set completion as a *pretraining-free channel*: the aux-only model never sees an arrival yet matches
  the arrival-trained base stacker; try more aux families (21.04 context rows with temporal tasks + relatives excluded,
  paralog / km families), longer training, and a listwise head that takes transfer as an offset.
- Train the stacker's channel-reliability on a *pseudo-temporal* split whose inputs are clean for it (seq views + 19.04
  aux), or fit the stacker on temporal dev with cross-fitting instead of `past`, to avoid the reliability shift.
- Arrival memorisation is a general hazard for any learned model trained on `past` (ML-6, LT-7, IR-7): always use
  Louvain groups and compare to a self-supervised twin.
