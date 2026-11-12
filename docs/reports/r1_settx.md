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
  pre-2021 donor gene sets): a strong single model, but it adds nothing to the stacker. Verdict: promising, needs work
  as a channel.** Alone nuAP 0.124 / 0.130 (transfer 0.096 / 0.104, +0.028 / +0.026, CI excludes 0; stackP_base 0.119 / 0.134),
  but AP 0.114 / 0.099 is **below** transfer 0.123 / 0.119, so under the both-AP-and-nuAP rule it does not beat transfer
  (part of the nuAP gain is degree down-weighting). Increment over stackP_bdv: AP +0.003 / -0.001, nuAP +0.000 / -0.001
  (CIs straddle 0); over stackP_base: AP +0.004 / +0.002, nuAP +0.002 / +0.001 (CIs straddle 0). Only AUROC moves
  (+0.003 / +0.004, confirm CI [+0.002, +0.006]): long-tail re-ranking, as with the donors.
- Negative control `settx_auxperm` (same recipe, view rows permuted): alone nuAP 0.115 / 0.125, i.e. ~70 % of the
  gain over transfer comes from the learned context-feature combination plus per-gene propensities learned from the aux
  sets, not from gene biology; the real views add +0.009 [+0.004, +0.014] dev / +0.005 [-0.000, +0.011] confirm nuAP.

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
3. **Arrival training hurts on temporal, self-supervised training helps** (table below): the variants that see past
   arrivals score nuAP 0.063-0.076; the aux-only twin 0.124 / 0.130.
4. **Past-vs-temporal reliability shift explains the stacker damage.** On past OOF the arrival-trained settx beats
   transfer (nuAP 0.074 vs 0.046) but on temporal it is below it (0.063-0.076 vs 0.096-0.104); the past-trained
   LightGBM therefore ranks `r_settx` as its top feature and degrades. Probable causes: arrival memorisation that is
   release-specific; and inputs that postdate the past seeds (HPO 2021-01, MSigDB 2021-03, GO/Reactome/DepMap views are
   <= 21.04 but > 19.04), which are clean for temporal but give the past OOF scores look-ahead.
5. **Negative control (perm, arrival-trained).** Permuted views give *higher* alone nuAP than real views (0.089 / 0.085)
   but lower zero-support AUROC (0.59 vs 0.67-0.68): with meaningless gene features the model leans on the transfer
   context features; with real features it trades head accuracy for a long-tail signal that is no better than popularity
   (zero-support AUROC popularity 0.68-0.69).

## Results (temporal; dev = 53, confirm = 138 tasks; dev / confirm)
Channel alone (`python -m modeng.settx.evaluate`; per-task parquet with z_* columns in `results/temporal/<split>/settx_<v>.parquet`).
Zero-support = rankable genes with transfer_jaccard == 0 (60-61 % of targets fall there); AUROC within that stratum.

| method | AP | nuAP | AUROC | R@100 | onnet AP | onnet nuAP | sa AP | zero-support AUROC |
|---|---|---|---|---|---|---|---|---|
| transfer_jaccard | 0.123 / 0.119 | 0.096 / 0.104 | 0.682 / 0.705 | 0.257 / 0.283 | 0.134 / 0.128 | 0.133 / 0.128 | 0.031 / 0.058 | 0.50 / 0.50 |
| popularity | 0.020 / 0.019 | 0.011 / 0.012 | 0.777 / 0.787 | 0.090 / 0.097 | 0.025 / 0.022 | 0.018 / 0.014 | 0.021 / 0.022 | 0.684 / 0.691 |
| settx (clean views, past+aux) | 0.055 / 0.048 | 0.063 / 0.072 | 0.764 / 0.778 | 0.135 / 0.150 | 0.064 / 0.055 | 0.089 / 0.088 | 0.018 / 0.022 | 0.667 / 0.673 |
| settx_all (transductive) | 0.055 / 0.048 | 0.076 / 0.070 | 0.776 / 0.787 | 0.139 / 0.155 | 0.065 / 0.056 | 0.100 / 0.088 | 0.015 / 0.018 | 0.683 / 0.686 |
| settx_perm (control) | 0.069 / 0.061 | 0.089 / 0.085 | 0.728 / 0.743 | 0.142 / 0.159 | 0.084 / 0.071 | 0.128 / 0.107 | 0.021 / 0.023 | 0.591 / 0.609 |
| **settx_auxonly** (clean, aux only) | 0.114 / 0.099 | **0.124 / 0.130** | 0.787 / 0.799 | 0.240 / 0.259 | 0.134 / 0.111 | 0.175 / 0.151 | 0.022 / 0.041 | 0.661 / 0.679 |
| settx_auxperm (control) | 0.105 / 0.096 | 0.115 / 0.125 | 0.777 / 0.788 | 0.237 / 0.259 | 0.123 / 0.106 | 0.165 / 0.146 | 0.024 / 0.039 | 0.641 / 0.657 |

Past OOF (cross-fitted, Louvain folds; 1,786 past tasks): transfer AP 0.051 nuAP 0.046; settx 0.043 / 0.074; settx_all
0.044 / 0.074; settx_perm 0.038 / 0.064; settx_auxonly 0.044 / 0.064.

Stacker (LightGBM trained on 900 past episodes, `--train past`; reference rows re-used from `results/temporal/*/stackP_{base,bdv}.parquet`, 04:12 / 04:14 runs):

| stack | AP | nuAP | AUROC | R@100 | onnet AP | onnet nuAP |
|---|---|---|---|---|---|---|
| stackP_base | 0.078 / 0.081 | 0.119 / 0.134 | 0.779 / 0.792 | 0.223 / 0.236 | 0.096 / 0.088 | 0.186 / 0.161 |
| stackP_base_settxaux | 0.082 / 0.083 | 0.120 / 0.135 | 0.789 / 0.804 | 0.232 / 0.241 | 0.103 / 0.091 | 0.188 / 0.161 |
| stackP_bdv (reference) | 0.102 / 0.091 | 0.124 / 0.151 | 0.810 / 0.825 | 0.252 / 0.260 | 0.121 / 0.099 | 0.173 / 0.171 |
| stackP_bdv_settx | 0.089 / 0.081 | 0.104 / 0.135 | 0.811 / 0.828 | 0.226 / 0.237 | 0.112 / 0.089 | 0.151 / 0.155 |
| stackP_bdv_settxaux | 0.104 / 0.090 | 0.124 / 0.150 | 0.814 / 0.829 | 0.255 / 0.262 | 0.124 / 0.098 | 0.174 / 0.170 |
| stackP_bdv_settxauxperm (control) | 0.094 / 0.090 | 0.115 / 0.150 | 0.812 / 0.826 | 0.256 / 0.258 | 0.117 / 0.098 | 0.169 / 0.171 |

Paired cluster bootstrap (5,000), delta vs reference [95 % CI]:
- bdv_settx - bdv: AP -0.013 [-0.022, -0.005] / -0.010 [-0.018, -0.004]; nuAP -0.019 [-0.031, -0.009] / -0.016 [-0.026, -0.008].
- bdv_settxaux - bdv: AP +0.003 [-0.001, +0.007] / -0.001 [-0.005, +0.003]; nuAP +0.000 [-0.009, +0.009] / -0.001 [-0.008, +0.004];
  AUROC +0.003 [-0.001, +0.007] / +0.004 [+0.002, +0.006].
- base_settxaux - base: AP +0.004 [-0.003, +0.011] / +0.002 [-0.001, +0.005]; nuAP +0.002 [-0.005, +0.008] / +0.001 [-0.005, +0.007].
- settx_auxonly alone - transfer_jaccard: nuAP +0.028 [+0.018, +0.039] / +0.026 [+0.017, +0.037] (AP lower, see table).
- settx_auxonly - settx_auxperm alone: nuAP +0.009 [+0.004, +0.014] / +0.005 [-0.000, +0.011].
- Negative control bdv_settxauxperm - bdv: AP -0.007 / -0.001, nuAP -0.009 / -0.000 (no gain, as required; slight dev harm).
- Positive control: stackP_bdv / stackP_base reproduce the brief's post-fix references (dev nuAP 0.124 / 0.119, confirm 0.151 / 0.134).
- `settx_all` stacker run (transductive arm) was killed when the lead limited RAM use to one stacker at a time; alone it
  equals `settx` within noise, so the same damage is expected.

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
- Up to four stackers ran in parallel at first (load ~157, RAM 83/90 GB machine-wide); reduced to one at a time after the lead's broadcast.

## New ideas
- Self-supervised set completion as a *pretraining-free channel*: the aux-only model never sees an arrival yet matches
  the arrival-trained base stacker; try more aux families (21.04 context rows with temporal tasks + relatives excluded,
  paralog / km families), longer training, and a listwise head that takes transfer as an offset.
- Train the stacker's channel-reliability on a *pseudo-temporal* split whose inputs are clean for it (seq views + 19.04
  aux), or fit the stacker on temporal dev with cross-fitting instead of `past`, to avoid the reliability shift.
- Arrival memorisation is a general hazard for any learned model trained on `past` (ML-6, LT-7, IR-7): always use
  Louvain groups and compare to a self-supervised twin.
