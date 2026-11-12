# r6 P1b: the full witness recipe trained at scale on `temporal_wide`, 2026-11-12 14:20-15:20
Pre-registration (committed before any scoring): `docs/preregistration_wide_full.md`, commit `768c27d`.
Leakage class: temporal-clean / vintage as v1clean (no transductive inputs); `--train wide` = same-window training with the
relative-target guard (12,202 positive pairs dropped in 506 train tasks). **wide_confirm was not scored** (flag never set;
the driver asserts it). Temporal confirm (138) is spent: descriptive only.

## 1. Port of the ext witnesses to `temporal_wide`
`B.name == 'temporal_wide'` now maps to the 21.04 inputs, by dict entries only (backward compatible; no other code path changed):
- `ext/donors_a.py` `PT_BY_BENCH["temporal_wide"] = "a_pt21"` (dA_pt; also `lead_ptridge` pt_ridge / pt_ridge_pos)
- `ext/gwas_d.py` `CUTOFF["temporal_wide"] = "2021-04-30"` (gwas_catlag)
- `ext/priors_h.py` `HPO` / `REL["temporal_wide"]` = temporal's (HPO 2021-01-15, OT 21.04 evidence); `_hpo` reads the 2021 file format for both
- `ext/disease_c.py` `PREV["temporal_wide"] = "1904"` (dc_trend / dc_delta only; dc_ridge has no name dependence)
- `lead_cs.py` (cs_donor), `lead_cleanviews.py` (vc_*), base / donor / view channels: no name dependence.

Check `scripts/r6_wide_port_check.py` (`results/temporal_wide/port_check.txt`): all 21 ext channels, computed without cache on
3 temporal dev tasks and 3 wide_dev tasks under the name `temporal` and under `temporal_wide`, are bit-identical (NaN-equal).
Counterfactual with an unmapped name: gwas_catlag changes (would have used the full 2026 catalog = leak), dA_pt / pt_ridge
become NaN, priors / ph raise; cs_donor, dc_ridge, vc_* unchanged. No stale ext caches for temporal_wide existed.

## 2. Runs (`scripts/r6_wide_full.py`, analysis `scripts/r6_wide_full_analyze.py`, outputs `results/r6w/`)
Wide arms: 600 rounds, **1,500 negatives** (the `run_wide` default, not reduced for memory), <= 60 positives, deg^-0.35;
839,966 training rows. Peak RSS 8.6 GB (v1clean, 51 ch.) / 4.0 GB (v2), 45-420 s each at 8 threads (first run computes the channels).
- `W_v1clean` = 51 channels of `results/v1clean_channels.txt`; `W_v2` = 23 channels (`results/v2_channels.txt`): v1clean
  minus all 20 view similarities (14 vmean/vmax + 6 vc_*), cs_donor, dA_pt, 6 pri_*/hz_*.
- `W_v1perm` = v2 + the 28 dropped channels gene-permuted per task (same width 51; `ext/wide_perm.py`).
- Seeds: LightGBM seed 0 and 1 (negatives unchanged); `@avg` = per-task metrics averaged over the two.
- Comparators: cv-trained `stack_v1clean` (temporal dev = existing OOF row, reproduced exactly 0.1771 / confirm 0.1497;
  wide_dev = trained on all 53 temporal dev); past twins `stackP_v1clean` / `stackP_v2` (priors and ph are NaN on `past`).

| set | method | AP | nuAP | onnet nuAP | sa nuAP | deg-matched AP |
|---|---|---|---|---|---|---|
| temporal dev (53) | transfer_jaccard | 0.123 | 0.096 | 0.133 | 0.040 | 0.188 |
| | stack_v1clean (cv OOF) | 0.177 | 0.152 | 0.212 | 0.093 | 0.255 |
| | stackP_v1clean / stackP_v2 | 0.153 / 0.158 | 0.158 / 0.162 | 0.204 / 0.225 | 0.068 / 0.070 | 0.245 / 0.248 |
| | stackW_wclean (r6, no ext) | 0.119 | 0.129 | 0.177 | 0.055 | |
| | **W_v1clean** s0 / s1 | **0.191 / 0.188** | **0.180 / 0.181** | 0.229 / 0.236 | 0.079 / 0.092 | 0.273 / 0.276 |
| | W_v2 s0 / s1 | 0.181 / 0.185 | 0.176 / 0.180 | 0.233 / 0.238 | 0.073 / 0.077 | 0.269 / 0.265 |
| | W_v1perm | 0.185 | 0.178 | 0.235 | 0.070 | 0.268 |
| wide_dev (45) | transfer_jaccard | 0.100 | 0.129 | 0.155 | 0.034 | 0.184 |
| | stack_v1clean (dev-trained) | 0.124 | 0.157 | 0.200 | 0.075 | 0.251 |
| | stackP_v1clean / stackP_v2 | 0.129 / 0.128 | 0.175 / 0.177 | 0.205 / 0.203 | 0.055 / 0.067 | 0.286 / 0.265 |
| | **W_v1clean** s0 / s1 | 0.124 / 0.132 | 0.183 / 0.184 | 0.211 / 0.205 | 0.071 / 0.088 | 0.275 / 0.284 |
| | W_v2 s0 / s1 | 0.124 / 0.126 | 0.178 / 0.185 | 0.220 / 0.224 | 0.076 / 0.063 | 0.271 / 0.272 |
| | W_v1perm | 0.112 | 0.172 | 0.205 | 0.052 | 0.251 |
| temporal confirm (138, descriptive) | stack_v1clean | 0.150 | 0.173 | 0.205 | 0.090 | 0.253 |
| | W_v1clean s0 / s1 | 0.172 / 0.172 | 0.204 / 0.206 | 0.225 / 0.225 | 0.090 / 0.095 | 0.273 / 0.275 |
| | W_v2 s0 / s1 | 0.165 / 0.165 | 0.197 / 0.196 | 0.211 / 0.211 | 0.081 / 0.085 | 0.264 / 0.265 |
| | W_v1perm | 0.162 | 0.193 | 0.211 | 0.085 | 0.262 |

Paired cluster bootstrap (10,000; full table `results/r6w/compare.tsv`), seed-averaged arms:

| comparison | temporal dev dAP / dnuAP | wide_dev dAP / dnuAP | confirm dAP / dnuAP (descr.) |
|---|---|---|---|
| W_v2 vs cv v1clean | +0.006 [-0.008, +0.019] / **+0.025 [+0.008, +0.046]** | +0.002 [-0.036, +0.040] / +0.025 [-0.018, +0.083] | **+0.015 [+0.006, +0.023]** / **+0.023 [+0.011, +0.035]** |
| W_v1clean vs cv v1clean | +0.012 [-0.003, +0.027] / **+0.028 [+0.013, +0.046]** | +0.005 [-0.014, +0.025] / +0.027 [-0.003, +0.065] | **+0.022 [+0.012, +0.032]** / **+0.032 [+0.020, +0.043]** |
| W_v1clean vs transfer_jaccard | **+0.066 [+0.047, +0.088]** / **+0.084** | **+0.029 [+0.004, +0.058]** / **+0.054 [+0.025, +0.092]** | **+0.053** / **+0.101** |
| W_v1clean vs past twin | **+0.037 [+0.022, +0.052]** / **+0.022** | -0.001 [-0.027, +0.025] / +0.009 [-0.016, +0.036] | **+0.040** / **+0.031** |
| W_v2 vs W_v1clean | -0.007 [-0.015, +0.002] / -0.003 | -0.003 [-0.025, +0.017] / -0.002 | **-0.007 [-0.012, -0.002]** / **-0.009 [-0.015, -0.002]** |
| W_v1clean vs W_v1perm (s0) | +0.006 [-0.001, +0.015] / +0.002 | **+0.012 [+0.003, +0.023]** / +0.011 [-0.000, +0.026] | **+0.010 [+0.004, +0.015]** / **+0.011 [+0.004, +0.018]** |
| W_v2 vs W_v1perm (s0) | -0.004 [-0.009, +0.001] / -0.002 | **+0.012 [+0.000, +0.025]** / +0.006 | +0.002 [-0.002, +0.006] / +0.004 [-0.001, +0.009] |
| seed 1 vs seed 0 (v1clean; v2) | -0.003; +0.003 / +0.001; +0.004 | +0.008; +0.002 / +0.001; +0.007 | 0.000; 0.000 / +0.002; -0.001 |

Degree-matched AP moves with AP (W_v1clean vs cv v1clean: dev +0.020 [+0.009, +0.032], wide_dev +0.028 [+0.005, +0.055]).
sa nuAP (strict-a, 28 / 10 tasks) does not improve over cv v1clean (dev -0.008 to -0.019, CIs touch 0).

## 3. Decision (pre-registered rule, `results/r6w/decision.tsv`)
**Neither wide-trained arm is the new candidate.** Both pass the nuAP condition on both sets (nuAP gains +0.025 to +0.028), but the
AP condition fails: AP point estimates are positive on both sets (W_v1clean +0.012 dev / +0.005 wide_dev; W_v2 +0.006 / +0.002),
with 95 % CIs including 0 on both. By the rule this is "fail", not "underpowered" (that label needed a CI excluding 0 on one set).

## 4. Reading (descriptive)
- **Scale buys nuAP, not AP, relative to the 53-dev cv recipe.** The robust effect of wide training on the full recipe is
  +0.025-0.03 nuAP (non-hub targets) on dev and confirm, and the same size on wide_dev (CI just touching 0). AP gains are
  +0.005-0.02 and only significant on spent confirm.
- **The ext witnesses are what made the wide stack competitive:** W_v1clean vs the r6 modet-free `stackW_wclean`: temporal dev
  AP 0.191 vs 0.119. Wide-trained W_v1clean now beats transfer_jaccard on wide_dev AP (+0.029, CI > 0), which r6's
  wclean stack did not.
- **The audit's pruning does not transfer to scale.** With 712 training tasks the dropped channels (views, priors, cs_donor,
  dA_pt) carry information: v1clean beats its same-width permuted control on confirm (+0.010 AP, +0.011 nuAP) and on wide_dev
  (+0.012 AP), and W_v2 is below W_v1clean on confirm (-0.007 AP, CI < 0). The audit's "views hurt" was an overfitting effect
  of 53 training diseases. Pruning vs noise: v2 ~= v1perm on temporal (width costs nothing at scale), v2 > v1perm on wide_dev
  (+0.012 AP): on wide_dev the permuted channels cost what the real ones gain, i.e. there the 28 channels are close to noise.
- **Past twin vs wide on wide_dev: no difference** (AP -0.001, nuAP +0.009, n.s.); the wide advantage over the past twin is
  confined to the temporal panel (+0.037 AP dev, +0.040 confirm). Same-window training may partly fit panel-specific label
  dynamics (the panel diseases are relatives of wide_train hubs even after the guard); wide_dev is the cleaner test and is flat.
- wide_dev (45 tasks, median 6 targets) cannot resolve AP differences below ~0.03-0.04; seed noise is up to 0.008 AP there.

## Files
`src/modeng/ext/{donors_a,gwas_d,priors_h,disease_c}.py` (mapping), `src/modeng/ext/wide_perm.py`, `scripts/r6_wide_port_check.py`,
`scripts/r6_wide_full.py`, `scripts/r6_wide_full_analyze.py`, `results/v2_channels.txt`, `results/r6w/{summary,compare,decision}.tsv`,
`results/r6w/imp_*.tsv` (gain; W_v1clean top: dc_ridge, dc_ridge_pos, transfer_jaccard, hz_gwas, dA_pt), `results/r6w/run_all.log`,
per-task parquets / scores in `results/temporal/{dev,confirm}/` and `results/temporal_wide/wide_dev/` (`stackW_W_*`, `stackP_v1clean`, `stackP_v2`,
`stack_v1clean`). `results/temporal/{dev,confirm}/stack_v1clean.scores.npy` re-saved by a write=False rerun (identical metrics).
