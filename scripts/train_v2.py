"""Train and save modeng-v2 (= r7 arm D, `stackW_P4D_v1clean_nolab`) from configs/modeng_v2.yaml.

Writes <model_root>/modeng_v2_<vintage>/ (on /mnt, never in git):
  model.txt            LightGBM text model (feature names embedded)
  features.json        feature order (channels, r_<channels>, log_nseeds)
  config.yaml          copy of the config; model_meta.json holds its sha256 (predict refuses a mismatching config)
  calibration.json     isotonic map (x, y thresholds) fitted on temporal dev + wide_dev candidate pairs, shrink
  eval_<bench>_<split>.parquet  per-task metrics of this model; eval_compare.tsv paired cluster bootstrap vs transfer_jaccard
  model_card.json / MODEL_CARD.md
Checks: the trained model reproduces the stored r7 arm-D scores (results/<bench>/<split>/stackW_P4D_v1clean_nolab.scores.npy,
float16) and their per-task AP.

Leakage: wide_confirm (spent by r7) is used ONLY inside relative_target_guard, as in r7 (it drops training pairs; no scoring);
this is needed to reproduce arm D exactly. Training never reads wide_confirm labels as training rows.
Usage: OMP_NUM_THREADS=8 PYTHONPATH=src python scripts/train_v2.py [--model-dir DIR]
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, "/home/ph/code/modeng/src")
import numpy as np, pandas as pd

from modeng import tasks as T, stack as S, bench as BM, data as D
from modeng.compare import paired_bootstrap
from modeng.predict import CONFIG, load_config, feature_names, default_model_dir

R7_METHOD = "stackW_P4D_v1clean_nolab"

# r7 (docs/reports/r7_wide_confirm.md): the one pre-registered scoring of wide_confirm (35 diseases, 28 clusters), vs transfer_jaccard
R7 = dict(fold="wide_confirm (35 fresh off-panel diseases, 28 ontology clusters, 297 targets; OT 21.04 -> 25.09)",
          preregistration="docs/preregistration_wide.md Amendment 1, commit 2fb1b77 (2026-11-12 15:16)",
          ap=0.118, ap_transfer=0.091, dAP=0.027, dAP_ci=[0.004, 0.049],
          nuap=0.126, nuap_transfer=0.084, dnuAP=0.042, dnuAP_ci=[0.006, 0.086],
          deg_matched_ap=0.259, d_deg_matched_ap=0.087, d_deg_matched_ap_ci=[0.043, 0.133],
          onnet_ap=0.152, d_onnet_ap=0.055, d_onnet_ap_ci=[0.013, 0.103], precision_at_20=0.076, auroc=0.87, recall_at_100=0.39,
          caveat="one of four pre-registered arms, uncorrected; does not survive a Bonferroni correction for four arms "
                 "(98.75 % CIs: AP [-0.002, +0.054], nuAP [-0.002, +0.097])")


def git_sha():
    return subprocess.run(["git", "-C", "/home/ph/code/modeng", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def cand_pairs(B, Sc):
    xs, ys, tid = [], [], []
    for i, t in enumerate(B.tasks):
        ok = np.ones(B.n_genes, bool); ok[t.exclude] = False
        y = np.zeros(B.n_genes, bool); y[t.targets] = True
        xs.append(Sc[i, ok]); ys.append(y[ok]); tid.append(np.full(ok.sum(), i))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(tid)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model-dir", default=None)
    a = ap.parse_args()
    t0 = time.time()
    cfg, chash = load_config()
    tr = cfg["training"]
    out = Path(a.model_dir) if a.model_dir else default_model_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    names = list(cfg["channels"])
    feats = feature_names(names)
    assert len(names) == 47 and len(feats) == 95
    S.POS_DEG_POWER = tr["pos_deg_power"]
    params = dict(tr["lightgbm"])

    # ---- training set: temporal_wide wide_train with the relative-target guard (evals as r7 arm D)
    W = T.load(tr["benchmark"])
    Wt = W.split(tr["split"])
    Tm = T.load("temporal")
    evals = [("temporal", Tm.split("dev")), ("temporal", Tm.split("confirm")), ("temporal_wide", W.split("wide_dev"))]
    wc = [t for t in W.tasks if t.split == "wide_confirm"]          # spent fold: guard only (drops pairs), never scored here
    guard_sets = [E for _, E in evals] + [T.Benchmark(W.name, wc, W.context, W.context_ids, W.network, W.union_network,
                                                      W.leakage_class, W.degree)]
    drop, ndrop = S.relative_target_guard(Wt, guard_sets) if tr["guard"] else ({}, 0)
    mw, rw, nw = S.build_feats(Wt, tr["split"], names)
    X, y, w = S.training_rows(Wt, np.arange(len(Wt.tasks)), mw, rw, nw, tr["negatives_per_task"], seed=tr["sampling_seed"],
                              max_pos=tr["max_pos_per_task"], drop=drop)
    del mw, rw
    print(f"[train] {len(Wt.tasks)} tasks, {len(names)} channels, {len(y)} rows, {int(y.sum())} pos; guard dropped {ndrop} "
          f"positive pairs in {len(drop)} tasks ({time.time()-t0:.0f}s)", flush=True)
    import lightgbm as lgb
    model = lgb.train(params, lgb.Dataset(X, y, weight=w, feature_name=feats), tr["rounds"])
    n_rows, n_pos = len(y), int(y.sum())
    del X, y, w
    model.save_model(str(out / "model.txt"))
    json.dump(feats, open(out / "features.json", "w"))
    shutil.copy(CONFIG, out / "config.yaml")

    # ---- evaluation sets (benchmark path, channel caches) + reproduction of the stored r7 arm-D scores
    rows, per, repro, scores = [], {}, [], {}
    for bench, Bs in evals:
        split = Bs.tasks[0].split
        ms, rs, ns_ = S.build_feats(Bs, split, names)
        Sc = np.zeros((len(Bs.tasks), Bs.n_genes), np.float32)
        S.predict_all(model, Bs, np.arange(len(Bs.tasks)), ms, rs, ns_, Sc)
        del ms, rs
        scores[(bench, split)] = (Bs, Sc)
        np.save(out / f"scores_{bench}_{split}.npy", Sc)
        df = BM.evaluate(Sc, Bs, jobs=8)
        df.to_parquet(out / f"eval_{bench}_{split}.parquet", index=False)
        per[(bench, split)] = df
        ref = np.load(BM.RESULTS / bench / split / f"{R7_METHOD}.scores.npy").astype(np.float32)
        ref_df = pd.read_parquet(BM.RESULTS / bench / split / f"{R7_METHOD}.parquet")
        repro.append(dict(bench=bench, split=split, max_abs_diff_vs_r7_float16=float(np.abs(Sc.astype(np.float16).astype(np.float32) - ref).max()),
                          max_abs_ap_diff_vs_r7=float(np.abs(df.ap.to_numpy() - ref_df.ap.to_numpy()).max()),
                          ap=float(df.ap.mean()), r7_ap=float(ref_df.ap.mean())))
        tj = pd.read_parquet(BM.RESULTS / bench / split / "transfer_jaccard.parquet")
        for m in ("ap", "nuap", "onnet_ap", "onnet_nuap"):
            r = paired_bootstrap(df, tj, m, 10000)
            rows.append(dict(bench=bench, split=split, status="spent (descriptive)" if split == "confirm" else "dev", **r))
        print(f"[eval] {bench}/{split}: AP {df.ap.mean():.4f} nuAP {df.nuap.mean():.4f} | {repro[-1]}", flush=True)
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(out / "eval_compare.tsv", sep="\t", index=False, float_format="%.4f")
    rp = pd.DataFrame(repro); rp.to_csv(out / "repro_vs_r7.tsv", sep="\t", index=False)
    assert (rp.max_abs_ap_diff_vs_r7 < 1e-6).all() and (rp.max_abs_diff_vs_r7_float16 <= 1e-3).all(), f"does not reproduce r7 arm D:\n{rp}"

    # ---- calibration: isotonic on dev candidate pairs (temporal dev + wide_dev; both out-of-sample for a wide_train model)
    from sklearn.isotonic import IsotonicRegression
    xs, ys, cl = [], [], []
    for key in (("temporal", "dev"), ("temporal_wide", "wide_dev")):
        Bs, Sc = scores[key]
        x_, y_, t_ = cand_pairs(Bs, Sc)
        xs.append(x_); ys.append(y_); cl.append(np.array([f"{key[1]}:{Bs.tasks[i].cluster}" for i in t_]))
    xd, yd, cd = np.concatenate(xs), np.concatenate(ys), np.concatenate(cl)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(xd, yd)
    ix, iy = iso.X_thresholds_.astype(float), iso.y_thresholds_.astype(float)
    interp_diff = float(np.abs(np.interp(xd.astype(float), ix, iy) - iso.predict(xd)).max())   # sklearn returns float32
    print(f"[calibration] np.interp vs IsotonicRegression.predict max diff {interp_diff:.2e}", flush=True)
    assert interp_diff < 1e-6
    shrink = float(cfg["calibration"]["shrink"])
    # dev-only shrink estimate: isotonic cross-fit over 5 cluster groups
    uc = np.unique(cd); rng = np.random.default_rng(0); rng.shuffle(uc)
    fold = {c: k % 5 for k, c in enumerate(uc)}; fd = np.array([fold[c] for c in cd])
    pp = np.zeros_like(xd, dtype=float)
    for k in range(5):
        m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(xd[fd != k], yd[fd != k]); pp[fd == k] = m.predict(xd[fd == k])
    crossfit = {str(tau): dict(predicted=float(pp[pp >= tau].mean()), realised=float(yd[pp >= tau].mean()), n=int((pp >= tau).sum()),
                               shrink=float(yd[pp >= tau].mean() / pp[pp >= tau].mean())) for tau in (0.1, 0.2, 0.3) if (pp >= tau).any()}
    # descriptive: realised precision of the shipped map (shrink x iso) on the spent temporal confirm
    Bc, Scc = scores[("temporal", "confirm")]
    xc, yc, _ = cand_pairs(Bc, Scc)
    pc = shrink * np.interp(xc, ix, iy)
    conf_cal = {str(tau): dict(predicted=float(pc[pc >= tau].mean()), realised=float(yc[pc >= tau].mean()), n=int((pc >= tau).sum()))
                for tau in (0.1, 0.2, 0.3) if (pc >= tau).any()}
    json.dump(dict(x=ix.tolist(), y=iy.tolist(), shrink=shrink, fit_on=cfg["calibration"]["fit_on"], n_pairs=int(len(xd)),
                   n_pos=int(yd.sum()), dev_crossfit=crossfit, temporal_confirm_descriptive=conf_cal),
              open(out / "calibration.json", "w"), indent=1)

    # ---- meta + model card
    def met(key):
        df = per[key]
        return dict(n=int(df.ap.notna().sum()), ap=float(df.ap.mean()), nuap=float(df.nuap.mean()), auroc=float(df.auroc.mean()),
                    recall_at_100=float(df["recall@100"].mean()), onnet_ap=float(df.onnet_ap.mean()))
    vs = {f"{r.bench}/{r.split}/{r.metric}": dict(delta=r.delta, ci=[r.ci_lo, r.ci_hi], model=r.mean_a, transfer=r.mean_b, status=r.status)
          for r in cmp_.itertuples()}
    imp = pd.Series(model.feature_importance("gain"), index=feats).sort_values(ascending=False)
    meta = dict(model_id="modeng-v2", recipe_of=cfg["recipe_of"], config_sha256=chash, config=cfg, git=git_sha(),
                trained=time.strftime("%Y-%m-%d %H:%M:%S"), lightgbm=lgb.__version__, n_train_tasks=len(Wt.tasks), n_rows=n_rows,
                n_pos=n_pos, guard_dropped_pairs=int(ndrop), guard_tasks=len(drop), n_features=len(feats))
    json.dump(meta, open(out / "model_meta.json", "w"), indent=1)
    card = dict(model_id="modeng-v2", summary="LightGBM stacker over 47 temporal-clean witness channels, trained on 712 OT 21.04 -> "
                "25.09 diseases (temporal_wide wide_train); ranks the genes a disease will gain in Open Targets' genetic sets.",
                intended_use="rank candidate disease genes for one disease given its known genes (seeds); module calls via calibrated P",
                data=dict(training=f"{len(Wt.tasks)} wide_train diseases, {n_rows} rows ({n_pos} positives), guard dropped {ndrop} positive pairs",
                          vintages=cfg["data_vintages"], leakage_class=cfg["leakage_class"], leakage_notes=cfg["leakage_notes"]),
                dates=dict(seeds_and_inputs="<= OT 21.04 (2021-04)", targets="OT 25.09 additions", trained=meta["trained"]),
                metrics=dict(fresh_confirmation_r7=R7, this_model=dict({f"{b}/{s}": met((b, s)) for b, s in per}),
                             vs_transfer_jaccard_cluster_bootstrap=vs,
                             note="temporal dev (53) and wide_dev (45) were used for selection during the sprint (dev numbers); "
                                  "temporal confirm (138) is spent (59+ variants scored on it): descriptive only."),
                calibration=dict(map=f"{shrink} x isotonic(score)", dev_crossfit=crossfit, temporal_confirm_descriptive=conf_cal),
                top_features_gain=imp.head(15).round(1).to_dict(),
                known_limits=[
                    "Strict-a (genetic discoveries first seen 2022+, modet's primary endpoint): tie with modet's headline in every variant; "
                    "the gain comes from arrivals with older evidence somewhere (literature, phenotypes, ClinVar-type knowledge).",
                    "Label-process caveat: the model predicts arrivals in Open Targets' genetic sets (an OT pipeline outcome), not causal biology; "
                    "label-process channels were removed, but the targets themselves are OT label arrivals.",
                    f"Calibration: P = {shrink} x isotonic map fitted on dev (shrink 0.8 carried over from modeng-v1, where confirm realised ~0.72x). "
                    f"For this model the dev cross-fit shrink at tau 0.2 is {crossfit.get('0.2', {}).get('shrink', float('nan')):.2f} and on the spent "
                    f"temporal confirm the shrunk map is conservative (P >= 0.2: predicted {conf_cal.get('0.2', {}).get('predicted', float('nan')):.2f}, "
                    f"realised {conf_cal.get('0.2', {}).get('realised', float('nan')):.2f}); treat P as approximate and window-specific "
                    "(21.04 -> 25.09, ~4 years; a lower bound rather than an estimate on panel-like diseases).",
                    "Fresh confirmation is narrow: n = 35 diseases, one of four arms, not Bonferroni-robust; a larger fresh fold (next OT release) is pending.",
                    "Hub genes: deg^-0.35 positive weighting trades some hub-gene AP for non-hub (nuAP) gains.",
                    "vintage 2609 mode mixes 26.09 labels with 21.04 side inputs (distribution shift; production only, never evaluation).",
                    "Diseases in modet's sealed fold (and relatives) are refused by predict unless --allow-sealed."])
    json.dump(card, open(out / "model_card.json", "w"), indent=1, default=float)
    md = [f"# modeng-v2 model card ({meta['trained']}, git {meta['git'][:7]}, config sha256 {chash[:12]})", "",
          card["summary"], "", f"Recipe: r7 arm D `{R7_METHOD}` (reproduced: per-task AP max diff "
          f"{rp.max_abs_ap_diff_vs_r7.max():.1e}). Leakage class: {cfg['leakage_class']}.", "",
          "## Data and dates", f"- Training: {card['data']['training']}.",
          *[f"- {k}: {v}" for k, v in cfg["data_vintages"].items()], *[f"- {x}" for x in cfg["leakage_notes"]], "",
          "## Fresh confirmation (r7, pre-registered, wide_confirm, vs transfer_jaccard)",
          f"- AP {R7['ap']:.3f} vs {R7['ap_transfer']:.3f}: +{R7['dAP']:.3f} [{R7['dAP_ci'][0]:+.3f}, {R7['dAP_ci'][1]:+.3f}]; "
          f"nuAP {R7['nuap']:.3f} vs {R7['nuap_transfer']:.3f}: +{R7['dnuAP']:.3f} [{R7['dnuAP_ci'][0]:+.3f}, {R7['dnuAP_ci'][1]:+.3f}]",
          f"- {R7['caveat']}.", "",
          "## This model on dev sets (cluster bootstrap 10,000 vs transfer_jaccard; dev = used for selection, confirm = spent, descriptive)",
          "| set | metric | model | transfer | delta [95 % CI] |", "|---|---|---|---|---|",
          *[f"| {r.bench}/{r.split} ({r.status}) | {r.metric} | {r.mean_a:.3f} | {r.mean_b:.3f} | {r.delta:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] |"
            for r in cmp_.itertuples()], "",
          "## Calibration", f"- P = {shrink} x isotonic(score), fitted on {len(xd):,} dev candidate pairs (temporal dev + wide_dev).",
          *[f"- dev cross-fit tau {k}: predicted {v['predicted']:.3f}, realised {v['realised']:.3f} (shrink {v['shrink']:.2f}, n {v['n']})" for k, v in crossfit.items()],
          *[f"- temporal confirm (descriptive) P >= {k}: predicted {v['predicted']:.3f}, realised {v['realised']:.3f} (n {v['n']})" for k, v in conf_cal.items()], "",
          "## Known limits", *[f"- {x}" for x in card["known_limits"]], "",
          "## Files", "model.txt, features.json, config.yaml, model_meta.json, calibration.json, eval_*.parquet, eval_compare.tsv, repro_vs_r7.tsv, scores_*.npy"]
    (out / "MODEL_CARD.md").write_text("\n".join(md) + "\n")
    print(f"[done] {out} ({time.time()-t0:.0f}s)")
    print(rp.to_string(index=False))


if __name__ == "__main__":
    main()
