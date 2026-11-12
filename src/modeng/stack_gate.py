"""Disease-type-gated stacker (r3 strict-a; ideas E-B2(c) task-context features and E-B3 arrival-class head, cheapest versions).

Variant of `modeng.stack.run` (stack.py behaviour unchanged; this module reuses its features, folds, LGB params and rows):
  (a) task-context features appended to every (task, gene) row, so trees can gate channel groups by disease type:
      ctx_mix_gwas      share of seeds that are GWAS winners (OT 21.04 ot_genetics_portal L2G >= 0.5) in the task family
      ctx_mix_mend      share of seeds with ClinVar(eva >= 0.5) / curated evidence in the task family (OT 21.04)
      ctx_mix_gwasonly  share of seeds with GWAS-winner but no Mendelian-type evidence
      ctx_fam_gwas_n    log1p(# genes with any ot_genetics_portal evidence in the task family, 21.04)
      ctx_fam_mend_n    log1p(# genes with Mendelian-type evidence in the task family, 21.04)
      ctx_cat_has       the disease (ontology-weighted) has own GWAS-Catalog hits added <= seed release (gwas_catlag row not NaN)
      ctx_cat_n         log1p(# genes with gwas_catlag > 0)
      ctx_hpo_share     share of seeds with an HPO 2021-01 gene annotation
      ctx_seed_logdeg   mean log1p(STRING degree) of the seeds
      (n_seeds is already in the base feature set as log_nseeds)
      optional (--area): 4 binary flags for modet's primary therapeutic area (nervous system, cardiovascular, genetic/congenital, other)
      optional (--ctx-perm): context rows permuted across tasks (same-width negative control)
      Leakage class: temporal-clean (seed-release evidence, HPO 2021-01, catalog <= 2021-04-30, seeds only).
  (b) arrival-class head: a second LightGBM on the same features whose positives are only the class-A targets of the training
      tasks (modet `arrival_devall_t0_string115_phys_notm_400`, arrival == 'A' = genetic arrivals first seen 2022+); other
      targets are dropped (not negatives). Trained in the same disease-grouped folds (dev OOF) / on all dev (confirm); only dev
      labels are used. Blend: final = (1 - a) * log p_main + a * log p_head; `a` chosen on dev OOF only (scripts/r3_strict_a.py).
  (b') --a-weight f: class-A positives of the dev training tasks get f x weight in the main model (arrival-class-aware
      weighting instead of a separate head).
Usage: python -m modeng.stack_gate --channels <v1 list> --tag gate_ctx --pos-deg-power 0.35 [--area] [--ctx-perm] [--head]
Scores always saved (results/temporal/<split>/stack_<tag>.scores.npy, and ..._headA.scores.npy for the head).
"""
from __future__ import annotations

import argparse, os, time
import numpy as np, pandas as pd

from . import tasks as T, bench as BM, channels as CH, data as D
from . import stack as SK

ARRIVAL = D.MODET / "data/processed/groundtruth/arrival_devall_t0_string115_phys_notm_400.parquet"
PANEL = D.MODET / "data/processed/groundtruth/panel_allgene_t0_string115_phys_notm_400.parquet"
AREAS = ("EFO_0000618", "EFO_0000319", "OTAR_0000018")
CTX_NAMES = ["ctx_mix_gwas", "ctx_mix_mend", "ctx_mix_gwasonly", "ctx_fam_gwas_n", "ctx_fam_mend_n", "ctx_cat_has", "ctx_cat_n",
             "ctx_hpo_share", "ctx_seed_logdeg"]


def context(B, split_tag, area=False):
    from .ext import priors_h as PH
    mats, dids = PH._ev_mats(PH.REL[B.name])
    W, Mn, Gs = mats["gwas_win"], mats["mend"], mats["gwas_score"]
    hpo = np.asarray(PH._hpo(B.name).sum(1)).ravel() > 0
    cat = CH.compute(B, "gwas_catlag", split_tag)
    deg = np.asarray(B.degree, np.float32)
    rows = []
    for i, t in enumerate(B.tasks):
        F = PH._family_rows(t, dids)
        s = t.seeds
        if len(F):
            gw = np.asarray(W[F].sum(0)).ravel() > 0; me = np.asarray(Mn[F].sum(0)).ravel() > 0
            ga = np.asarray((Gs[F] > 0).sum(0)).ravel() > 0
        else:
            gw = me = ga = np.zeros(B.n_genes, bool)
        c = cat[i]; has = bool(np.isfinite(c).any())
        r = [gw[s].mean(), me[s].mean(), (gw & ~me)[s].mean(), np.log1p(ga.sum()), np.log1p(me.sum()), float(has),
             np.log1p((np.nan_to_num(c) > 0).sum()), hpo[s].mean(), np.log1p(deg[s]).mean()]
        rows.append(r)
    X = np.asarray(rows, np.float32)
    names = list(CTX_NAMES)
    if area:
        p = pd.read_parquet(PANEL).set_index("disease_id")["area"]
        a = np.array([p.get(t.disease_id, "") for t in B.tasks])
        A = np.stack([(a == x) for x in AREAS] + [~np.isin(a, AREAS)], 1).astype(np.float32)
        X = np.hstack([X, A]); names += [f"ctx_area_{x}" for x in AREAS] + ["ctx_area_other"]
    return X, names


def class_a_targets(B):
    arr = pd.read_parquet(ARRIVAL, columns=["disease_id", "gene", "arrival"])
    a = arr[arr.arrival == "A"].groupby("disease_id")["gene"].apply(set).to_dict()
    out = []
    for t in B.tasks:
        g = D.gene_idx(a.get(t.disease_id, set()))
        out.append(np.intersect1d(g, t.targets))
    return out


class Feats:
    """build_feats + task-context columns (task_X appends ctx[i])."""
    def __init__(self, B, split_tag, names, area=False, ctx=True, perm=False):
        self.m, self.r, self.ns = SK.build_feats(B, split_tag, names)
        if ctx:
            self.ctx, self.ctx_names = context(B, split_tag, area)
            if perm:
                self.ctx = self.ctx[np.random.default_rng(1).permutation(len(self.ctx))]
        else:
            self.ctx, self.ctx_names = np.zeros((len(B.tasks), 0), np.float32), []

    def X(self, i, genes):
        base = SK.task_X(i, genes, self.m, self.r, self.ns)
        if self.ctx.shape[1] == 0:
            return base
        return np.hstack([base, np.repeat(self.ctx[i][None, :], len(genes), 0)])


def rows(B, idx, F, neg=3000, seed=0, pos_override=None, upweight=None, factor=1.0):
    """as stack.training_rows; pos_override[i] replaces the positives (other targets are excluded, not negatives);
    upweight[i] = positives whose weight is multiplied by `factor` (after the deg^-power normalisation)."""
    rng = np.random.default_rng(seed)
    Xs, ys, ws = [], [], []
    deg = np.asarray(B.degree, np.float32)
    for i in idx:
        t = B.tasks[i]
        tg = t.targets if pos_override is None else pos_override[i]
        if len(tg) == 0:
            continue
        excl = np.zeros(B.n_genes, bool); excl[t.exclude] = True; excl[t.targets] = True
        cand = np.flatnonzero(~excl)
        negs = rng.choice(cand, min(neg, len(cand)), replace=False)
        g = np.concatenate([tg, negs])
        Xs.append(F.X(i, g)); ys.append(np.r_[np.ones(len(tg)), np.zeros(len(negs))])
        wp = 1.0 / np.maximum(deg[tg], 1.0) ** SK.POS_DEG_POWER
        wp = wp / wp.mean()
        if upweight is not None and factor != 1.0:
            wp = np.where(np.isin(tg, upweight[i]), wp * factor, wp)
        ws.append(np.r_[wp, np.ones(len(negs))])
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(ws)


def predict(model, B, idx, F, out):
    allg = np.arange(B.n_genes)
    for i in idx:
        out[i] = model.predict(F.X(i, allg))


def run(channels, tag, rounds=400, neg=3000, folds=5, area=False, ctx=True, perm=False, head=False, a_weight=1.0):
    os.environ["MODENG_SAVE_SCORES"] = "1"
    t0 = time.time()
    names = CH.all_channel_names(tuple(channels.split(",")))
    method = f"stack_{tag}"
    Bfull = T.load("temporal")
    Bd, Bc = Bfull.split("dev"), Bfull.split("confirm")
    Fd = Feats(Bd, "dev", names, area, ctx, perm)
    print(f"[gate] {len(names)} channels + {len(Fd.ctx_names)} ctx ({time.time()-t0:.0f}s)", flush=True)
    clusters = np.array([t.cluster or t.disease_id for t in Bd.tasks])
    uc = np.unique(clusters); rng = np.random.default_rng(0); rng.shuffle(uc)
    fold_of = {c: k % folds for k, c in enumerate(uc)}
    fid = np.array([fold_of[c] for c in clusters])
    posA = class_a_targets(Bd) if (head or a_weight != 1.0) else None
    uw = dict(upweight=posA, factor=a_weight)
    S = np.zeros((len(Bd.tasks), Bd.n_genes), np.float32); H = np.zeros_like(S)
    for k in range(folds):
        tr, te = np.flatnonzero(fid != k), np.flatnonzero(fid == k)
        m = SK.fit(*rows(Bd, tr, Fd, neg, **uw), rounds); predict(m, Bd, te, Fd, S)
        if head:
            mh = SK.fit(*rows(Bd, tr, Fd, neg, pos_override=posA), rounds); predict(mh, Bd, te, Fd, H)
        print(f"  fold {k} ({time.time()-t0:.0f}s)", flush=True)
    SK._record(S, Bd, "temporal", "dev", method, t0, True)
    if head:
        SK._record(H, Bd, "temporal", "dev", method + "_headA", t0, False)
    alli = np.arange(len(Bd.tasks))
    m = SK.fit(*rows(Bd, alli, Fd, neg, **uw), rounds)
    Fc = Feats(Bc, "confirm", names, area, ctx, perm)
    Sc = np.zeros((len(Bc.tasks), Bc.n_genes), np.float32); predict(m, Bc, np.arange(len(Bc.tasks)), Fc, Sc)
    SK._record(Sc, Bc, "temporal", "confirm", method, t0, True)
    if head:
        mh = SK.fit(*rows(Bd, alli, Fd, neg, pos_override=posA), rounds)
        Hc = np.zeros_like(Sc); predict(mh, Bc, np.arange(len(Bc.tasks)), Fc, Hc)
        SK._record(Hc, Bc, "temporal", "confirm", method + "_headA", t0, False)
    cols = names + ["r_" + n for n in names] + ["log_nseeds"] + Fd.ctx_names
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    imp.to_csv(BM.RESULTS / f"importance_{method}.tsv", sep="\t")
    return imp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--pos-deg-power", type=float, default=0.35)
    ap.add_argument("--area", action="store_true"); ap.add_argument("--no-ctx", action="store_true")
    ap.add_argument("--ctx-perm", action="store_true"); ap.add_argument("--head", action="store_true")
    ap.add_argument("--a-weight", type=float, default=1.0, help="multiply training weight of class-A (dev-label) positives")
    a = ap.parse_args()
    SK.POS_DEG_POWER = a.pos_deg_power
    imp = run(a.channels, a.tag, area=a.area, ctx=not a.no_ctx, perm=a.ctx_perm, head=a.head, a_weight=a.a_weight)
    print(imp.head(30).to_string())
