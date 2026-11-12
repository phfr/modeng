"""Export modeng-v1 disease modules for subex (r4).

Per temporal *confirm* disease (138), into data/modules/<method>/<disease_id>/ (gitignored):
  scores.tsv + scores.json   subex score artefact, TSV alternative of artefacts_v1 (gene_id, score, calib_p, is_seed; sidecar task_id, namespace)
  module_tau{T}.tsv          module gene list at calibrated tau T (seeds + candidates with P >= T): gene_id, symbol, role, score, calib_p, rank
  module_tau{T}.npz          the same set as an artefacts_v1 `kind = "module"` file on subex's graph t0_string115_phys_notm_400
                             (genes not on that graph cannot be decomposed; listed in __meta__.info.off_graph), input to `subex decompose`

Calibration (as scripts/module_cutoff.py, refit for modeng-v1): isotonic map score -> P(future disease gene) fitted on the dev split
scores of stack_v3ptr_p0.35 (dev scores are out-of-fold), times a dev-only shrink of 0.8 (FINAL report sections 4b, 9).
module(d, tau) = seeds + candidate genes (not excluded) with 0.8 * iso(score) >= tau.

Comparators at the same sizes (same number of added genes per disease and tau): transfer_jaccard and modet's headline (modet-U-cov-Lb),
top-N candidates by their own score. Only scores / npz are written for comparators (no score TSV).

Leakage: seeds, labels, graph (STRING 11.5 T0) and GO / Reactome (T0, used inside subex) are <= 2021-04; the modeng-v1 score uses the
inputs of stack_v3ptr_p0.35 (temporal-clean channels as in the FINAL report); targets (OT 25.09 additions) are used only in evaluation.
Usage: python scripts/export_modules.py
"""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "/home/ph/code/modeng/src")
from sklearn.isotonic import IsotonicRegression
from modeng import tasks as T, bench as BM, data as D
from modeng.methods import baselines as BL

METHOD = "stack_v3ptr_p0.35"
SHRINK = 0.8
TAUS = (0.2, 0.3)
GRAPH = "t0_string115_phys_notm_400"
SUBEX_GRAPH = Path("/home/ph/code/subex/data/networks") / f"{GRAPH}.parquet"
OUT = Path("/home/ph/code/modeng/data/modules")


def cand_mask(B, t):
    ok = np.ones(B.n_genes, bool); ok[t.exclude] = False
    return ok


def fit_iso():
    Bd = T.load("temporal").split("dev")
    S = np.load(BM.RESULTS / "temporal" / "dev" / f"{METHOD}.scores.npy").astype(np.float32)
    xs, ys = [], []
    for i, t in enumerate(Bd.tasks):
        ok = cand_mask(Bd, t); y = np.zeros(Bd.n_genes, bool); y[t.targets] = True
        xs.append(S[i, ok]); ys.append(y[ok])
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(np.concatenate(xs), np.concatenate(ys))


def write_module_npz(path, genes, seeds, prob, nodes, node_pos, info):
    """artefacts_v1 kind = "module" (schema 1), restricted to graph nodes; written atomically."""
    on = [g for g in genes if g in node_pos]
    on_seeds = [g for g in seeds if g in node_pos]
    added = sorted(set(on) - set(on_seeds))
    vprob = np.zeros(len(nodes));
    for g in on:
        vprob[node_pos[g]] = prob[g]
    arrays = {"node_ids": np.asarray(nodes, dtype=str), "set__genes": np.asarray(sorted(on), dtype=str),
              "set__seeds": np.asarray(sorted(on_seeds), dtype=str), "set__added": np.asarray(added, dtype=str),
              "set__connectors": np.asarray([], dtype=str), "set__dropped": np.asarray([], dtype=str),
              "vec__prob": vprob, "vec__z": np.full(len(nodes), np.nan), "vec__pval": np.ones(len(nodes)), "vec__qval": np.ones(len(nodes))}
    info = dict(info, off_graph=sorted(set(genes) - set(on)), n_off_graph=len(set(genes) - set(on)))
    meta = dict(kind="module", schema=1, graph=GRAPH, n_nodes=len(nodes), info=info,
                provenance=dict(tool="modeng scripts/export_modules.py", model_id="modeng-v1", recipe=METHOD))
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, __meta__=np.array(json.dumps(meta)), **arrays)
    tmp.replace(path)


def main():
    B = T.load("temporal").split("confirm")
    U = D.universe(); sym = D.symbols().reindex(U).fillna("").to_numpy()
    iso = fit_iso()
    S = {"modeng": np.load(BM.RESULTS / "temporal" / "confirm" / f"{METHOD}.scores.npy").astype(np.float32),
         "transfer": BL.transfer_jaccard(B.tasks, B), "headline": BL.modet_headline(B.tasks, B)}
    g = pd.read_parquet(SUBEX_GRAPH, columns=["a", "b"])
    nodes = sorted(set(g.a.astype(str)) | set(g.b.astype(str))); node_pos = {n: i for i, n in enumerate(nodes)}
    rows = []
    for i, t in enumerate(B.tasks):
        ok = cand_mask(B, t); cand = np.flatnonzero(ok)
        s = S["modeng"][i]; p = np.zeros(B.n_genes); p[cand] = SHRINK * iso.predict(s[cand]); p[t.seeds] = 1.0
        d = OUT / "modeng" / t.disease_id; d.mkdir(parents=True, exist_ok=True)
        is_seed = np.zeros(B.n_genes, bool); is_seed[t.seeds] = True
        pd.DataFrame({"gene_id": U, "score": s, "calib_p": np.round(p, 5), "is_seed": is_seed.astype(int)}).to_csv(d / "scores.tsv", sep="\t", index=False)
        (d / "scores.json").write_text(json.dumps(dict(task_id=t.disease_id, namespace="ensembl_gene", recipe=f"modeng-v1 ({METHOD})", scale="raw",
                                                       name=t.name, calibration=f"isotonic on dev OOF x {SHRINK} (component calib_p)")))
        for tau in TAUS:
            add = cand[p[cand] >= tau]; n_add = len(add)
            for m in ("modeng", "transfer", "headline"):
                sm = S[m][i]
                if np.isnan(sm).all():
                    continue
                if m == "modeng":
                    sel = add[np.argsort(-sm[add], kind="stable")]
                else:
                    sv = np.where(np.isnan(sm[cand]), -np.inf, sm[cand])
                    sel = cand[np.argsort(-sv, kind="stable")[:n_add]]
                dm = OUT / m / t.disease_id; dm.mkdir(parents=True, exist_ok=True)
                genes = np.concatenate([t.seeds, sel])
                mod = pd.DataFrame({"gene_id": U[genes], "symbol": sym[genes], "role": ["seed"] * len(t.seeds) + ["added"] * len(sel),
                                    "score": sm[genes], "calib_p": p[genes] if m == "modeng" else np.nan,
                                    "rank": [0] * len(t.seeds) + list(range(1, len(sel) + 1)), "on_graph": [U[x] in node_pos for x in genes]})
                mod.to_csv(dm / f"module_tau{tau}.tsv", sep="\t", index=False)
                prob = {U[x]: (1.0 if is_seed[x] else float(p[x])) for x in genes}
                write_module_npz(dm / f"module_tau{tau}.npz", list(U[genes]), list(U[t.seeds]), prob, nodes, node_pos,
                                 dict(method=f"{m} top-N" if m != "modeng" else f"modeng-v1 calibrated P >= {tau} (shrink {SHRINK})",
                                      task_id=t.disease_id, name=t.name, tau=tau, n_added=n_add))
                rows.append(dict(method=m, disease_id=t.disease_id, name=t.name, tau=tau, n_seeds=len(t.seeds), n_added=n_add,
                                 n_added_on_graph=int(mod[(mod.role == "added")].on_graph.sum()), n_targets=len(t.targets),
                                 hits=int(np.isin(sel, t.targets).sum())))
    df = pd.DataFrame(rows); df.to_csv(OUT / "modules_index.tsv", sep="\t", index=False)
    print(df.groupby(["method", "tau"])[["n_added", "n_added_on_graph", "hits"]].agg(["median", "sum"]))


if __name__ == "__main__":
    main()
