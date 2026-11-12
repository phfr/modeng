"""modeng-v2 inference: rank candidate disease genes for ONE disease with the saved model (scripts/train_v2.py).

    python -m modeng.predict --disease EFO_0000270 [--seeds genes.txt] [--vintage 2104|2609] --out ranked.tsv
                             [--module module.tsv --tau 0.2] [--no-subex] [--model-dir DIR]

What it does
  1. Builds a single ad-hoc `Task`: seeds = the genes in --seeds (Ensembl ids or HGNC symbols, one per line, '#' comments),
     else the disease's own gene set in the vintage's OT label release. Context = that release's disease x gene matrix with the
     modet sealed ids (and relatives) removed, as in the benchmarks. Masking ("self+desc", as `tasks.attach_families`):
     the disease's own row and its ontology descendants keep only the seed genes (every other gene pair of those rows is
     masked); nothing else is masked. With seeds taken from the release itself this masks nothing (own row = seeds).
  2. Computes the 47 channels with the existing channel code (`channels.compute(..., cache=False)`) under benchmark name
     `temporal_wide`, so every side input is the 21.04-vintage one the model was trained with (PubTator <= 2021-03, HPO 2021-01,
     OT 21.04 evidence, STRING 11.5).
  3. Applies the saved LightGBM model; calibrated P = shrink (0.8) x isotonic(score) (isotonic fitted on dev scores);
     per-gene witness-group contributions from LightGBM pred_contrib (log-odds units, summed over a channel and its rank).
  4. Optional module: seeds + candidates with calibrated P >= tau (TSV + subex artefacts_v1 module npz); if subex is importable,
     its `leiden_anno` decomposition (read-only use of subex, as scripts/decompose_modules.py).

Vintages
  2104 (default) = reproducible research mode: identical inputs to the benchmarks; a temporal / wide disease run with the benchmark
       seeds and mask reproduces the benchmark score (tests/test_predict.py).
  2609 = "current" production mode: seeds and context from OT 26.09; side inputs stay 21.04-vintage (the model was trained on them;
       mixing vintages is a distribution shift: popularity and transfer see a larger context). The 26.09 labels contain every
       arrival up to 2026-09, so NO evaluation of any kind may use this mode; it is for producing candidate lists only.
"""
from __future__ import annotations

import argparse, hashlib, json, sys, uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D, tasks as T, channels as CH
from .stack import pct_rank, task_X

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "modeng_v2.yaml"
BENCH_NAME = "temporal_wide"          # ext channel mappings -> 21.04-vintage side inputs
SUBEX_GRAPH = "t0_string115_phys_notm_400"


# ------------------------------------------------------------------------------------------------------------ config / model
def load_config(path: Path = CONFIG) -> tuple[dict, str]:
    import yaml
    cfg = yaml.safe_load(open(path))
    h = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    return cfg, h


def feature_names(channels: list) -> list:
    return list(channels) + ["r_" + c for c in channels] + ["log_nseeds"]


def default_model_dir(cfg: dict) -> Path:
    return Path(cfg["model_root"]) / f"modeng_v2_{cfg['vintage']}"


class Model:
    """The saved modeng-v2 model: booster, feature names, config, isotonic map, witness groups."""

    def __init__(self, model_dir: Path | None = None, check_config: bool = True):
        import lightgbm as lgb
        cfg, h = load_config()
        self.dir = Path(model_dir) if model_dir else default_model_dir(cfg)
        meta = json.load(open(self.dir / "model_meta.json"))
        if check_config and meta["config_sha256"] != h:
            raise RuntimeError(f"config hash mismatch: {CONFIG} ({h[:12]}) != model ({meta['config_sha256'][:12]}); retrain or "
                               "pass the matching config")
        self.cfg, self.meta = meta["config"], meta
        self.booster = lgb.Booster(model_file=str(self.dir / "model.txt"))
        self.channels = list(self.cfg["channels"])
        self.features = json.load(open(self.dir / "features.json"))
        assert self.features == feature_names(self.channels) == self.booster.feature_name(), "feature list mismatch"
        iso = json.load(open(self.dir / "calibration.json"))
        self.iso_x, self.iso_y, self.shrink = np.asarray(iso["x"]), np.asarray(iso["y"]), float(iso["shrink"])
        grp = {}
        for g, members in self.cfg["witness_groups"].items():
            for c in members:
                grp[c] = g
        missing = [c for c in self.channels + ["log_nseeds"] if c not in grp]
        assert not missing, f"channels without witness group: {missing}"
        self.group_of = [grp[f[2:]] if f.startswith("r_") else grp[f] for f in self.features]
        self.groups = list(self.cfg["witness_groups"])

    def calibrate(self, s: np.ndarray) -> np.ndarray:
        """shrink x isotonic(score) (np.interp == sklearn IsotonicRegression(out_of_bounds='clip').predict)."""
        return self.shrink * np.interp(s, self.iso_x, self.iso_y)


# ------------------------------------------------------------------------------------------------------------ context / task
@lru_cache(None)
def context(vintage: str):
    """(L, ids) of the vintage's OT label release without the modet sealed ids + relatives (as the benchmarks)."""
    if vintage not in ("2104", "2609"):
        raise ValueError(f"vintage {vintage!r}: 2104 (research) or 2609 (current / production)")
    L, ids = D.label_matrix(vintage)
    keep = ~ids.isin(D.excluded_task_ids())
    return L[np.flatnonzero(keep)].tocsr(), ids[keep]


def norm_id(d: str) -> str:
    return d.strip().replace(":", "_")


@lru_cache(None)
def _dmap():
    if str(D.MODET / "src") not in sys.path:
        sys.path.append(str(D.MODET / "src"))           # modet imported read-only
    from modet_data import arrival as A
    return A.disease_map()


def _to26(x: str):
    try:
        return _dmap().to26(x)
    except ValueError:
        return None


def resolve_id(disease_id: str, release: str) -> str:
    """The disease's id in OT `release` (ids change between releases, e.g. asthma EFO_0000270 (21.04) -> MONDO_0004979
    (25.09 / 26.09)); via modet's 26.09 obsolete-term map. Returns the input id if no unique counterpart exists."""
    d = norm_id(disease_id)
    _, ids = D.label_matrix(release)
    if d in ids:
        return d
    d26 = _to26(d) or d
    if d26 in ids:
        return d26
    back = [x for x in ids if x.split("_")[0] in ("EFO", "MONDO", "Orphanet", "HP", "DOID", "OTAR") and _to26(x) == d26]
    return back[0] if len(back) == 1 else d


def read_genes(path) -> np.ndarray:
    """Ensembl ids or HGNC symbols (one per line / whitespace separated; '#' comments) -> universe indices."""
    toks = [x for line in open(path) for x in line.split("#")[0].split()]
    return genes_to_idx(toks)


def genes_to_idx(toks) -> np.ndarray:
    U = D.universe()
    sym = D.symbols()
    by_sym = {str(s).upper(): g for g, s in sym.items() if isinstance(s, str)}
    out, bad = [], []
    for x in toks:
        g = x if x.startswith("ENSG") else by_sym.get(x.upper())
        if g is None or g not in U:
            bad.append(x)
        else:
            out.append(g)
    if bad:
        print(f"[predict] {len(bad)} genes not in the 19,255-gene universe, ignored: {bad[:10]}", file=sys.stderr)
    return D.gene_idx(out)


def seeds_from_release(disease_id: str, vintage: str) -> np.ndarray:
    L, ids = D.label_matrix(vintage)
    i = ids.get_indexer([disease_id])[0]
    if i < 0:
        raise KeyError(f"{disease_id} has no gene set in OT {vintage}; pass --seeds")
    return np.sort(L[i].indices.astype(np.int64))


def make_task(disease_id: str, seeds: np.ndarray, vintage: str = "2104", mask_genes: np.ndarray | None = None,
              name: str = "") -> T.Task:
    """Ad-hoc Task. mask_genes = genes whose pairs in the disease's own + descendant context rows are masked
    (default: every non-seed gene, i.e. those rows are reduced to the seeds). Benchmarks mask exactly their targets;
    passing a benchmark task's targets reproduces its masking."""
    L, ids = context(vintage)
    n = L.shape[1]
    seeds = np.unique(np.asarray(seeds, np.int64))
    if len(seeds) == 0:
        raise ValueError("no seeds")
    mg = np.setdiff1d(np.arange(n), seeds) if mask_genes is None else np.setdiff1d(np.asarray(mask_genes, np.int64), seeds)
    t = T.Task(disease_id, name, "adhoc", seeds, mg, seeds.copy(), cluster=disease_id)
    T.attach_families([t], L, ids, "self+desc")
    return t


def adhoc_bench(task: T.Task, vintage: str = "2104") -> T.Benchmark:
    L, ids = context(vintage)
    net = T.NETWORK["temporal"]
    B = T.Benchmark(BENCH_NAME, [task], L, ids, net, T.UNION["temporal"], f"adhoc vintage {vintage}")
    B.degree = D.degree(net)
    return B


def features(B: T.Benchmark, channels: list) -> np.ndarray:
    """[n_genes, n_features] for the single task of B (channels computed fresh, no disk cache)."""
    assert len(B.tasks) == 1
    tag = f"adhoc_{uuid.uuid4().hex}"          # unique: ext modules memoise in-process by (name, split_tag)
    mats = [CH.compute(B, c, tag, cache=False) for c in channels]
    ranks = [pct_rank(m) for m in mats]
    ns = np.log1p(np.array([len(B.tasks[0].seeds)], np.float32))
    return task_X(0, np.arange(B.n_genes), mats, ranks, ns)


# ------------------------------------------------------------------------------------------------------------ prediction
def predict(model: Model, disease_id: str, seeds=None, vintage: str = "2104", mask_genes=None, allow_sealed: bool = False,
            contrib: bool = True) -> dict:
    """Returns dict(task, scores [n_genes], calib_p, contrib DataFrame [n_genes x groups] or None, table (ranked candidates))."""
    disease_id = resolve_id(disease_id, vintage)
    if disease_id in D.excluded_task_ids() and not allow_sealed:
        raise PermissionError(f"{disease_id} is a modet sealed-fold disease or relative: not usable here (--allow-sealed only "
                              "for production use, never for development / evaluation)")
    if seeds is None:
        seeds = seeds_from_release(disease_id, vintage)
    names = D.disease_names("2609")
    t = make_task(disease_id, seeds, vintage, mask_genes, name=str(names.get(disease_id, "")))
    B = adhoc_bench(t, vintage)
    X = features(B, model.channels)
    s = model.booster.predict(X)
    p = model.calibrate(s)
    U = D.universe()
    sym = D.symbols().reindex(U).fillna("").to_numpy()
    cand = np.setdiff1d(np.arange(B.n_genes), t.seeds)
    order = cand[np.argsort(-s[cand], kind="stable")]
    tab = pd.DataFrame({"rank": np.arange(1, len(order) + 1), "gene_id": U[order], "symbol": sym[order],
                        "score": s[order], "calib_p": np.round(p[order], 5), "on_network": B.degree[order] > 0})
    C = None
    if contrib:
        raw = model.booster.predict(X, pred_contrib=True)[:, :-1]           # last column = bias
        G = pd.DataFrame(raw, columns=model.features).T.groupby(model.group_of, sort=False).sum().T
        C = G.reindex(columns=model.groups)
        Co = C.to_numpy()[order]
        top = []
        for row in Co:
            k = np.argsort(-row)[:3]
            top.append(";".join(f"{model.groups[j]}:{row[j]:+.2f}" for j in k if row[j] > 0))
        tab["top_witnesses"] = top
        for j, g in enumerate(model.groups):
            tab[f"c_{g}"] = np.round(Co[:, j], 4)
    return dict(task=t, bench=B, scores=s, calib_p=p, contrib=C, table=tab, symbols=sym)


def module_table(res: dict, tau: float) -> pd.DataFrame:
    t, s, p, sym = res["task"], res["scores"], res["calib_p"], res["symbols"]
    U = D.universe()
    cand = np.setdiff1d(np.arange(len(s)), t.seeds)
    add = cand[p[cand] >= tau]
    add = add[np.argsort(-s[add], kind="stable")]
    genes = np.concatenate([t.seeds, add])
    return pd.DataFrame({"gene_id": U[genes], "symbol": sym[genes], "role": ["seed"] * len(t.seeds) + ["added"] * len(add),
                         "score": s[genes], "calib_p": np.r_[np.ones(len(t.seeds)), np.round(p[add], 5)],
                         "rank": np.r_[np.zeros(len(t.seeds), int), np.arange(1, len(add) + 1)]})


def write_module_npz(path: Path, mod: pd.DataFrame, info: dict) -> tuple[Path, list, dict]:
    """subex artefacts_v1 kind="module" (schema 1) on graph t0_string115_phys_notm_400 (as scripts/export_modules.py)."""
    g = pd.read_parquet(Path("/home/ph/code/subex/data/networks") / f"{SUBEX_GRAPH}.parquet", columns=["a", "b"])
    nodes = sorted(set(g.a.astype(str)) | set(g.b.astype(str)))
    node_pos = {n: i for i, n in enumerate(nodes)}
    genes = list(mod.gene_id)
    seeds = list(mod.gene_id[mod.role == "seed"])
    prob = dict(zip(mod.gene_id, mod.calib_p))
    on = [x for x in genes if x in node_pos]
    on_seeds = [x for x in seeds if x in node_pos]
    vprob = np.zeros(len(nodes))
    for x in on:
        vprob[node_pos[x]] = prob[x]
    arrays = {"node_ids": np.asarray(nodes, dtype=str), "set__genes": np.asarray(sorted(on), dtype=str),
              "set__seeds": np.asarray(sorted(on_seeds), dtype=str), "set__added": np.asarray(sorted(set(on) - set(on_seeds)), dtype=str),
              "set__connectors": np.asarray([], dtype=str), "set__dropped": np.asarray([], dtype=str),
              "vec__prob": vprob, "vec__z": np.full(len(nodes), np.nan), "vec__pval": np.ones(len(nodes)), "vec__qval": np.ones(len(nodes))}
    off = sorted(set(genes) - set(on))
    meta = dict(kind="module", schema=1, graph=SUBEX_GRAPH, n_nodes=len(nodes), info=dict(info, off_graph=off, n_off_graph=len(off)),
                provenance=dict(tool="modeng.predict", model_id="modeng-v2"))
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, __meta__=np.array(json.dumps(meta)), **arrays)
    tmp.replace(path)
    return path, on, node_pos


def decompose(module_npz: Path) -> Path | None:
    """subex leiden_anno submodules (subex imported read-only; same path as `subex decompose --method leiden_anno --overlap none`)."""
    try:
        from subex.cli import graph, submodule_table
        from subex.annot import load_library
        from subex.io import load_module, save_submodules
        from subex.stages import decompose_stage
    except Exception as e:                                 # subex not importable: skip, say so
        print(f"[predict] subex not importable ({type(e).__name__}: {e}); no submodules", file=sys.stderr)
        return None
    G = graph(SUBEX_GRAPH)
    mr = load_module(module_npz, G)
    if len(mr.genes) < 3:
        print("[predict] module has < 3 graph genes; no submodules", file=sys.stderr)
        return None
    er, ov = decompose_stage(G, mr, load_library(G), overlap=None, module_kind="connected", method="leiden_anno")
    out = module_npz.with_name(module_npz.name.replace(".module.npz", ".sub.npz"))
    save_submodules(out, er, G, provenance=dict(module=str(module_npz), overlap="none", method="leiden_anno", tool="modeng.predict"),
                    overlap=ov)
    submodule_table(G, er, ov).to_csv(out.with_suffix(".tsv"), sep="\t", index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m modeng.predict", description=__doc__.split("\n\n")[0])
    ap.add_argument("--disease", required=True, help="EFO / MONDO / ... id (EFO_0000270 or EFO:0000270)")
    ap.add_argument("--seeds", help="seed genes file (Ensembl ids or symbols); default = the disease's OT gene set of --vintage")
    ap.add_argument("--vintage", default="2104", choices=("2104", "2609"),
                    help="2104 = reproducible research mode (default); 2609 = current OT 26.09 seeds / context (production only, never evaluation)")
    ap.add_argument("--out", required=True, help="ranked candidates TSV (seeds not listed); sidecar <out>.json")
    ap.add_argument("--module", help="module TSV (seeds + candidates with calibrated P >= tau); also <stem>.module.npz and subex submodules")
    ap.add_argument("--tau", type=float, default=None, help="module threshold on calibrated P (default from config: 0.2)")
    ap.add_argument("--no-subex", action="store_true", help="skip the subex leiden_anno decomposition")
    ap.add_argument("--no-contrib", action="store_true", help="skip pred_contrib explanations (faster)")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--allow-sealed", action="store_true", help="permit modet sealed-fold diseases (production use only)")
    a = ap.parse_args(argv)
    M = Model(a.model_dir)
    seeds = read_genes(a.seeds) if a.seeds else None
    res = predict(M, a.disease, seeds, a.vintage, allow_sealed=a.allow_sealed, contrib=not a.no_contrib)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(out, sep="\t", index=False, float_format="%.6g")
    t = res["task"]
    U = D.universe()
    side = dict(model="modeng-v2", model_dir=str(M.dir), config_sha256=M.meta["config_sha256"], disease_id=t.disease_id, name=t.name,
                vintage=a.vintage, seeds_source=a.seeds or f"OT {a.vintage} gene set", n_seeds=len(t.seeds), seeds=list(U[t.seeds]),
                n_masked_pairs=int(len(t.mask_r)), fam_rows=int(len(t.fam_rows)),
                leakage=("research mode: 21.04 inputs" if a.vintage == "2104" else
                         "PRODUCTION mode: OT 26.09 seeds/context; outputs must not be used to evaluate anything"),
                calibration=f"{M.shrink} x isotonic(score), fitted on dev; P = probability of becoming an OT genetic disease gene "
                            "within ~4 years (21.04 -> 25.09 window)")
    if a.module:
        tau = a.tau if a.tau is not None else float(M.cfg["calibration"]["module_tau_default"])
        mod = module_table(res, tau)
        mp = Path(a.module)
        mp.parent.mkdir(parents=True, exist_ok=True)
        mod.to_csv(mp, sep="\t", index=False, float_format="%.6g")
        side.update(module=str(mp), tau=tau, module_n_added=int((mod.role == "added").sum()))
        if not a.no_subex:
            npz = mp.with_name(mp.stem + ".module.npz")
            write_module_npz(npz, mod, dict(task_id=t.disease_id, name=t.name, tau=tau, method=f"modeng-v2 calibrated P >= {tau}"))
            sub = decompose(npz)
            side.update(module_npz=str(npz), submodules=str(sub) if sub else None)
    Path(str(out) + ".json").write_text(json.dumps(side, indent=1))
    top = res["table"].head(10)
    print(f"{t.disease_id} {t.name}: {len(t.seeds)} seeds, vintage {a.vintage}; top 10:")
    print(top[["rank", "symbol", "score", "calib_p"] + (["top_witnesses"] if "top_witnesses" in top else [])].to_string(index=False))


if __name__ == "__main__":
    main()
