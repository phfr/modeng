"""Gene kernels that lift transfer to zero-support genes (implementer B, round 1: EV-2 / EV-8-proxy / ML-1 / view lifts).

Channels (all exact w.r.t. the task masks, see below):
  lift_<view>        score(g) = sum_h K_v[g, h] * T[h], K_v = row-normalised cosine kNN graph (k=20, self excluded) in gene view v
                     (data/views/<v>.npz); T = transfer_jaccard (masked exactly by baselines._transfer). Genes outside the view: NaN.
  lift_paralog       same with K = Ensembl Compara 103 human within-species paralogs (weight = mean % identity / 100),
                     pairs < 1 Mb apart on the same chromosome dropped (tandem arrays = locus prior; critic_bio).
  lift_paralog_all   paralog kernel without the locus filter.
  lift_afdb          K = AFDB/Foldseek structural-cluster co-membership (Barrio-Hernandez 2023, v3 file 3; GO column dropped),
                     same-locus pairs dropped. Proxy for EV-8 (shared fold/domain family).
  coldstart_profile  ML-1 (ridge form): gene profile factors from a truncated SVD (rank 128) of a SAFE context = context minus the
                     union of the masked pairs of ALL tasks of the benchmark; ridge from all 15 views -> gene factor, 5-fold
                     cross-fitted over genes (a gene's imputed profile never uses its own column); score = transfer weights
                     (top-50 Jaccard, as transfer_jaccard) applied to the imputed matrix U S Vhat^T.
  coldstart_profile_t0  same with only the seq / pre21 / annot views (esmc, prott5, esm2_15b, depmap20q4, phylo_npp, go_t0, reactome_t0).
  *_perm (negative controls): lift_perm_<view> / lift_perm_paralog: the kernel with gene identities permuted (K -> P K P^T).

Masking: a lift is a fixed, label-free linear map applied to the exactly masked transfer score T, so lifted scores are as
masked as T. coldstart uses only the safe context (subset of every task's L_t) plus label-free content.
Leakage class: seq / annot (GO/Reactome T0) / pre21 views, Compara 103 (2021-02): temporal-clean (AFDB clusters and ESM-2/ProtT5
are post-T0 label-free tools). Modern views (hpa_*, gtex, ps_*): transductive on temporal, fair on forward/holdout as
label-free cell data (Perturb-seq CD4 2025 is post-25.09 -> transductive on forward too).
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .. import channels as CH
from .. import data as D

ROOT = Path(__file__).resolve().parents[3]
KC = ROOT / "data/cache/kernels_b"
KNN = 20
T0_VIEWS = ("esmc", "prott5", "esm2_15b", "depmap20q4", "phylo_npp", "go_t0", "reactome_t0")


# ------------------------------------------------------------------------------------------------ kernels
def _rownorm(K):
    K = sp.csr_matrix(K, dtype=np.float32)
    K.setdiag(0); K.eliminate_zeros()
    s = np.asarray(K.sum(1)).ravel()
    return (sp.diags(1.0 / np.where(s > 0, s, 1)) @ K).tocsr().astype(np.float32)


def _view_knn(view, k=KNN, chunk=2048):
    X, m = CH.load_view(view)
    idx = np.flatnonzero(m)
    Xn = X[idx].astype(np.float32)
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-6
    rows, cols, vals = [], [], []
    for s in range(0, len(idx), chunk):
        C = Xn[s:s + chunk] @ Xn.T
        C[np.arange(C.shape[0]), np.arange(s, s + C.shape[0])] = -np.inf
        top = np.argpartition(-C, k, axis=1)[:, :k]
        rows.append(np.repeat(idx[s:s + C.shape[0]], k)); cols.append(idx[top].ravel())
        vals.append(np.maximum(np.take_along_axis(C, top, 1).ravel(), 0))
    n = X.shape[0]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def _gene_loc():
    z = pd.read_csv("/mnt/msadb/modet/downloads/gnomad_v2.1.1/gnomad.v2.1.1.lof_metrics.by_gene.txt.bgz", sep="\t",
                    compression="gzip", usecols=["gene_id", "chromosome", "start_position"]).drop_duplicates("gene_id")
    u = D.universe()
    z = z[z.gene_id.isin(u)]
    chrom = pd.Series("", index=u, dtype=object); pos = pd.Series(np.nan, index=u)
    chrom[z.gene_id.values] = z.chromosome.astype(str).values; pos[z.gene_id.values] = z.start_position.values
    return chrom.to_numpy(), pos.to_numpy()


def _drop_local(a, b, w, dist=1_000_000):
    chrom, pos = _gene_loc()
    local = (chrom[a] == chrom[b]) & (chrom[a] != "") & (np.abs(pos[a] - pos[b]) < dist)
    return a[~local], b[~local], w[~local], int(local.sum())


COMPARA = Path("/mnt/msadb/modet/downloads/ensembl103_compara/Compara.103.protein_default.homologies.tsv.gz")


def _paralog_wide_tsv(min_id=30.0):
    """within_species_paralog + other_paralog (older duplications), human-human, mean identity >= min_id."""
    f = KC / "hs_paralogs_wide.tsv"
    if not f.exists():
        rows = []
        with gzip.open(COMPARA, "rt") as fh:
            fh.readline()
            for l in fh:
                p = l.split("\t")
                if p[2] == "homo_sapiens" and p[7] == "homo_sapiens" and p[4] in ("within_species_paralog", "other_paralog") \
                        and (float(p[3]) + float(p[8])) / 2 >= min_id:
                    rows.append("\t".join((p[0], p[5], p[3], p[8], p[4])))
        f.write_text("\n".join(rows) + "\n")
    return f


def _paralog_pairs(wide=False):
    src = _paralog_wide_tsv() if wide else KC / "hs_paralogs.tsv"   # hs_paralogs.tsv = within_species_paralog only (13,319 rows)
    df = pd.read_csv(src, sep="\t", header=None, names=["a", "b", "ida", "idb", "ds"])
    u = D.universe()
    a, b = u.get_indexer(df.a), u.get_indexer(df.b)
    ok = (a >= 0) & (b >= 0) & (a != b)
    w = ((df.ida + df.idb) / 200.0).to_numpy()[ok]
    return a[ok], b[ok], w


def _afdb_pairs(max_size=200):
    h = pd.read_parquet("/home/ph/code/modet/data/processed/hgnc_protein_coding.parquet", columns=["ensembl_gene_id", "uniprot_ids"])
    acc2g = {}
    for g, ids in zip(h.ensembl_gene_id, h.uniprot_ids):
        if isinstance(ids, str):
            for a in ids.replace(",", "|").replace(";", "|").split("|"):
                if a.strip():
                    acc2g[a.strip()] = g
    f = "/mnt/msadb/subex/raw/afdb_clusters_v3/3-sapId_sapGO_repId_cluFlag_LCAtaxId.tsv.gz"
    df = pd.read_csv(f, sep="\t", header=None, usecols=[0, 2], names=["acc", "rep"])   # column 2 (2022 GO) never loaded
    df["g"] = df.acc.map(acc2g)
    u = D.universe()
    df["gi"] = u.get_indexer(df.g.fillna("x"))
    df = df[df.gi >= 0].drop_duplicates(["rep", "gi"])
    a, b = [], []
    for _, grp in df.groupby("rep"):
        g = grp.gi.to_numpy()
        if 2 <= len(g) <= max_size:
            aa, bb = np.meshgrid(g, g); sel = aa != bb
            a.append(aa[sel]); b.append(bb[sel])
    a, b = np.concatenate(a), np.concatenate(b)
    return a, b, np.ones(len(a))


def kernel(name):
    """Row-normalised gene x gene kernel (rows = scored gene), cached."""
    KC.mkdir(parents=True, exist_ok=True)
    f = KC / f"K_{name}.npz"
    if f.exists():
        return sp.load_npz(f).tocsr()
    n = len(D.universe())
    if name.startswith("perm_"):
        K = kernel(name[5:])
        p = np.random.default_rng(1).permutation(n)
        K = K[p][:, p]
    elif name in ("paralog", "paralog_all", "paralog_wide", "afdb"):
        a, b, w = _afdb_pairs() if name == "afdb" else _paralog_pairs(wide=(name == "paralog_wide"))
        if name != "paralog_all":
            a, b, w, nl = _drop_local(a, b, w)
            print(f"[kernels_b] {name}: dropped {nl} same-locus pairs", flush=True)
        K = sp.csr_matrix((w, (a, b)), shape=(n, n))
        K = K.maximum(K.T)
    else:
        K = _view_knn(name)
    K = _rownorm(K)
    sp.save_npz(f, K)
    return K


def lift(B, split_tag, kname):
    T = CH.compute(B, "transfer_jaccard", split_tag)          # exact masked transfer
    K = kernel(kname)
    out = np.asarray((K @ T.T).T, np.float32)                  # out[i, g] = sum_h K[g, h] T[i, h]
    out = np.round(out, 6)                                     # quantise (leakage rule 1; T itself is already quantised)
    out[:, np.asarray(K.getnnz(1)) == 0] = np.nan
    return out


# ------------------------------------------------------------------------------------------------ ML-1 cold start
def transfer_weights(tasks, B, k=50):
    """Sparse [n_tasks, n_context_rows] neighbour weights exactly as baselines._transfer (Jaccard on masked sizes, own row out)."""
    from ..methods.baselines import _indicator
    L = B.context
    sizes = L.getnnz(1).astype(np.float32)
    inter = (_indicator([t.seeds for t in tasks], B.n_genes) @ L.T).toarray()
    pos = {d: i for i, d in enumerate(B.context_ids)}
    r, c, v = [], [], []
    for i, t in enumerate(tasks):
        sz = sizes - B.row_masked_counts(t)
        jac = inter[i] / np.maximum(len(t.seeds) + sz - inter[i], 1)
        own = pos.get(t.disease_id)
        if own is not None:
            jac[own] = 0
        cand = np.flatnonzero(jac > 0)
        top = cand[np.argsort(-jac[cand])[:k]]
        r.append(np.full(len(top), i)); c.append(top); v.append(jac[top])
    return sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(len(tasks), L.shape[0]))


def _content(views):
    blocks = []
    for v in views:
        X, m = CH.load_view(v)
        X = X.astype(np.float32).copy()
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-6
        X = (X - mu) / sd / np.sqrt(X.shape[1])               # each view gets equal total variance
        X[~m] = 0
        blocks += [X, m[:, None].astype(np.float32)]
    return np.hstack(blocks)


def coldstart_factors(bench_name, views, rank=128, alpha=10.0, folds=5):
    from sklearn.decomposition import TruncatedSVD
    from .. import tasks as TK
    tag = "all" if len(views) > 8 else "t0"
    f = KC / f"coldstart_{bench_name}_{tag}.npz"
    if f.exists():
        z = np.load(f); return z["US"], z["Vhat"]
    Bf = TK.load(bench_name)
    L = Bf.context.tocsr().astype(np.float32)
    mr = np.concatenate([t.mask_r for t in Bf.tasks if t.mask_r is not None and len(t.mask_r)])
    mc = np.concatenate([t.mask_c for t in Bf.tasks if t.mask_c is not None and len(t.mask_c)])
    M = sp.csr_matrix((np.ones(len(mr), np.float32), (mr, mc)), shape=L.shape); M.data[:] = 1
    Ls = (L - L.multiply(M)).tocsr(); Ls.eliminate_zeros()           # safe context: no task's hidden pair
    assert np.asarray(Ls[mr, mc]).ravel().max() == 0, "safe context still holds a masked pair"
    print(f"[coldstart] {bench_name}: context nnz {L.nnz}, safe nnz {Ls.nnz}", flush=True)
    svd = TruncatedSVD(rank, random_state=0, algorithm="randomized", n_iter=7).fit(Ls)
    V = svd.components_.T.astype(np.float32)                     # gene factors [n_genes, rank]; Ls ~ (Ls V) V^T
    US = (Ls @ V).astype(np.float32)                             # disease factors [n_rows, rank]
    X = _content(views)
    Vhat = np.zeros_like(V)
    fold = np.random.default_rng(0).integers(0, folds, X.shape[0])
    for k in range(folds):
        tr, te = fold != k, fold == k
        Xt = X[tr]; mu = Xt.mean(0); Xc = Xt - mu
        W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1], dtype=np.float32), Xc.T @ (V[tr] - V[tr].mean(0)))
        Vhat[te] = (X[te] - mu) @ W + V[tr].mean(0)
    # quality: held-out-gene correlation of reconstructed column norms (diagnostic only)
    r = np.corrcoef(np.linalg.norm(V, axis=1), np.linalg.norm(Vhat, axis=1))[0, 1]
    print(f"[coldstart] {bench_name}/{tag}: |V| vs |Vhat| corr (gene-cross-fitted) {r:.3f}", flush=True)
    np.savez(f, US=US, Vhat=Vhat)
    return US, Vhat


def coldstart(B, split_tag, views):
    US, Vhat = coldstart_factors(B.name, views)
    W = transfer_weights(B.tasks, B)
    return np.asarray((W @ US) @ Vhat.T, np.float32)


# ------------------------------------------------------------------------------------------------ registry
CHANNELS = {}
for _v in CH.view_names():
    CHANNELS[f"lift_{_v}"] = (lambda v: lambda B, s: lift(B, s, v))(_v)
    CHANNELS[f"lift_perm_{_v}"] = (lambda v: lambda B, s: lift(B, s, "perm_" + v))(_v)
for _k in ("paralog", "paralog_all", "paralog_wide", "afdb"):
    CHANNELS[f"lift_{_k}"] = (lambda v: lambda B, s: lift(B, s, v))(_k)
    CHANNELS[f"lift_perm_{_k}"] = (lambda v: lambda B, s: lift(B, s, "perm_" + v))(_k)
CHANNELS["coldstart_profile"] = lambda B, s: coldstart(B, s, tuple(CH.view_names()))
CHANNELS["coldstart_profile_t0"] = lambda B, s: coldstart(B, s, T0_VIEWS)

LIFT_VIEWS = [f"lift_{v}" for v in CH.view_names()]
LIFT_HOMOLOGY = ["lift_paralog", "lift_paralog_wide", "lift_afdb"]
LIFT_T0 = [f"lift_{v}" for v in T0_VIEWS]
COLDSTART = ["coldstart_profile", "coldstart_profile_t0"]
ALL = LIFT_VIEWS + LIFT_HOMOLOGY + COLDSTART
PERM = [f"lift_perm_{v}" for v in CH.view_names()] + ["lift_perm_paralog", "lift_perm_paralog_wide", "lift_perm_afdb"]
