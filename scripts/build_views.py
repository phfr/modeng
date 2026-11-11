"""Build multi-view gene embeddings aligned to the 19,255-gene universe.

Output: data/views/<name>.npz with X (float32 [19255, d], z-scored, 0 for missing) and mask (bool [19255]).
Views are reduced with PCA to <= 256 dims. Sources are read-only (modet / subex data roots).
Leakage class of every view is recorded in data/views/MANIFEST.tsv: 'seq' (sequence only), 'pre21' (dated <= 2021-04),
'modern' (post-2021, label-free experimental data), 'annot' (curated annotation; may encode disease knowledge).
"""
import sys, os, numpy as np, pandas as pd
from pathlib import Path

OUT = Path("/home/ph/code/modeng/data/views"); OUT.mkdir(parents=True, exist_ok=True)
U = pd.read_parquet("/home/ph/code/modet/data/processed/hgnc_protein_coding.parquet")
ENSG = U.ensembl_gene_id.values; idx_ensg = {g: i for i, g in enumerate(ENSG)}
idx_sym = {s: i for i, s in enumerate(U.symbol.values)}
acc2i = {}
for i, a in enumerate(U.uniprot_ids.fillna("")):
    for x in str(a).split("|"):
        if x and x not in acc2i: acc2i[x] = i
SD = "/mnt/msadb/subex/derived/"; MF = "/home/ph/code/modet/data/processed/features/"


def pca(M, d=256):
    M = np.nan_to_num(M.astype(np.float32))
    M = (M - M.mean(0)) / (M.std(0) + 1e-6)
    if M.shape[1] <= d: return M
    from sklearn.utils.extmath import randomized_svd
    Uu, S, _ = randomized_svd(M, n_components=d, random_state=0)
    return (Uu * S).astype(np.float32)


def save(name, rows_idx, M, cls, src, d=256):
    rows_idx = np.asarray(rows_idx); ok = rows_idx >= 0
    Z = pca(M[ok], d)
    X = np.zeros((len(ENSG), Z.shape[1]), np.float32); mask = np.zeros(len(ENSG), bool)
    X[rows_idx[ok]] = Z; mask[rows_idx[ok]] = True
    X[mask] = (X[mask] - X[mask].mean(0)) / (X[mask].std(0) + 1e-6)
    np.savez_compressed(OUT / f"{name}.npz", X=X, mask=mask)
    with open(OUT / "MANIFEST.tsv", "a") as f:
        f.write(f"{name}\t{cls}\t{X.shape[1]}\t{mask.sum()}\t{src}\n")
    print(name, X.shape, mask.sum(), flush=True)


def from_npy(stem):
    M = np.load(SD + stem + ".npy")
    g = [l.strip() for l in open(SD + stem + "_genes.txt")]
    return [idx_ensg.get(x, -1) for x in g], M


def main(which):
    if "esmc" in which:
        d = pd.read_parquet(MF + "esmc.parquet")
        key = d.columns[0] if not d.columns[0].startswith("esmc") else None
        ids = d[key].values if key else d.index.values
        save("esmc", [idx_ensg.get(x, -1) for x in ids], d.filter(like="esmc_").values, "seq", MF + "esmc.parquet")
    if "prott5" in which:
        import h5py
        f = h5py.File("/mnt/msadb/subex/raw/uniprot_prott5_embeddings/per-protein.h5")
        rows, vecs, seen = [], [], set()
        for k in f.keys():
            i = acc2i.get(k, -1)
            if i >= 0 and i not in seen: seen.add(i); rows.append(i); vecs.append(f[k][:])
        save("prott5", rows, np.stack(vecs), "seq", "uniprot per-protein.h5")
    if "esm2" in which:
        import torch
        D = torch.load("/mnt/msadb/subex/raw/uce_esm2_protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt", weights_only=True)
        rows, vecs = [], []
        for s, v in D.items():
            i = idx_sym.get(s, -1)
            if i >= 0: rows.append(i); vecs.append(v.float().numpy())
        save("esm2_15b", rows, np.stack(vecs), "seq", "UCE ESM2-15B")
    for stem, name, cls in [("depmap20q4", "depmap20q4", "pre21"), ("expr_hpa_sc", "hpa_sc", "modern"), ("expr_gtex", "gtex", "modern"),
                            ("expr_hpa_tissue", "hpa_tissue", "modern"),
                            ("ps_k562_gwps_pc10", "ps_k562", "modern"), ("ps_rpe1_ess_pc10", "ps_rpe1", "modern"),
                            ("ps_cd4_2025", "ps_cd4", "modern"), ("ps_hepg2_nadig", "ps_hepg2", "modern"), ("ps_jurkat_nadig", "ps_jurkat", "modern")]:
        if name in which:
            try:
                rows, M = from_npy(stem)
                if M.shape[0] != len(rows): M = M.T
                save(name, rows, M, cls, SD + stem)
            except Exception as e:
                print("FAIL", name, e, flush=True)
    if "phylo" in which:
        d = pd.read_csv("/mnt/msadb/subex/raw/cladeoscope/9606_NPP.tsv", sep="\t", index_col=0)
        save("phylo_npp", [idx_sym.get(s, -1) for s in d.index], d.values, "seq", "CladeOScope NPP")
    for nm, fn, col in [("go", MF + "go_t0.parquet", "go_id"), ("reactome", MF + "reactome_t0.parquet", "pathway_id")]:
        if nm in which:
            d = pd.read_parquet(fn); d = d[d.ensembl_gene_id.isin(idx_ensg)]
            terms = {t: j for j, t in enumerate(d[col].unique())}
            from scipy.sparse import csr_matrix
            A = csr_matrix((np.ones(len(d), np.float32), (d.ensembl_gene_id.map(idx_ensg).values, d[col].map(terms).values)),
                           shape=(len(ENSG), len(terms)))
            A = A.multiply(1.0 / np.sqrt(np.asarray(A.sum(0)).ravel() + 1)).tocsr()
            from sklearn.utils.extmath import randomized_svd
            Uu, S, _ = randomized_svd(A, n_components=128, random_state=0)
            rows = np.where(np.asarray(A.sum(1)).ravel() > 0)[0]
            save(nm + "_t0", rows, (Uu * S)[rows], "annot", fn, d=128)


if __name__ == "__main__":
    which = sys.argv[1:] or ["esmc", "prott5", "esm2", "depmap20q4", "hpa_sc", "gtex", "hpa_tissue", "ps_k562", "ps_rpe1",
                             "ps_cd4", "ps_hepg2", "ps_jurkat", "phylo", "go", "reactome"]
    main(which)
