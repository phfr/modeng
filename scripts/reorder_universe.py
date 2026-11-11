"""One-off fix (2026-11-12 01:30): views/donors were built in HGNC-parquet row order; the harness universe is sorted ENSG
(modeng.data.universe()). Permute every data/views/*.npz and data/donors/*.npz to the harness order (idempotent via marker)."""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path
sys.path.insert(0, "/home/ph/code/modeng/src")
from modeng import data as D
h = pd.read_parquet("/home/ph/code/modet/data/processed/hgnc_protein_coding.parquet").ensembl_gene_id.values
u = D.universe().values
perm = pd.Index(h).get_indexer(u)          # new[i] = old[perm[i]]
assert (perm >= 0).all()
inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))  # old col j -> new col inv[j]
root = Path("/home/ph/code/modeng/data")
for f in sorted((root / "views").glob("*.npz")):
    z = dict(np.load(f))
    if "order" in z: continue
    np.savez_compressed(f, X=z["X"][perm], mask=z["mask"][perm], order=np.array("sorted_ensg")); print("view", f.name)
for f in sorted((root / "donors").glob("*.npz")):
    z = dict(np.load(f))
    if "order" in z: continue
    M = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    M = M[:, perm].tocsr()
    np.savez_compressed(f, data=M.data, indices=M.indices, indptr=M.indptr, shape=M.shape, names=z["names"], order=np.array("sorted_ensg")); print("donor", f.name)
