"""Build OT 23.06 genetic gene sets (modet rules: GWAS/ClinVar score >= 0.5, curated any score), propagated with the
25.09 ontology (so rows align with the forward context = OT 25.09 sets). Used by ext/disease_c.py as the 'previous
release' of the forward benchmark (23.06 -> 25.09 history). Output: data/cache/labels_2306p2509.npz + _ids.parquet.
Source: /mnt/msadb/modet/downloads/v2/ot2306_assoc (OT 23.06 associationByDatasourceDirect, EBI FTP)."""
import sys, numpy as np, pandas as pd, pyarrow.dataset as ds, scipy.sparse as sp
sys.path.insert(0, "/home/ph/code/modeng/src")
from modeng import data as D

GWAS = {"ot_genetics_portal", "gwas_credible_sets"}; CLINVAR = {"eva"}
CURATED = {"genomics_england", "gene2phenotype", "clingen", "orphanet", "uniprot_variants", "uniprot_literature", "gene_burden"}
GEN = GWAS | CLINVAR | CURATED
t = ds.dataset("/mnt/msadb/modet/downloads/v2/ot2306_assoc").to_table(
    columns=["datasourceId", "diseaseId", "targetId", "score"], filter=ds.field("datasourceId").isin(list(GEN))).to_pandas()
print(t.datasourceId.value_counts())
strong = np.where(t.datasourceId.isin(GWAS | CLINVAR), t.score >= 0.5, True)
t = t[strong]
on = D.ontology("2509")
desc = dict(zip(on.id, on.descendants)); anc = dict(zip(on.id, on.ancestors))
known = set(on.id)
print("direct diseases", t.diseaseId.nunique(), "in 25.09 ontology", t.diseaseId.isin(known).mean())
t = t[t.diseaseId.isin(known)]
by = t.groupby("diseaseId")["targetId"].agg(set).to_dict()
ids = sorted(set(by) | {a for d in by for a in anc.get(d, [])})
U = D.universe(); rows, cols, keep = [], [], []
for d in ids:
    g = set(by.get(d, set()))
    for c in desc.get(d, []):
        g |= by.get(c, set())
    ix = U.get_indexer(sorted(g)); ix = np.unique(ix[ix >= 0])
    if len(ix) < 3:
        continue
    rows.append(np.full(len(ix), len(keep))); cols.append(ix); keep.append(d)
r, c = np.concatenate(rows), np.concatenate(cols)
L = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(keep), len(U)))
sp.save_npz(D.CACHE / "labels_2306p2509.npz", L)
pd.DataFrame({"disease_id": keep}).to_parquet(D.CACHE / "labels_2306p2509_ids.parquet")
print("rows", L.shape, "nnz", L.nnz)
