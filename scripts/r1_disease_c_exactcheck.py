"""Implementer C: verify dc_ridge equals a from-scratch ridge on context_for(task) with fam/masked rows removed (5 tasks)."""
import numpy as np, scipy.sparse as sp
from modeng import tasks as T, channels as CH
B = T.load("temporal").split("dev")
S = CH.compute(B, "dc_ridge", "dev")
for i in [0, 7, 19, 33, 50]:
    t = B.tasks[i]
    Lt = B.context_for(t).tocsr()
    F = set(t.mask_r.tolist()) | set(t.fam_rows.tolist()); own = B.own_row(t)
    if own >= 0: F.add(own)
    K = np.array(sorted(set(range(Lt.shape[0])) - F))
    L0 = B.context.tocsr()
    nrm = np.sqrt(np.maximum(L0.getnnz(1), 1))          # norms of the (identical on K) rows
    X = (sp.diags(1 / nrm[K]) @ Lt[K]).tocsr()
    s = np.zeros(B.n_genes); s[t.seeds] = 1
    w = np.linalg.solve((X @ X.T).toarray() + np.eye(len(K)), X @ s)
    ref = np.asarray(X.T @ w).ravel()
    print(t.disease_id, "max|diff|", float(np.abs(np.round(ref, 6) - S[i]).max()), "masked rows in K:", int(np.isin(t.mask_r, K).sum()))
