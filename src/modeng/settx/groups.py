"""Leak-aware fold groups for past episodes.

Past episodes are ontology-propagated: an ancestor's arrivals contain its descendants' arrivals, and siblings often get
the same arrivals from one source update. modet clusters cover only panel diseases (most past tasks are singletons), so
cluster folds leak shared arrivals across folds (held-out past AP 0.31 vs transfer 0.04 in the first tune run).
Groups = Louvain communities of the graph whose edges join tasks that are ontology ancestor/descendant or whose target
sets overlap (Jaccard >= 0.1), weight = target Jaccard + 0.1; communities are packed greedily into K folds.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .. import data as D
from ..methods import baselines as BL


def leak_folds(tasks, k=3, seed=0, jthr=0.1):
    import networkx as nx
    n = len(tasks); G = tasks[0].seeds.max()  # placeholder
    ids = [t.disease_id for t in tasks]; pos = {d: i for i, d in enumerate(ids)}
    ng = int(max(max(t.targets.max(), t.seeds.max()) for t in tasks)) + 1
    Tm = BL._indicator([t.targets for t in tasks], ng)
    inter = (Tm @ Tm.T).toarray()
    sz = np.asarray(Tm.sum(1)).ravel()
    jac = inter / np.maximum(sz[:, None] + sz[None] - inter, 1)
    np.fill_diagonal(jac, 0)
    g = nx.Graph(); g.add_nodes_from(range(n))
    r, c = np.nonzero(np.triu(jac >= jthr))
    for a, b in zip(r, c):
        g.add_edge(int(a), int(b), weight=float(jac[a, b]) + 0.1)
    for i, d in enumerate(ids):
        for x in D.descendants(d):
            j = pos.get(x)
            if j is not None and j != i:
                w = float(jac[i, j]) + 0.1
                if g.has_edge(i, j):
                    g[i][j]["weight"] = max(g[i][j]["weight"], w)
                else:
                    g.add_edge(i, j, weight=w)
    comms = nx.community.louvain_communities(g, weight="weight", resolution=1.0, seed=seed)
    comms = sorted(comms, key=len, reverse=True)
    fold = np.zeros(n, int); load = np.zeros(k)
    for cm in comms:
        f = int(np.argmin(load)); fold[list(cm)] = f; load[f] += len(cm)
    tot = sum(d["weight"] for _, _, d in g.edges(data=True))
    cut = sum(d["weight"] for a, b, d in g.edges(data=True) if fold[a] != fold[b])
    # cross-fold arrival sharing: per task, fraction of its targets that are targets of an other-fold task with jac>=jthr
    return fold, dict(n_comms=len(comms), largest=len(comms[0]), cut_frac=cut / max(tot, 1e-9), fold_sizes=load.tolist())
