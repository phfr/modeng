"""r6: channels alone on temporal dev (all tasks and covered tasks), incl. permuted controls. Dev only."""
import numpy as np, pandas as pd
from modeng import tasks as T, channels as CH, bench as BM
B = T.load("temporal").split("dev")
out = []
for c in ["fg_hit", "fg_hit_perm", "fg_hit_cs2g", "fg_hit_cs2g_perm", "gnomad_loeuf", "gnomad_loeuf_perm", "gwas_catlag", "transfer_jaccard"]:
    S = CH.compute(B, c, "dev").copy()
    cov = ~np.all(np.isnan(S), 1)
    S = np.where(np.isnan(S), -1e9, S)
    for scope, m in (("all", np.ones(len(cov), bool)), ("covered_fg", ~np.all(np.isnan(CH.compute(B, "fg_hit", "dev")), 1))):
        Bs = T.Benchmark(B.name, [t for t, k in zip(B.tasks, m) if k], B.context, B.context_ids, B.network, B.union_network, B.leakage_class, B.degree)
        df = BM.evaluate(S[m], Bs, jobs=4)
        r = dict(channel=c, scope=scope, n=int(m.sum()), n_cov=int((cov & m).sum()))
        for k in ("ap", "nuap", "auroc", "recall@100", "sa_ap", "sa_nuap"):
            if k in df: r[k] = df[k].mean()
        out.append(r)
df = pd.DataFrame(out); df.to_csv("/home/ph/code/modeng/results/finngen/alone_temporal_dev.tsv", sep="\t", index=False, float_format="%.4f")
print(df.to_string())
