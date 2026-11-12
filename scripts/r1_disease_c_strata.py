"""Implementer C: channel minus transfer_jaccard by seed-count stratum (temporal dev+confirm pooled, forward dev)."""
import pandas as pd, numpy as np
from modeng.bench import RESULTS
rows = []
for bench, splits in (("temporal", ("dev", "confirm")), ("forward", ("dev", "confirm"))):
    for ch in ("dc_trend", "dc_directed", "dc_learned", "dc_ridge", "dc_ridge_pos"):
        for s in splits:
            try:
                a = pd.read_parquet(RESULTS / bench / s / f"{ch}.parquet"); b = pd.read_parquet(RESULTS / bench / s / "transfer_jaccard.parquet")
            except FileNotFoundError:
                continue
            if "sa_ap" not in b: a["sa_ap"] = np.nan; b["sa_ap"] = np.nan
            m = a.merge(b[["disease_id", "nuap", "ap", "sa_ap"]], on="disease_id", suffixes=("", "_tj"))
            m["stratum"] = pd.cut(m.n_seeds, [0, 10, 30, 100, 1e9], labels=["<=10", "11-30", "31-100", ">100"])
            for st, g in m.groupby("stratum", observed=True):
                rows.append(dict(bench=bench, split=s, ch=ch, stratum=st, n=len(g), d_nuap=(g.nuap - g.nuap_tj).mean(),
                                 d_ap=(g.ap - g.ap_tj).mean(), d_sa_ap=(g.sa_ap - g.sa_ap_tj).mean()))
df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
