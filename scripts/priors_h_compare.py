"""implementer H: paired cluster bootstrap (modeng.compare) of hS_<a> vs hS_<b> on nuap / ap / gw_nuap / ng_nuap."""
import sys
from modeng.compare import paired_bootstrap, load
bench = sys.argv[1]; pairs = [p.split(":") for p in sys.argv[2:]]
for split in ("dev", "confirm"):
    for a, b in pairs:
        try:
            A, B = load(bench, split, "hS_" + a), load(bench, split, "hS_" + b)
        except FileNotFoundError:
            continue
        for m in ("nuap", "ap", "auroc", "recall@100", "onnet_nuap") + (("gw_nuap", "ng_nuap") if "gw_nuap" in A else ()):
            r = paired_bootstrap(A, B, m, 5000)
            print(f"{bench}\t{split}\t{a}\tvs\t{b}\t{m}\t{r['mean_a']:.4f}\t{r['mean_b']:.4f}\t{r['delta']:+.4f}\t[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]\tP<=0 {r['p_le0']:.3f}\twin {r['win_rate']:.2f}\tn={r['n_tasks']}")
