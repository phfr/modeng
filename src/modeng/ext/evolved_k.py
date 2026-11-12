"""F2-3 evolved transfer kernel (round-2 island 2): channel `evo_transfer` = the program selected by the evolve loop
(`src/modeng/evolve/cand_17.py`, chosen on 600 `past` episodes by nuAP alone; docs/reports/r2_evolve.md).

score(g) = (deg(g)+1)^-0.6 * [ sum_{r in top-200 cosine rows} cos_r^2 L_t[r,g]  +  1e-3 * T(g) / max T ],
cos_r = |seeds & row r| / sqrt(|seeds| |row r|) on the exactly masked context L_t (B.context_for via the target-free
harness), own row excluded; T = sum over ALL rows with cos > 0 of cos_r L_t[r,g] (orders the genes no top-200 neighbour
carries). deg = degree in the benchmark's network (B.degree).
Inputs: seeds, masked context, network degree only. Leakage class = that of the benchmark context (temporal-clean on
past/temporal/forward; holdout-current on holdout). Exact masking, no float correction (no residue route).
"""
import numpy as np

from ..evolve.harness import SafeBench
from ..evolve import cand_17


def evo_transfer(B, split_tag):
    sb = SafeBench(B)
    return np.asarray(cand_17.score(sb.tasks, sb), np.float32)


CHANNELS = {"evo_transfer": evo_transfer}
