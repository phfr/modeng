"""Same-width permuted control for the v2 pruning test (r6 P1b, lead): `perm_<c>` = channel <c> with its gene axis permuted
per task (fixed seed per channel; keeps each task's score distribution and NaN share, destroys gene identity). Defined for
the 28 channels that v2 drops from v1clean (results/v1clean_channels.txt minus results/v2_channels.txt)."""
import zlib
import numpy as np
from pathlib import Path

_R = Path(__file__).resolve().parents[3] / "results"


def _dropped():
    v1 = (_R / "v1clean_channels.txt").read_text().strip().split(",")
    v2 = set((_R / "v2_channels.txt").read_text().strip().split(","))
    return [c for c in v1 if c not in v2]


def _make(c):
    def fn(B, split_tag):
        from modeng.channels import compute
        S = compute(B, c, split_tag).copy()
        rng = np.random.default_rng(zlib.crc32(c.encode()))
        for i in range(S.shape[0]):
            S[i] = S[i, rng.permutation(S.shape[1])]
        return S
    return fn


CHANNELS = {"perm_" + c: _make(c) for c in _dropped()} if (_R / "v2_channels.txt").exists() else {}
