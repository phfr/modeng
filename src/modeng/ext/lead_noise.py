"""Same-width negative controls for increment tests (lead): channels of pure noise, and cs_donor with its gene axis permuted
per task (keeps the score distribution, destroys gene identity)."""
import numpy as np


def _noise(B, split_tag):
    return np.random.default_rng(12345).random((len(B.tasks), B.n_genes), dtype=np.float32)


def _cs_perm(B, split_tag):
    from modeng.channels import compute
    S = compute(B, "cs_donor", split_tag).copy()
    rng = np.random.default_rng(7)
    for i in range(S.shape[0]):
        S[i] = S[i, rng.permutation(S.shape[1])]
    return S


CHANNELS = {"noise1": _noise, "cs_donor_perm": _cs_perm}
