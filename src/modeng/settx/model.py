"""settx model: gene encoder over views (no gene-id embedding) + seed set encoder + candidate->seed cross-attention."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CTX = 5   # per-(episode, gene) context features: log1p(tj), pct(tj), tj==0, log1p(pop), log1p(deg) (train.ctx_tensor)


class GeneEncoder(nn.Module):
    def __init__(self, dims, d_view=32, d=128, drop=0.1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(k, d_view) for k in dims])
        nv = len(dims)
        self.mlp = nn.Sequential(nn.Linear(nv * d_view + nv, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d),
                                 nn.LayerNorm(d))
        self.view_drop = 0.15

    def forward(self, Xs, M):
        """Xs: list of [n, d_v]; M: [n, n_views] (1 = view present)."""
        if self.training and self.view_drop > 0:
            M = M * (torch.rand_like(M) > self.view_drop).float()
        parts = [p(x) * M[:, i:i + 1] for i, (p, x) in enumerate(zip(self.proj, Xs))]
        return self.mlp(torch.cat(parts + [M], 1))


class SetTX(nn.Module):
    def __init__(self, dims, d=128, heads=4, drop=0.1, use_ctx=True, resid=False):
        super().__init__()
        self.resid = resid
        if resid:   # separate context-only path; the set/gene path is an additive correction
            self.ctx_head = nn.Sequential(nn.Linear(N_CTX + 1, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        self.enc = GeneEncoder(dims, d=d, drop=drop)
        self.use_ctx = use_ctx
        self.seed_sa = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.xattn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        nin = 4 * d + 1 + (N_CTX if use_ctx else 0)
        self.head = nn.Sequential(nn.Linear(nin, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d), nn.GELU(),
                                  nn.Linear(d, 1))

    def encode_seeds(self, hs, smask):
        """hs [B, S, d] seed embeddings, smask [B, S] True = padding. -> tokens [B, S, d], pooled [B, d]."""
        tok = self.seed_sa(hs, src_key_padding_mask=smask)
        z, _ = self.pool(self.pool_q.expand(hs.shape[0], -1, -1), tok, tok, key_padding_mask=smask)
        return tok, z[:, 0]

    def score(self, hc, tok, z, smask, ctx, nseed):
        """hc [B, C, d] candidate embeddings, ctx [B, C, N_CTX], nseed [B] -> logits [B, C]."""
        a, _ = self.xattn(hc, tok, tok, key_padding_mask=smask)
        zc = z[:, None].expand_as(hc)
        parts = [hc, a, zc, hc * zc, torch.log1p(nseed)[:, None, None].expand(-1, hc.shape[1], 1)]
        if self.use_ctx:
            parts.append(ctx)
        out = self.head(torch.cat(parts, -1)).squeeze(-1)
        if self.resid:
            out = out + self.ctx_head(torch.cat([ctx, parts[4]], -1)).squeeze(-1)
        return out
