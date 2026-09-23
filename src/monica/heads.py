"""Typed decision heads: noul (binary), choice (mutually exclusive criteria),
score (ordered levels). All consume the shared pooled audio descriptor +
question text vectors. No generation anywhere.

Order invariance: option logits are computed per-option independently and
softmaxed across the set, so shuffling criteria order cannot change the
probability assigned to a key (only the index of `score` levels matters).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_entropy_confidence(probs: torch.Tensor) -> torch.Tensor:
    """Distribution concentration in [0,1]: 1 - H(p)/log(K). 0 == uniform."""
    k = probs.shape[-1]
    if k <= 1:
        return torch.zeros_like(probs[..., 0])
    ent = -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(-1)
    return (1.0 - ent / math.log(k)).clamp(0.0, 1.0)


class PoolAttention(nn.Module):
    """Question-conditioned attention pooling over audio windows."""

    def __init__(self, d_q: int, d_a: int, d_h: int = 512):
        super().__init__()
        self.d_h = d_h
        self.wq = nn.Linear(d_q, d_h, bias=False)
        self.wk = nn.Linear(d_a, d_h, bias=False)
        self.wv = nn.Linear(d_a, d_h, bias=False)
        self.scale = 1.0 / math.sqrt(d_h)

    def kv(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute keys/values for an audio encoding (shared by all questions)."""
        return self.wk(A), self.wv(A)

    def forward(self, u: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        q = self.wq(u)  # [Q, d_h]
        logits = K @ q.T * self.scale  # [W, Q]
        if valid is not None:
            logits = logits.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        attn = torch.softmax(logits, dim=0)  # softmax over windows
        z = attn.T @ V  # [Q, d_h]
        return z


class QuestionHeads(nn.Module):
    """Pools audio by question query, then scores (audio-context, option-text)
    pairs with a shared MLP. noul = single option; choice/score = K options."""

    """Full head stack: pool attention + option scorer + ordinal bias."""

    def __init__(self, d_a: int, d_q: int = 512, d_h: int = 512, d_stats: int = 64):
        super().__init__()
        self.d_a = d_a
        self.d_q = d_q
        self.stats_dim = d_stats
        self.stats_proj = nn.Linear(d_stats, d_h)
        self.pool = PoolAttention(d_q, d_a, d_h)
        self.scorer = nn.Sequential(
            nn.Linear(d_h * 2 + d_q * 2, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )
        self.noul_opt = nn.Parameter(torch.randn(d_q) * 0.02)
        self.level_bias = nn.Parameter(torch.zeros(16))

    def forward_pooled(self, z: torch.Tensor, stats: torch.Tensor, u_opt: torch.Tensor):
        """z [Q,d_h]; stats [d_stats]; u_opt [Q,K,d_q] -> logits [Q,K]."""
        if u_opt.dim() == 2:
            u_opt = u_opt.unsqueeze(1)
        Q, K, _ = u_opt.shape
        z_ = z.unsqueeze(1).expand(-1, K, -1)
        u_ = u_opt
        feats = torch.cat([z_, u_, z_ * u_, (z_ - u_).abs()], dim=-1)
        return self.scorer(feats).squeeze(-1)
