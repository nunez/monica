"""Temporal pooling of audio encoder states into fixed-size question-agnostic
descriptors.

The audio encoding (selected intermediate layers + final projection) is pooled
into N fixed time windows (mean+std per window) plus global per-layer stats.
Question-conditioned cross-attention then runs over the small window sequence
(not raw frames), so scoring many questions is cheap and every question reads
the SAME shared encoding (encoded once per request).
"""

from __future__ import annotations

import torch


def window_pool(hidden: torch.Tensor, n_windows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hidden [L,d] -> mean [W,d], std [W,d], valid [W] (masked windows).

    Window boundaries are equal spans over the valid token axis. Empty windows
    (from very short utterances) are marked invalid and carry zeros.
    """
    L, d = hidden.shape
    W = n_windows
    mean = torch.zeros(W, d, dtype=torch.float32)
    sq = torch.zeros(W, d, dtype=torch.float32)
    cnt = torch.zeros(W, 1, dtype=torch.float32)
    if L > 0:
        idx = (torch.arange(L) * W / L).long().clamp(max=W - 1)
        x = hidden.float()
        mean.index_add_(0, idx, x)
        sq.index_add_(0, idx, x * x)
        cnt.index_add_(0, idx, torch.ones(L, 1, dtype=torch.float32))
    cnt_safe = cnt.clamp(min=1)
    mean = mean / cnt_safe
    var = (sq / cnt_safe - mean * mean).clamp(min=0)
    std = torch.sqrt(var)
    valid = (cnt.squeeze(1) > 0)
    return mean.to(hidden.dtype), std.to(hidden.dtype), valid


def subset_A(A_full: torch.Tensor, layer_ids, lw: int = 1024, out_dim: int = 1536,
             n_layers: int = 12) -> torch.Tensor:
    """Slice a full-layer windowed A (layout: [layer means..., final mean | layer stds..., final std])
    down to a chosen layer subset."""
    dims = [lw] * n_layers + [out_dim]
    offs = [0]
    for d in dims:
        offs.append(offs[-1] + d)
    total = offs[-1]
    idxs = list(layer_ids) + [n_layers]
    means = torch.cat([A_full[..., offs[i]:offs[i] + dims[i]] for i in idxs], dim=-1)
    stds = torch.cat([A_full[..., total + offs[i]:total + offs[i] + dims[i]] for i in idxs], dim=-1)
    return torch.cat([means, stds], dim=-1)


def pool_encoding(enc, layer_ids, n_windows: int = 20) -> dict:
    """Build pooled window descriptor A [W, d_a] (per-window mean+std per layer)
    plus a global per-layer stats vector. All derived from one AudioEncoding."""
    lw = enc.layers.shape[-1]
    dims = [lw] * len(layer_ids) + [enc.final.shape[-1]]
    d_a = sum(dims)
    means = torch.zeros(n_windows, d_a, dtype=torch.float32)
    stds = torch.zeros(n_windows, d_a, dtype=torch.float32)
    valid = None
    offs, off = [], 0
    for d in dims:
        offs.append(off)
        off += d
    states = [enc.layers[i].float() for i in layer_ids]
    states.append(enc.final.float())
    for h, o, d in zip(states, offs, dims):
        h = h[: enc.token_len]
        m, s, valid = window_pool(h, n_windows)
        means[:, o:o + d] = m
        stds[:, o:o + d] = s
    A = torch.cat([means, stds], dim=1)  # [W, 2*d_a] fp32

    stats = []
    for i in range(enc.layers.shape[0]):
        h = enc.layers[i][: enc.token_len].float()
        stats.append(h.mean(0) if enc.token_len else torch.zeros(lw))
        stats.append(h.std(0) if enc.token_len > 1 else torch.zeros(lw))
    hf = enc.final[: enc.token_len].float()
    stats.append(hf.mean(0) if enc.token_len else torch.zeros(enc.final.shape[-1]))
    stats.append(hf.std(0) if enc.token_len > 1 else torch.zeros(enc.final.shape[-1]))
    stats_all = torch.cat([s.float() for s in stats])

    return {
        "A": A.half(),
        "valid": valid,
        "stats": stats_all.half(),
        "token_len": enc.token_len,
    }
