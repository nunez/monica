"""Text encoders for Jev question instructions and criteria.

Option 1 (default): compact question encoder — frozen Gemma 4 unified token
embedding table + small trainable transformer. New emotion words / criteria
phrasings generalize through the frozen multilingual token embeddings; no
retraining per question.

Option 2 (comparison): the Gemma 4 shared text trunk (`Gemma4TextModel`,
with per-layer embeddings) used prefill-only — no generation.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict

import torch
import torch.nn as nn

EMBED_KEY = "model.language_model.embed_tokens.weight"


class GemmaEmbeddingTable:
    """Frozen unified embedding table extracted from the Gemma 4 checkpoint."""

    def __init__(self, model_dir: str, device: str = "mps", dtype=torch.float32):
        from safetensors import safe_open

        path = os.path.join(model_dir, "model.safetensors")
        with safe_open(path, framework="pt") as f:
            w = f.get_slice(EMBED_KEY)[:]
        self.weight = w.to(device=device, dtype=dtype)  # [vocab, 1536]
        self.dim = w.shape[1]

    def embed(self, ids: torch.Tensor) -> torch.Tensor:  # [.., L, dim]
        return self.weight[ids.long()]


class TextEncoder(nn.Module):
    """Frozen Gemma embeddings -> trainable down-proj + small transformer -> pooled text vector."""

    def __init__(self, emb_dim: int = 1536, out_dim: int = 512, layers: int = 2, heads: int = 8):
        super().__init__()
        self.down = nn.Linear(emb_dim, out_dim, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=heads, dim_feedforward=out_dim * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(out_dim)
        self.pool_mix = nn.Parameter(torch.zeros(2, out_dim))  # mean/max blend

    def forward(self, ids: torch.Tensor, emb_table: GemmaEmbeddingTable) -> torch.Tensor:
        # ids: [B, L]
        pad = ids == 0
        x = emb_table.embed(ids).to(self.down.weight.dtype)
        x = self.down(x)
        x = self.encoder(x, src_key_padding_mask=pad)
        x = self.norm(x)
        m = (~pad).unsqueeze(-1)
        mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
        xa = x.masked_fill(pad.unsqueeze(-1), float("-inf")).max(1).values
        xa = torch.nan_to_num(xa, neginf=0.0)
        w = torch.sigmoid(self.pool_mix)  # [2,d]
        return w[0] * mean + w[1] * xa


class TextEncoderWithTable(nn.Module):
    """TextEncoder bundled with its frozen embedding table + string cache."""

    def __init__(self, model_dir: str, device: str = "mps", out_dim: int = 512):
        super().__init__()
        self.device = torch.device(device)
        self.table = None  # GemmaEmbeddingTable, registered as buffer container
        self.enc = TextEncoder(emb_dim=1536, out_dim=out_dim)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_max = 8192

    def set_table(self, table: GemmaEmbeddingTable):
        self.table = table

    @torch.no_grad()
    def encode_strings(self, strings: list[str], tok, max_len: int = 64) -> torch.Tensor:
        """Returns [K, out_dim] for K strings; results cached by (string, max_len)."""
        keys = [hashlib.sha256(f"{max_len}|{s}".encode()).hexdigest() for s in strings]
        misses = [(i, k) for i, k in enumerate(keys) if k not in self._cache]
        if misses:
            batch = [strings[i] for i, _ in misses]
            ids = tok(batch, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=max_len)["input_ids"].to(self.device)
            vecs = self.enc(ids, self.table).float().cpu()
            for (i, k), v in zip(misses, vecs):
                self._cache[k] = v
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)
        return torch.stack([self._cache[k] for k in keys])


def make_tokenizer(model_dir: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class TrunkTextEncoder(nn.Module):
    """Option 2: Gemma 4 text trunk, prefill-only (no generation), pooled.

    Loads the text trunk (`model.language_model.*`) with per-layer input
    embeddings (PLE) active. Text-only prefill — audio is not passed.
    """

    def __init__(self, model_dir: str, device: str = "mps", pool: str = "last"):
        super().__init__()
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
        from safetensors import safe_open

        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.config = Gemma4TextConfig(**cfg["text_config"])
        self.model = Gemma4TextModel(self.config)
        prefix = "model.language_model."
        sd = {}
        with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    sd[k[len(prefix):]] = f.get_slice(k)[:]
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not missing, f"trunk weights missing: {missing[:5]}"
        self.model.to(device=device, dtype=torch.bfloat16).eval()
        self.pool = pool
        self.out_dim = self.config.hidden_size

    @torch.no_grad()
    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=ids.to(self.model.device),
                         attention_mask=mask.to(self.model.device),
                         output_hidden_states=False, return_dict=True)
        hs = out.last_hidden_state[0].float()
        m = mask[0].unsqueeze(-1).float()
        return ((hs * m).sum(0) / m.sum().clamp(min=1)).cpu()
