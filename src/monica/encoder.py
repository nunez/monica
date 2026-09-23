"""Gemma 4 E2B audio encoder extraction (non-generative).

Loads ONLY the `model.audio_tower.*` weights (12 layers, hidden 1024,
projected to 1536) from the Gemma 4 E2B checkpoint. Generation is never
involved. Intermediate layer states are captured for the layer study and the
optional layer-mix used by the decision heads.

Intended use: encode each audio section exactly once per request; every
question is scored from the shared encoding.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from .frontend import GemmaFrontend, AUDIO_TOKEN_MS

ALL_LAYERS = tuple(range(12))
DEFAULT_LAYERS = (2, 5, 8, 11)  # replaced by layer-study result in config


@dataclass
class AudioEncoding:
    layers: torch.Tensor  # [L, T, 1024] fp16 cpu (captured encoder layer states)
    final: torch.Tensor   # [T, 1536] fp16 cpu (post output_proj, LM space)
    token_len: int        # valid tokens (T) — padded tail masked out
    layer_ids: tuple
    n_audio_tokens_ms: float = AUDIO_TOKEN_MS
    stats: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return self.layers.shape[0] * self.layers.shape[-1] + self.final.shape[-1]


class GemmaAudioEncoder:
    """Standalone audio tower with layer-state capture."""

    def __init__(self, model_dir: str, device: str = "mps", dtype: str = "float32"):
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        cfg_all = json.load(open(os.path.join(model_dir, "config.json")))
        from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel

        self.config = Gemma4AudioConfig(**cfg_all["audio_config"])
        self.n_layers = self.config.num_hidden_layers
        self.hidden = self.config.hidden_size
        self.out_dim = self.config.output_proj_dims

        self.model = Gemma4AudioModel(self.config)
        self._load_audio_weights()
        self.model.to(device=self.device, dtype=self.dtype).eval()
        self.frontend = GemmaFrontend(model_dir)
        self.n_encodes = 0  # request-level counter (once-per-request audit)

        # hook to capture per-layer states
        self._captured: dict[int, torch.Tensor] = {}
        for i, layer in enumerate(self.model.layers):
            layer.register_forward_hook(self._make_hook(i))

    def _load_audio_weights(self) -> None:
        from safetensors import safe_open

        path = os.path.join(self.model_dir, "model.safetensors")
        prefix = "model.audio_tower."
        sd: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt") as f:
            names = [k for k in f.keys() if k.startswith(prefix)]
            for k in names:
                sd[k[len(prefix):]] = f.get_tensor(k)  # lazy per-tensor load
        missing, unexpected = self.model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, (missing[:3], unexpected[:3])

    def _make_hook(self, i: int):
        def hook(_mod, _inp, out):
            self._captured[i] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    @torch.no_grad()
    def encode(self, wav: np.ndarray) -> AudioEncoding:
        """wav: float32 mono 16 kHz, <= 30 s (audioio-enforced)."""
        feats, mask = self.frontend(wav)
        feats = feats.to(self.device, dtype=self.dtype)
        mask = mask.to(self.device) if mask is not None else None
        self._captured = {}
        out = self.model(input_features=feats, attention_mask=mask)
        final = out.last_hidden_state[0].float().cpu().half()  # (T,1536)
        layers = torch.stack([self._captured[i][0].float().cpu().half() for i in range(self.n_layers)])
        tok_len = int(mask[0].sum().item()) if mask is not None else final.shape[0]
        self.n_encodes += 1
        return AudioEncoding(layers=layers, final=final, token_len=tok_len, layer_ids=ALL_LAYERS)

    @torch.no_grad()
    def encode_pooled(self, wav: np.ndarray, layer_ids=DEFAULT_LAYERS, n_windows: int = 20):
        """Encode once and build the windowed audio descriptor consumed by heads."""
        enc = self.encode(wav)
        return pooling.pool_encoding(enc, layer_ids, n_windows)
