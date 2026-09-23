"""Gemma 4 E2B audio frontend: waveform -> mel features for the audio tower.

Intensity preservation: no loudness normalization is applied here. Waveforms
arrive in [-1, 1] from audioio (peak-limited only if needed) and are passed
to the Gemma frontend with scaling factor 1.0, so loudness and dynamic range
reach the encoder intact.
"""

from __future__ import annotations

import numpy as np
import torch

from .audioio import TARGET_SR

AUDIO_TOKEN_MS = 40.0  # Gemma 4 audio: 40 ms per soft token


class GemmaFrontend:
    def __init__(self, model_dir: str):
        import json
        import os
        from transformers.models.gemma4.feature_extraction_gemma4 import (
            Gemma4AudioFeatureExtractor,
        )

        cfg = json.load(open(os.path.join(model_dir, "processor_config.json")))
        self.fx = Gemma4AudioFeatureExtractor(**cfg["feature_extractor"])
        self.sampling_rate = int(cfg["feature_extractor"]["sampling_rate"])
        assert self.sampling_rate == TARGET_SR

    def __call__(self, wav: np.ndarray) -> tuple[torch.Tensor, torch.Tensor | None]:
        """16 kHz float32 mono -> (input_features [1,T,128], attention_mask or None).

        Single unpadded items need no mask (encoder treats all frames valid).
        """
        feats = self.fx(wav[None, :], sampling_rate=self.sampling_rate, return_tensors="pt")
        return feats["input_features"], feats.get("attention_mask")
