"""Battery evaluation: run packed question sets through MonicaModel.score on
precomputed pooled features (mirrors the serve-time computation exactly)."""

from __future__ import annotations

import numpy as np
import torch

from .audioio import AudioQuality
from . import pooling


def stub_quality(q: torch.Tensor) -> AudioQuality:
    q = q.tolist()
    return AudioQuality(
        duration_s=q[0] * 30.0, sample_rate_orig=16000, channels=1,
        peak=q[6], rms_db=q[3] * 60 - 60, noise_floor_db=-60.0, snr_db=q[4] * 40,
        speech_seconds=q[2] * 30.0, speech_ratio=q[1], clipping_ratio=q[5],
        clipped=q[5] >= 0.02, silent=q[1] < 0.02, low_information=False, music_probable=False,
    )


def make_pooled(model, rec, layer_ids_full=True):
    if layer_ids_full:
        A = pooling.subset_A(rec["A"].unsqueeze(0)[0], model.cfg.layer_ids)
    else:
        A = rec["A"]
    return {
        "A": A,
        "valid": rec["valid"],
        "stats": rec["stats"],
        "token_len": rec["token_len"],
        "_quality": stub_quality(rec["q"]),
    }


def build_noul_questions(items):
    return {f"n{i}": {"type": "noul", "instructions": it["text"]} for i, it in enumerate(items)}


def build_choice_questions(panels):
    return {f"c{i}": {"type": "choice", "instructions": pl.get("instruction",
            "What primary sentiment is expressed by the speaker's voice?"),
            "criteria": {k: pl["descriptions"][k] for k in pl["panel"]}}
            for i, pl in enumerate(panels)}


def build_score_questions(scales):
    qs = {}
    for i, (word, sc) in enumerate(scales.items()):
        qs[f"s{i}"] = {"type": "score",
                       "instructions": f"How {word} does the speaker sound?",
                       "criteria": sc["levels"]}
    return qs


@torch.no_grad()
def run_clip(model, rec, questions):
    pooled = make_pooled(model, rec)
    return model.score(pooled, questions)


def run_battery(model, split_feats, questions, ids=None):
    """Returns per-question lists: probs, argmax, plus clip meta."""
    out: dict[str, list] = {"probs": [], "meta": []}
    n = len(ids) if ids is not None else split_feats["A"].shape[0]
    for i in range(n):
        rec = {"A": split_feats["A"][i], "valid": split_feats["valid"][i],
               "stats": split_feats["stats"][i], "token_len": split_feats["token_len"][i],
               "q": split_feats["q"][i]}
        answers = run_clip(model, rec, questions)
        out["probs"].append({k: v for k, v in answers.items()})
        out["meta"].append({"id": ids[i] if ids else i})
    return out
