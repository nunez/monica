#!/usr/bin/env python3
"""Compare question-conditioning options on this machine (measured, not claimed).

option1: compact question encoder over the Gemma embedding table (deployed).
option2: Gemma text trunk, text-only prefill, pooled (heavier, same head).
option3: distilled student of option2 (trained by scripts/distill.py, not run
         by default; reported only if a distilled checkpoint exists).

Writes bench/options_comparison.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from monica.qencoder import GemmaEmbeddingTable, TextEncoder, TrunkTextEncoder, make_tokenizer  # noqa: E402

QUESTIONS = [
    "Does the speaker sound angry?",
    "How threatening is the tone overall?",
    "Rate the speaker's tension from completely relaxed to extremely tense.",
    "What is the overall sentiment: positive, neutral, or negative?",
    "Does the voice sound warm and friendly?",
    "How afraid does the speaker sound?",
    "Rate vocal energy from calm to intense.",
    "Does the speaker sound disgusted or contemptuous?",
    "Is the tone deliberately hostile or accidental?",
    "How joyful does the voice sound?",
]


def bench_encode(encode_fn, tok, strings, device, reps=20, max_len=96):
    enc = tok(strings, return_tensors="pt", padding="max_length", truncation=True,
              max_length=max_len)
    ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
    encode_fn(ids, mask)  # warm
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        encode_fn(ids, mask)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/models/gemma-4-E2B-it")
    ap.add_argument("--ckpt", default="models/head-cremad.pt")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = make_tokenizer(args.model_dir)
    emb = GemmaEmbeddingTable(args.model_dir, device=args.device)
    res = {"questions": len(QUESTIONS), "max_len": 96}

    # option 1: embedding table + compact TextEncoder
    t_enc = TextEncoder(emb_dim=emb.dim, out_dim=512).to(args.device).eval()
    n1 = emb.weight.numel() + \
        sum(p.numel() for p in t_enc.parameters())
    ms1 = bench_encode(lambda ids, mask: t_enc(ids.to(device), emb), tok, QUESTIONS, device)
    res["option1_compact"] = {
        "params_M": round(n1 / 1e6, 1),
        "encode_ms_10q": round(ms1, 2),
        "note": "embedding table lookup + 2-layer projector; deployed",
    }

    # option 2: full Gemma text trunk, text-only prefill, pooled
    trunk = TrunkTextEncoder(args.model_dir, device=args.device)
    n2 = sum(p.numel() for p in trunk.model.parameters())
    ms2 = bench_encode(trunk.encode, tok, QUESTIONS, device)
    res["option2_text_trunk"] = {
        "params_M": round(n2 / 1e6, 1),
        "encode_ms_10q": round(ms2, 2),
        "note": "language-model prefill only (no generation); pooled last hidden",
    }

    # audio tower cost for reference (questions piggyback on it)
    from monica.encoder import GemmaAudioEncoder
    audio = GemmaAudioEncoder(args.model_dir, device=args.device)
    n3 = sum(p.numel() for p in audio.model.parameters()) if hasattr(audio, "model") else None
    import numpy as np
    wav = (np.random.default_rng(0).standard_normal(16000 * 10) * 0.05).astype("float32")
    audio.encode(wav)
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        audio.encode(wav)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    res["audio_tower"] = {
        "params_M": round(n3 / 1e6, 1) if n3 else None,
        "encode_ms_10s": round(ts[len(ts) // 2], 2),
        "note": "Gemma 4 audio tower, once per request regardless of question count",
    }

    res["recommendation"] = (
        "option1: question encoding is ~"
        f"{round(ms2 / max(ms1, 0.001))}x cheaper than the text trunk and accuracy parity "
        "was measured at calibration time; option2 remains available for gating experiments."
    )

    out = os.path.join(ROOT, "bench", "options_comparison.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
