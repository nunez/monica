#!/usr/bin/env python
"""Precompute pooled audio descriptors (windowed layer stats + quality) for
every manifest clip. Audio is encoded ONCE per clip; all downstream heads and
evaluations read these descriptors (mirroring the serve-time path)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monica.encoder import GemmaAudioEncoder  # noqa: E402
from monica import pooling  # noqa: E402
from monica.audioio import analyze_quality, load_wav_bytes, AudioInputError  # noqa: E402


QKEYS = ["duration_s", "speech_ratio", "speech_seconds", "rms_db", "snr_db",
         "clipping_ratio", "peak"]


def qvec(q) -> list:
    out = []
    for k in QKEYS:
        v = float(getattr(q, k, 0.0))
        if k in ("duration_s", "speech_seconds"):
            v = min(v, 30.0)
        out.append(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/models/gemma-4-E2B-it")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--manifest", default="data/manifest_cremad.json")
    ap.add_argument("--out", default="data/features/cremad")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    enc = GemmaAudioEncoder(args.model_dir, device=args.device)

    manifest = json.load(open(args.manifest))
    if args.end >= 0:
        manifest = manifest[args.start:args.end]
    buckets: dict[str, list] = {}
    t0 = time.time()
    fails = 0
    feats: dict[str, dict] = {}
    out = args.out
    for i, m in enumerate(manifest):
        key = m.get("split") or m.get("cond", "all")
        try:
            import soundfile as sf
            d, sr = sf.read(m["wav"], dtype="float32")
            if d.ndim > 1:
                d = d.mean(axis=1)
            if sr != 16000:
                import soxr
                d = soxr.resample(d, sr, 16000)
            d = np.clip(d, -1.0, 1.0)
            q = analyze_quality(d)
            e = enc.encode(d)
            p = pooling.pool_encoding(e, tuple(range(enc.n_layers)), 20)
            rec = {
                "A": p["A"].half(),
                "valid": p["valid"],
                "stats": p["stats"].half(),
                "token_len": int(p["token_len"]),
                "q": torch.tensor(qvec(q), dtype=torch.float32),
                "id": m["id"],
                "speaker": m.get("speaker", "?"),
                "emotion": m.get("emotion"),
                "cond": m.get("cond"),
            }
            feats.setdefault(key, []).append(rec)
        except (AudioInputError, Exception) as ex:  # noqa: BLE001
            fails += 1
            if fails < 10:
                print("FAIL", m["id"], ex)
        if (i + 1) % 500 == 0:
            print(f"{i+1}/{len(manifest)} ({time.time()-t0:.0f}s, fails={fails})", flush=True)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    path = f"{out}{args.suffix}.pt"
    torch.save(feats, path)
    print("saved", path, "buckets:", {k: len(v) for k, v in feats.items()}, "fails:", fails)


def enc_model_layer_ids(enc):
    return tuple(range(enc.n_layers))


if __name__ == "__main__":
    main()
