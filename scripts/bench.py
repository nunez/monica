#!/usr/bin/env python3
"""Machine-readable benchmark: cold start, warm request latency, peak RSS, sizes.

Usage: PYTHONPATH=src .venv/bin/python scripts/bench.py [--audio-src data/raw/1001_101_01.wav]
Writes bench/bench_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

MODEL_DIR = os.environ.get("MONICA_BASE_MODEL", "data/models/gemma-4-E2B-it")
CKPT = "models/head-cremad.pt"
CALIB = "models/calibration.json"

CHOICE = {
    "type": "choice",
    "instructions": "How threatening is the tone overall?",
    "criteria": {"calm": "calm, no threat", "uneasy": "slightly uneasy",
                 "tense": "tense", "threatening": "actively threatening"},
}
CHOICE2 = {
    "type": "choice",
    "instructions": "What is the overall sentiment?",
    "criteria": {"positive": "positive", "neutral": "neutral", "negative": "negative"},
}
SCORE = {
    "type": "score",
    "instructions": "Rate vocal tension.",
    "criteria": ["completely relaxed", "slightly tense", "noticeably tense",
                 "very tense", "extremely tense"],
}
NOUL = {"type": "noul", "instructions": "Does the speaker sound angry?"}


def question_set(n: int) -> dict:
    kinds = [CHOICE, CHOICE2, SCORE, NOUL]
    out = {}
    for i in range(n):
        out[f"q{i}"] = dict(kinds[i % len(kinds)])
        if out[f"q{i}"]["type"] == "noul":
            out[f"q{i}"]["instructions"] = f"Does the speaker sound tense or agitated #{i}?"
    return out


def load_reference_wav(path: str) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    # resample to 16k with soxr
    import soxr

    return soxr.resample(x, sr, 16000).astype(np.float32)


def make_variants(x: np.ndarray) -> dict:
    out = {"3s": x[: 3 * 16000], "10s": x[: 10 * 16000]}
    reps = int(np.ceil(30.0 * 16000 / len(x)))
    out["30s"] = np.tile(x, reps)[: 30 * 16000]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-src", default="data/raw/1001/1001_101_01.wav")
    ap.add_argument("--reps", type=int, default=7)
    args = ap.parse_args()

    src = os.path.join(ROOT, args.audio_src)
    if not os.path.exists(src):
        raise SystemExit(f"reference wav not found: {src}")

    import torch  # noqa: F401  (imported once to include in RSS)
    from monica.model import MonicaModel, MonicaConfig

    res = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "mps",
        },
        "files": {
            "base_model": {"path": MODEL_DIR,
                          "bytes": sum(os.path.getsize(os.path.join(dp, f))
                                       for dp, _, fs in os.walk(os.path.join(ROOT, MODEL_DIR))
                                       for f in fs if f.endswith((".safetensors", ".json", ".model"))),
                          },
            "head_checkpoint": {"path": CKPT, "bytes": os.path.getsize(os.path.join(ROOT, CKPT))},
            "calibration": {"path": CALIB, "bytes": os.path.getsize(os.path.join(ROOT, CALIB))},
        },
    }

    # ---- cold start (fresh process: import + load + warmup)
    t0 = time.perf_counter()
    cfg = MonicaConfig(model_dir=os.path.join(ROOT, MODEL_DIR), ckpt=os.path.join(ROOT, CKPT),
                    device="mps", calibration_path=os.path.join(ROOT, CALIB))
    model = MonicaModel(cfg)
    cold_model_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    model.warmup()
    cold_warmup = time.perf_counter() - t0
    res["cold_start"] = {
        "model_load_s": round(cold_model_load, 2),
        "warmup_s": round(cold_warmup, 2),
        "total_s": round(cold_model_load + cold_warmup, 2),
    }

    x = load_reference_wav(src)
    variants = make_variants(x)
    res["audio"] = {"source": args.audio_src, "source_seconds": round(len(x) / 16000, 2)}

    import base64
    import io

    import soundfile as sf

    def data_url(audio: np.ndarray) -> str:
        buf = io.BytesIO()
        sf.write(buf, audio, 16000, format="WAV", subtype="PCM_16")
        return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()

    # ---- warm latency grid
    grid = {}
    for dur_name, wav in variants.items():
        for nq in (1, 10, 32):
            req = {"model": "monica-bench", "state": {"audio": data_url(wav)},
                   "questions": question_set(nq)}
            lats = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                resp, _ = model.handle(req)
                lats.append((time.perf_counter() - t0) * 1000)
            lats.sort()
            grid[f"{dur_name}_{nq}q"] = {
                "min_ms": round(lats[0], 1),
                "median_ms": round(lats[len(lats) // 2], 1),
                "p95_ms": round(lats[min(len(lats) - 1, int(len(lats) * 0.95))], 1),
                "encode_ms_median": resp["usage"]["encode_ms"],
                "score_ms_median": resp["usage"]["score_ms"],
                "rtf": round(lats[len(lats) // 2] / 1000 / float(dur_name.rstrip("s")), 4),
            }
    res["latency_grid"] = grid

    # ---- encoding cost vs question count (same audio, encoded once per request)
    res["encoding_amortization"] = {
        "audio_seconds": 30.0,
        "encode_ms_by_request": {
            "1q": grid["30s_1q"]["encode_ms_median"],
            "10q": grid["30s_10q"]["encode_ms_median"],
            "32q": grid["30s_32q"]["encode_ms_median"],
        },
        "audio_encoder_calls_per_request": 1,
        "generated_tokens": 0,
    }

    ru = resource.getrusage(resource.RUSAGE_SELF)
    res["memory"] = {"peak_rss_bytes": int(ru.ru_maxrss),  # bytes on macOS
                    "peak_rss_gb": round(ru.ru_maxrss / 1e9, 2)}

    out = os.path.join(ROOT, "bench", "bench_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
