import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import io
import base64

import numpy as np
import pytest
import soundfile as sf

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)


def make_wav(data: np.ndarray, sr: int = 16000, subtype: str = "PCM_16") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def data_url(raw: bytes, mime: str = "audio/wav") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def voiced_like(dur=3.0, sr=16000, f0=140.0, seed=0) -> np.ndarray:
    """Synthetic voice-like signal: harmonic stack + syllabic amplitude modulation."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    sig = np.zeros_like(t)
    for h in range(1, 8):
        sig += (0.8 / h) * np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6))
    syl = 0.5 + 0.5 * (np.sin(2 * np.pi * 4.2 * t) ** 2)
    sig = sig * syl
    sig += rng.standard_normal(len(sig)) * 0.015
    sig = sig / (np.abs(sig).max() + 1e-9) * 0.8
    return sig.astype(np.float32)


def music_like(dur=4.0, sr=16000) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    sig = np.zeros_like(t)
    for f0 in (220.0, 277.2, 329.6, 440.0):
        sig += 0.25 * np.sin(2 * np.pi * f0 * t + 1.5 * np.sin(2 * np.pi * 0.5 * t))
    sig *= (0.6 + 0.4 * np.sin(2 * np.pi * 0.25 * t))
    return (sig / (np.abs(sig).max() + 1e-9) * 0.7).astype(np.float32)


MODEL_DIR = os.environ.get("MONICA_BASE_MODEL", "data/models/gemma-4-E2B-it")
CKPT = os.environ.get("MONICA_HEAD_CKPT", "models/head-cremad.pt")
CALIB = os.environ.get("MONICA_CALIB", "models/calibration.json")


def model_available() -> bool:
    return os.path.isdir(MODEL_DIR) and os.path.exists(CKPT)


@pytest.fixture(scope="session")
def client():
    if not model_available():
        pytest.skip("trained checkpoint / base model not present")
    from fastapi.testclient import TestClient
    from monica.server import create_app

    os.environ["MONICA_CALIB"] = CALIB if os.path.exists(CALIB) else ""
    app = create_app()
    with TestClient(app) as c:
        yield c
