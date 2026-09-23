"""WAV input/output: data-URL parsing, decoding, safety, quality analysis.

Design rules:
- Audio arrives as a base64 WAV data URL (or a local file path served by the
  server). One section of audio per request; never silently truncated.
- Sound is preserved faithfully: stereo is summed to mono (amplitude summed,
  not averaged; clips only if the summed peak exceeds full scale), resampling
  is high-quality, and loudness handling is *reference-based*: quiet signals
  are brought up toward a target loudness; signals at or above it keep their
  original level (peak-clipped only when needed to avoid digital clipping).
"""

from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

TARGET_SR = int(os.environ.get("MONICA_TARGET_SR", "16000"))
MAX_AUDIO_SECONDS = float(os.environ.get("MONICA_MAX_AUDIO_SECONDS", "120"))
TARGET_RMS = float(os.environ.get("MONICA_TARGET_RMS", "0.10"))  # -20 dBFS reference
PEAK_CEILING = 0.999
CLIPPING_RATIO_WARN = float(os.environ.get("MONICA_CLIP_RATIO", "0.02"))

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>audio/(?:wav|x-wav|wave|vnd\.wave))\s*(?:;[^,]*)?;base64,(?P<data>.*)$",
    re.IGNORECASE | re.DOTALL,
)


class AudioInputError(ValueError):
    """Raised for invalid or unusable audio input. Carries an HTTP-safe code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class AudioQuality:
    duration_s: float
    sample_rate_orig: int
    channels: int
    peak: float
    rms_db: float
    noise_floor_db: float
    snr_db: float
    speech_seconds: float
    speech_ratio: float
    clipping_ratio: float
    clipped: bool
    silent: bool
    music_probable: bool
    low_information: bool
    notes: list = field(default_factory=list)


def parse_data_url(url: str) -> tuple[str, bytes]:
    """Parse a base64 WAV data URL. Returns (media_type, raw_bytes)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise AudioInputError("malformed_data_url", "audio must be a base64 data: URL")
    m = _DATA_URL_RE.match(url)
    if not m:
        raise AudioInputError(
            "unsupported_media_type",
            "only base64 WAV data URLs are supported (audio/wav)",
        )
    try:
        raw = base64.b64decode(m.group("data"), validate=True)
    except Exception as e:  # noqa: BLE001
        raise AudioInputError("malformed_data_url", f"invalid base64 payload: {e}") from e
    if not raw:
        raise AudioInputError("empty_audio", "decoded audio payload is empty")
    return m.group("mime"), raw


def mono_downmix(data: np.ndarray) -> np.ndarray:
    """Mix multi-channel audio down to mono.

    Averaging keeps duplicated/panned stereo at a level consistent with a mono
    original (summing would inflate correlated content by up to +6 dB and push
    it past full scale); single-sided panned content is compensated afterwards
    by the reference-loudness stage which amplifies quiet signals back toward
    TARGET_RMS.
    """
    if data.ndim == 1:
        return data
    if data.shape[1] == 1:
        return data[:, 0]
    return data.mean(axis=1)


def normalize_intensity(x: np.ndarray) -> np.ndarray:
    """Reference-based loudness handling (not full normalization).

    Quiet audio is amplified toward TARGET_RMS; audio at or above the reference
    keeps its original level. Gain is capped so the peak never exceeds
    PEAK_CEILING.
    """
    rms = float(np.sqrt((x**2).mean()) + 1e-12)
    if rms >= TARGET_RMS:
        return x
    gain = min(TARGET_RMS / rms, PEAK_CEILING / max(float(np.abs(x).max()), 1e-9))
    return (x * gain).astype(np.float32)


def decode_audio(raw: bytes) -> tuple[np.ndarray, dict]:
    """Decode WAV bytes -> float32 mono at TARGET_SR + format metadata."""
    bio = io.BytesIO(raw)
    try:
        info = sf.info(bio)
    except Exception as e:  # noqa: BLE001
        raise AudioInputError("decode_error", f"not decodable as WAV: {e}") from e
    if info.duration > MAX_AUDIO_SECONDS:
        raise AudioInputError(
            "audio_too_long",
            f"audio section is {info.duration:.2f}s; the limit is {MAX_AUDIO_SECONDS:g}s "
            "(input is not silently truncated)",
        )
    bio.seek(0)
    try:
        data, sr = sf.read(bio, dtype="float32", always_2d=True)
    except Exception as e:  # noqa: BLE001
        raise AudioInputError("decode_error", f"WAV decode failed: {e}") from e

    meta = {"orig_sr": int(sr), "orig_channels": int(info.channels), "duration_s": float(info.duration)}

    if data.size == 0 or data.shape[0] == 0:
        raise AudioInputError("empty_audio", "audio has zero frames")
    if np.isnan(data).any() or np.isinf(data).any():
        raise AudioInputError("corrupted_audio", "audio contains NaN/Inf samples")

    data = mono_downmix(data)
    data = normalize_intensity(data.astype(np.float32))

    dur = len(data) / sr
    if dur < 0.02:
        raise AudioInputError("empty_audio", "audio shorter than 20 ms")

    if sr != TARGET_SR:
        data = _resample(data, sr, TARGET_SR)

    return data.astype(np.float32, copy=False), meta


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    try:
        import soxr

        return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)
    except ImportError:
        import torch
        import torchaudio

        t = torch.from_numpy(x).unsqueeze(0)
        t = torchaudio.functional.resample(t, sr_in, sr_out)
        return t.squeeze(0).numpy()


def analyze_quality(x: np.ndarray, sr: int = TARGET_SR) -> AudioQuality:
    """Clipping, silence, speech-ratio (energy/tonality VAD), SNR, music heuristic.

    VAD: frames above a noise-floor-relative threshold that are tonal
    (low spectral flatness) with dominant energy in the 80-4000 Hz speech band
    count as speech-like. Used only for gating low-information input.
    """
    n = len(x)
    frame, hop = int(0.02 * sr), int(0.01 * sr)
    if n < frame:
        x = np.concatenate([x, np.zeros(frame - n, dtype=np.float32)])
    frames = _stft_frames(x, frame, hop)
    frame_rms = np.sqrt((frames**2).sum(axis=1) + 1e-12)
    flatness = _spectral_flatness(frames)
    band_ratio = _speech_band_ratio(frames, sr)

    floor = np.percentile(frame_rms, 10) + 1e-8
    thresh = max(floor * 10 ** (3 / 20), 1e-5)
    voiced = (frame_rms > thresh) & (flatness < 0.35) & (band_ratio > 0.3)
    voiced = _hysteresis(voiced, on=2, off=5)
    speech_seconds = int(voiced.sum()) * hop / sr
    duration = n / sr
    speech_ratio = speech_seconds / max(duration, 1e-6)

    active_rms = frame_rms[voiced] if voiced.any() else np.zeros(1)
    active_db = 20 * np.log10(max(active_rms.mean(), 1e-9))
    floor_db = 20 * np.log10(max(floor, 1e-9))
    snr_db = float(active_db - floor_db)

    peak = float(np.abs(x).max())
    clip_ratio = float((np.abs(x) >= 0.999).mean())
    rms_db = 20 * np.log10(max(np.sqrt((x**2).mean()), 1e-9))

    tonal_ratio = float((flatness < 0.12).mean()) if len(flatness) else 0.0
    mod = _syllabic_modulation_energy(x, sr)
    music_probable = bool(tonal_ratio > 0.6 and mod < 0.05 and duration > 1.0)

    silent = bool(speech_ratio < 0.02 or duration < 0.05)
    clipped = clip_ratio >= CLIPPING_RATIO_WARN
    low_information = bool(silent or music_probable or speech_ratio < 0.10)

    return AudioQuality(
        duration_s=round(duration, 3),
        sample_rate_orig=int(sr),
        channels=1,
        peak=round(peak, 4),
        rms_db=round(float(rms_db), 2),
        noise_floor_db=round(float(floor_db), 2),
        snr_db=round(snr_db, 2),
        speech_seconds=round(float(speech_seconds), 3),
        speech_ratio=round(float(speech_ratio), 3),
        clipping_ratio=round(clip_ratio, 5),
        clipped=bool(clipped),
        silent=bool(silent),
        music_probable=music_probable,
        low_information=bool(low_information),
    )


def _stft_frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    nfft = 512 if frame <= 512 else 1024
    win = np.hanning(frame).astype(np.float32)
    n_frames = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    idx = np.clip(idx, 0, len(x) - 1)
    xf = x[idx] * win
    return np.abs(np.fft.rfft(xf, n=nfft, axis=1)).astype(np.float32)


def _spectral_flatness(frames: np.ndarray) -> np.ndarray:
    eps = 1e-12
    geo = np.exp(np.log(frames + eps).mean(axis=1))
    ari = frames.mean(axis=1) + eps
    return (geo / ari).astype(np.float32)


def _speech_band_ratio(frames: np.ndarray, sr: int) -> np.ndarray:
    freqs = np.fft.rfftfreq((frames.shape[1] - 1) * 2, 1 / sr)
    band = (freqs >= 80) & (freqs <= 4000)
    return (frames[:, band].sum(axis=1) / (frames.sum(axis=1) + 1e-12)).astype(np.float32)


def _hysteresis(mask: np.ndarray, on: int, off: int) -> np.ndarray:
    out = mask.astype(bool).copy()
    state, run = False, 0
    for i in range(len(out)):
        if mask[i]:
            run += 1
        else:
            run = 0 if not state else -1
        if not state and run >= on:
            state = True
        elif state and run <= -off:
            state = False
        out[i] = state
    return out


def _syllabic_modulation_energy(x: np.ndarray, sr: int) -> float:
    """Relative energy of amplitude modulation in the 3-8 Hz syllabic band."""
    env = np.abs(x)
    env = np.convolve(env, np.hanning(int(0.02 * sr) + 1) / max(1, int(0.02 * sr)), mode="same")
    env = env - env.mean()
    if len(env) < sr // 4:
        return 0.0
    m = np.fft.rfft(env * np.hanning(len(env)))
    freqs = np.fft.rfftfreq(len(env), 1 / sr)
    band = (freqs >= 3) & (freqs <= 8)
    total = (np.abs(m) ** 2).sum() + 1e-12
    return float((np.abs(m[band]) ** 2).sum() / total)


def load_wav_bytes(raw: bytes) -> tuple[np.ndarray, AudioQuality]:
    data, meta = decode_audio(raw)
    q = analyze_quality(data)
    q.sample_rate_orig = meta["orig_sr"]
    q.channels = meta["orig_channels"]
    return data, q
