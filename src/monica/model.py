"""MonicaModel: encode audio once per request; score all typed questions from
the shared encoding. Non-generative end to end.

Response shape is the Monica System One answer structure. `usage` carries
latency breakdown + audio quality flags (extension fields); `calibration`
declares whether returned probabilities are calibrated.
"""

from __future__ import annotations

import hashlib
import math
import os
import time

MAX_FILE_BYTES = int(os.environ.get("MONICA_MAX_FILE_BYTES", str(256 * 1024 * 1024)))
from dataclasses import dataclass, field

import numpy as np
import torch

from . import pooling
from .audioio import AudioInputError, AudioQuality, analyze_quality, decode_audio, parse_data_url
from .calibration import Calibrator
from .encoder import GemmaAudioEncoder
from .heads import QuestionHeads
from .qencoder import GemmaEmbeddingTable, TextEncoder, make_tokenizer

MAX_LEVELS = 16


@dataclass
class MonicaConfig:
    model_dir: str
    ckpt: str
    device: str = "mps"
    layer_ids: tuple = (2, 5, 8, 11)
    n_windows: int = 20
    d_q: int = 512
    d_h: int = 512
    max_questions: int = 64
    low_info_min_speech_s: float = 0.5
    low_info_min_ratio: float = 0.10
    severe_clip_ratio: float = 0.05
    calibration_path: str | None = None
    snr_conf_caps: tuple = ((8.0, 0.35), (16.0, 0.6))  # (snr_db threshold, confidence cap)


class MonicaModel:
    def __init__(self, cfg: MonicaConfig, training: bool = False):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.encoder = GemmaAudioEncoder(cfg.model_dir, device=cfg.device)
        self.tokenizer = make_tokenizer(cfg.model_dir)
        self.emb_table = GemmaEmbeddingTable(cfg.model_dir, device=cfg.device)
        self.text_encoder = TextEncoder(emb_dim=self.emb_table.dim, out_dim=cfg.d_q).to(self.device)
        # pooled A = per-window [layer means ... final mean | layer stds ... final std]
        self.d_a = 2 * (len(cfg.layer_ids) * self.encoder.hidden + self.encoder.out_dim)
        # head stats vector = 8-dim normalized audio quality summary
        self.stats_dim = 7
        self.heads = QuestionHeads(self.d_a, cfg.d_q, cfg.d_h, self.stats_dim).to(self.device)
        if cfg.calibration_path:
            self.calibrator = Calibrator.load(cfg.calibration_path)
        else:
            self.calibrator = Calibrator({"noul": 1.0, "choice": 1.0, "score": 1.0})
            self._uncalibrated = True
        self._uncalibrated = cfg.calibration_path is None
        if not training:
            self.load_head_ckpt(cfg.ckpt)
            self.eval()
        self.n_audio_encodes = 0
        self.training_mode = training
        from collections import OrderedDict

        self._txt_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self._txt_cache_max = 8192

        n_train = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        n_train += sum(p.numel() for p in self.heads.parameters() if p.requires_grad)
        self.trainable_params = n_train

    def load_head_ckpt(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.text_encoder.load_state_dict(ck["text_encoder"])
        self.heads.load_state_dict(ck["heads"])
        self.stats_mean = ck["stats_mean"].detach().clone().float().to(self.device)
        self.stats_std = ck["stats_std"].detach().clone().float().to(self.device).clamp(min=1e-6)
        if "calibration" in ck:
            self.calibrator = Calibrator(ck["calibration"])
            self._uncalibrated = False

    def eval(self):
        self.text_encoder.eval()
        self.heads.eval()
        return self

    # ------------------------------------------------------------------ text
    def encode_texts(self, strings: list[str], max_len: int = 96) -> torch.Tensor:
        ids = self.tokenizer(strings, return_tensors="pt", padding="max_length",
                             truncation=True, max_length=max_len)["input_ids"].to(self.device)
        with torch.no_grad():
            return self.text_encoder(ids, self.emb_table)

    def encode_texts_cached(self, strings: list[str], max_len: int = 96) -> torch.Tensor:
        """Reusable question/criterion text representation cache (content-hash keyed)."""
        keys = [hashlib.sha256(f"{max_len}|{s}".encode()).hexdigest() for s in strings]
        miss_i = [i for i, k in enumerate(keys) if k not in self._txt_cache]
        if miss_i:
            vecs = self.encode_texts([strings[i] for i in miss_i], max_len=max_len).float().cpu()
            for i, v in zip(miss_i, vecs):
                self._txt_cache[keys[i]] = v
                while len(self._txt_cache) > self._txt_cache_max:
                    self._txt_cache.popitem(last=False)
        return torch.stack([self._txt_cache[k].float() for k in keys])

    def warmup(self):
        rng = np.random.default_rng(0)
        wav = (rng.standard_normal(int(1.0 * 16000)) * 0.05).astype(np.float32)
        self.audio_encode(wav)
        self.encode_texts_cached(["Does the speaker sound happy?", "warm"])
        self.encode_texts_cached(["neutral", "happy", "warm"])
        torch.mps.synchronize() if self.device.type == "mps" else None

    # ------------------------------------------------------------------ audio
    def audio_encode(self, wav: np.ndarray) -> dict:
        enc = self.encoder.encode(wav)
        pooled = pooling.pool_encoding(enc, self.cfg.layer_ids, self.cfg.n_windows)
        self.n_audio_encodes += 1
        return pooled

    def stats_vec(self, quality: AudioQuality) -> torch.Tensor:
        return torch.tensor([
            min(quality.duration_s, 30.0) / 30.0, quality.speech_ratio,
            min(quality.speech_seconds, 30.0) / 30.0, (quality.rms_db + 60) / 60.0,
            max(0.0, min(quality.snr_db, 40.0)) / 40.0, quality.clipping_ratio,
            quality.peak,
        ], dtype=torch.float32)

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score(self, pooled: dict, questions: dict) -> dict:
        """questions: name -> validated spec dict. Returns answers dict (raw probs)."""
        names = list(questions)
        if not names:
            return {}
        instrs = [questions[n]["instructions"] for n in names]
        crits = [questions[n].get("criteria") for n in names]

        # batch-encode all texts with caching (cache lives outside per-request)
        all_strings = instrs + [c for cl in crits if isinstance(cl, dict) for c in cl.values()] + \
                    [c for cl in crits if isinstance(cl, list) for c in cl]
        if getattr(self, "training_mode", False):
            u_all = self.encode_texts(all_strings)
        else:
            u_all = self.encode_texts_cached(all_strings).to(self.device)
        p = 0
        U = u_all[p:p + len(names)]; p += len(names)
        opts: list[torch.Tensor | None] = []
        for cl in crits:
            if cl is None:
                opts.append(None)
            elif isinstance(cl, dict):
                k = len(cl)
                opts.append(u_all[p:p + k].unsqueeze(0)); p += k
            else:
                k = len(cl)
                opts.append(u_all[p:p + k].unsqueeze(0)); p += k

        A = (pooled["A"].float().to(self.device) - self.stats_mean[: self.d_a]) / self.stats_std[: self.d_a]
        valid = pooled["valid"].to(self.device)
        stats = ((self.stats_vec(pooled["_quality"]).to(self.device) - self.stats_mean[-7:]) / self.stats_std[-7:])

        z = self.heads.pool(U, *self.heads.pool.kv(A), valid=valid)
        z = z + self.heads.stats_proj(stats).unsqueeze(0)

        answers: dict[str, dict] = {}
        for i, n in enumerate(names):
            spec = questions[n]
            t = spec["type"]
            if t == "noul":
                u = U[i] + self.heads.noul_opt
                uo = u.reshape(1, 1, -1)
                logits = self.heads.forward_pooled(z[i:i + 1], stats, uo)[0]
                answers[n] = {"type": "noul", "_logit": float(logits[0])}
            elif t == "choice":
                uo = opts[i][0].reshape(1, -1, self.cfg.d_q)
                logits = self.heads.forward_pooled(z[i:i + 1], stats, uo)[0]
                answers[n] = {"type": "choice", "_logits": logits.cpu().numpy(),
                             "keys": list(spec["criteria"])}
            elif t == "score":
                uo = opts[i][0].reshape(1, -1, self.cfg.d_q)
                k = uo.shape[1]
                logits = self.heads.forward_pooled(z[i:i + 1], stats, uo)[0] + self.heads.level_bias[:k]
                answers[n] = {"type": "score", "_logits": logits.cpu().numpy(),
                             "criteria": spec["criteria"]}
        return answers

    # ------------------------------------------------------------- response build
    def _cap_conf(self, conf: float, quality: AudioQuality) -> float:
        caps = {t: c for t, c in self.cfg.snr_conf_caps}
        for thr in sorted(caps):
            if quality.snr_db < thr:
                conf = min(conf, caps[thr])
        if quality.clipped:
            conf = min(conf, 0.7)
        return conf

    def build_response(self, raw: dict, questions: dict, quality: AudioQuality,
                       low_information: bool, timings: dict, model_id: str = "monica/gemma4-e2b-systemone-v1") -> dict:
        answers = {}
        for name, a in raw.items():
            if a["type"] == "noul":
                if low_information:
                    p = 0.5
                else:
                    t = self.calibrator.temps.get("noul", 1.0)
                    x = max(-40.0, min(40.0, a["_logit"] / t))
                    p = 1.0 / (1.0 + math.exp(-x))
                answers[name] = {"type": "noul", "noul": round(p, 4)}
            elif a["type"] == "choice":
                probs = self._calib_probs(a["_logits"], "choice", low_information)
                probs = _round_probs(probs)
                k = len(probs)
                conf = 0.0 if low_information else round(self._cap_conf(_negentropy(probs), quality), 4)
                answers[name] = {
                    "type": "choice",
                    "choice": a["keys"][int(np.argmax(probs))],
                    "confidence": conf,
                    "probabilities": {key: float(v) for key, v in zip(a["keys"], probs)},
                }
            elif a["type"] == "score":
                probs = self._calib_probs(a["_logits"], "score", low_information)
                probs = _round_probs(probs)
                levels = len(a["criteria"])
                score = round(float(np.sum(probs * np.arange(levels))), 2)
                conf = 0.0 if low_information else round(self._cap_conf(_negentropy(probs), quality), 4)
                answers[name] = {
                    "type": "score",
                    "score": score,
                    "confidence": conf,
                    "legend": {str(i): a["criteria"][i] for i in range(levels)},
                    "probabilities": {str(i): float(probs[i]) for i in range(levels)},
                }
        confs = [abs(v["noul"] * 2 - 1) for v in answers.values() if v.get("type") == "noul"]
        confs += [v["confidence"] for v in answers.values() if v.get("type") in ("choice", "score")]
        resp = {
            "model": model_id,
            "confidence": round(max(confs), 4) if confs else 0.0,
            "answers": answers,
            "audio": self._audio_block(quality),
            "usage": {
                "audio_seconds": quality.duration_s,
                "speech_seconds": quality.speech_seconds,
                "audio_encoder_calls": timings.get("n_audio_encodes", 1),
                "generated_tokens": 0,
                "latency_ms": int(timings.get("total_ms", 0)),
                "n_questions": len(questions),
                "low_information": low_information,
                "clipped": quality.clipped,
                "music_probable": quality.music_probable,
            },
            "calibration": {
                "status": "uncalibrated" if self._uncalibrated else "calibrated",
                "method": "none" if self._uncalibrated else "temperature_scaling",
            },
        }
        return resp

    def _calib_probs(self, logits: np.ndarray, kind: str, low_info: bool) -> np.ndarray:
        if low_info:
            k = len(logits)
            return np.full(k, 1.0 / k)
        t = self.calibrator.temps.get(kind, 1.0)
        e = np.exp((logits - logits.max()) / t)
        return e / e.sum()

    # ---------------------------------------------------------------- request
    def _audio_block(self, q: AudioQuality) -> dict:
        return {
            "duration_s": q.duration_s,
            "channels": q.channels,
            "sample_rate_orig": q.sample_rate_orig,
            "peak": q.peak,
            "rms_db": q.rms_db,
            "noise_floor_db": q.noise_floor_db,
            "snr_db": q.snr_db,
            "speech_seconds": q.speech_seconds,
            "speech_ratio": q.speech_ratio,
            "clipping_ratio": q.clipping_ratio,
            "clipped": q.clipped,
            "music_probable": q.music_probable,
            "low_information": q.low_information,
        }

    def handle(self, request: dict) -> tuple[dict, AudioQuality]:
        t0 = time.perf_counter()
        state = request.get("state") or {}
        audio = state.get("audio")
        if not audio or not isinstance(audio, str):
            raise AudioInputError("empty_audio", "state.audio missing or empty")
        if audio.startswith("file://") or (os.path.sep in audio and not audio.startswith("data:")):
            path = audio[len("file://"):] if audio.startswith("file://") else audio
            if not os.path.isfile(path):
                raise AudioInputError("file_not_found", f"audio file not found: {path}")
            size = os.path.getsize(path)
            if size > MAX_FILE_BYTES:
                raise AudioInputError(
                    "audio_too_large",
                    f"audio file is {size} bytes; the limit is {MAX_FILE_BYTES} bytes "
                    "(input is not silently truncated)",
                )
            with open(path, "rb") as f:
                raw = f.read()
            if not raw:
                raise AudioInputError("empty_audio", "audio file is empty")
        else:
            _, raw = parse_data_url(audio)
        wav, meta = decode_audio(raw)
        quality = analyze_quality(wav)
        quality.sample_rate_orig = meta["orig_sr"]
        quality.channels = meta["orig_channels"]

        if not request["questions"]:
            resp = self.build_response({}, request["questions"], quality, True,
                                       {"total_ms": (time.perf_counter() - t0) * 1000},
                                       model_id=request.get("model") or "monica/gemma4-e2b-systemone-v1")
            resp["usage"]["audio_encoder_calls"] = 0
            resp["usage"]["encode_ms"] = 0
            resp["usage"]["score_ms"] = 0
            resp["usage"]["low_information"] = False
            return resp, quality

        low_info = bool(
            quality.silent or quality.music_probable
            or quality.speech_seconds < self.cfg.low_info_min_speech_s
            or quality.speech_ratio < self.cfg.low_info_min_ratio
            or quality.clipping_ratio >= self.cfg.severe_clip_ratio
        )
        enc0 = self.n_audio_encodes
        t_enc0 = time.perf_counter()
        pooled = None
        if not low_info:
            pooled = self.audio_encode(wav)
            pooled["_quality"] = quality
        t_enc1 = time.perf_counter()
        if low_info:
            raw_answers = {
                n: {"type": spec["type"], "_logit": 0.0,
                   "_logits": np.zeros(len(spec.get("criteria") or [0])),
                   "keys": list(spec.get("criteria") or {}),
                   "criteria": spec.get("criteria")}
                for n, spec in request["questions"].items()
            }
        else:
            raw_answers = self.score(pooled, request["questions"])
        t1 = time.perf_counter()
        resp = self.build_response(raw_answers, request["questions"], quality, low_info,
                                    {"total_ms": (t1 - t0) * 1000,
                                     "n_audio_encodes": self.n_audio_encodes - enc0,
                                     "encode_ms": (t_enc1 - t_enc0) * 1000,
                                     "score_ms": (t1 - t_enc1) * 1000},
                                    model_id=request.get("model") or "monica/gemma4-e2b-systemone-v1")
        resp["usage"]["encode_ms"] = int((t_enc1 - t_enc0) * 1000)
        resp["usage"]["score_ms"] = int((t1 - t_enc1) * 1000)
        return resp, quality


def _round_probs(p: np.ndarray, decimals: int = 4) -> np.ndarray:
    r = np.round(np.asarray(p, dtype=float), decimals)
    diff = round(1.0 - float(r.sum()), decimals + 4)
    r[int(np.argmax(r))] = round(float(r[int(np.argmax(r))] + diff), decimals)
    return np.clip(r, 0.0, 1.0)


def _negentropy(p: np.ndarray) -> float:
    k = len(p)
    if k <= 1:
        return 0.0
    e = -(np.clip(p, 1e-9, None) * np.log(np.clip(p, 1e-9, None))).sum()
    return float(np.clip(1.0 - e / math.log(k), 0.0, 1.0))
