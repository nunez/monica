"""Calibration: temperature scaling per output head, fitted on held-out
speaker-independent validation data. Reports ECE / Brier / RPS.

Choice additionally supports identity-keyed VECTOR scaling: a (gain, bias)
pair registered per exact option criterion TEXT (per-option logits are
conditioned on the option text, so calibration must be keyed by that text),
applied when an option's description matches a registered entry; other slots
fall back to plain temperature inside the same softmax.

`confidence` fields are distribution concentration (1 - H(p)/log K), NOT a
verified accuracy probability; calibration applies to probability values.
"""

from __future__ import annotations

import json
import math

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def temperature_scale(logits: np.ndarray, temp: float) -> np.ndarray:
    return softmax(logits / temp)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Fit a single temperature minimizing NLL (labels int)."""
    if grid is None:
        grid = np.exp(np.linspace(np.log(0.3), np.log(4.0), 60))
    best_t, best_nll = 1.0, np.inf
    labels = labels.astype(int)
    for t in grid:
        p = temperature_scale(logits, t)
        nll = -np.log(p[np.arange(len(labels)), labels] + 1e-12).mean()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error for multiclass probs (confidence = max prob)."""
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(labels)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def ece_binary(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    n = len(labels)
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.sum() / n * abs(labels[m].mean() - probs[m].mean())
    return float(e)


def brier_binary(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(((probs - labels.astype(float)) ** 2).mean())


def brier_multiclass(probs: np.ndarray, labels: np.ndarray) -> float:
    k = probs.shape[1]
    oh = np.zeros_like(probs)
    oh[np.arange(len(labels)), labels.astype(int)] = 1.0
    return float(((probs - oh) ** 2).sum(axis=-1).mean())


def rps(probs: np.ndarray, labels: np.ndarray) -> float:
    """Ranked probability score for ordered levels (lower is better)."""
    labels = labels.astype(int)
    cdf_p = probs.cumsum(axis=-1)
    cdf_y = (np.arange(probs.shape[1])[None, :] >= labels[:, None]).astype(float)
    return float(((cdf_p - cdf_y) ** 2).sum(axis=-1).mean())


def brier_ordinal_soft(probs: np.ndarray, soft_labels: np.ndarray) -> float:
    return float(((probs - soft_labels) ** 2).sum(axis=-1).mean())


class Calibrator:
    def __init__(self, temps: dict, choice_vector: dict | None = None):
        # temps: {"noul": t, "choice": t, "score": t}
        # choice_vector: {"entries": [{"emotion": e, "desc": text, "u": g, "b": b}, ...]}
        self.temps = temps
        self.choice_vector = self._validated_vector(choice_vector)

    @staticmethod
    def _validated_vector(cv):
        if not cv or "entries" not in cv:
            return None
        ents = [e for e in cv["entries"]
                if all(k in e for k in ("emotion", "desc", "u", "b")) and e.get("desc")]
        return {"entries": ents, "fit": cv.get("fit", {})} if ents else None

    @property
    def method(self) -> str:
        return "vector_scaling+temperature_fallback" if self.choice_vector else "temperature_scaling"

    def choice_slot_params(self, descs):
        """Per-slot (gain, bias) so probs = softmax(gain*l + bias).
        A slot matches a registered entry on exact normalized criterion text;
        unmatched slots keep 1/T with zero bias."""
        t = self.temps.get("choice", 1.0)
        g = np.full(len(descs), 1.0 / t)
        bi = np.zeros(len(descs))
        if not self.choice_vector:
            return g, bi
        by_desc = {str(e["desc"]).strip().lower(): e for e in self.choice_vector["entries"]}
        for j, desc in enumerate(descs):
            e = by_desc.get(str(desc).strip().lower()) if desc is not None else None
            if e is not None:
                g[j] = float(np.exp(e["u"]))
                bi[j] = float(e["b"])
        return g, bi

    def apply_choice(self, logits: np.ndarray, descs) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        g, bi = self.choice_slot_params(descs)
        x = logits * g + bi
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    def apply(self, kind: str, probs: np.ndarray) -> np.ndarray:
        return softmax(np.log(np.clip(probs, 1e-9, 1.0)) / self.temps.get(kind, 1.0))

    @classmethod
    def fit(cls, data: dict) -> tuple["Calibrator", dict]:
        """data: {kind: {"logits": np [N,K], "labels": np [N]}}."""
        temps, metrics = {}, {}
        for kind, d in data.items():
            t = fit_temperature(d["logits"], d["labels"])
            temps[kind] = t
            raw_p = softmax(d["logits"])
            cal_p = softmax(d["logits"] / t)
            lab = d["labels"].astype(int)
            if kind == "noul":
                metrics[kind] = {
                    "ece_raw": ece(np.stack([1 - raw_p[:, 1], raw_p[:, 1]], -1), lab),
                    "ece_calibrated": ece(np.stack([1 - cal_p[:, 1], cal_p[:, 1]], -1), lab),
                    "brier_raw": brier_binary(raw_p[:, 1], lab),
                    "brier_calibrated": brier_binary(cal_p[:, 1], lab),
                    "acc": float(((cal_p[:, 1] >= 0.5).astype(int) == lab).mean()),
                }
            else:
                m = {"ece_raw": ece(raw_p, lab), "ece_calibrated": ece(cal_p, lab)}
                if kind == "score":
                    m["rps_raw"] = rps(raw_p, lab)
                    m["rps_calibrated"] = rps(cal_p, lab)
                    m["level_mae"] = float(np.abs(cal_p.argmax(-1) - lab).mean())
                else:
                    m["brier_raw"] = brier_multiclass(raw_p, lab)
                    m["brier_calibrated"] = brier_multiclass(cal_p, lab)
                    m["acc"] = float((cal_p.argmax(-1) == lab).mean())
                metrics[kind] = m
        return cls(temps), metrics

    def save(self, path: str):
        out = {"method": self.method, "temperatures": self.temps}
        if self.choice_vector:
            out["choice_vector"] = self.choice_vector
        json.dump(out, open(path, "w"), indent=2)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        d = json.load(open(path))
        return cls(d["temperatures"], d.get("choice_vector"))
