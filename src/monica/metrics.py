"""Evaluation metrics: accuracy/F1, stability (packing / permutation), latency
stats, plus re-exported calibration metrics."""

from __future__ import annotations

import numpy as np

from .calibration import ece, ece_binary, brier_binary, brier_multiclass, rps  # noqa: F401


def accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    return float((pred == gold).mean())


def macro_f1(pred: np.ndarray, gold: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = ((pred == c) & (gold == c)).sum()
        fp = ((pred == c) & (gold != c)).sum()
        fn = ((pred != c) & (gold == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC with tie handling
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    _, counts = np.unique(np.concatenate([pos, neg]), return_counts=True)
    # average ranks for ties
    svals = np.concatenate([pos, neg])
    sorted_v = np.sort(svals)
    for v, c in zip(*np.unique(svals, return_counts=True)):
        idx = np.where(sorted_v == v)[0]
        avg = (idx.min() + idx.max()) / 2 + 1
        mask = svals == v
        ranks[mask] = avg
    rpos = ranks[: len(pos)].sum()
    n1, n0 = len(pos), len(neg)
    return float((rpos - n1 * (n1 + 1) / 2) / (n1 * n0))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * np.log(a / b)).sum(axis=-1)  # noqa: E731
    return float(0.5 * (kl(p, m) + kl(q, m)).mean())


def stability_report(packed_probs: np.ndarray, separate_probs: np.ndarray) -> dict:
    """Compare probability distributions from packed vs separate evaluation."""
    d = np.abs(packed_probs - separate_probs)
    return {
        "max_prob_delta": float(d.max()),
        "mean_prob_delta": float(d.mean()),
        "argmax_agreement": float((packed_probs.argmax(-1) == separate_probs.argmax(-1)).mean()),
        "js_divergence": js_divergence(packed_probs, separate_probs),
    }


def latency_stats(times_ms: list[float]) -> dict:
    a = np.array(times_ms, dtype=float)
    return {
        "n": int(len(a)),
        "mean_ms": round(float(a.mean()), 1),
        "median_ms": round(float(np.median(a)), 1),
        "p90_ms": round(float(np.percentile(a, 90)), 1),
        "p99_ms": round(float(np.percentile(a, 99)), 1),
        "min_ms": round(float(a.min()), 1),
        "max_ms": round(float(a.max()), 1),
    }
