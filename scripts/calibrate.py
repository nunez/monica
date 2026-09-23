#!/usr/bin/env python
"""Fit temperature-scaling calibration on held-out VAL speakers for each
question type. Saves models/calibration.json + a report with ECE/Brier before
and after."""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monica.battery_eval import run_clip, make_pooled, build_noul_questions  # noqa: E402
from monica.calibration import Calibrator  # noqa: E402
from monica.model import MonicaModel, MonicaConfig  # noqa: E402
from monica.audioio import AudioQuality  # noqa: E402
from monica import pooling  # noqa: E402

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sadness"]


def stub_quality(q8):
    q = q8.tolist()
    return AudioQuality(duration_s=q[0] * 30, sample_rate_orig=16000, channels=max(1, round(q[7] * 2)),
                        peak=q[6], rms_db=q[3] * 60 - 60, noise_floor_db=-60, snr_db=q[4] * 40,
                        speech_seconds=q[2] * 30, speech_ratio=q[1], clipping_ratio=q[5],
                        clipped=q[5] >= 0.02, silent=False, low_information=False, music_probable=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/models/gemma-4-E2B-it")
    ap.add_argument("--ckpt", default="models/head-cremad.pt")
    ap.add_argument("--features", nargs="+", default=["data/features/cremad.pt"])
    ap.add_argument("--battery", default="data/battery")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="models/calibration.json")
    ap.add_argument("--report", default="data/calibration_report.json")
    ap.add_argument("--max-clips", type=int, default=400)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    layer_ids = tuple(ck["meta"]["layer_ids"])
    cfg = MonicaConfig(model_dir=args.model_dir, ckpt=args.ckpt, device=args.device, layer_ids=layer_ids)
    model = MonicaModel(cfg)  # temp=1 (uncalibrated) — we fit from raw logits
    model.load_head_ckpt(args.ckpt)

    import glob as _glob
    fin = args.features if isinstance(args.features, (list, tuple)) else args.features.replace(",", " ").split()
    paths = []
    for p0 in fin:
        paths.extend(sorted(_glob.glob(p0)) or [p0])
    merged = {}
    for p in paths:
        for k, v in torch.load(p).items():
            merged.setdefault(k, []).extend(v)
    val = merged["val"]  # list of per-clip records
    n = min(len(val), args.max_clips)

    noul_items = json.load(open(f"{args.battery}/noul_train.json"))
    panels = json.load(open(f"{args.battery}/choice_train.json"))
    scales = json.load(open(f"{args.battery}/score_scales.json"))["train"]

    noul_qs = build_noul_questions(noul_items)
    choice_qs = {f"c{i}": {"type": "choice", "instructions": "What primary sentiment is expressed by the speaker's voice?",
                 "criteria": {k: pl["descriptions"][k] for k in pl["panel"]}} for i, pl in enumerate(panels)}
    score_qs = {f"s{i}": {"type": "score", "instructions": f"How {w} does the speaker sound?", "criteria": sc["levels"]}
                for i, (w, sc) in enumerate(scales.items())}

    noul_logits, noul_labels = [], []
    ch_logits, ch_labels = [], []
    sc_logits, sc_labels = [], []

    for i in range(n):
        rec = val[i]
        pooled = make_pooled(model, rec)
        emo = rec["emotion"]
        if emo not in EMOTIONS:
            continue
        e = EMOTIONS.index(emo)
        ans = model.score(pooled, {**noul_qs, **choice_qs, **score_qs})
        for k, it in zip(noul_qs, noul_items):
            noul_logits.append([ -ans[k]["_logit"], ans[k]["_logit"]])
            noul_labels.append(1 if emo in it["true_emotions"] else 0)
        for k, pl in zip(choice_qs, panels):
            pos = pl["panel"]
            if emo in pos and len(pos) == 6:
                ch_logits.append(np.asarray(ans[k]["_logits"], dtype=np.float64))
                ch_labels.append(pos.index(emo))
        for k, (w, sc) in zip(score_qs, scales.items()):
            center = sc["map"].get(emo, 2.0)
            lv = min(max(int(round(center)), 0), len(sc["levels"]) - 1)
            sc_logits.append(np.asarray(ans[k]["_logits"], dtype=np.float64))
            sc_labels.append(lv)

    data = {
        "noul": {"logits": np.array(noul_logits), "labels": np.array(noul_labels)},
        "choice": {"logits": np.array(ch_logits), "labels": np.array(ch_labels)},
        "score": {"logits": np.array(sc_logits), "labels": np.array(sc_labels)},
    }
    cal, metrics = Calibrator.fit(data)
    cal.save(args.out)
    json.dump({"temperatures": cal.temps, "metrics": metrics, "n_val_clips": n}, open(args.report, "w"), indent=2)
    print(json.dumps(metrics, indent=2))


def val_recs(v):
    for i in range(len(v["ids"])):
        yield {"id": v["ids"][i], "emotion": v["emotion"][i]}


if __name__ == "__main__":
    main()
