#!/usr/bin/env python
"""Fit identity-keyed vector scaling per option criterion TEXT and merge into
models/calibration.json.

For each distinct battery panel text set (size == len(EMOTIONS)), extract val
canonical logits per option text, run speaker-grouped CV over L2, refit on
full val, register one (u, b) entry per option text. Test is never consulted.
Backs up the previous calibration.json.

Run:  PYTHONPATH=src .venv/bin/python scripts/upgrade_calibration.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from monica.battery_eval import make_pooled  # noqa: E402
from monica.calibration import Calibrator, fit_temperature  # noqa: E402
from monica.model import MonicaModel, MonicaConfig  # noqa: E402
from vector_calibrate import EMOTIONS, fit_linear, apply_linear, nll  # noqa: E402


def extract_panel(model, recs, panel, descs):
    qs = {"c0": {"type": "choice",
                 "instructions": "What primary sentiment is expressed by the speaker's voice?",
                 "criteria": {k: descs[k] for k in panel}}}
    X, y, ids = [], [], []
    for r in recs:
        if r.get("emotion") not in panel:
            continue
        ans = model.score(make_pooled(model, r), qs)
        X.append(np.asarray(ans["c0"]["_logits"], dtype=np.float64))
        y.append(EMOTIONS.index(r["emotion"]))
        ids.append(r["id"])
    return np.stack(X), np.array(y), ids


def main():
    ck = torch.load("models/head-cremad.pt", map_location="cpu")
    cfg = MonicaConfig(model_dir="data/models/gemma-4-E2B-it", ckpt="models/head-cremad.pt",
                       device="cpu", layer_ids=tuple(ck["meta"]["layer_ids"]))
    model = MonicaModel(cfg)
    model.load_head_ckpt("models/head-cremad.pt")

    feats = {}
    for p in sorted(glob.glob("data/features/cremad_*.pt")):
        for k, v in torch.load(p, weights_only=False).items():
            feats.setdefault(k, []).extend(v)

    seen_descs, panels = set(), []
    for pl in json.load(open("data/battery/choice_train.json")):
        if len(pl["panel"]) != len(EMOTIONS):
            continue
        key = tuple(sorted(pl["descriptions"][e].strip().lower() for e in pl["panel"]))
        if key in seen_descs:
            continue
        seen_descs.add(key)
        panels.append(pl)

    l2_grid = [1e-5, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1]
    entries, fitinfo = [], []
    for pl in panels:
        X, y, ids = extract_panel(model, feats["val"], pl["panel"], pl["descriptions"])
        spk = np.array([cid.split("_")[0] for cid in ids])
        uspk = sorted(set(spk.tolist()))
        folds = [[] for _ in range(4)]
        for i, s in enumerate(spk):
            folds[uspk.index(s) % 4].append(i)
        cv = {}
        for l2 in l2_grid:
            vals = []
            for f in range(4):
                te = np.array(folds[f]); tr = np.setdiff1d(np.arange(len(X)), te)
                cal = fit_linear(X[tr], y[tr], len(pl["panel"]), mode="vector", l2=l2)
                vals.append(nll(apply_linear(cal, X[te]), y[te]))
            cv[l2] = round(float(np.mean(vals)), 5)
        best_l2 = min(cv, key=cv.get)
        cal = fit_linear(X, y, len(pl["panel"]), mode="vector", l2=best_l2)
        for e, u, b in zip(pl["panel"], cal["a"], cal["b"]):
            entries.append({"emotion": e, "desc": pl["descriptions"][e], "u": float(u), "b": float(b)})
        fitinfo.append({"tag": pl.get("tag"), "panel": pl["panel"], "l2": best_l2, "cv_nll": cv,
                        "n_val_windows": int(len(X))})
        print(pl.get("tag"), "l2", best_l2, "cv", cv, flush=True)

    old = json.load(open("models/calibration.json.bak")) if os.path.exists("models/calibration.json.bak") \
        else json.load(open("models/calibration.json"))
    if not os.path.exists("models/calibration.json.bak"):
        json.dump(old, open("models/calibration.json.bak", "w"), indent=2)

    calib = Calibrator(old["temperatures"], {"entries": entries, "fit": {"grid": fitinfo}})
    calib.save("models/calibration.json")
    print(json.dumps({"n_entries": len(entries),
                      "entries": [{"emotion": e["emotion"], "u": round(e["u"], 3), "b": round(e["b"], 3)}
                                  for e in entries]}, indent=2))


if __name__ == "__main__":
    main()
