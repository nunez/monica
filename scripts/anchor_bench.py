#!/usr/bin/env python
"""Bench EmbeddingGemma 2 anchor cosines against the monica choice head on the
same CREMA-D speaker-independent clips.

Compares three systems on identical clips:
  1. anchor      — EmbeddingGemma 2 zero-shot cosine vs emotion anchors
  2. head        — monica choice head (trained on CREMA-D train split)
  3. stack       — logistic stack of [head logits, anchor cosines] (fit on val)

Run in main venv (needs embeddinggemma_cos.pt from scripts/embedgemma_anchor.py):
    PYTHONPATH=src .venv/bin/python scripts/anchor_bench.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monica.battery_eval import make_pooled  # noqa: E402
from monica.model import MonicaModel, MonicaConfig  # noqa: E402
from monica.metrics import accuracy, macro_f1, ece, brier_multiclass  # noqa: E402

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sadness"]
EIDX = {e: i for i, e in enumerate(EMOTIONS)}
CRIT_DESC = {
    "angry": "Anger or hostility is expressed", "disgust": "Disgust or revulsion is expressed",
    "fear": "Fear or apprehension is expressed", "happy": "Happiness or enjoyment is expressed",
    "neutral": "No strong emotion is expressed", "sadness": "Sadness or dejection is expressed",
}


def softmax(x, t=1.0):
    e = np.exp((x - x.max(-1, keepdims=True)) / t)
    return e / e.sum(-1, keepdims=True)


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log((a + 1e-12) / (b + 1e-12)), -1).mean())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def fit_temp(logits, labels, lo=0.1, hi=20.0):
    def nll(t):
        return float(-np.log(softmax(logits, t)[np.arange(len(labels)), labels] + 1e-12).mean())
    best, bt = 1e9, 1.0
    for t in np.geomspace(lo, hi, 200):
        v = nll(t)
        if v < best:
            best, bt = v, float(t)
    return bt


def block(name, P, L, K=6):
    pred = P.argmax(-1)
    per = {EMOTIONS[c]: round(float(
        ((pred == c) & (L == c)).sum() / max((pred == c).sum(), 1)), 3) for c in range(K)}
    return {"n": int(len(L)), "acc": round(accuracy(pred, L), 4),
            "macro_f1": round(macro_f1(pred, L, K), 4),
            "ece": round(ece(P, L), 4), "brier": round(brier_multiclass(P, L), 4),
            "per_class_precision": per}


def stack_fit(Xtr, ytr, K=6, steps=800, lr=0.05, wd=1e-4, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.zeros(Xtr.shape[1], K, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(Xt @ W + b, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach().numpy(), b.detach().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cos", default="bench/embedgemma_cos.pt")
    ap.add_argument("--model-dir", default="data/models/gemma-4-E2B-it")
    ap.add_argument("--ckpt", default="models/head-cremad.pt")
    ap.add_argument("--calibration", default="models/calibration.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="bench/anchor_bench.json")
    args = ap.parse_args()

    cos = torch.load(args.cos, weights_only=False)
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = MonicaConfig(model_dir=args.model_dir, ckpt=args.ckpt, device=args.device,
                       layer_ids=tuple(ck["meta"]["layer_ids"]),
                       calibration_path=args.calibration if os.path.exists(args.calibration) else None)
    model = MonicaModel(cfg)
    model.load_head_ckpt(args.ckpt)
    temp_choice = model.calibrator.temps.get("choice", 1.0)

    import glob as _glob
    feats = {}
    for p in sorted(_glob.glob("data/features/cremad_*.pt")):
        for k, v in torch.load(p, weights_only=False).items():
            feats.setdefault(k, []).extend(v)
    by_id = {r["id"]: r for r in feats["train"] + feats["val"] + feats["test"]}

    qs = {"c0": {"type": "choice",
                 "instructions": "What primary sentiment is expressed by the speaker's voice?",
                 "criteria": {k: CRIT_DESC[k] for k in EMOTIONS}}}

    res = {}
    head_logits, anchor_state = {}, {}
    for sp in ("val", "test"):
        ids, emos = cos[sp]["ids"], cos[sp]["emotions"]
        hl, keep = [], []
        for i, cid in enumerate(ids):
            r = by_id.get(cid)
            if r is None or r["emotion"] not in EIDX:
                continue
            pooled = make_pooled(model, r)
            ans = model.score(pooled, qs)
            hl.append(np.asarray(ans["c0"]["_logits"], dtype=np.float64))
            keep.append(i)
        hl = np.stack(hl)
        L = np.array([EIDX[emos[i]] for i in keep])
        cosd = cos[sp]["cos_desc"][keep].numpy()
        coss = cos[sp]["cos_simple"][keep].numpy()
        head_logits[sp], anchor_state[sp] = (hl, L), (cosd, coss)

    # temperature for anchors fitted on val NLL
    L_val, L_test = head_logits["val"][1], head_logits["test"][1]
    t_desc = fit_temp(anchor_state["val"][0], L_val)
    t_simp = fit_temp(anchor_state["val"][1], L_val)

    P_head_val = softmax(head_logits["val"][0], temp_choice)

    res["anchor_desc_raw_t10"] = block("a", softmax(anchor_state["test"][0], 10.0), L_test)
    res["anchor_desc_valfit"] = {"tau": round(t_desc, 3),
                                  **block("a", softmax(anchor_state["test"][0], t_desc), L_test)}
    res["anchor_simple_valfit"] = {"tau": round(t_simp, 3),
                                    **block("a", softmax(anchor_state["test"][1], t_simp), L_test)}
    res["head"] = block("h", softmax(head_logits["test"][0], temp_choice), L_test)
    res["head_val"] = block("h", P_head_val, L_val)

    # agreement
    P_a = softmax(anchor_state["test"][0], t_desc)
    P_h = softmax(head_logits["test"][0], temp_choice)
    res["agreement"] = {"js": round(js(P_a, P_h), 4),
                       "mean_max_prob_delta": round(float(np.abs(P_a - P_h).max(-1).mean()), 4),
                       "argmax_agreement": round(float((P_a.argmax(-1) == P_h.argmax(-1)).mean()), 4)}
    corr = {}
    for c, e in enumerate(EMOTIONS):
        x, y = anchor_state["test"][0][:, c], head_logits["test"][0][:, c]
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        corr[e] = round(float(np.corrcoef(rx, ry)[0, 1]), 3)
    res["per_emotion_spearman_cos_vs_logit"] = corr

    # stacks (fit on val, test on test)
    for name, Xf in (("stack_head_only", lambda hl, cd, cs: hl),
                     ("stack_anchor_only", lambda hl, cd, cs: cd),
                     ("stack_both", lambda hl, cd, cs: np.concatenate([hl, cd], -1))):
        Xtr = np.concatenate([Xf(head_logits["val"][0], anchor_state["val"][0], anchor_state["val"][1]),
                              np.ones((len(L_val), 1))], -1)
        Xte = np.concatenate([Xf(head_logits["test"][0], anchor_state["test"][0], anchor_state["test"][1]),
                              np.ones((len(L_test), 1))], -1)
        W, b = stack_fit(Xtr, L_val)
        Pte = softmax(Xte @ W + b, 1.0)
        res[name] = block("s", Pte, L_test)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    for k, v in res.items():
        print(k, json.dumps(v, default=str))


if __name__ == "__main__":
    main()
