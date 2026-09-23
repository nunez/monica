#!/usr/bin/env python
"""Layer study: evaluate intermediate and final audio encoder layers for
emotion signal with linear probes (mean+std pooling per layer).

ASR-oriented final layers may suppress prosody, so we compare ALL encoder
layers and the projected final state and choose the subset for the heads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sadness"]
EIDX = {e: i for i, e in enumerate(EMOTIONS)}


def parse_stats(stats: torch.Tensor):
    """stats layout: [l0 mean(1024), l0 std, ..., l11 mean, l11 std, final mean(1536), final std]"""
    nl, lw, fd = 12, 1024, 1536
    per = []
    for i in range(nl):
        o = i * 2 * lw
        per.append(stats[:, o:o + lw])
        per.append(stats[:, o + lw:o + 2 * lw])
    fo = nl * 2 * lw
    per.append(stats[:, fo:fo + fd])
    per.append(stats[:, fo + fd:fo + 2 * fd])
    return per


def probe(Xtr, ytr, Xva, yva, epochs=300, lr=0.08, d=0):
    Xtr = torch.nn.functional.normalize((Xtr - Xtr.mean(0)) / Xtr.std(0).clamp(min=1e-6), dim=-1)
    Xva = torch.nn.functional.normalize((Xva - Xtr.mean(0)) / Xtr.std(0).clamp(min=1e-6), dim=-1)
    W = torch.zeros(Xtr.shape[1], 6, requires_grad=True)
    b = torch.zeros(6, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=1e-5)
    n = Xtr.shape[0]
    ytr1 = torch.tensor(ytr, dtype=torch.long)
    for ep in range(epochs):
        idx = torch.randint(0, n, (min(512, n),))
        x = Xtr[idx]
        y = ytr1[idx]
        logits = x @ W + b
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float(((Xva @ W + b).argmax(-1) == torch.tensor(yva)).float().mean())
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/cremad.pt")
    ap.add_argument("--out", default="data/layer_study.json")
    args = ap.parse_args()

    import glob as _glob
    paths = sorted(_glob.glob(args.features)) or args.features.split(",")
    merged = {}
    for p in paths:
        for k, v in torch.load(p).items():
            merged.setdefault(k, []).extend(v)
    tr, va = merged["train"], merged["val"]

    def prep(recs):
        ids = [r for r in recs if r["emotion"] in EIDX]
        stats = torch.stack([r["stats"].float() for r in ids])
        y = np.array([EIDX[r["emotion"]] for r in ids])
        return stats, y

    S_tr, y_tr = prep(tr)
    S_va, y_va = prep(va)

    per_tr = parse_stats(S_tr)
    per_va = parse_stats(S_va)
    results = []
    for i in range(12):
        acc = probe(per_tr[i * 2], y_tr, per_va[i * 2], y_va)
        acc_s = probe(torch.cat([per_tr[i * 2], per_tr[i * 2 + 1]], -1), y_tr,
                      torch.cat([per_va[i * 2], per_va[i * 2 + 1]], -1), y_va)
        results.append({"layer": i, "acc_mean": round(acc, 4), "acc_mean_std": round(acc_s, 4)})
    acc_f = probe(per_tr[-2], y_tr, per_va[-2], y_va)
    acc_fs = probe(torch.cat([per_tr[-2], per_tr[-1]], -1), y_tr,
                   torch.cat([per_va[-2], per_va[-1]], -1), y_va)
    results.append({"layer": "final_proj", "acc_mean": round(acc_f, 4), "acc_mean_std": round(acc_fs, 4)})

    ranked = sorted(results, key=lambda r: -r["acc_mean_std"])
    top = [r["layer"] for r in ranked if isinstance(r["layer"], int)][:4]
    out = {"per_layer": results, "ranking": ranked[:6], "recommended_layers": sorted(top),
           "note": "probe accuracy of time-pooled layer statistics; higher layers carry more emotion-aligned signal"}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
