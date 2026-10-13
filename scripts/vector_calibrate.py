#!/usr/bin/env python
"""Prototype: replace choice temperature scaling with per-emotion VECTOR scaling.

p = softmax(exp(a_e) * l_e + b_e) over the panel, keyed by option emotion.
Order-equivariant by construction (per-option logits are order-independent),
so it serves any panel containing the calibrated emotions; unknown options
fall back to the plain temperature.

Selection: speaker-grouped CV on val (never touches test). Variants:
  temp   — temperature grid
  vector — diagonal exp(a) + bias b (2K params), L2 toward identity
  matrix — full W + b (K^2 + K params), L2 toward init

Held out: CREMA test (canonical panel), RAVDESS cross-corpus (mapped),
option-permutation stability on battery perm panels.

Run:  PYTHONPATH=src .venv/bin/python scripts/vector_calibrate.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monica.battery_eval import make_pooled  # noqa: E402
from monica.model import MonicaModel, MonicaConfig  # noqa: E402
from monica.calibration import ece, ece_binary, brier_multiclass, fit_temperature  # noqa: E402
from monica.metrics import accuracy, macro_f1  # noqa: E402

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sadness"]
EIDX = {e: i for i, e in enumerate(EMOTIONS)}
CRIT_DESC = {
    "angry": "Anger or hostility is expressed", "disgust": "Disgust or revulsion is expressed",
    "fear": "Fear or apprehension is expressed", "happy": "Happiness or enjoyment is expressed",
    "neutral": "No strong emotion is expressed", "sadness": "Sadness or dejection is expressed",
}
RAV_MAP = {"angry": "angry", "disgusted": "disgust", "fearful": "fear", "happy": "happy",
           "neutral": "neutral", "sad": "sadness"}


def softmax(x, t=1.0):
    e = np.exp((x - x.max(-1, keepdims=True)) / t)
    return e / e.sum(-1, keepdims=True)


def extract(model, recs, panel):
    qs = {"c0": {"type": "choice",
                 "instructions": "What primary sentiment is expressed by the speaker's voice?",
                 "criteria": {k: CRIT_DESC[k] for k in panel}}}
    X, y, ids = [], [], []
    for r in recs:
        e = r.get("emotion")
        if e not in panel:
            continue
        ans = model.score(make_pooled(model, r), qs)
        lg = np.asarray(ans["c0"]["_logits"], dtype=np.float64)
        if len(lg) != len(panel):
            continue
        X.append(lg); y.append(EMOTIONS.index(e)); ids.append(r["id"])
    return np.stack(X), np.array(y), ids


def fit_linear(Xtr, ytr, K, mode="vector", l2=1e-3, steps=600, lr=0.05, seed=0, t0=1.0):
    g = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    u0 = -np.log(t0)
    if mode == "vector":
        p = torch.nn.Parameter(torch.full((K,), u0, dtype=torch.float32))
        b = torch.nn.Parameter(torch.zeros(K, dtype=torch.float32))
        opt = torch.optim.Adam([p, b], lr=lr, weight_decay=0.0)
        for _ in range(steps):
            logits = Xt * p.exp() + b
            loss = torch.nn.functional.cross_entropy(logits, yt) + l2 * ((p - u0) ** 2).sum() + l2 * (b ** 2).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        return {"mode": "vector", "a": p.detach().numpy(), "b": b.detach().numpy()}
    W0 = np.diag(np.exp(np.full(K, u0)))
    W = torch.nn.Parameter(torch.tensor(W0, dtype=torch.float32))
    b = torch.nn.Parameter(torch.zeros(K, dtype=torch.float32))
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=0.0)
    W0t = torch.tensor(W0, dtype=torch.float32)
    for _ in range(steps):
        logits = Xt @ W + b
        loss = torch.nn.functional.cross_entropy(logits, yt) + l2 * ((W - W0t) ** 2).sum() + l2 * (b ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return {"mode": "matrix", "W": W.detach().numpy(), "b": b.detach().numpy()}


def apply_linear(cal, X, panel=None):
    """panel: emotions naming each logit slot (identity keying). Default canonical."""
    if cal["mode"] == "vector":
        a = cal["a"]; b = cal["b"]
        if panel is not None:
            idx = [EMOTIONS.index(e) if e in EIDX else -1 for e in panel]
            a = np.where(np.array(idx) >= 0, a[np.array(idx) + 0 * np.arange(len(idx))], 1.0)
            b = np.where(np.array(idx) >= 0, b[np.array(idx) + 0 * np.arange(len(idx))], 0.0)
        return softmax(X * np.exp(a) + b)
    return softmax(X @ cal["W"] + cal["b"])


def block(P, L, K):
    pred = P.argmax(-1)
    return {"n": int(len(L)), "acc": round(accuracy(pred, L), 4),
            "macro_f1": round(macro_f1(pred, L, K), 4),
            "ece": round(ece(P, L), 4), "brier": round(brier_multiclass(P, L), 4)}


def nll(P, L):
    return float(-np.log(P[np.arange(len(L)), L] + 1e-12).mean())


def extract_battery(model, feats, noul_items, panels, scales):
    noul_qs = {f"n{i}": {"type": "noul", "instructions": it["text"]} for i, it in enumerate(noul_items)}
    sc_qs = {f"s{i}": {"type": "score", "instructions": f"How {w} does the speaker sound?", "criteria": sc["levels"]}
             for i, (w, sc) in enumerate(scales.items())}
    qs = {**noul_qs, **sc_qs}
    out = {"noul": {}, "score": {}}
    emo_of = {}
    noul_true = [set(it["true_emotions"]) for it in noul_items]
    for sp in ("val", "test"):
        recs = feats[sp]
        nl, nb = [], []
        for r in recs:
            e = r.get("emotion")
            if e not in EIDX:
                continue
            ans = model.score(make_pooled(model, r), qs)
            nl.append([ans[f"n{i}"]["_logit"] for i in range(len(noul_items))])
            nb.append([1 if e in noul_true[i] else 0 for i in range(len(noul_items))])
            for i, (w, sc) in enumerate(scales.items()):
                lg = np.asarray(ans[f"s{i}"]["_logits"], dtype=np.float64)
                lv = min(max(int(round(sc["map"].get(e, 2.0))), 0), len(sc["levels"]) - 1)
                out["score"].setdefault(w, {}).setdefault(sp, {"X": [], "y": []})
                out["score"][w][sp]["X"].append(lg)
                out["score"][w][sp]["y"].append(lv)
        out["noul"][sp] = {"logits": np.stack(nl), "labels": np.stack(nb)}
    for w in out["score"]:
        for sp in out["score"][w]:
            out["score"][w][sp]["X"] = np.stack(out["score"][w][sp]["X"])
            out["score"][w][sp]["y"] = np.array(out["score"][w][sp]["y"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/models/gemma-4-E2B-it")
    ap.add_argument("--ckpt", default="models/head-cremad.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ravdess-features", default="data/features/ravdess.pt")
    ap.add_argument("--out", default="bench/vector_calib.json")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = MonicaConfig(model_dir=args.model_dir, ckpt=args.ckpt, device=args.device,
                      layer_ids=tuple(ck["meta"]["layer_ids"]))
    model = MonicaModel(cfg)
    model.load_head_ckpt(args.ckpt)  # raw logits, no calibration

    feats = {}
    for p in sorted(glob.glob("data/features/cremad_*.pt")):
        for k, v in torch.load(p, weights_only=False).items():
            feats.setdefault(k, []).extend(v)

    cache = "bench/vector_calib_logits.pt"
    if os.path.exists(cache):
        pack = torch.load(cache, weights_only=False)
        Xval, yval, idsval = pack["Xval"], pack["yval"], pack["idsval"]
        Xte, yte, idste = pack["Xte"], pack["yte"], pack["idste"]
        bat = pack["bat"]
    else:
        Xval, yval, idsval = extract(model, feats["val"], EMOTIONS)
        Xte, yte, idste = extract(model, feats["test"], EMOTIONS)
        noul_items = json.load(open("data/battery/noul_train.json"))
        panels = json.load(open("data/battery/choice_train.json"))
        scales = json.load(open("data/battery/score_scales.json"))["train"]
        bat = extract_battery(model, feats, noul_items, panels, scales)
        torch.save({"Xval": Xval, "yval": yval, "idsval": idsval, "Xte": Xte, "yte": yte, "idste": idste,
                   "bat": bat}, cache)
    K = len(EMOTIONS)
    print(f"val {Xval.shape} test {Xte.shape}")

    spk = np.array([r["speaker"] for r in feats["val"] if r["emotion"] in EIDX])
    spk = spk[: len(Xval)]
    uspk = sorted(set(spk))
    folds = [[] for _ in range(4)]
    for i, s in enumerate(spk):
        folds[uspk.index(s) % 4].append(i)

    l2_grid = [1e-5, 1e-4, 1e-3, 1e-2, 3e-2]
    cv = {"temp": [], "vector": [], "matrix": []}
    for f in range(4):
        te = np.array(folds[f]); tr = np.setdiff1d(np.arange(len(Xval)), te)
        t = fit_temperature(Xval[tr], yval[tr], grid=np.exp(np.linspace(np.log(0.3), np.log(10.0), 80)))
        cv["temp"].append(nll(softmax(Xval[te] / t, ), yval[te]))
        for mode in ("vector", "matrix"):
            best = min((nll(apply_linear(fit_linear(Xval[tr], yval[tr], K, mode=mode, l2=l2), Xval[te]), yval[te]), l2)
                       for l2 in l2_grid)
            cv[mode].append(best[0])
            if f == 0:
                print(f"cv fold0 {mode} {best[1]:.1e} nll {best[0]:.4f}")
    cv_summary = {m: round(float(np.mean(v)), 4) for m, v in cv.items()}
    best_mode = min(cv_summary, key=cv_summary.get)
    print("cv nll:", cv_summary, "-> selected", best_mode)

    res = {"cv_val_nll": cv_summary, "selected": best_mode}

    # deployed-style baselines on test
    t_dep = fit_temperature(Xval, yval, grid=np.exp(np.linspace(np.log(0.3), np.log(10.0), 80)))
    t_old = json.load(open("models/calibration.json"))["temperatures"]["choice"]
    res["temp_grid_test"] = block(softmax(Xte / t_dep), yte, K)
    res["temp_deployed_test"] = block(softmax(Xte / t_old), yte, K)

    # full-val fits
    fits = {}
    for mode in ("vector", "matrix"):
        best = min(((l2, nll(apply_linear(fit_linear(Xval, yval, K, mode=mode, l2=l2), Xval), yval))
                    for l2 in l2_grid), key=lambda z: z[1])
        fits[mode] = fit_linear(Xval, yval, K, mode=mode, l2=best[0])
        res[f"{mode}_test"] = block(apply_linear(fits[mode], Xte), yte, K)
    res["temp_grid_test"]["tau"] = round(t_dep, 3)

    # cross-corpus RAVDESS
    if os.path.exists(args.ravdess_features):
        rv = torch.load(args.ravdess_features, weights_only=False).get("all", [])
        rv_recs = []
        for r in rv:
            e = RAV_MAP.get(r.get("emotion"))
            if e:
                rr = dict(r); rr["emotion"] = e
                rv_recs.append(rr)
        Xr, yr, _ = extract(model, rv_recs, EMOTIONS)
        res["ravdess_temp_grid"] = block(softmax(Xr / t_dep), yr, K)
        res["ravdess_old_temp"] = block(softmax(Xr / t_old), yr, K)
        for mode in ("vector", "matrix"):
            res[f"ravdess_{mode}"] = block(apply_linear(fits[mode], Xr), yr, K)

    # permutation stability: identity-keyed scaling on permuted/subset panels
    perms = json.load(open("data/battery/choice_perm6.json"))
    Pcanon = apply_linear(fits["vector"], Xte)
    perm_check = []
    for pl in perms:
        panel = pl["panel"]
        if not all(e in EIDX for e in panel) or len(panel) < 2:
            continue
        Xp, yp, idsp = extract(model, feats["test"][:300], panel)
        Pv = apply_linear(fits["vector"], Xp, panel=panel)
        ke = [e for e in EMOTIONS if e in panel]
        back = np.stack([Pv[:, panel.index(e)] for e in ke], -1)
        cidx = {cid: i for i, cid in enumerate(idste)}
        rows = [cidx[c] for c in idsp]
        Psub = Pcanon[rows][:, [EMOTIONS.index(e) for e in ke]]
        def js(a, b):
            m = 0.5 * (a + b)
            kl = lambda x, y: np.sum(x * np.log((x + 1e-12) / (y + 1e-12)), -1)
            return float((0.5 * kl(a, m) + 0.5 * kl(b, m)).mean())
        perm_check.append({"panel": panel,
                           "identity_keyed_js_vs_canonical": round(js(back, Psub), 8),
                           "identity_keyed_argmax_agreement": round(float((back.argmax(-1) == Psub.argmax(-1)).mean()), 4)})
    res["permutation_panel_checks"] = perm_check

    # noul / score: check temperature grid ceiling
    t_old_n, t_old_s = (json.load(open("models/calibration.json"))["temperatures"][k] for k in ("noul", "score"))
    grid_wide = np.exp(np.linspace(np.log(0.3), np.log(50.0), 120))
    nv, nt = bat["noul"]["val"], bat["noul"]["test"]
    X2v = np.stack([-nv["logits"].ravel(), nv["logits"].ravel()], -1)
    t_noul = fit_temperature(X2v, nv["labels"].ravel(), grid=grid_wide)
    sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -40, 40)))
    pw_t = []
    for c in range(nv["logits"].shape[1]):
        lab = nv["labels"][:, c]
        if len(set(lab.tolist())) < 2:
            continue
        tw = fit_temperature(np.stack([-nv["logits"][:, c], nv["logits"][:, c]], -1), lab, grid=grid_wide)
        pw_t.append((c, tw))
    res["noul"] = {
        "deployed_t": t_old_n, "wide_fit_t": round(t_noul, 3),
        "ece_deployed": round(float(ece_binary(sig(nt["logits"] / t_old_n).ravel(), nt["labels"].ravel())), 4),
        "ece_wide": round(float(ece_binary(sig(nt["logits"] / t_noul).ravel(), nt["labels"].ravel())), 4),
        "ece_wide_perword": round(float(np.mean([ece_binary(sig(nt["logits"][:, c] / tw), nt["labels"][:, c])
                                                  for c, tw in pw_t])), 4),
        "ece_deployed_perword": round(float(np.mean([ece_binary(sig(nt["logits"][:, c] / t_old_n), nt["labels"][:, c])
                                                      for c, _ in pw_t])), 4),
    }
    score_res = {}
    for w, d in bat["score"].items():
        Ks = d["val"]["X"].shape[1]
        t_w = fit_temperature(d["val"]["X"], d["val"]["y"], grid=grid_wide)
        out_w = {}
        for tag, tt in (("wide", t_w), ("deployed", t_old_s)):
            P = softmax(d["test"]["X"] / tt)
            exp = (P * np.arange(Ks)).sum(-1)
            out_w[tag] = {"t": round(tt, 2),
                          "mae_exp": round(float(np.abs(exp - d["test"]["y"]).mean()), 3),
                          "mae_argmax": round(float(np.abs(P.argmax(-1) - d["test"]["y"]).mean()), 3),
                          "rps": round(float(((P.cumsum(-1) - (np.arange(Ks)[None, :] >= d["test"]["y"][:, None]).astype(float)) ** 2).sum(-1).mean()), 4)}
        score_res[w] = out_w
    res["score_wide_temp"] = score_res

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, default=str)
    torch.save({"choice_vector": fits["vector"], "choice_matrix": fits["matrix"],
                "temp": t_dep, "emotions": EMOTIONS, "crit_desc": CRIT_DESC},
               "models/vector-scaling.pt")
    for k, v in res.items():
        print(k, json.dumps(v, default=str)[:400])


if __name__ == "__main__":
    main()
