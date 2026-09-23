#!/usr/bin/env python
"""Full evaluation suite: accuracy + calibration per question type, packing /
permutation stability, robustness conditions, audio-format robustness,
cross-corpus check (RAVDESS), latency/RTF, and repetition stability.

Writes machine-readable bench/eval_results.json and prints a summary table."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monica import pooling  # noqa: E402
from monica.battery_eval import make_pooled, stub_quality  # noqa: E402
from monica.model import MonicaModel, MonicaConfig, _round_probs  # noqa: E402
from monica.metrics import accuracy, macro_f1, binary_auc, brier_binary, brier_multiclass  # noqa: E402
from monica.metrics import ece, ece_binary, rps, stability_report, latency_stats  # noqa: E402
from monica.audioio import AudioQuality  # noqa: E402

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sadness"]
EIDX = {e: i for i, e in enumerate(EMOTIONS)}
RAV_MAP = {"angry": "angry", "disgusted": "disgust", "fearful": "fear", "happy": "happy",
           "neutral": "neutral", "sad": "sadness", "calm": "calm", "surprised": "surprise"}
CRIT_DESC = {
    "angry": "Anger or hostility is expressed", "disgust": "Disgust or revulsion is expressed",
    "fear": "Fear or apprehension is expressed", "happy": "Happiness or enjoyment is expressed",
    "neutral": "No strong emotion is expressed", "sadness": "Sadness or dejection is expressed",
    "calm": "Calmness or tranquility is expressed", "surprise": "Surprise or astonishment is expressed",
}


def softmax_t(logits, t):
    e = np.exp((logits - logits.max()) / t)
    return e / e.sum(-1, keepdims=True)


def noul_p(lg, t):
    return 1.0 / (1.0 + np.exp(-np.clip(lg / t, -40.0, 40.0)))


def run_recs(model, recs, questions, raw=True):
    out = []
    for r in recs:
        pooled = make_pooled(model, r)
        ans = model.score(pooled, questions)
        out.append((r, ans))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/models/gemma-4-E2B-it")
    ap.add_argument("--ckpt", default="models/head-cremad.pt")
    ap.add_argument("--calibration", default="models/calibration.json")
    ap.add_argument("--features", default="data/features/cremad_*.pt")
    ap.add_argument("--robust-features", default="data/features/robust.pt")
    ap.add_argument("--ravdess-features", default="data/features/ravdess.pt")
    ap.add_argument("--battery", default="data/battery")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="bench/eval_results.json")
    ap.add_argument("--max-clips", type=int, default=400)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = MonicaConfig(model_dir=args.model_dir, ckpt=args.ckpt, device=args.device,
                    layer_ids=tuple(ck["meta"]["layer_ids"]),
                    calibration_path=args.calibration if os.path.exists(args.calibration) else None)
    model = MonicaModel(cfg)
    model.load_head_ckpt(args.ckpt)
    temps = model.calibrator.temps

    import glob as _glob
    paths = sorted(_glob.glob(args.features)) or args.features.split(",")
    feats = {}
    for p in paths:
        for k, v in torch.load(p).items():
            feats.setdefault(k, []).extend(v)
    test = feats["test"]
    bat = {k: json.load(open(f"{args.battery}/{k}.json"))
           for k in ("noul_train", "noul_eval_unseen", "choice_train", "choice_eval_perm",
                     "choice_perm6", "score_scales")}

    results: dict = {"meta": {"ckpt_meta": ck["meta"], "calibrated": not getattr(model, "_uncalibrated", False)}}

    # ------------------------------------------------------------- 1. noul
    def eval_noul(items, recs, tag):
        qs = {f"n{i}": {"type": "noul", "instructions": it["text"]} for i, it in enumerate(items)}
        probs, labels, per_word = [], [], {}
        for r, ans in run_recs(model, recs, qs):
            emo = r.get("emotion")
            for i, it in enumerate(items):
                trues = [t for t in it["true_emotions"] if t in EIDX or t == "calm" or t == "surprise"]
                if emo is None:
                    continue
                p = noul_p(ans[f"n{i}"]["_logit"], temps.get("noul", 1.0))
                lab = 1 if emo in it["true_emotions"] else 0
                probs.append(p); labels.append(lab)
                per_word.setdefault(it["word"], []).append((emo, p))
        probs = np.array(probs); labels = np.array(labels)
        m = {"n": len(labels), "acc": round(accuracy((probs >= 0.5).astype(int), labels), 4),
            "brier": round(brier_binary(probs, labels), 4), "ece": round(ece_binary(probs, labels), 4),
            "auc": round(binary_auc(probs, labels), 4)}
        true_of = {it["word"]: set(it["true_emotions"]) for it in items}
        wpos = {}
        for w, v in per_word.items():
            vals = [p for e, p in v if e in true_of.get(w, set())]
            wpos[w] = round(float(np.mean(vals)), 3) if vals else None
        m["per_word_positive"] = wpos
        return m

    test_recs = test
    results["noul_test_seen"] = eval_noul(bat["noul_train"], test_recs, "seen")
    unseen = [it for it in bat["noul_eval_unseen"] if any(t in EIDX for t in it["true_emotions"])]
    results["noul_test_unseen_words"] = eval_noul(unseen, test_recs, "unseen")

    # ------------------------------------------------------------ 2. choice
    def eval_choice(panels, recs):
        probs_all, labels_all, preds = [], [], []
        pos_maps = []
        n_tested = 0
        for r, ans in run_recs(model, recs, {f"c{i}": {"type": "choice",
                        "instructions": "What primary sentiment is expressed by the speaker's voice?",
                        "criteria": {k: pl["descriptions"][k] for k in pl["panel"]}}
                        for i, pl in enumerate(panels)}):
            emo = r.get("emotion")
            for i, pl in enumerate(panels):
                if emo not in pl["panel"]:
                    continue
                lg = np.asarray(ans[f"c{i}"]["_logits"], dtype=np.float64)
                if len(lg) != 6:
                    continue
                p = softmax_t(lg, temps.get("choice", 1.0))
                probs_all.append(p); labels_all.append(pl["panel"].index(emo)); n_tested += 1
        P = np.array(probs_all); L = np.array(labels_all)
        return {"n": n_tested, "acc": round(accuracy(P.argmax(-1), L), 4),
                "macro_f1": round(macro_f1(P.argmax(-1), L, 6), 4),
                "nll": round(float(-np.log(P[np.arange(len(L)), L] + 1e-9).mean()), 4),
                "brier": round(brier_multiclass(P, L), 4), "ece": round(ece(P, L), 4)}

    results["choice_test"] = eval_choice(bat["choice_train"], test_recs)

    # permutation stability: fixed panel + shuffled criteria order
    base_panel = {"panel": ["happy", "sadness", "angry", "neutral"],
                  "descriptions": {k: CRIT_DESC[k] for k in ["happy", "sadness", "angry", "neutral"]}}
    perm_recs = test_recs[: args.max_clips]
    probs_by_order = []
    for pl in [base_panel] + bat["choice_perm6"]:
        qs = {"c0": {"type": "choice", "instructions": "What primary sentiment is expressed by the speaker's voice?",
                     "criteria": {k: pl["descriptions"][k] for k in pl["panel"]}}}
        rows = []
        for r, ans in run_recs(model, perm_recs, qs):
            lg = ans["c0"]["_logits"]
            rows.append(softmax_t(lg, temps.get("choice", 1.0)))
        probs_by_order.append((pl["panel"], np.array(rows)))
    base_keys, base_P = probs_by_order[0]
    perm_stab = []
    for keys, P in probs_by_order[1:]:
        aligned = np.stack([P[:, keys.index(k)] for k in base_keys], -1)
        perm_stab.append(stability_report(base_P, aligned))
    results["choice_order_stability"] = {
        "n_orders": len(probs_by_order),
        "mean_max_prob_delta": round(float(np.mean([s["max_prob_delta"] for s in perm_stab])), 5),
        "mean_js": round(float(np.mean([s["js_divergence"] for s in perm_stab])), 5),
        "argmax_agreement": round(float(np.mean([s["argmax_agreement"] for s in perm_stab])), 4),
    }

    # ------------------------------------------------------------- 3. score
    def eval_score(scales, recs):
        rows = []
        for r, ans in run_recs(model, recs, {f"s{i}": {"type": "score",
                        "instructions": f"How {w} does the speaker sound?", "criteria": sc["levels"]}
                        for i, (w, sc) in enumerate(scales.items())}):
            emo = r.get("emotion")
            if emo not in EMOTIONS:
                continue
            for i, (w, sc) in enumerate(scales.items()):
                lg = ans[f"s{i}"]["_logits"]
                p = softmax_t(lg, temps.get("score", 1.0))
                gold = min(max(sc["map"].get(emo, 2.0), 0), len(sc["levels"]) - 1)
                rows.append((w, p, gold))
        by_word = {}
        for w, p, gold in rows:
            by_word.setdefault(w, {"P": [], "G": []})
            by_word[w]["P"].append(p); by_word[w]["G"].append(gold)
        out = {}
        for w, d in by_word.items():
            P = np.array(d["P"]); G = np.array(d["G"])
            exp = (P * np.arange(P.shape[1])).sum(-1)
            out[w] = {"n": len(G), "level_mae": round(float(np.abs(exp - G).mean()), 3),
                      "level_rmse": round(float(np.sqrt(((exp - G) ** 2).mean())), 3),
                      "rps": round(rps(P, np.clip(G.round().astype(int), 0, P.shape[1] - 1)), 4),
                      "ece_argmax": round(ece(P, np.clip(G.round().astype(int), 0, P.shape[1] - 1)), 4)}
        return out

    scales_train = bat["score_scales"]["train"]
    scales_eval = bat["score_scales"]["eval"]
    results["score_test_train_scales"] = eval_score(scales_train, test_recs)
    results["score_test_eval_scales"] = eval_score(scales_eval, test_recs)

    # ------------------------------------------------- 4. packed vs separate
    noul_items = bat["noul_train"][::4][:8]
    pack_qs = {f"n{i}": {"type": "noul", "instructions": it["text"]} for i, it in enumerate(noul_items)}
    pack_ch = {"c0": {"type": "choice", "instructions": "What primary sentiment is expressed by the speaker's voice?",
                      "criteria": {k: CRIT_DESC[k] for k in ["happy", "sadness", "angry", "neutral"]}}}
    pack_sc = {"s0": {"type": "score", "instructions": "How frustrated does the speaker sound?",
                      "criteria": scales_train["frustrated"]["levels"]}}
    all_qs = {**pack_qs, **pack_ch, **pack_sc}
    pack_recs = test_recs[:150]
    packed_p = {"noul": [[] for _ in pack_qs], "choice": [], "score": []}
    sep_p = {"noul": [[] for _ in pack_qs], "choice": [], "score": []}
    for r in pack_recs:
        pooled = make_pooled(model, r)
        ans_p = model.score(pooled, all_qs)
        for i in range(len(pack_qs)):
            packed_p["noul"][i].append(noul_p(ans_p[f"n{i}"]["_logit"], temps.get("noul", 1.0)))
        packed_p["choice"].append(softmax_t(ans_p["c0"]["_logits"], temps.get("choice", 1.0)))
        packed_p["score"].append(softmax_t(ans_p["s0"]["_logits"], temps.get("score", 1.0)))
        ans_s = model.score(pooled, {**pack_qs})
        for i in range(len(pack_qs)):
            sep_p["noul"][i].append(noul_p(ans_s[f"n{i}"]["_logit"], temps.get("noul", 1.0)))
        ans_c = model.score(pooled, pack_ch)
        sep_p["choice"].append(softmax_t(ans_c["c0"]["_logits"], temps.get("choice", 1.0)))
    results["packed_vs_separate"] = {
        "n_clips": len(pack_recs),
        "choice": stability_report(np.array(packed_p["choice"]), np.array(sep_p["choice"])),
    }
    # clean noul packed delta (packed vs separate elementwise)
    sep_mat = np.stack(sep_p["noul"], axis=1)  # [questions, clips]
    packed_mat = np.stack(packed_p["noul"], axis=1)
    results["packed_vs_separate"]["noul"] = {
        "max_abs_delta": round(float(np.abs(packed_mat - sep_mat).max()), 6),
        "mean_abs_delta": round(float(np.abs(packed_mat - sep_mat).mean()), 6),
        "sign_flip_rate": round(float(((packed_mat >= 0.5) != (sep_mat >= 0.5)).mean()), 5),
    }

    # ---------------------------------------------------------- 5. robustness
    # live path: decode + analyze + score via model.handle on real variant wavs
    import soundfile as sf
    rob_manifest = json.load(open("data/robust_manifest.json"))
    rob_qs = {f"n{i}": {"type": "noul", "instructions": it["text"]}
              for i, it in enumerate(bat["noul_train"][:8])}
    rob_qs["c0"] = pack_ch["c0"]
    rob_rows = []
    for item in rob_manifest:
        raw = open(item["wav"], "rb").read()
        import base64 as b64
        req = {"model": "gemma-audio-emotion",
               "state": {"audio": "data:audio/wav;base64," + b64.b64encode(raw).decode()},
               "questions": {**rob_qs,
                            "c0": {"type": "choice",
                                   "instructions": "What primary sentiment is expressed by the speaker's voice?",
                                   "criteria": {k: CRIT_DESC[k] for k in ["happy", "sadness", "angry", "neutral"]}}}}
        try:
            resp, q = model.handle(req)
            probs = [ans["noul"] for k, ans in resp["answers"].items() if ans["type"] == "noul"]
            ch = resp["answers"].get("c0", {})
            rob_rows.append({"cond": item["cond"], "low_info": resp["usage"]["low_information"],
                              "music": resp["usage"]["music_probable"], "clipped": resp["usage"]["clipped"],
                              "max_noul": float(max(probs)) if probs else None,
                              "speech_ratio": q.speech_ratio,
                              "choice_conf": ch.get("confidence")})
        except Exception as e:  # noqa: BLE001
            print("ROB ERR", item["cond"], item["wav"], repr(str(e))[:120])
            rob_rows.append({"cond": item["cond"], "error": str(e)})
    rob_summary = {}
    for cond in sorted({x["cond"] for x in rob_rows}):
        xs = [x for x in rob_rows if x["cond"] == cond and "error" not in x]
        scored = [x for x in xs if x.get("max_noul") is not None]
        rob_summary[cond] = {"n": len(xs),
                              "low_info_rate": round(float(np.mean([x["low_info"] for x in xs])), 3),
                              "music_rate": round(float(np.mean([x.get("music", False) for x in xs])), 3),
                              "max_noul_scored": round(float(max([x["max_noul"] for x in scored], default=0.0)), 3),
                              "max_choice_conf": round(float(max([x.get("choice_conf") or 0.0 for x in xs])), 3)}
    results["robustness"] = rob_summary

    # ------------------------------------------------- 6. audio format matrix
    import soxr
    wav_by_id = {m["id"]: m["wav"] for m in json.load(open("data/manifest_cremad.json"))}
    fmt_base_recs = [r for r in test_recs if r["emotion"] in EIDX][:6]
    qs_fmt = dict(rob_qs)
    def handle_bytes(raw):
        req = {"model": "gemma-audio-emotion",
               "state": {"audio": "data:audio/wav;base64," + b64.b64encode(raw).decode()},
               "questions": qs_fmt}
        resp, q = model.handle(req)
        probs = {k: (ans.get("noul") if ans["type"] == "noul"
                     else tuple(ans["probabilities"][kk] for kk in sorted(ans["probabilities"])))
                 for k, ans in resp["answers"].items()}
        return probs, q
    def variant_bytes(raw, kind):
        d, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if d.ndim > 1:
            d = d.mean(axis=1)
        out = io.BytesIO()
        if kind == "mono16_16k":
            sf.write(out, d, sr, format="WAV", subtype="PCM_16")
        elif kind == "stereo16":
            sf.write(out, np.stack([d, d], axis=1), sr, format="WAV", subtype="PCM_16")
        elif kind == "sr48k":
            sf.write(out, soxr.resample(d, sr, 48000).astype(np.float32), 48000, format="WAV", subtype="PCM_16")
        elif kind == "sr22k":
            sf.write(out, soxr.resample(d, sr, 22050).astype(np.float32), 22050, format="WAV", subtype="PCM_16")
        elif kind == "sr8k":
            sf.write(out, soxr.resample(d, sr, 8000).astype(np.float32), 8000, format="WAV", subtype="PCM_16")
        elif kind == "mono24":
            sf.write(out, d, sr, format="WAV", subtype="PCM_24")
        elif kind == "float":
            sf.write(out, d, sr, format="WAV", subtype="FLOAT")
        return out.getvalue()
    fmt_deltas = {}
    rep_delta = 0.0
    for r in fmt_base_recs:
        raw = open(wav_by_id[r["id"]], "rb").read()
        base_p, _ = handle_bytes(raw)
        for kind in ("stereo16", "sr48k", "sr22k", "sr8k", "mono24", "float"):
            try:
                pv, _ = handle_bytes(variant_bytes(raw, kind))
            except Exception as e:  # noqa: BLE001
                fmt_deltas.setdefault(kind, []).append({"id": r["id"], "error": str(e)})
                continue
            for k in base_p:
                if k in pv and isinstance(base_p[k], float):
                    fmt_deltas.setdefault(kind + "_noul", []).append(abs(pv[k] - base_p[k]))
                elif k in pv:
                    fmt_deltas.setdefault(kind + "_dist", []).append(max(abs(a - b) for a, b in zip(pv[k], base_p[k])))
        p2, _ = handle_bytes(raw)
        for k in base_p:
            if isinstance(base_p[k], float):
                rep_delta = max(rep_delta, abs(p2[k] - base_p[k]))
            else:
                rep_delta = max(rep_delta, max(abs(a - b) for a, b in zip(p2[k], base_p[k])))
    fmt_summary = {k: round(float(np.mean(v)), 5) for k, v in fmt_deltas.items() if v}
    results["audio_formats"] = {"mean_abs_prob_delta_by_variant": fmt_summary,
                                 "n_clips": len(fmt_base_recs)}
    results["repetition_stability"] = {"max_abs_delta_identical_input": round(float(rep_delta), 8)}

    # ------------------------------------------------------------- 7. latency
    lat_recs = test_recs[:60]
    enc_times, score_times = {3.0: [], 10.0: [], 30.0: []}, {}
    import random as _random
    rng = _random.Random(0)
    # synthesize target durations from test speech by concatenation
    dur_wav = {}
    for target in (3.0, 10.0, 30.0):
        wavs = []
        for r in lat_recs[:12]:
            import soundfile as sf2
            d, sr = sf2.read(r.get("wav", ""), dtype="float32") if r.get("wav") else (None, None)
            if d is None:
                break
            wavs.append(d)
        cat = np.concatenate(wavs) if wavs else np.zeros(1)
        dur_wav[target] = (cat[: int(target * 16000)] if len(cat) >= int(target * 16000) else
                            np.pad(cat, (0, int(target * 16000) - len(cat)))).astype(np.float32)
    # need model device encoder for timing
    from monica.model import MonicaModel as _JM
    for target, wav in dur_wav.items():
        for _ in range(5):
            t0 = time.perf_counter(); model.audio_encode(wav); torch.mps.synchronize()
            enc_times[target].append((time.perf_counter() - t0) * 1000)
    # question scaling
    q10 = {f"n{i}": {"type": "noul", "instructions": bat["noul_train"][i]["text"]} for i in range(10)}
    q32 = dict(q10)
    for i in range(10, 22):
        q32[f"c{i}"] = pack_ch["c0"] if False else {"type": "choice",
             "instructions": "What primary sentiment is expressed by the speaker's voice?",
             "criteria": {k: CRIT_DESC[k] for k in ["happy", "sadness", "angry", "neutral"]}}
    for i in range(22, 32):
        q32[f"s{i}"] = {"type": "score", "instructions": "How frustrated does the speaker sound?",
                         "criteria": scales_train["frustrated"]["levels"]}
    for label, qset in (("q1", {k: v for k, v in q10.items() if k == "n0"}), ("q10", q10), ("q32", q32)):
        times = []
        for wav in dur_wav.values():
            pooled = model.audio_encode(wav)
            pooled["_quality"] = stub_quality(torch.zeros(8))
            for _ in range(5):
                t0 = time.perf_counter(); model.score(pooled, qset); torch.mps.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
        score_times[label] = latency_stats(times)
    results["latency"] = {
        "audio_encode_ms": {str(k): latency_stats(v) for k, v in enc_times.items()},
        "audio_seconds": [3.0, 10.0, 30.0],
        "rtf_encode": {str(k): round(np.mean(v) / 1000 / k, 4) for k, v in enc_times.items()},
        "scoring_ms": score_times,
    }

    # ------------------------------------------- 8. repetition stability (live)

    # ------------------------------------------------ 9. RAVDESS cross-corpus
    if os.path.exists(args.ravdess_features):
        rv = torch.load(args.ravdess_features)
        rv_recs = rv.get("all", [])
        rv_map_recs = []
        for r in rv_recs:
            e = RAV_MAP.get(r.get("emotion"))
            if e:
                rr = dict(r); rr["emotion"] = e
                rv_map_recs.append(rr)
        noul_unseen = [it for it in bat["noul_eval_unseen"] if it["word"] in ("calm", "surprised")]
        if noul_unseen:
            results["ravdess_unseen_words"] = eval_noul(noul_unseen, rv_map_recs, "rav")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in results.items() if k != "robustness"}, indent=2)[:4000])
    print("saved", args.out)


if __name__ == "__main__":
    main()
