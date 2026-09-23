#!/usr/bin/env python
"""Prepare manifests, speaker-independent splits, question batteries, and
robustness audio sets for training/evaluation.

Primary corpus: CREMA-D (ODbL per original distribution; this mirror is
Apache-tagged — see docs/LICENSES.md). Secondary (research-only, benchmark
only): RAVDESS (CC BY-NC-SA). Splits are BY SPEAKER so that no speaker ever
appears in both train and eval.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import glob

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sadness"]
EMO_CANON = {"anger": "angry", "angry": "angry", "sad": "sadness", "sadness": "sadness",
             "fear": "fear", "fearful": "fear", "disgust": "disgust", "disgusted": "disgust",
             "happy": "happy", "neutral": "neutral", "calm": "calm", "surprised": "surprise",
             "surprise": "surprise"}

WORD_SETS = {
    "angry": ["angry", "hostile", "furious", "irritated"],
    "disgust": ["disgusted", "repulsed", "revolted"],
    "fear": ["fearful", "afraid", "scared", "terrified"],
    "happy": ["happy", "joyful", "cheerful", "glad"],
    "neutral": ["neutral", "flat", "unemotional"],
    "sadness": ["sad", "unhappy", "sorrowful", "dejected"],
}
EVAL_WORDS = {
    "relieved": ["happy", "neutral"],
    "anxious": ["fear"],
    "excited": ["happy"],
    "gloomy": ["sadness"],
    "contemptuous": ["disgust"],
    "content": ["happy", "neutral"],
    "frightened": ["fear"],
    "pleased": ["happy"],
    # RAVDESS-only words (cross-corpus eval):
    "calm": ["calm"],
    "surprised": ["surprise"],
}

NOUL_TEMPLATES_A = [
    "Does the speaker sound {w}?",
    "Is the speaker expressing {w}?",
    "Does the voice of the speaker convey {w}?",
    "To what degree does the speaker sound {w}?",
]
NOUL_TEMPLATES_B = [
    "Is there any sign of {w} in the way the speaker sounds?",
    "Does the speaker's voice come across as {w}?",
    "Would you say the speaker sounds {w}?",
]

CRIT_DESC = {
    "angry": "Anger or hostility is expressed",
    "disgust": "Disgust or revulsion is expressed",
    "fear": "Fear or apprehension is expressed",
    "happy": "Happiness or enjoyment is expressed",
    "neutral": "No strong emotion is expressed",
    "sadness": "Sadness or dejection is expressed",
}
CRIT_DESC_ALT = {
    "angry": "The voice sounds mad or aggressive",
    "disgust": "The voice sounds repulsed",
    "fear": "The voice sounds afraid or nervous",
    "happy": "The voice sounds glad or pleased",
    "neutral": "The voice sounds matter-of-fact",
    "sadness": "The voice sounds down or sorrowful",
}

CHOICE_PANELS_TRAIN = [
    ["angry", "disgust", "fear", "happy", "neutral", "sadness"],
    ["happy", "sadness", "angry", "neutral"],
    ["fear", "angry", "sadness"],
    ["happy", "neutral"],
    ["angry", "neutral"],
]

# ordinal score scales: emotion -> center level (0..4)
SCALES = {
    "frustrated": {"angry": 4.0, "disgust": 2.0, "fear": 3.0, "happy": 2.5, "sadness": 1.0, "neutral": 0.5},
    "intense":    {"angry": 4.0, "disgust": 2.5, "fear": 3.5, "happy": 3.0, "sadness": 1.5, "neutral": 0.5},
    "energetic":  {"angry": 3.5, "disgust": 2.0, "fear": 3.0, "happy": 4.0, "sadness": 1.0, "neutral": 0.5},
    "agitated":   {"angry": 4.0, "disgust": 2.5, "fear": 3.5, "happy": 2.0, "sadness": 1.0, "neutral": 0.5},
}
SCALE_LEVELS = {
    "frustrated": ["not frustrated", "slightly frustrated", "moderately frustrated", "very frustrated", "extremely frustrated"],
    "intense": ["not intense", "slightly intense", "moderately intense", "very intense", "extremely intense"],
    "energetic": ["not energetic", "slightly energetic", "moderately energetic", "very energetic", "extremely energetic"],
    "agitated": ["not agitated", "slightly agitated", "moderately agitated", "very agitated", "extremely agitated"],
}
EVAL_SCALE_WORDS = {
    "strained": {"angry": 4.0, "disgust": 2.5, "fear": 3.5, "happy": 2.0, "sadness": 1.0, "neutral": 0.5},
    "tense": {"angry": 3.5, "disgust": 2.0, "fear": 4.0, "happy": 1.5, "sadness": 1.0, "neutral": 0.5},
}
EVAL_SCALE_LEVELS = {
    "strained": ["not strained", "slightly strained", "somewhat strained", "quite strained", "very strained"],
    "tense": ["not tense", "a little tense", "somewhat tense", "quite tense", "very tense"],
}


def soft_level_target(center: float, K: int = 5, width: float = 1.0) -> np.ndarray:
    idx = np.arange(K, dtype=np.float64)
    t = np.clip(1.0 - np.abs(idx - center) / width, 0.0, None) ** 1.5
    return t / t.sum()


def speaker_split(actors, seed=7):
    a = sorted(set(actors))
    random.Random(seed).shuffle(a)
    n = len(a)
    n_tr = int(round(n * 0.75))
    n_va = int(round(n * 0.12))
    return {"train": a[:n_tr], "val": a[n_tr:n_tr + n_va], "test": a[n_tr + n_va:]}


def prep_cremad(out_dir: str, raw_dir: str) -> dict:
    manifest = []
    audio_root = os.path.join(out_dir, "audio", "cremad")
    os.makedirs(audio_root, exist_ok=True)
    for split in ("train", "validation", "test"):
        path = glob.glob(f"{raw_dir}/cremad_parquet/data/{split}-*.parquet")
        for p in path:
            t = pq.read_table(p)
            n = t.num_rows
            files = [c.as_py() for c in t.column("file")]
            emos = [c.as_py() for c in t.column("emotion")]
            auds = t.column("audio").to_pylist()
            for fn, emo, aud in zip(files, emos, auds):
                cid = os.path.basename(str(fn)).split(".")[0]
                spk = cid.split("_")[0]
                b = aud["bytes"] if isinstance(aud, dict) else aud
                data, sr = sf.read(io.BytesIO(b), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                dur = len(data) / sr
                wav_dir = os.path.join(audio_root, spk)
                os.makedirs(wav_dir, exist_ok=True)
                wpath = os.path.join(wav_dir, f"{cid}.wav")
                if not os.path.exists(wpath):
                    sf.write(wpath, data, sr, subtype="PCM_16")
                manifest.append({"id": cid, "speaker": spk, "emotion": EMO_CANON.get(emo.lower(), emo.lower()),
                                "dur": round(dur, 3), "src": "cremad", "wav": wpath})
    spk_map = speaker_split([m["speaker"] for m in manifest])
    lab = {"train": set(spk_map["train"]), "val": set(spk_map["val"]), "test": set(spk_map["test"])}
    for m in manifest:
        m["split"] = "train" if m["speaker"] in lab["train"] else "val" if m["speaker"] in lab["val"] else "test"
    return {"manifest": manifest, "splits": spk_map}


def prep_ravdess(out_dir: str, raw_dir: str) -> list:
    manifest = []
    audio_root = os.path.join(out_dir, "audio", "ravdess")
    os.makedirs(audio_root, exist_ok=True)
    for p in sorted(glob.glob(f"{raw_dir}/ravdess_parquet/data/*.parquet")):
        t = pq.read_table(p)
        cols = {c: t.column(c).to_pylist() for c in t.column_names if c != "audio"}
        auds = t.column("audio").to_pylist()
        for i in range(t.num_rows):
            if cols.get("vocal_channel") and str(cols["vocal_channel"][i]) != "speech":
                continue
            spk = str(cols["actor"][i])
            emo = str(cols["emotion"][i])
            b = auds[i]["bytes"] if isinstance(auds[i], dict) else auds[i]
            data, sr = sf.read(io.BytesIO(b), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            cid = f"rav_{spk}_{emo}_{i}"
            wav_dir = os.path.join(audio_root, spk)
            os.makedirs(wav_dir, exist_ok=True)
            wpath = os.path.join(wav_dir, f"{cid}.wav")
            if not os.path.exists(wpath):
                sf.write(wpath, data, sr, subtype="PCM_16")
            manifest.append({"id": cid, "speaker": f"rav_{spk}", "emotion": EMO_CANON.get(emo.lower(), emo.lower()),
                             "dur": round(len(data) / sr, 3), "src": "ravdess", "wav": wpath})
    return manifest


def write_battery(out_dir: str):
    bat = os.path.join(out_dir, "battery")
    os.makedirs(bat, exist_ok=True)
    noul_train = []
    for emo, words in WORD_SETS.items():
        for w in words:
            for t in NOUL_TEMPLATES_A:
                noul_train.append({"text": t.format(w=w), "true_emotions": [emo],
                                  "emotion": EMO_CANON.get(emo.lower(), emo.lower()), "word": w, "template": t})
    noul_eval_unseen = []
    for w, trues in EVAL_WORDS.items():
        for t in NOUL_TEMPLATES_B:
            noul_eval_unseen.append({"text": t.format(w=w), "true_emotions": trues, "word": w, "template": t})
    json.dump(noul_train, open(f"{bat}/noul_train.json", "w"), indent=1)
    json.dump(noul_eval_unseen, open(f"{bat}/noul_eval_unseen.json", "w"), indent=1)

    panels = []
    for i, panel in enumerate(CHOICE_PANELS_TRAIN):
        for descset, tag in ((CRIT_DESC, "A"), (CRIT_DESC_ALT, "B")):
            panels.append({"panel": panel, "descriptions": {k: descset[k] for k in panel}, "tag": f"{i}{tag}"})
    panels_eval = []
    for i, panel in enumerate(CHOICE_PANELS_TRAIN):
        shuffled = panel + []
        random.Random(11 + i).shuffle(shuffled)
        panels_eval.append({"panel": shuffled, "descriptions": {k: CRIT_DESC_ALT[k] for k in panel},
                            "orig": panel, "tag": f"perm{i}"})
    json.dump(panels, open(f"{bat}/choice_train.json", "w"), indent=1)
    json.dump(panels_eval, open(f"{bat}/choice_eval_perm.json", "w"), indent=1)

    scales_train = {w: {"map": m, "levels": SCALE_LEVELS[w]} for w, m in SCALES.items()}
    scales_eval = {w: {"map": m, "levels": EVAL_SCALE_LEVELS[w]} for w, m in EVAL_SCALE_WORDS.items()}
    json.dump({"train": scales_train, "eval": scales_eval}, open(f"{bat}/score_scales.json", "w"), indent=1)

    # permutation battery for option-order stability (choice panels)
    perm_battery = []
    base = {"happy": CRIT_DESC["happy"], "sadness": CRIT_DESC["sadness"],
            "angry": CRIT_DESC["angry"], "neutral": CRIT_DESC["neutral"]}
    for r in range(6):
        keys = list(base)
        random.Random(100 + r).shuffle(keys)
        perm_battery.append({"panel": keys, "descriptions": {k: base[k] for k in keys}})
    json.dump(perm_battery, open(f"{bat}/choice_perm6.json", "w"), indent=1)


def make_robust(out_dir: str, manifest: list, rng: random.Random):
    """Synthetic robustness conditions on real eval speakers."""
    rob = os.path.join(out_dir, "robust")
    os.makedirs(rob, exist_ok=True)
    by_spk = {}
    for m in manifest:
        if m["split"] == "test":
            by_spk.setdefault(m["speaker"], []).append(m)
    items = []
    ev_speakers = sorted(by_spk)[:8]
    others = [m for m in manifest if m["split"] == "test" and m["speaker"] not in ev_speakers]

    def load(m):
        d, sr = sf.read(m["wav"], dtype="float32")
        return d, sr

    for m in [manifest[i] for i in rng.sample(range(len(manifest)), 40) if manifest[i]["split"] == "test"]:
        d, sr = load(m)
        base = {"id": m["id"], "speaker": m["speaker"], "emotion": m["emotion"], "src": m["src"], "dur": m["dur"]}
        # silence
        w = os.path.join(rob, f"silence_{m['id']}.wav")
        sf.write(w, np.zeros(int(3 * sr), np.float32), sr, subtype="PCM_16")
        items.append({**base, "cond": "silence", "wav": w})
        # clipped
        gain = 5.0
        cw = np.clip(d * gain, -1.0, 1.0)
        w = os.path.join(rob, f"clip_{m['id']}.wav")
        sf.write(w, cw.astype(np.float32), sr, subtype="PCM_16")
        items.append({**base, "cond": "clipped", "wav": w})
        # babble noise mixed at SNR 5/10/20 dB
        for snr in (5, 10, 20):
            pool = rng.sample(others, min(6, len(others)))
            bab = np.zeros_like(d)
            for p in pool:
                o, _ = load(p)
                L = min(len(o), len(bab))
                bab[:L] += o[:L]
            bab = bab / (np.abs(bab).max() + 1e-9)
            sp = np.sqrt((d ** 2).mean() / (10 ** (snr / 10)) / ((bab ** 2).mean() + 1e-12))
            mix = d + sp * bab
            mix = np.clip(mix, -1.0, 1.0)
            w = os.path.join(rob, f"babble{snr}_{m['id']}.wav")
            sf.write(w, mix.astype(np.float32), sr, subtype="PCM_16")
            items.append({**base, "cond": f"babble_{snr}", "wav": w})
        # telephone band-pass
        spec = np.fft.rfft(d)
        freqs = np.fft.rfftfreq(len(d), 1 / sr)
        band = ((freqs >= 300) & (freqs <= 3400)).astype(np.float32)
        d2 = np.fft.irfft(spec * band, n=len(d))
        d2 = d2 / (np.abs(d2).max() + 1e-9) * np.abs(d).max()
        w = os.path.join(rob, f"phone_{m['id']}.wav")
        sf.write(w, d2.astype(np.float32), sr, subtype="PCM_16")
        items.append({**base, "cond": "phone_band", "wav": w})
        # very short speech (first 0.4 s)
        w = os.path.join(rob, f"short400_{m['id']}.wav")
        sf.write(w, d[: int(0.4 * sr)], sr, subtype="PCM_16")
        items.append({**base, "cond": "short_0.4s", "wav": w, "dur": 0.4})
    # pure synthetic music (tonal, low syllabic modulation)
    t = np.arange(int(6 * 16000)) / 16000
    chord = np.zeros_like(t)
    for f0 in (196.0, 246.9, 293.7, 392.0):
        chord += np.sin(2 * np.pi * f0 * t + 2 * np.sin(2 * np.pi * 5.5 * t))
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 0.4 * t)
    music = (chord / 4 * env).astype(np.float32) * 0.5
    w = os.path.join(rob, "music_synth.wav")
    sf.write(w, music, 16000, subtype="PCM_16")
    items.append({"id": "music_synth", "speaker": "synth", "emotion": None, "cond": "music", "wav": w, "dur": 6.0})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()
    out = args.data_dir
    os.makedirs(out, exist_ok=True)

    cream = prep_cremad(out, f"{out}/raw")
    json.dump(cream["manifest"], open(f"{out}/manifest_cremad.json", "w"))
    json.dump(cream["splits"], open(f"{out}/speaker_splits_cremad.json", "w"), indent=1)
    print("CREMA-D:", len(cream["manifest"]), "clips;",
          {k: len(v) for k, v in cream["splits"].items()}, "speakers")

    rv = prep_ravdess(out, f"{out}/raw")
    json.dump(rv, open(f"{out}/manifest_ravdess.json", "w"))
    print("RAVDESS speech:", len(rv))

    write_battery(out)

    rng = random.Random(3)
    rob = make_robust(out, cream["manifest"], rng)
    json.dump(rob, open(f"{out}/robust_manifest.json", "w"))
    print("robustness clips:", len(rob))

    # leakage assertion
    sp_tr = set(cream["splits"]["train"]); sp_te = set(cream["splits"]["test"]); sp_va = set(cream["splits"]["val"])
    assert not (sp_tr & sp_te) and not (sp_tr & sp_va) and not (sp_te & sp_va), "speaker leakage!"
    print("speaker independence: OK")


if __name__ == "__main__":
    main()
