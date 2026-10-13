#!/usr/bin/env python
"""EmbeddingGemma 2 zero-shot anchor probe over CREMA-D val/test splits.

Embeds each clip and each emotion anchor sentence into the shared 768-d space,
stores cosine matrices for offline scoring/stacking against the monica heads.

Run in .venv-eg2 (needs embedding_gemma2 support):
    .venv-eg2/bin/python scripts/embedgemma_anchor.py
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sadness"]
ANCHOR_DESC = {
    "angry": "Anger or hostility is expressed",
    "disgust": "Disgust or revulsion is expressed",
    "fear": "Fear or apprehension is expressed",
    "happy": "Happiness or enjoyment is expressed",
    "neutral": "No strong emotion is expressed",
    "sadness": "Sadness or dejection is expressed",
}
ANCHOR_SIMPLE = {
    "angry": "angry", "disgust": "disgusted", "fear": "afraid",
    "happy": "happy", "neutral": "neutral", "sadness": "sad",
}


def encode_audio(model, rows, batch_size, tag):
    embs, t_sum, n = [], 0.0, 0
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        t0 = time.perf_counter()
        e = model.encode([{"audio": [r["wav"]]} for r in chunk],
                         normalize_embeddings=True, batch_size=len(chunk))
        dt = time.perf_counter() - t0
        embs.append(np.asarray(e, dtype=np.float32).reshape(len(chunk), -1))
        t_sum += dt
        n += len(chunk)
        if (i // batch_size) % 20 == 0:
            print(f"  {tag}: {n}/{len(rows)}  ({t_sum / n:.3f} s/clip)", flush=True)
    return np.concatenate(embs), t_sum / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/models/embeddinggemma-2")
    ap.add_argument("--manifest", default="data/manifest_cremad.json")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-clips", type=int, default=0)
    ap.add_argument("--out", default="bench/embedgemma_cos.pt")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model_dir,
                                model_kwargs={"torch_dtype": torch.bfloat16},
                                config_kwargs={"vision_config": None})

    def enc_texts(texts):
        return np.asarray(model.encode([f"task: classification | query: {t}" for t in texts],
                                         prompt_name="Classification",
                                         normalize_embeddings=True), dtype=np.float32)

    anchor_desc = enc_texts([ANCHOR_DESC[e] for e in EMOTIONS])
    anchor_simple = enc_texts([ANCHOR_SIMPLE[e] for e in EMOTIONS])

    man = json.load(open(args.manifest))
    splits = {}
    for sp in ("val", "test"):
        rows = [r for r in man if r["split"] == sp]
        if args.max_clips:
            rows = rows[:args.max_clips]
        t0 = time.perf_counter()
        emb, per_clip = encode_audio(model, rows, args.batch_size, sp)
        print(f"{sp}: {len(rows)} clips, {per_clip:.3f} s/clip", flush=True)
        splits[sp] = {
            "ids": [r["id"] for r in rows],
            "emotions": [r["emotion"] for r in rows],
            "emb": torch.from_numpy(emb),
            "cos_desc": torch.from_numpy(emb @ anchor_desc.T),
            "cos_simple": torch.from_numpy(emb @ anchor_simple.T),
            "ms_per_clip_encode": round(per_clip * 1000, 1),
        }

    torch.save({
        "meta": {"model": args.model_dir, "emotions": EMOTIONS,
                 "anchor_desc": ANCHOR_DESC, "anchor_simple": ANCHOR_SIMPLE},
        "anchor_desc": torch.from_numpy(anchor_desc),
        "anchor_simple": torch.from_numpy(anchor_simple),
        **splits,
    }, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
