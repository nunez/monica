#!/usr/bin/env python3
"""Option 3: distill Gemma text-trunk question embeddings into the compact
question encoder (option 1), then fine-tune heads on precomputed features.

Teacher: TrunkTextEncoder (Gemma 4 text trunk, text-only prefill, pooled).
Student: TextEncoder (embedding table + 2-layer MLP).

Training data are question/criterion strings from data/battery/*.json plus
instruction templates — text only, no audio needed for the distill step.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/distill.py [--steps 2000]
Writes models/question_encoder_distilled.pt. After distillation, re-run
train_heads.py with --init-qenc models/question_encoder_distilled.pt to fine-tune
heads on student embeddings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from monica.qencoder import GemmaEmbeddingTable, TextEncoder, TrunkTextEncoder, make_tokenizer  # noqa: E402


TEMPLATES = [
    "Does the speaker sound {w}?",
    "How {w} does the voice sound?",
    "Rate the speaker's {w} from low to high.",
    "Is the tone {w}?",
    "{w}",
]


def question_strings(battery_dir: str) -> list[str]:
    out: list[str] = []
    if os.path.isdir(battery_dir):
        for fn in os.listdir(battery_dir):
            if not fn.endswith(".json"):
                continue
            data = json.load(open(os.path.join(battery_dir, fn)))
            items = data if isinstance(data, list) else data.get("items", [])
            for it in items:
                if isinstance(it, str):
                    out.append(it)
                elif isinstance(it, dict):
                    if "instructions" in it:
                        out.append(it["instructions"])
                    crit = it.get("criteria") or it.get("options") or {}
                    if isinstance(crit, dict):
                        out += [f"{k}: {v}" for k, v in crit.items()]
                    elif isinstance(crit, list):
                        out += list(crit)
    words = ["angry", "calm", "anxious", "happy", "sad", "fearful", "disgusted",
             "warm", "hostile", "tense", "relaxed", "joyful", "contemptuous",
             "threatening", "uneasy", "neutral", "excited", "gloomy"]
    for w in words:
        for t in TEMPLATES:
            out.append(t.format(w=w))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/models/gemma-4-E2B-it")
    ap.add_argument("--battery", default="data/battery")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--out", default="models/question_encoder_distilled.pt")
    args = ap.parse_args()

    device = torch.device(args.device)
    tok = make_tokenizer(args.model_dir)
    emb = GemmaEmbeddingTable(args.model_dir, device=args.device)

    strings = question_strings(args.battery)
    print(f"distill texts: {len(strings)}")
    enc = tok(strings, return_tensors="pt", padding="max_length", truncation=True,
              max_length=args.max_len).to(device)

    teacher = TrunkTextEncoder(args.model_dir, device=args.device).eval()
    student = TextEncoder(emb_dim=emb.dim, out_dim=512).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    with torch.no_grad():
        teacher_out = []
        for i in range(0, len(strings), args.batch):
            teacher_out.append(teacher.encode(enc.input_ids[i:i + args.batch],
                                              enc.attention_mask[i:i + args.batch]).to(device))
        T = torch.cat(teacher_out)  # (N, trunk_hidden)
    proj = torch.nn.Linear(T.shape[1], 512).to(device)
    opt2 = torch.optim.AdamW(list(student.parameters()) + list(proj.parameters()), lr=args.lr)

    t0 = time.time()
    for step in range(args.steps):
        i = (step * args.batch) % len(strings)
        ids = enc.input_ids[i:i + args.batch]
        mask = enc.attention_mask[i:i + args.batch]
        s = student(ids, emb)
        t = proj(T[i:i + args.batch])
        loss = 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        if step % 200 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}  {time.time()-t0:.0f}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"text_encoder": student.state_dict(), "steps": args.steps,
                "teacher": "gemma4-text-trunk-pooled"}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
