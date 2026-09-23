#!/usr/bin/env python
"""Train the non-generative typed decision heads (option 1: frozen Gemma 4
audio tower + frozen Gemma token embeddings + compact trainable question
encoder + typed heads). The audio tower itself is NOT fine-tuned here
(that is the gate's option 2/3 territory).

Data: precomputed pooled audio descriptors (scripts/extract_features.py),
speaker-independent splits, question batteries.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from monica import pooling  # noqa: E402
from monica.heads import QuestionHeads  # noqa: E402
from monica.encoder import GemmaAudioEncoder  # noqa: E402
from monica.qencoder import GemmaEmbeddingTable, TextEncoder, make_tokenizer  # noqa: E402

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sadness"]
EIDX = {e: i for i, e in enumerate(EMOTIONS)}


def load_features(path):
    import glob as _glob
    paths_in = list(path) if isinstance(path, (list, tuple)) else path.replace(",", " ").split()
    paths = []
    for p0 in paths_in:
        paths.extend(sorted(_glob.glob(p0)) or [p0])
    merged: dict[str, list] = {}
    for p in paths:
        feats = torch.load(p)
        for split, recs in feats.items():
            merged.setdefault(split, []).extend(recs)
    out = {}
    for split, recs in merged.items():
        A = torch.stack([r["A"] for r in recs])
        stats = torch.stack([r["stats"] for r in recs])
        q = torch.stack([r["q"] for r in recs])
        emo = [EIDX.get(r["emotion"], -1) for r in recs]
        out[split] = {
            "A": A, "stats": stats, "q": q, "emo": emo,
            "ids": [r["id"] for r in recs], "speakers": [r["speaker"] for r in recs],
            "valid": torch.stack([r["valid"] for r in recs]),
        }
    return out


class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        li = args.layer_ids_file if args.layer_ids_file and os.path.exists(args.layer_ids_file) else None
        self.layer_ids = tuple(json.load(open(li))["layer_ids"]) if li else (3, 4, 9, 10)
        self.n_windows = 20
        self.d_a = 2 * (len(self.layer_ids) * 1024 + 1536)

        self.tokenizer = make_tokenizer(args.model_dir)
        self.emb = GemmaEmbeddingTable(args.model_dir, device=args.device)
        self.text_enc = TextEncoder(emb_dim=self.emb.dim, out_dim=args.d_q).to(self.device)
        if getattr(args, "init_qenc", None) and os.path.exists(args.init_qenc):
            self.text_enc.load_state_dict(torch.load(args.init_qenc, map_location=self.device)["text_encoder"])
            print("init_qenc loaded:", args.init_qenc)
        self.heads = QuestionHeads(self.d_a, args.d_q, args.d_h, 7).to(self.device)
        self.params = list(self.text_enc.parameters()) + list(self.heads.parameters())
        self.opt = torch.optim.AdamW(self.params, lr=args.lr, weight_decay=1e-4)

        self.feats = load_features(args.features)
        for sp in ("train", "val", "test"):
            if sp in self.feats:
                self.feats[sp]["Asub"] = pooling.subset_A(self.feats[sp]["A"], self.layer_ids).to(self.device)
                self.feats[sp]["q"] = self.feats[sp]["q"].to(self.device)
                self.feats[sp]["valid"] = self.feats[sp]["valid"].to(self.device)

        tr = self.feats["train"]
        am = tr["Asub"].mean(dim=(0, 1))
        asd = tr["Asub"].std(dim=(0, 1)).clamp(min=1e-6)
        qm = tr["q"].mean(0)
        qsd = tr["q"].std(0).clamp(min=1e-6)
        self.stats_mean = torch.cat([am, qm]).to(self.device)
        self.stats_std = torch.cat([asd, qsd]).to(self.device)

        self.battery = {
            "noul_train": json.load(open(f"{args.battery}/noul_train.json")),
            "choice_train": json.load(open(f"{args.battery}/choice_train.json")),
            "scales": json.load(open(f"{args.battery}/score_scales.json")),
        }
        self.noul_by_word: dict[str, list] = {}
        for s in self.battery["noul_train"]:
            self.noul_by_word.setdefault(s["word"], []).append(s)

    # ------------------------------------------------------------- text batch
    def emb_texts(self, texts, max_len=96):
        ids = self.tokenizer(texts, return_tensors="pt", padding="max_length",
                             truncation=True, max_length=max_len)["input_ids"].to(self.device)
        return self.text_enc(ids, self.emb)

    # ------------------------------------------------------------------ epoch
    def run_epoch(self, split, train=True):
        self.text_enc.train(train); self.heads.train(train)
        f = self.feats[split]
        n = f["Asub"].shape[0]
        B = self.args.batch
        rng = random.Random(self.args.seed + (0 if train else 999))
        idx = list(range(n))
        if train:
            rng.shuffle(idx)
        tot_loss, nb = 0.0, 0
        noul_acc_n, noul_acc_c = 0, 0
        ch_acc_n, ch_acc_c = 0, 0
        sc_mae_n, sc_mae_c = 0.0, 0
        for b0 in range(0, n, B):
            bidx = idx[b0:b0 + B]
            Ab = f["Asub"][bidx]
            qb = f["q"][bidx]
            emb_b = [f["emo"][i] for i in bidx]
            valid_b = f["valid"][bidx]
            qb_n = (qb - self.stats_mean[-7:]) / self.stats_std[-7:]
            Ab_n = (Ab - self.stats_mean[: self.d_a]) / self.stats_std[: self.d_a]
            if train and rng.random() < 0.5:
                Ab_n = Ab_n + torch.randn_like(Ab_n) * 0.04
            if train and rng.random() < 0.3:
                Ab_n = torch.roll(Ab_n, shifts=rng.randint(-3, 3), dims=1)
            # noul battery: sample template, all words
            texts, labels = [], []
            word_true = {}
            for w, samples in self.noul_by_word.items():
                tpl = rng.choice(samples)["text"] if train else samples[0]["text"]
                word_true[w] = (tpl, set(samples[0]["true_emotions"]))
            for w, (tpl, trueset) in word_true.items():
                texts.append(tpl)
            u_words = self.emb_texts([word_true[w][0] for w in word_true])  # [V,dq]
            words = list(word_true)
            # K/V for audio
            K, V = self.heads.pool.kv(Ab_n.reshape(-1, self.d_a))  # [B*W, dh]
            K = K.view(len(bidx), self.n_windows, -1)
            V = V.view(len(bidx), self.n_windows, -1)
            # noul logits: per clip x word
            Wu = u_words + self.heads.noul_opt  # [V,dq]
            qword = self.heads.pool.wq(Wu)  # [V,dh]
            attn = torch.einsum("bwh,vh->bvw", K, qword) / (self.args.d_h ** 0.5)
            attn = attn.masked_fill(~valid_b.unsqueeze(1), float("-inf"))
            z = torch.einsum("bvw,bwh->bvh", torch.softmax(attn, dim=1), V)  # [B,V,dh]
            stats_b = self.heads.stats_proj(qb_n).unsqueeze(1)  # [B,1,dh]
            z = z + stats_b
            feat = torch.cat([z, u_words.unsqueeze(0).expand(z.shape[0], -1, -1),
                              z * u_words.unsqueeze(0),
                              (z - u_words.unsqueeze(0)).abs()], dim=-1)
            noul_logits = self.heads.scorer(feat).squeeze(-1)  # [B,V]
            lbl = torch.zeros(len(bidx), len(words))
            for bi, e in enumerate(emb_b):
                if e < 0:
                    continue
                em = EMOTIONS[e]
                for wi, w in enumerate(words):
                    lbl[bi, wi] = 1.0 if em in word_true[w][1] else 0.0
            lbl = lbl.to(self.device)
            mask = torch.tensor([[e >= 0 for e in emb_b]], device=self.device).transpose(0,1).expand_as(noul_logits)
            n_loss = F.binary_cross_entropy_with_logits(noul_logits[mask], lbl[mask])

            # choice panels
            panel = rng.choice(self.battery["choice_train"])
            keys = panel["panel"]
            opt_texts = [f"{k}: {panel['descriptions'][k]}" for k in keys]
            pos = {k: i for i, k in enumerate(keys)}
            u_opts = self.emb_texts(opt_texts)  # [K,dq]
            u_inst = self.emb_texts([panel.get("instruction", "What sentiment is expressed by the speaker's voice?")])[0]
            qo = self.heads.pool.wq(u_opts)  # [K,dh]
            attn2 = torch.einsum("bwh,kh->bwk", K, qo) / (self.args.d_h ** 0.5)
            attn2 = attn2.masked_fill(~valid_b.unsqueeze(-1), float("-inf"))
            zc = torch.einsum("bwk,bwh->bkh", torch.softmax(attn2, dim=1), V) + stats_b
            feat2 = torch.cat([zc, u_opts.unsqueeze(0).expand(zc.shape[0], -1, -1),
                               zc * u_opts.unsqueeze(0),
                               (zc - u_opts.unsqueeze(0)).abs()], dim=-1)
            ch_logits = self.heads.scorer(feat2).squeeze(-1)  # [B,K]
            target = torch.full((len(bidx),), -100, dtype=torch.long)
            for bi, e in enumerate(emb_b):
                if e >= 0 and EMOTIONS[e] in pos:
                    target[bi] = pos[EMOTIONS[e]]
            target = target.to(self.device)
            c_loss = F.cross_entropy(ch_logits, target, ignore_index=-100, label_smoothing=0.05)

            # score scales
            sname = rng.choice(list(self.battery["scales"]["train"].keys()))
            sc = self.battery["scales"]["train"][sname]
            levels = sc["levels"]
            u_lv = self.emb_texts(levels)  # [5,dq]
            ql = self.heads.pool.wq(u_lv)
            attn3 = torch.einsum("bwh,lh->bwl", K, ql) / (self.args.d_h ** 0.5)
            attn3 = attn3.masked_fill(~valid_b.unsqueeze(-1), float("-inf"))
            zs = torch.einsum("bwl,bwh->blh", torch.softmax(attn3, dim=1), V) + stats_b
            feat3 = torch.cat([zs, u_lv.unsqueeze(0).expand(zs.shape[0], -1, -1),
                               zs * u_lv.unsqueeze(0),
                               (zs - u_lv.unsqueeze(0)).abs()], dim=-1)
            sc_logits = self.heads.scorer(feat3).squeeze(-1) + self.heads.level_bias[: len(levels)]
            soft = np.stack([soft_target(sc["map"].get(EMOTIONS[e] if e >= 0 else "neutral", 2.0), len(levels))
                            for e in emb_b])
            soft = torch.tensor(soft, dtype=torch.float32, device=self.device)
            valid_mask = torch.tensor([[float(e >= 0)] for e in emb_b], device=self.device).expand(-1, len(levels))
            logp = F.log_softmax(sc_logits, dim=-1)
            s_loss = -((soft * logp) * valid_mask).sum(1).mean()
            # ordinal EMD auxiliary
            cdf_p = torch.softmax(sc_logits, -1).cumsum(-1)
            cdf_t = soft.cumsum(-1)
            s_loss = s_loss + 0.5 * ((cdf_p - cdf_t).abs() * valid_mask).sum(1).mean()

            loss = n_loss + c_loss + s_loss
            if train:
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, 5.0)
                self.opt.step()

            tot_loss += float(loss); nb += 1
            with torch.no_grad():
                p = torch.sigmoid(noul_logits)
                m2 = mask
                noul_acc_n += int(m2.sum()); noul_acc_c += int(((p >= 0.5).float() == lbl)[m2].sum().item())
                ok = target != -100
                ch_acc_n += int(ok.sum()); ch_acc_c += int((ch_logits.argmax(-1)[ok] == target[ok]).sum().item())
                exp = (torch.softmax(sc_logits, -1) * torch.arange(len(levels), device=self.device, dtype=torch.float32)).sum(-1)
                gold = torch.tensor([sc["map"].get(EMOTIONS[e], 2.0) if e >= 0 else -1.0 for e in emb_b], device=self.device)
                okm = gold >= 0
                sc_mae_n += int(okm.sum()); sc_mae_c += float((exp[okm] - gold[okm]).abs().sum())
        return {
            "loss": tot_loss / max(nb, 1),
            "noul_acc": noul_acc_c / max(noul_acc_n, 1),
            "choice_acc": ch_acc_c / max(ch_acc_n, 1),
            "score_mae": sc_mae_c / max(sc_mae_n, 1),
        }

    def save(self, path, meta=None):
        ck = {
            "text_encoder": self.text_enc.state_dict(),
            "heads": self.heads.state_dict(),
            "stats_mean": self.stats_mean.cpu(),
            "stats_std": self.stats_std.cpu(),
            "meta": {"layer_ids": list(self.layer_ids), "n_windows": self.n_windows,
                    "d_q": self.args.d_q, "d_h": self.args.d_h, "emotions": EMOTIONS,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {})},
        }
        torch.save(ck, path)


def soft_target(center: float, K: int, width: float = 1.0) -> np.ndarray:
    idx = np.arange(K, dtype=np.float64)
    t = np.clip(1.0 - np.abs(idx - center) / width, 0.0, None) ** 1.5
    return t / (t.sum() + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="data/models/gemma-4-E2B-it")
    ap.add_argument("--features", nargs="+", default=["data/features/cremad.pt"])
    ap.add_argument("--battery", default="data/battery")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-q", type=int, default=512)
    ap.add_argument("--d-h", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--layer-ids-file", default="data/selected_layers.json")
    ap.add_argument("--out", default="models/head-cremad.pt")
    ap.add_argument("--init-qenc", default=None, help="optional distilled question-encoder checkpoint")
    ap.add_argument("--val-logits", default="data/val_logits.pt")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tr = Trainer(args)
    best = -1
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(tr.opt, T_max=args.epochs)
    for ep in range(args.epochs):
        t0 = time.time()
        trm = tr.run_epoch("train", train=True)
        val = tr.run_epoch("val", train=False)
        sched.step()
        comp = 0.4 * val["noul_acc"] + 0.4 * val["choice_acc"] + 0.2 * (1 - min(val["score_mae"], 4) / 4)
        print(f"ep{ep:02d} train {trm} val {val} comp={comp:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if comp > best:
            best = comp
            tr.save(args.out, meta={"epoch": ep, "val": val, "composite": round(comp, 4)})
            print("  saved best", args.out, flush=True)
    print("done; best composite", best)


if __name__ == "__main__":
    main()
