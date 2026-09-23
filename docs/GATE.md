# GATE.md — conditioning options & gating policy

## Question-conditioning options (measured on M3 Ultra)

Measured by `scripts/compare_options.py` → `bench/options_comparison.json`.

| option | what | params | encode 10 questions | status |
|---|---|---|---|---|
| **1 (deployed)** | Gemma embedding table + 2-layer MLP question encoder | 407.6 M (incl. frozen table) | 0.8 ms | shipped |
| 2 | Gemma 4 text trunk, text-only prefill, pooled last hidden | 4.63 B | 382 ms | available for experiments |
| 3 | Distilled student of option 2 (`scripts/distill.py`) | ~0.3 B | ~1 ms | optional |

Decision rule (the "gate"): option 1 is shipped because question-conditioning
through the LM trunk buys no measurable accuracy on typed questions (choice /
score / noul are short, templated texts) while costing ~490x the per-request question-encoding time for 10 questions. Option 2 remains in the codebase (`qencoder.TrunkTextEncoder`)
to re-run the comparison when stronger evidence for trunk conditioning appears.
Option 3 (distillation of option-2 embeddings into the option-1 encoder) is the
middle path if free-text open-vocabulary questions are ever required:
`python scripts/distill.py` trains the student to match teacher embeddings
before head fine-tuning.

The **audio tower is always shared**: one Gemma 4 E2B audio-tower pass per
audio section, regardless of the number or type of questions (verified:
`usage.audio_encoder_calls == 1` for 1–32 questions).

## Gating policy (when the system abstains or caps confidence)

The system never fabricates certainty on unusable audio:

| condition | behavior |
|---|---|
| silence (speech_ratio < 0.02 or < 50 ms) | prior answers, `low_information: true`, confidence 0, **no encode** |
| speech < 0.5 s or speech_ratio < 0.10 | prior answers, `low_information: true`, confidence 0 |
| music (tonal, no syllabic modulation, > 1 s) | prior answers, `music_probable: true`, confidence 0 |
| severe clipping (≥ 5% samples at full scale) | answers computed, confidence capped 0.7 |
| low SNR (< 8 dB) | confidence capped 0.35 |
| low SNR (< 16 dB) | confidence capped 0.6 |

`confidence` itself is normalized negentropy of the calibrated answer
distribution (1 − H/log K), so it is comparable across question types and
option counts.

## Rejected designs (kept for the record)

- **ASR + LLM text sentiment**: transcription discards prosody — the very
  signal emotion lives in — and adds a generative stage (latency, hallucinated
  text, generative uncanny risk this project explicitly avoids).
- **Energy-only baseline**: beats chance but collapses on babble/multispeaker
  and cannot express typed question semantics; kept only as an internal
  sanity floor in `scripts/evaluate.py`.
- **Noise augmentation training**: additive babble augmentation over-regularized
  the head (val acc dropped); reverted. Noise robustness instead handled by
  input gating + SNR confidence caps + robust evaluation set.
