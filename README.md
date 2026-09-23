# monica

Local, non-generative audio-sentiment service — a Monica-style
**System One**: one WAV section in, calibrated probabilities out. No
transcription, no generation, no cloud. The foundation is a frozen
**Gemma 4 E2B** audio tower; a compact question-conditioned decision head
scores typed questions over a single shared audio encoding.

## Weights

| what | where | size |
|---|---|---|
| Gemma 4 E2B base (frozen) | `hf download google/gemma-4-E2B-it` (https://huggingface.co/google/gemma-4-E2B-it) | ~10 GB |
| monica heads + calibration + layer selection | `hf download datumsteve/monica` (https://huggingface.co/datumsteve/monica, public) | ~72 MB |

## Setup

Requires macOS with Apple Silicon (MPS), Python 3.12+, and the `hf` CLI
(https://huggingface.co/cli) for weights.

```bash
git clone https://github.com/nunez/monica && cd monica
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# frozen Gemma 4 E2B base model (~10 GB, Apache-2.0)
hf download google/gemma-4-E2B-it --local-dir data/models/gemma-4-E2B-it

# trained heads + calibration + layer selection (~72 MB, public HF repo)
hf download datumsteve/monica --include "models/*" "data/selected_layers.json" \
  --local-dir .

# offline check (needs the two downloads above; no network at runtime)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHONPATH=src python -m pytest tests -q
```

## Running the server

```bash
./scripts/server.sh start      # uvicorn monica.server:app on 0.0.0.0:8910
./scripts/server.sh status     # pid + live health check
./scripts/server.sh stop
./scripts/server.sh restart
./scripts/server.sh log        # last 40 lines
```

Direct invocation (same thing):

```bash
PYTHONPATH=src python -m uvicorn monica.server:app --host 0.0.0.0 --port 8910
```

Environment variables:

| var | default | meaning |
|---|---|---|
| `MONICA_HOST` / `MONICA_PORT` | `127.0.0.1` / `8910` | bind (server.sh defaults host to `0.0.0.0`) |
| `MONICA_BASE_MODEL` | `data/models/gemma-4-E2B-it` | Gemma 4 E2B checkout |
| `MONICA_HEAD_CKPT` | `models/head-cremad.pt` | trained heads |
| `MONICA_CALIB` | `models/calibration.json` | temperature calibration |
| `MONICA_DEVICE` | `mps` | `mps` or `cpu` |
| `MONICA_MAX_AUDIO_SECONDS` | `120` | input length limit (rejected, never truncated) |
| `MONICA_MAX_BODY_BYTES` | 64 MiB | request body limit |
| `MONICA_MAX_FILE_BYTES` | 256 MiB | file-path input limit |
| `MONICA_TARGET_SR` | `16000` | internal sample rate |

## Query

```bash
curl -s localhost:8910/v1/systemone -H 'content-type: application/json' -d '{
  "model": "monica/gemma4-e2b-systemone-v1",
  "state": {"audio": "data:audio/wav;base64,<...>"},
  "questions": {
    "feel":   {"type": "noul",   "instructions": "Does the speaker sound angry?"},
    "threat": {"type": "choice", "instructions": "How threatening is the tone overall?",
               "criteria": {"calm": "calm, no threat", "uneasy": "slightly uneasy",
                           "tense": "tense", "threatening": "actively threatening"}},
    "tension":{"type": "score",  "instructions": "Rate vocal tension.",
               "criteria": ["completely relaxed", "slightly tense", "noticeably tense",
                           "very tense", "extremely tense"]}
  }
}'
```

`state.audio` is a base64 WAV data URL **or** a local file path on the server
(`file:///path/to.wav` or `/path/to.wav`). WAV: any sample rate (HQ-resampled
to 16 kHz), mono/stereo (mixed down), PCM 8/16/24-bit, float 32/64. 0–64
questions per request; the audio is encoded exactly once per request regardless
of question count.

### Response shape

```json
{
  "model": "...",
  "confidence": 0.83,
  "answers": {
    "feel":    {"type": "noul", "noul": 0.91},
    "threat":  {"type": "choice", "choice": "threatening", "confidence": 0.77,
                "probabilities": {"calm": 0.03, "uneasy": 0.08, "tense": 0.12, "threatening": 0.77}},
    "tension": {"type": "score", "score": 3.8, "confidence": 0.66,
                "legend": {"0": "completely relaxed", "...": "..."},
                "probabilities": {"0": 0.01, "...": "...", "4": 0.61}}
  },
  "audio": {"duration_s": 4.2, "speech_ratio": 0.86, "snr_db": 21.3, "clipped": false,
            "music_probable": false, "low_information": false, "...": "..."},
  "usage": {"audio_encoder_calls": 1, "generated_tokens": 0, "latency_ms": 140, "...": "..."},
  "calibration": {"status": "calibrated", "method": "temperature_scaling"}
}
```

## Design rules

- **One encoding per section.** The audio tower runs once per request; every
  question is scored from the same pooled encoding. Question text goes through
  a tiny embedding-table + MLP question encoder (not the LM); answers are
  linear/attention head outputs — there is no autoregressive decode anywhere.
- **No silent truncation.** Audio longer than 120 s (or a request body over
  the limit) is rejected, not trimmed.
- **Intensity preserved.** Stereo is mixed down consistently; quiet audio is
  brought toward a −20 dBFS reference, loud audio keeps its level; peak
  ceiling avoids digital clipping. Resampling is HQ soxiv-style (`soxr`).
- **Low-information honesty.** Silence, music, severe clipping, or sub-0.5 s
  speech return prior answers with `low_information: true` and `confidence: 0`
  without spending an encode. Low-SNR inputs have their confidence capped.
- **Calibrated probabilities.** Temperature scaling per question type, fitted
  on held-out speakers; `confidence` is normalized negentropy of the answer
  distribution.

## Pipeline

```
WAV -> decode/resample (audioio) -> Gemma 4 E2B audio tower (frozen, 12 layers)
    -> window pooling over layers (3,4,9,10)+proj -> 20x11264 pooled A
question text -> Gemma embedding table -> MLP question encoder (option 1)
    -> window attention + per-option scorer -> logits -> temperature -> probs
```

`docs/GATE.md` compares this (option 1) with a full Gemma text-trunk
conditioning option (option 2) and the distilled-student option (option 3),
with measured numbers in `bench/options_comparison.json`.

## Evaluation

See `docs/EVALUATION.md` and `bench/eval_results.json` for full numbers:
speaker-independent CREMA-D splits, unseen-word generalization, RAVDESS
cross-corpus, calibration (ECE), order stability, audio-format robustness,
and latency (bench/bench_results.json).

## Train / eval

```bash
./run_pipeline.sh            # extract features -> train -> calibrate -> evaluate
PYTHONPATH=src .venv/bin/python scripts/bench.py
PYTHONPATH=src .venv/bin/python scripts/compare_options.py
PYTHONPATH=src .venv/bin/python -m pytest tests -q
```

## Datasets & licenses

CREMA-D (training), RAVDESS (research-only benchmark). Gemma 4 E2B weights:
see `docs/LICENSES.md`.
