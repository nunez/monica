# Licenses & data provenance

## Code

This service code (monica/*, scripts, tests) is Apache-2.0.

## Foundation model

- **Gemma 4 E2B instruction-tuned** (`google/gemma-4-E2b-it`, single
  `model.safetensors`, ~10.3 GB). License: Apache 2.0. Used frozen (no
  fine-tuning of base weights; only adapter heads trained on top).
- The audio tower and embedding table are the base model's; question/head
  weights trained here are distributed with this project.

## Datasets

- **CREMA-D** (Ravanish et al.) — primary training/validation/test data.
  Open Database License (ODbL) / community attribution. 7,442 clips,
  6 emotion labels (angry, disgust, fear, happy, neutral, sad) + intensity.
  We use canonicalized label names (`anger`, `disgust`, `fear`, `happy`,
  `neutral`, `sad`).
- **RAVDESS** (Livingstone & Russo, 2018) — *research-only* cross-corpus
  benchmark, never used for training. CC BY-NC-SA 4.0 (non-commercial).
  Used in `scripts/evaluate.py` for unseen-speaker/cross-corpus metrics only.

## Third-party runtime

- torch (BSD), transformers (Apache-2.0), fastapi/uvicorn (MIT),
  soundfile (LGPL), soxr (LGPL), numpy (BSD), pyarrow (Apache-2.0),
  pytest (MIT), httpx (BSD).

## Notes

- All inference runs fully offline from local weights
  (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`).
- The RAVDESS non-commercial clause means a commercial deployment should
  re-validate on commercially licensed data (CREMA-D covers this).
