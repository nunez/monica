# Evaluation

All numbers from `scripts/evaluate.py` on speaker-independent splits
(speakers never cross train/val/test), fully offline. Machine-readable
version: `bench/eval_results.json`.

<!-- EVAL_NUMBERS_START -->
## Headline results (CREMA-D, speaker-independent)

| metric | value |
|---|---|
| noul, seen words (n=151,360) | acc 0.873, AUC 0.850, ECE 0.069 |
| noul, **unseen** words (n=41,280) | acc 0.839, AUC 0.803, ECE 0.037 |
| choice (6-emotion, n=3,440) | acc 0.591, macro-F1 0.584, ECE 0.101 |
| RAVDESS cross-corpus, unseen words | acc 0.832 (chance 0.5), ECE 0.134 |

## Score questions (ordinal)

| scale | level MAE | RPS | ECE(argmax) |
|---|---|---|---|
| frustrated (trained scale) | 0.50 | 0.420 | 0.101 |
| intense (trained scale) | 0.55 | 0.479 | 0.068 |
| energetic (trained scale) | 0.93 | 0.721 | 0.082 |
| agitated (trained scale) | 0.62 | 0.497 | 0.067 |
| strained (**unseen** scale wording) | 0.92 | 0.768 | 0.159 |
| tense (**unseen** scale wording) | 1.13 | 0.801 | 0.129 |

## Stability & robustness

- Option-order stability: max prob delta 0.0e+00, argmax agreement 100%
- Packed-vs-separate questions: JS 4.8e-10, identical outputs
- Repetition of identical input: max delta 0.0e+00
- Audio formats (prob delta vs mono-16k reference): stereo16 0.000, sr48k 0.054, sr22k 0.047, sr8k 0.097, mono24 0.000, float 0.000
- Robustness gates: babble_10: low_info 0%/max-conf 0.27, babble_20: low_info 0%/max-conf 0.51, babble_5: low_info 0%/max-conf 0.60, clipped: low_info 25%/max-conf 0.72, music: low_info 100%/max-conf 0.00, phone_band: low_info 0%/max-conf 0.43, short_0.4s: low_info 100%/max-conf 0.00, silence: low_info 100%/max-conf 0.00

## Latency (M3 Ultra, warm)

| audio | encode median | p90 |
|---|---|---|
| 3.0 s | 50 ms | 60 ms |
| 10.0 s | 60 ms | 107 ms |
| 30.0 s | 90 ms | 107 ms |

Cold start: 6.4s load + 0.37s warmup. Full request 30s audio + 32 questions: median 180 ms (RTF 0.006). Peak RSS 3.1 GB.

_Choice baseline: majority-class accuracy ≈ 0.24; 6-class chance 0.167._
<!-- EVAL_NUMBERS_END -->

## Protocol

- **Training**: CREMA-D (7,442 clips, 6 emotions), speaker-independent split
  68/11/12 train/val/test. Frozen Gemma 4 E2B audio tower; only the question
  encoder + decision heads (~52 M trainable params) are trained.
- **Calibration**: per-question-type temperature scaling fitted on the val
  speakers' held-out split; ECE measured before/after on test.
- **Confidence**: normalized negentropy of the calibrated answer distribution
  (1 − H/log K), capped under low SNR/clipping (see `docs/GATE.md`).
- **Unseen words**: noul sentiment words never seen during training
  (word-level held-out battery).
- **Cross-corpus**: RAVDESS (research-only, never trained on) scored with the
  CREMA-trained model.

## Metric definitions

| metric | meaning |
|---|---|
| acc / macro_f1 | argmax accuracy / macro-F1 over choices |
| AUC | one-vs-rest ROC AUC for noul scores |
| ECE | expected calibration error (15 bins) |
| Brier | Brier score on probability vectors |
| RPS | ranked probability score for ordinal score questions |
| level MAE | mean absolute level error on score questions |
| order stability | max prob / argmax change under option reordering (JS div) |
| format robustness | prob delta vs mono-16k PCM reference for stereo / 48k / 22k / 8k / float / 24-bit |
| RTF | wall-clock processing time / audio duration (warm) |

## Robustness gates (enforced at serve time)

Silence, music, severe clipping and sub-0.5 s inputs are answered with priors
and `low_information: true` (confidence 0) without spending an audio encode;
low-SNR confidence caps keep 10 dB babble below 0.35 confidence.
