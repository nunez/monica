# EmbeddingGemma 2 probe (CREMA-D, speaker-independent)

Experiment: can `google/embeddinggemma-2` (unified 768-d audio+text embedding
space) help the monica System One? Scripts: `scripts/embedgemma_anchor.py`
(.venv-eg2), `scripts/anchor_bench.py` (main venv). Raw outputs:
`bench/embedgemma_cos.pt`, `bench/anchor_bench.json`.

All on identical 978 test / 896 val clips (val used only for temperature /
linear probes; chance = 0.167).

| system | acc | macro-F1 | ECE |
|---|---|---|---|
| EG2 zero-shot cosine vs description anchors | 0.166 | 0.114 | 0.004 |
| EG2 zero-shot cosine vs single-word anchors | 0.161 | 0.080 | 0.012 |
| EG2 cosine, logistic stack alone | 0.212 | 0.136 | 0.032 |
| EG2 768-d full logistic probe | 0.407 | 0.388 | — |
| monica head (deployed, calibrated) | 0.608 | 0.600 | 0.121 |
| head logits + logistic refit on val | **0.684** | **0.683** | **0.039** |
| head logits + EG2 cosines (stack both) | 0.679 | 0.677 | 0.035 |

Agreement head vs anchor: argmax 0.126, JS 0.097, per-emotion Spearman
0.03–0.19 — statistically orthogonal.

## Verdict

- **Zero-shot fails at chance.** Cosine to emotion anchor text carries no
  usable affect signal. Expected: CREMA-D clips are the same sentences across
  emotions, and EG2's audio encoder is trained for semantic content/sound
  events (MAEB ~49), not vocal prosody. Anchor temperature fits collapse to
  ~1.1 (uniform distributions).
- **No complementarity to stack.** Adding EG2 cosines to head logits does not
  improve over head alone (0.679 vs 0.684); correlation ~0.05 confirms there
  is no shared affect axis.
- **Some affect exists in the space (0.407 probed)** but far below the
  Gemma 4 E2B tower; not worth an extra 300M-param encoder at serve time.
- Confirms the monica bet: affect lives in the audio tower's intermediate
  layers, not in a semantic audio-text shared space. The Speech2Vec-style
  alignment target would likely wash out prosody — skip it.

## Side finding (worth acting on)

A plain logistic refit of the **existing** head logits on the val split gives
+7.6 acc points and ECE 0.121 → 0.039. Temperature scaling alone is leaving
accuracy on the table; consider fitting a per-panel linear rescoring layer
(same ~36 params, no architecture change) at calibration time.
