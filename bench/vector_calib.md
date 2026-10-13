# Choice-head scaling prototype: vector/matrix calibration beats temperature

Scripts: `scripts/vector_calibrate.py` (logit cache: `bench/vector_calib_logits.pt`,
artifact: `models/vector-scaling.pt`, results: `bench/vector_calib.json`).

Idea: temperature scaling is the 1-parameter special case of an affine map on
logits. Because choice logits are per-option and order-independent, we can fit
an identity-keyed affine layer on held-out val speakers:

- **vector**: p = softmax(exp(a_e)·l_e + b_e), 2K params, order-equivariant
- **matrix**: p = softmax(W·l + b) with panel canonicalized by emotion, then
  un-canonicalized — generalizes exactly for panels over the calibrated set

Selection: 4-fold **speaker-grouped** CV on val, never touching test.

## Choice (CREMA-D test, n=1720 panels; chance 0.167)

| scaling | acc | macro-F1 | ECE | Brier |
|---|---|---|---|---|
| temperature (deployed t=2.163) | 0.6128 | 0.6047 | 0.1257 | 0.5824 |
| temperature (refit t=1.935) | 0.6128 | 0.6047 | 0.0936 | 0.5729 |
| **vector** | **0.6785** | **0.6789** | 0.0749 | 0.4895 |
| **matrix** | 0.6779 | 0.6768 | **0.0316** | **0.4650** |

CV val NLL: temp 1.309 → vector 1.097 → matrix 1.052 (matrix selected).
Argmax accuracy is temperature-invariant — the +6.5 points come purely from
per-emotion identity information that a single scalar cannot express.

## RAVDESS cross-corpus (never trained on, mapped to 6; n=480)

| scaling | acc | macro-F1 | ECE |
|---|---|---|---|
| temperature | 0.4625 | 0.2677 | 0.0802 |
| vector | 0.5917 | 0.3160 | 0.1166 |
| matrix | **0.6104** | 0.3156 | 0.1503 |

+15 acc points and +5 macro-F1 cross-corpus. ECE degrades modestly
(miscalibrated confidence from the domain gap; SNR caps unchanged).

## Order/permutation stability

Identity-keyed scaling on all battery perm/subset panels: argmax agreement
1.0 vs canonical, JS ≈ 0.019 (explained entirely by subset renormalization,
not order sensitivity).

## noul / score: leave temperature as-is

- noul: deployed t=4.0 gives ECE 0.069 (matches suite). Wide-grid refit
  (t=5.8) **worsens** ECE to 0.130; per-word temps 0.139. Grid ceiling at
  4.0 is harmless — no win, skip.
- score: per-word wide temps (5.3–7.2) trade tiny RPS gains for worse
  exp-MAE (e.g. frustrated 0.4126/0.693 vs deployed 0.4203/0.642). Wash.

## Deployed

Vector scaling is live, **keyed by exact option criterion text** (not emotion
key): the first integration keyed slots by emotion identity, but option logits
are conditioned on the option's own text — applying a fit from one text variant
to a paraphrase panel *hurt* (0A-fit on 0B text: acc 0.494 vs temp 0.570).
Each distinct battery panel text set is fitted and registered separately;
`calibration.json` method reports `vector_scaling+temperature_fallback`.

Full-suite effect (CREMA-D test, n=3,440, both panel text sets):
acc 0.591 → **0.656**, macro-F1 0.584 → **0.655**, NLL 1.352 → **1.103**,
ECE 0.101 → **0.068**. Order stability, packed-vs-separate, determinism all
unchanged (exact). Unknown option text = pre-change temperature behavior.

Per-text self-fit test numbers: 0A temp 0.6128 → vec 0.6785;
0B temp 0.5698 → vec 0.6331.

To calibrate a new option text: extract val logits for it (one pass over val
features) and register its (u, b) via `scripts/upgrade_calibration.py`.

## Notes / open items

- Paraphrase fall-off is the cost of text keying: similar-but-unregistered
  wording gets no calibration. Candidate follow-up: embedding-keyed
  (u, b) predictor on top of qencoder option vectors.
- Cross-corpus (RAVDESS) choice accuracy benefits strongly (0.462 → 0.610
  with 0A-text fit) but ECE degrades out of domain (0.080 → 0.117).
