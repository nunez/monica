#!/bin/zsh
# Full offline pipeline: extract features -> train heads -> calibrate -> evaluate
# (features depend only on the frozen audio tower + audioio; delete
#  data/features/*.pt to force re-extraction)
set -e
export PYTHONPATH=src
E=(.venv/bin/python scripts/extract_features.py)
"$E[@]" --manifest data/manifest_cremad.json --out data/features/cremad --start 0 --end 1860 --suffix _a
"$E[@]" --manifest data/manifest_cremad.json --out data/features/cremad --start 1860 --end 3720 --suffix _b
"$E[@]" --manifest data/manifest_cremad.json --out data/features/cremad --start 3720 --end 5580 --suffix _c
"$E[@]" --manifest data/manifest_cremad.json --out data/features/cremad --start 5580 --suffix _d
"$E[@]" --manifest data/robust_manifest.json --out data/features/robust --suffix _robust
"$E[@]" --manifest data/manifest_ravdess.json --out data/features/ravdess --suffix _ravdess
.venv/bin/python scripts/train_heads.py --features data/features/cremad_a.pt data/features/cremad_b.pt data/features/cremad_c.pt data/features/cremad_d.pt --layer-ids-file data/selected_layers.json --epochs 30
.venv/bin/python scripts/calibrate.py --features data/features/cremad_a.pt data/features/cremad_b.pt data/features/cremad_c.pt data/features/cremad_d.pt --max-clips 450
.venv/bin/python scripts/evaluate.py
echo PIPELINE_DONE
