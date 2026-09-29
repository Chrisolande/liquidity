#!/usr/bin/env bash
# Pull all baseline outputs from Kaggle kernel → local pulled/
# Wipes pulled/ clean first so there's no stale data.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Cleaning pulled/ ==="
rm -rf pulled/
mkdir -p pulled/

echo "=== Pulling artifacts from Kaggle ==="

python3 kaggle_runner.py pull checkpoints/oof_champ_train.npy     pulled/oof_champ_train.npy
python3 kaggle_runner.py pull checkpoints/y_true.npy               pulled/y_true.npy
python3 kaggle_runner.py pull checkpoints/oof_raw_blend.npy        pulled/oof_raw_blend.npy
python3 kaggle_runner.py pull checkpoints/run_summary_baseline.json pulled/run_summary_baseline.json
python3 kaggle_runner.py pull submissions/submission_baseline.csv   pulled/submission_baseline.csv

echo ""
echo "=== pulled/ contents ==="
ls -lh pulled/

echo ""
echo "=== OOF Summary ==="
python3 - <<'EOF'
import numpy as np, json

oof  = np.load("pulled/oof_champ_train.npy")
y    = np.load("pulled/y_true.npy")
raw  = np.load("pulled/oof_raw_blend.npy")

import sys; sys.path.insert(0, ".")
from src.metrics import competition_score

rll, rauc, rcomp = competition_score(y, raw)
ll,  auc,  comp  = competition_score(y, oof)

print(f"  Raw blend  : Comp={rcomp:.5f} | AUC={rauc:.5f} | LL={rll:.5f}")
print(f"  Beta final : Comp={comp:.5f}  | AUC={auc:.5f}  | LL={ll:.5f}")

with open("pulled/run_summary_baseline.json") as f:
    meta = json.load(f)
print(f"\nSeeds used  : {meta['seeds']}")
print(f"Features    : {meta['feature_count']} ({meta['numeric_features']} numeric + {meta['categorical_features']} categorical)")
EOF
