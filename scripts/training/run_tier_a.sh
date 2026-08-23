#!/usr/bin/env bash
# Tier A: 5 folds x 2 arms = 10 runs (10-20 GPU-days on one 24-48 GB card).
# Run after verify_pipeline.sh has passed and prereg-v1 is tagged.
set -euo pipefail

DATASET=${DATASET:-501}
CNN_PLANS=${CNN_PLANS:?Set CNN_PLANS per frozen config}
PRIMUS_TRAINER=${PRIMUS_TRAINER:?Set PRIMUS_TRAINER per frozen config}

for FOLD in 0 1 2 3 4; do
  echo ">> [CNN] fold $FOLD"
  nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" -p "$CNN_PLANS" --npz
  echo ">> [Transformer] fold $FOLD"
  nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" -tr "$PRIMUS_TRAINER" --npz
done

echo ">> Tier A complete. Out-of-fold predictions live under \$nnUNet_results."
echo ">> Next: scripts/analysis/paired_analysis.py and scripts/analysis/occlusion_test.py"
