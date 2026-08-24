#!/usr/bin/env bash
# Tier A: 5 folds x 2 arms = 10 runs (10-20 GPU-days on one 24-48 GB card).
# Run after verify_pipeline.sh has passed and prereg-v1 is tagged.
set -euo pipefail

DATASET=${DATASET:-501}
CNN_PLANS=${CNN_PLANS:?Set CNN_PLANS per frozen config}
PRIMUS_TRAINER=${PRIMUS_TRAINER:?Set PRIMUS_TRAINER per frozen config}

NNUNET_COMMIT=$(pip freeze | grep nnunetv2 || echo "unknown")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

for FOLD in 0 1 2 3 4; do
  echo ">> [CNN] fold $FOLD"
  RUN_ID=$(python scripts/training/log_run.py start --arm cnn --fold "$FOLD" \
    --nnunet-commit "$NNUNET_COMMIT" --trainer-or-plans "$CNN_PLANS" --gpu "$GPU_NAME")
  nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" -p "$CNN_PLANS" --npz \
    && python scripts/training/log_run.py finish --run-id "$RUN_ID" --status completed \
    || { python scripts/training/log_run.py finish --run-id "$RUN_ID" --status crashed; exit 1; }

  echo ">> [Transformer] fold $FOLD"
  RUN_ID=$(python scripts/training/log_run.py start --arm transformer --fold "$FOLD" \
    --nnunet-commit "$NNUNET_COMMIT" --trainer-or-plans "$PRIMUS_TRAINER" --gpu "$GPU_NAME")
  nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" -tr "$PRIMUS_TRAINER" --npz \
    && python scripts/training/log_run.py finish --run-id "$RUN_ID" --status completed \
    || { python scripts/training/log_run.py finish --run-id "$RUN_ID" --status crashed; exit 1; }
done

echo ">> Tier A complete. Out-of-fold predictions live under \$nnUNet_results."
echo ">> Next: scripts/analysis/paired_analysis.py and scripts/analysis/occlusion_test.py"
