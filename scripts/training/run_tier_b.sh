#!/usr/bin/env bash
# Tier B: 15 runs on top of Tier A's 10.
#   leave-one-source-out, both arms   -> 2 arms x N sources (3 sources = 6 runs)
#   identity control, all five folds  -> 5 runs
#   seed replicates on the replicate fold, both arms, two extra seeds -> 4 runs
#
# Prerequisites: Tier A finished, prereg-v1 tagged, and
#   python scripts/data/make_loso_splits.py --emit-combined splits/splits_with_loso.json
#   cp splits/splits_with_loso.json $nnUNet_preprocessed/<Dataset>/splits_final.json
# so LOSO folds are addressable as nnU-Net fold indices 5, 6, ... (the five frozen CV folds
# keep indices 0-4 byte-identically; see make_loso_splits.py).
set -euo pipefail

# No --npz on any training command: see run_tier_a.sh.
DATASET=${DATASET:-501}
CNN_PLANS=${CNN_PLANS:?Set CNN_PLANS per frozen config}
PRIMUS_TRAINER=${PRIMUS_TRAINER:?Set PRIMUS_TRAINER per frozen config}
IDENTITY_TRAINER=${IDENTITY_TRAINER:-nnUNet_PrimusV2_Identity_Trainer}
LOSO_FOLDS_CSV=${LOSO_FOLDS_CSV:-splits/loso_folds.csv}
# Seed replicates: the extra seeds and the fold they run on come from the frozen config.
REPLICATE_FOLD=$(python -c "import yaml;print(yaml.safe_load(open('config/analysis_config.yaml'))['training']['seed_replicates']['fold'])")
EXTRA_SEEDS=$(python -c "import yaml;print(' '.join(map(str,yaml.safe_load(open('config/analysis_config.yaml'))['training']['seed_replicates']['extra_seeds'])))")

NNUNET_COMMIT=$(pip freeze | grep nnunetv2 || echo "unknown")
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")

train () {  # arm, fold-label, seed, trainer_or_plans, extra nnUNetv2_train args...
  local arm="$1" fold_label="$2" seed="$3" tp="$4"; shift 4
  echo ">> [$arm] $fold_label seed=$seed"
  local rid
  rid=$(python scripts/training/log_run.py start --arm "$arm" --fold "$fold_label" --seed "$seed" \
        --nnunet-commit "$NNUNET_COMMIT" --trainer-or-plans "$tp" --gpu "$GPU_NAME")
  if nnUNetv2_train "$@"; then
    python scripts/training/log_run.py finish --run-id "$rid" --status completed
  else
    python scripts/training/log_run.py finish --run-id "$rid" --status crashed
    exit 1
  fi
}

echo "===== Leave-one-source-out (both arms) ====="
[ -f "$LOSO_FOLDS_CSV" ] || { echo "Missing $LOSO_FOLDS_CSV — run make_loso_splits.py first"; exit 1; }
tail -n +2 "$LOSO_FOLDS_CSV" | while IFS=, read -r loso_index nnunet_fold source n_val n_train; do
  train cnn         "loso_${source}" default "$CNN_PLANS" \
        "$DATASET" 3d_fullres "$nnunet_fold" -p "$CNN_PLANS"
  train transformer "loso_${source}" default "$PRIMUS_TRAINER" \
        "$DATASET" 3d_fullres "$nnunet_fold" -tr "$PRIMUS_TRAINER"
done

echo "===== Identity control (H2b), all five folds ====="
export PRIMUS_BASE_TRAINER="$PRIMUS_TRAINER"
for FOLD in 0 1 2 3 4; do
  train identity_control "$FOLD" default "$IDENTITY_TRAINER" \
        "$DATASET" 3d_fullres "$FOLD" -tr "$IDENTITY_TRAINER"
done

echo "===== Seed replicates on fold $REPLICATE_FOLD, both arms ====="
# nnU-Net has no --seed flag, and it keys its output directory off the trainer class name:
# re-running the same command would reproduce the same initialization AND overwrite the first
# checkpoint. src/trainers/seed_variant_trainers.py generates one trainer class per replicate
# seed, which fixes both at once (different seed, different results directory).
for SEED in $EXTRA_SEEDS; do
  train cnn         "$REPLICATE_FOLD" "$SEED" "nnUNetTrainer_seed${SEED} / $CNN_PLANS" \
        "$DATASET" 3d_fullres "$REPLICATE_FOLD" -p "$CNN_PLANS" -tr "nnUNetTrainer_seed${SEED}"
  train transformer "$REPLICATE_FOLD" "$SEED" "nnUNet_Primus_seed${SEED}_Trainer" \
        "$DATASET" 3d_fullres "$REPLICATE_FOLD" -tr "nnUNet_Primus_seed${SEED}_Trainer"
done

echo ">> Tier B complete. Next:"
echo "   scripts/analysis/loso_analysis.py, identity_comparison.py, build_variance_tables.py"
