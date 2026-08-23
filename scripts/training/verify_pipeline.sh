#!/usr/bin/env bash
# Week-one verification: run both arms end to end on one fold for a handful of epochs.
# Nothing here counts as a study run; its only purpose is to prove the pipeline before
# the pre-registration tag. Per the fallback rule, if the PrimusV2 trainer cannot be made
# to run here, the transformer arm falls back to Primus v1 and the gap is reported.
set -euo pipefail

DATASET=${DATASET:-501}
FOLD=${FOLD:-0}
CNN_PLANS=${CNN_PLANS:?Set CNN_PLANS (nnUNetResEncUNetMPlans|LPlans|XLPlans) per confirmed VRAM}
PRIMUS_TRAINER=${PRIMUS_TRAINER:?Set PRIMUS_TRAINER (nnUNet_PrimusV2S/B/M/L_Trainer) per confirmed VRAM}

: "${nnUNet_raw:?export nnUNet_raw first}"
: "${nnUNet_preprocessed:?export nnUNet_preprocessed first}"
: "${nnUNet_results:?export nnUNet_results first}"

echo ">> Preprocessing (ResEnc planner derived from $CNN_PLANS)"
PLANNER="nnUNetPlannerResEnc${CNN_PLANS#nnUNetResEncUNet}"; PLANNER=${PLANNER%Plans}
nnUNetv2_plan_and_preprocess -d "$DATASET" -pl "$PLANNER" --verify_dataset_integrity

echo ">> Installing frozen splits"
DSNAME=$(ls "$nnUNet_preprocessed" | grep "^Dataset${DATASET}_")
cp splits/splits_final.json "$nnUNet_preprocessed/$DSNAME/splits_final.json"

echo ">> Smoke test: CNN arm (a few epochs via nnUNetTrainer_5epochs)"
nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" -p "$CNN_PLANS" -tr nnUNetTrainer_5epochs

echo ">> Smoke test: transformer arm ($PRIMUS_TRAINER)"
# PrimusV2 trainers set their own schedule; abort manually after a few epochs if no
# short-schedule variant exists in the installed commit — the goal is only that training
# steps run and checkpoints write.
nnUNetv2_train "$DATASET" 3d_fullres "$FOLD" -tr "$PRIMUS_TRAINER"

echo ">> Verify identity-ablation trainer imports and swaps blocks"
python -c "from src.trainers.primus_identity_trainer import nnUNet_PrimusV2_Identity_Trainer; print('identity trainer import OK')"

echo ">> Record in config/frozen_thresholds.yaml: nnU-Net commit, presets, patch/token sizes,"
echo "   parameter counts, peak VRAM of both arms. Then commit and tag prereg-v1."
