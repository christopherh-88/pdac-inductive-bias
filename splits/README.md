# Frozen splits

This directory holds the frozen cohort and split artifacts. Once `splits_final.json` is
committed and the repo is tagged `prereg-v1`, nothing here is ever regenerated or edited.

Produced in order by:

1. `scripts/data/deduplicate.py` → `cohort.csv`, `duplicates.csv`
2. `scripts/analysis/compute_strata.py` → `strata.csv` (+ freezes `config/frozen_thresholds.yaml`)
3. `scripts/data/make_splits.py` → `splits_final.json` (nnU-Net format), `fold_assignment.csv`

`splits_final.json` is copied into `nnUNet_preprocessed/Dataset501_PDAC/` after preprocessing
so both arms train on identical folds.
