#!/usr/bin/env python
"""Gather one out-of-fold prediction per case into a single directory, per arm.

nnU-Net writes each fold's held-out predictions to
  $nnUNet_results/<Dataset>/<Trainer>__<Plans>__<config>/fold_<k>/validation/<case>.nii.gz
so the cohort-wide out-of-fold set is the union over folds — which is exactly what a paired
analysis needs, and exactly what is easy to get subtly wrong by copying the wrong fold's
prediction over a case. This script copies (or symlinks) them into one flat directory and
refuses to overwrite a case with a second fold's prediction, so a splits/results mismatch
surfaces as an error instead of as a silently better-looking Dice.

Usage:
  python scripts/training/collect_oof.py \
      --results-dir $nnUNet_results/Dataset501_PDAC/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres \
      --out /preds/cnn --folds 0 1 2 3 4
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="the <Trainer>__<Plans>__<config> directory for one arm")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4"])
    ap.add_argument("--symlink", action="store_true", help="symlink instead of copying")
    ap.add_argument("--expect", type=int, default=None,
                    help="expected case count; fail if the union does not match")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seen = {}
    for fold in args.folds:
        vdir = args.results_dir / f"fold_{fold}" / "validation"
        if not vdir.exists():
            raise SystemExit(f"missing {vdir} — fold {fold} has not finished training/validation")
        files = sorted(vdir.glob("*.nii.gz"))
        if not files:
            raise SystemExit(f"{vdir} contains no predictions")
        for f in files:
            cid = f.name[: -len(".nii.gz")]
            if cid in seen:
                raise SystemExit(
                    f"case {cid} appears in fold {seen[cid]} AND fold {fold} — the results "
                    "directory does not match splits_final.json. Do not analyse this; fix the "
                    "splits installed in nnUNet_preprocessed and re-predict.")
            seen[cid] = fold
            dst = args.out / f.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if args.symlink:
                dst.symlink_to(f.resolve())
            else:
                shutil.copy(f, dst)
        print(f"fold {fold}: {len(files)} predictions")

    print(f"\n{len(seen)} out-of-fold predictions in {args.out}")
    if args.expect is not None and len(seen) != args.expect:
        raise SystemExit(f"expected {args.expect} cases, got {len(seen)} — a paired analysis "
                         "needs one out-of-fold prediction for every cohort case.")


if __name__ == "__main__":
    main()
