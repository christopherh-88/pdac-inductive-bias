#!/usr/bin/env python
"""Occlusion test, stage 2: re-infer every occluded image with the fold that held it out.

occlusion_test.py --stage occlude writes one directory per shell containing every cohort case.
Those cases do not share a fold, and predicting them all with a single fold's checkpoint would
mean predicting most cases with a model that trained on them — the occlusion Dice loss would
then be measured against a memorised case rather than a held-out one, which quietly destroys
the comparison. This script splits each shell directory by fold, runs `nnUNetv2_predict` once
per (arm, shell, fold) with `-f <that fold>`, and merges the results back into the flat layout
occlusion_test.py --stage score expects:

    <workdir>/pred_<arm>/shell_<lo>_<hi>/<case_id>.nii.gz

Usage:
  python scripts/training/predict_occlusion.py --workdir /path/occlusion --dataset 501 \
      --arm cnn:-p:nnUNetResEncUNetLPlans --arm tf:-tr:nnUNet_PrimusV2M_Trainer
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import yaml

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "config" / "analysis_config.yaml"))
SHELLS = [tuple(s) for s in CFG["hypotheses"]["h2"]["occlusion_shells_mm"]]


def parse_arm(spec: str):
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--arm expects NAME:FLAG:VALUE (e.g. cnn:-p:nnUNetResEncUNetLPlans), got {spec!r}")
    return tuple(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--dataset", default="501")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--arm", action="append", required=True, type=parse_arm,
                    metavar="NAME:FLAG:VALUE")
    ap.add_argument("--manifest", default=None, type=Path,
                    help="default: <workdir>/case_fold_manifest.csv from the occlude stage")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="extra args passed through to nnUNetv2_predict")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    manifest = args.manifest or (args.workdir / "case_fold_manifest.csv")
    if not manifest.exists():
        raise SystemExit(f"missing {manifest} — run occlusion_test.py --stage occlude first")
    folds = pd.read_csv(manifest)[["case_id", "fold"]].drop_duplicates("case_id")

    for arm, flag, value in args.arm:
        for lo, hi in SHELLS:
            src = args.workdir / "occluded" / f"shell_{lo}_{hi}" / "imagesTs"
            dst = args.workdir / f"pred_{arm}" / f"shell_{lo}_{hi}"
            dst.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                print(f"SKIP {arm} shell {lo}-{hi}: {src} missing")
                continue
            for fold, g in folds.groupby("fold"):
                cases = [c for c in g["case_id"]
                         if (src / f"{c}_0000.nii.gz").exists()
                         and not (dst / f"{c}.nii.gz").exists()]
                if not cases:
                    continue
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_in = Path(tmp) / "in"
                    tmp_out = Path(tmp) / "out"
                    tmp_in.mkdir()
                    for c in cases:
                        (tmp_in / f"{c}_0000.nii.gz").symlink_to(
                            (src / f"{c}_0000.nii.gz").resolve())
                    cmd = ["nnUNetv2_predict", "-i", str(tmp_in), "-o", str(tmp_out),
                           "-d", str(args.dataset), "-c", args.configuration,
                           "-f", str(int(fold)), flag, value, *args.extra]
                    print(f">> {arm} shell {lo}-{hi} fold {fold}: {len(cases)} cases")
                    print("   " + " ".join(cmd))
                    if args.dry_run:
                        continue
                    subprocess.run(cmd, check=True)
                    for f in tmp_out.glob("*.nii.gz"):
                        shutil.move(str(f), dst / f.name)
            n = len(list(dst.glob("*.nii.gz")))
            print(f"   {arm} shell {lo}-{hi}: {n} predictions in {dst}")

    print("\nNext: python scripts/analysis/occlusion_test.py --stage score "
          f"--workdir {args.workdir} --pred-base-cnn DIR --pred-base-tf DIR --refs DIR")


if __name__ == "__main__":
    main()
