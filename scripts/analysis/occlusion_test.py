#!/usr/bin/env python
"""H2 occlusion test, applied identically to both arms.

For each test case: replace spherical shells at 10-20, 20-40, 40-80 mm from the manual
lesion surface with the shell's own mean HU (structure removed, intensity statistics
preserved), write the occluded volumes, re-run inference for both arms, and record the Dice
loss per shell relative to the unoccluded out-of-fold prediction.

Stage 1 (this script, --stage occlude): builds occluded images into
  <workdir>/occluded/shell_<lo>_<hi>/imagesTs/<case>_0000.nii.gz
Stage 2 (external): run nnUNetv2_predict for each arm on each shell directory, using the
  fold that held the case out (the script writes a manifest mapping case -> fold).
Stage 3 (this script, --stage score): computes Dice loss per case/arm/shell + patient-level
  bootstrap CI of the between-arm contrast in the decisive 40-80 mm shell (margin: 2 points).

Usage:
  python scripts/analysis/occlusion_test.py --stage occlude --cohort splits/cohort.csv \
      --strata splits/strata.csv --workdir /path/occlusion
  python scripts/analysis/occlusion_test.py --stage score --workdir /path/occlusion \
      --pred-base-cnn DIR --pred-base-tf DIR --refs DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from tqdm import tqdm

from metrics import CFG, load_mask, dice, bootstrap_ci

SHELLS = [tuple(s) for s in CFG["hypotheses"]["h2"]["occlusion_shells_mm"]]
LESION = CFG["labels"]["panorama"]["pdac_lesion"]


def occlude_case(case_id, img_path, manual_path, out_dirs):
    img = sitk.ReadImage(str(img_path))
    hu = sitk.GetArrayFromImage(img).astype(np.float32)
    seg = sitk.ReadImage(str(manual_path))
    lesion = (sitk.GetArrayViewFromImage(seg) == LESION).astype(np.uint8)
    if not lesion.any():
        return False
    lesion_img = sitk.GetImageFromArray(lesion)
    lesion_img.SetSpacing(seg.GetSpacing())
    # SignedMaurerDistanceMap's return is an unnamed temporary; GetArrayViewFromImage on it
    # directly is a zero-copy view that goes stale as soon as the temporary is garbage-collected.
    dmap_img = sitk.SignedMaurerDistanceMap(
        lesion_img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    dmap = sitk.GetArrayFromImage(dmap_img)
    for (lo, hi), out_dir in zip(SHELLS, out_dirs):
        shell = (dmap > lo) & (dmap <= hi)
        occluded = hu.copy()
        if shell.any():
            occluded[shell] = hu[shell].mean()   # the shell's own mean HU
        out = sitk.GetImageFromArray(occluded)
        out.CopyInformation(img)
        # case_id passed explicitly rather than re-derived from img_path.stem: a naive
        # `.replace("_0000", "")` corrupts case IDs whose study-number suffix itself contains
        # that substring (e.g. "100000_00001" -> "1000001"), silently mismatching cases in
        # stage_score() (which just `continue`s on a missing file, no error raised).
        sitk.WriteImage(out, str(out_dir / f"{case_id}_0000.nii.gz"), useCompression=True)
    return True


def stage_occlude(args):
    cohort = pd.read_csv(args.cohort)
    cases = cohort[cohort["has_manual_lesion"]]
    fold_map = pd.read_csv("splits/fold_assignment.csv")[["case_id", "fold"]]
    out_dirs = []
    for lo, hi in SHELLS:
        d = args.workdir / "occluded" / f"shell_{lo}_{hi}" / "imagesTs"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs.append(d)
    n = 0
    for r in tqdm(cases.itertuples(), total=len(cases)):
        n += occlude_case(r.case_id, Path(r.path), Path(r.manual_label), out_dirs)
    fold_map.to_csv(args.workdir / "case_fold_manifest.csv", index=False)
    print(f"Occluded {n} cases x {len(SHELLS)} shells under {args.workdir}/occluded/")
    print("Now run nnUNetv2_predict per arm per shell dir with -f <fold holding the case out> "
          "(see case_fold_manifest.csv), outputs to <workdir>/pred_<arm>/shell_<lo>_<hi>/")


def stage_score(args):
    st = CFG["statistics"]
    margin = CFG["hypotheses"]["h2"]["margin_dice_points"] / 100.0
    dec_lo, dec_hi = CFG["hypotheses"]["h2"]["decisive_shell_mm"]
    patients = pd.read_csv(args.cohort)[["case_id", "patient_id"]]

    rows = []
    base = {"cnn": args.pred_base_cnn, "tf": args.pred_base_tf}
    for arm in ("cnn", "tf"):
        for lo, hi in SHELLS:
            pdir = args.workdir / f"pred_{arm}" / f"shell_{lo}_{hi}"
            for f in sorted(pdir.glob("*.nii.gz")):
                cid = f.name.replace(".nii.gz", "")
                ref_p = args.refs / f"{cid}.nii.gz"
                base_p = Path(base[arm]) / f"{cid}.nii.gz"
                if not (ref_p.exists() and base_p.exists()):
                    continue
                ref, _ = load_mask(ref_p)
                d_occ = dice(load_mask(f)[0], ref)
                d_base = dice(load_mask(base_p)[0], ref)
                rows.append({"case_id": cid, "arm": arm, "shell": f"{lo}-{hi}",
                             "dice_base": d_base, "dice_occluded": d_occ,
                             "dice_loss": d_base - d_occ})
    df = pd.DataFrame(rows).merge(patients, on="case_id", how="left")
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "occlusion_per_case.csv", index=False)

    # Decisive contrast: transformer loss - CNN loss in the 40-80 mm shell, bootstrap CI
    dec = df[df["shell"] == f"{dec_lo}-{dec_hi}"].pivot_table(
        index=["case_id", "patient_id"], columns="arm", values="dice_loss").reset_index().dropna()
    dec["contrast"] = dec["tf"] - dec["cnn"]
    lo_ci, hi_ci = bootstrap_ci(dec, lambda b: b["contrast"].mean(),
                                st["bootstrap_resamples"], st["bootstrap_seed"])
    supported = (dec["contrast"].mean() >= margin) and (lo_ci > 0)
    print(f"H2 decisive shell {dec_lo}-{dec_hi} mm: mean(tf loss - cnn loss) = "
          f"{dec['contrast'].mean():.4f}, CI [{lo_ci:.4f}, {hi_ci:.4f}], margin {margin:.2f} "
          f"-> {'SUPPORTED' if supported else 'NOT SUPPORTED'}")
    yaml.safe_dump({"h2_contrast_mean": float(dec["contrast"].mean()),
                    "h2_ci": [float(lo_ci), float(hi_ci)], "h2_supported": bool(supported),
                    "n_cases": int(len(dec))}, open(args.out / "summary.yaml", "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["occlude", "score"], required=True)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--strata", default="splits/strata.csv", type=Path)
    ap.add_argument("--refs", type=Path)
    ap.add_argument("--pred-base-cnn", type=Path)
    ap.add_argument("--pred-base-tf", type=Path)
    ap.add_argument("--out", default=Path("results/h2"), type=Path)
    args = ap.parse_args()
    if args.stage == "occlude":
        stage_occlude(args)
    else:
        missing = [f"--{n.replace('_', '-')}" for n in ("refs", "pred_base_cnn", "pred_base_tf")
                   if getattr(args, n) is None]
        if missing:
            ap.error(f"--stage score requires {', '.join(missing)}")
        stage_score(args)


if __name__ == "__main__":
    main()
