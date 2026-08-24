#!/usr/bin/env python
"""NIH negative-control harness: false positives per case where there is no lesion at all.

NIH Pancreas-CT cases (kidney donors / no pancreatic lesions) are never a tumor test set --
per the pre-registration they are used only to count false-positive lesions per case, in
domain and (separately, once leave-one-source-out predictions exist) under source shift. Since
these cases have no manual lesion label, the "reference" is an empty mask over the case's own
geometry, and every predicted foreground connected component >= 100 mm^3 (the frozen FP
definition in config/analysis_config.yaml) counts as a false positive by construction.

Usage:
  python scripts/analysis/nih_false_positives.py --pred-cnn DIR --pred-tf DIR \
      --cohort splits/cohort.csv [--source-col source] --out results/nih_fp
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

from metrics import CFG, false_positives, bootstrap_ci


def count_case(pred_path: Path):
    img = sitk.ReadImage(str(pred_path))
    arr = sitk.GetArrayViewFromImage(img) > 0
    spacing_zyx = tuple(reversed(img.GetSpacing()))
    ref = np.zeros_like(arr)  # NIH cases: no lesion, by construction
    return false_positives(arr, ref, spacing_zyx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-cnn", required=True, type=Path)
    ap.add_argument("--pred-tf", required=True, type=Path)
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--nih-value", default="NIH",
                    help="value in --source-col identifying the NIH negative-control cases")
    ap.add_argument("--out", default=Path("results/nih_fp"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort)
    if args.source_col not in cohort.columns:
        raise SystemExit(f"'{args.source_col}' not in {args.cohort} — set --source-col")
    nih = cohort[cohort[args.source_col] == args.nih_value]
    if len(nih) == 0:
        raise SystemExit(f"No cases with {args.source_col}=={args.nih_value!r} in {args.cohort}")

    rows = []
    for arm, pred_dir in (("cnn", args.pred_cnn), ("tf", args.pred_tf)):
        for _, r in nih.iterrows():
            p = pred_dir / f"{r['case_id']}.nii.gz"
            if not p.exists():
                continue
            rows.append({"case_id": r["case_id"], "patient_id": r["patient_id"], "arm": arm,
                         "fp_count": count_case(p)})
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "nih_fp_per_case.csv", index=False)

    st = CFG["statistics"]
    summary = {}
    for arm, g in df.groupby("arm"):
        lo, hi = bootstrap_ci(g, lambda b: b["fp_count"].mean(),
                              st["bootstrap_resamples"], st["bootstrap_seed"])
        summary[arm] = {
            "n_cases": int(len(g)),
            "mean_fp_per_case": float(g["fp_count"].mean()),
            "mean_fp_per_case_ci": [lo, hi],
            "pct_cases_with_any_fp": float((g["fp_count"] > 0).mean()),
        }
        print(f"{arm}: mean FP/case = {summary[arm]['mean_fp_per_case']:.3f} "
              f"CI [{lo:.3f}, {hi:.3f}], {summary[arm]['pct_cases_with_any_fp']:.0%} of cases "
              f"have >=1 FP  (n={summary[arm]['n_cases']})")

    yaml.safe_dump(summary, open(args.out / "summary.yaml", "w"))
    print(f"Wrote {args.out}/")


if __name__ == "__main__":
    main()
