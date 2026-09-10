#!/usr/bin/env python
"""NIH negative-control harness: false positives per case where there is no lesion at all.

NIH Pancreas-CT cases (kidney donors / no pancreatic lesions) are never a tumor test set --
per the pre-registration they are used only to count false-positive lesions per case, in
domain and (separately, once leave-one-source-out predictions exist) under source shift. Since
these cases have no manual lesion label, the "reference" is an empty mask over the case's own
geometry, and every predicted foreground connected component >= 100 mm^3 (the frozen FP
definition in config/analysis_config.yaml) counts as a false positive by construction.

Deliverable 8 asks for the rate in domain *and under source shift*, which is one prediction
set per (arm, condition): --pred ARM:CONDITION:DIR, repeatable. The in-domain condition uses
the Tier A cross-validation models' predictions on the NIH cases; the shifted condition uses
a leave-one-source-out model's. Every (arm, condition) is reported separately and the
in-domain -> shifted change is reported per arm, since the interesting quantity is whether an
arm starts hallucinating lesions when the acquisition source changes, not its absolute rate.

Usage:
  python scripts/analysis/nih_false_positives.py --pred cnn:in_domain:DIR \
      --pred cnn:source_shift:DIR --pred tf:in_domain:DIR --pred tf:source_shift:DIR \
      --cohort splits/cohort.csv --out results/nih_fp
  # short form, in-domain only:
  python scripts/analysis/nih_false_positives.py --pred-cnn DIR --pred-tf DIR \
      --cohort splits/cohort.csv --out results/nih_fp
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

from metrics import CFG, false_positives, bootstrap_ci, find_case_file


def count_case(pred_path: Path):
    img = sitk.ReadImage(str(pred_path))
    arr = sitk.GetArrayViewFromImage(img) > 0
    spacing_zyx = tuple(reversed(img.GetSpacing()))
    ref = np.zeros_like(arr)  # NIH cases: no lesion, by construction
    return false_positives(arr, ref, spacing_zyx)


def parse_pred(spec: str):
    parts = spec.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--pred expects ARM:CONDITION:DIR (e.g. cnn:in_domain:/preds/cnn), got {spec!r}")
    arm, condition, path = parts
    return arm, condition, Path(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", default=[], type=parse_pred,
                    metavar="ARM:CONDITION:DIR", help="repeatable")
    ap.add_argument("--pred-cnn", type=Path, help="shorthand for --pred cnn:in_domain:DIR")
    ap.add_argument("--pred-tf", type=Path, help="shorthand for --pred tf:in_domain:DIR")
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--nih-value", default="NIH",
                    help="value in --source-col identifying the NIH negative-control cases")
    ap.add_argument("--out", default=Path("results/nih_fp"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    preds = list(args.pred)
    if args.pred_cnn:
        preds.append(("cnn", "in_domain", args.pred_cnn))
    if args.pred_tf:
        preds.append(("tf", "in_domain", args.pred_tf))
    if not preds:
        ap.error("give at least one --pred ARM:CONDITION:DIR (or --pred-cnn/--pred-tf)")

    cohort = pd.read_csv(args.cohort)
    if args.source_col not in cohort.columns:
        raise SystemExit(f"'{args.source_col}' not in {args.cohort} — set --source-col")
    nih = cohort[cohort[args.source_col] == args.nih_value]
    if len(nih) == 0:
        raise SystemExit(f"No cases with {args.source_col}=={args.nih_value!r} in {args.cohort}")

    rows = []
    for arm, condition, pred_dir in preds:
        for _, r in nih.iterrows():
            p = find_case_file(pred_dir, r['case_id'])
            if p is None:
                continue
            rows.append({"case_id": r["case_id"], "patient_id": r["patient_id"], "arm": arm,
                         "condition": condition, "fp_count": count_case(p)})
    if not rows:
        raise SystemExit(
            f"No prediction files matched any NIH case_id under {args.pred_cnn} or {args.pred_tf} "
            "— check the directories and the '<case_id>.nii.gz' naming convention"
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "nih_fp_per_case.csv", index=False)

    st = CFG["statistics"]
    summary, conditions = {}, sorted(df["condition"].unique())
    for arm, g_arm in df.groupby("arm"):
        entry = {}
        for condition, g in g_arm.groupby("condition"):
            lo, hi = bootstrap_ci(g, lambda b: b["fp_count"].mean(),
                                  st["bootstrap_resamples"], st["bootstrap_seed"])
            entry[condition] = {
                "n_cases": int(len(g)),
                "mean_fp_per_case": float(g["fp_count"].mean()),
                "mean_fp_per_case_ci": [lo, hi],
                "pct_cases_with_any_fp": float((g["fp_count"] > 0).mean()),
            }
            print(f"{arm} / {condition}: mean FP/case = {entry[condition]['mean_fp_per_case']:.3f} "
                  f"CI [{lo:.3f}, {hi:.3f}], {entry[condition]['pct_cases_with_any_fp']:.0%} of "
                  f"cases have >=1 FP  (n={entry[condition]['n_cases']})")

        # Paired in-domain -> shifted change on the same NIH cases, where both exist.
        if "in_domain" in entry and len(entry) > 1:
            base = g_arm[g_arm["condition"] == "in_domain"].set_index("case_id")
            for condition in [c for c in entry if c != "in_domain"]:
                shifted = g_arm[g_arm["condition"] == condition].set_index("case_id")
                common = base.index.intersection(shifted.index)
                if not len(common):
                    continue
                paired = pd.DataFrame({
                    "patient_id": base.loc[common, "patient_id"],
                    "delta": shifted.loc[common, "fp_count"] - base.loc[common, "fp_count"],
                }).reset_index()
                lo, hi = bootstrap_ci(paired, lambda b: b["delta"].mean(),
                                      st["bootstrap_resamples"], st["bootstrap_seed"] + 1)
                entry[f"delta_{condition}_minus_in_domain"] = {
                    "n_cases": int(len(common)), "mean_delta_fp_per_case": float(paired["delta"].mean()),
                    "mean_delta_fp_per_case_ci": [lo, hi]}
                print(f"{arm}: FP/case change {condition} - in_domain = "
                      f"{paired['delta'].mean():+.3f} CI [{lo:+.3f}, {hi:+.3f}] "
                      f"(n={len(common)} paired cases)")

        # Backward-compatible flat keys: the in-domain rate stays at the top level per arm.
        if "in_domain" in entry:
            entry.update({k: v for k, v in entry["in_domain"].items()})
        summary[arm] = entry

    summary["conditions"] = conditions
    yaml.safe_dump(summary, open(args.out / "summary.yaml", "w"))
    print(f"\nWrote {args.out}/")


if __name__ == "__main__":
    main()
