#!/usr/bin/env python
"""Score Tier-A-lite fold-0 held-out predictions against ground truth, CPU-only, local.

NOT Tier A: this is a diagnostic preview over Dataset601_PDACTierALite's 7-case fold-0
validation split (32-case real-lesion pool, single P100, single fold) -- not the pre-registered
478-case cohort, and not statistically powered or fold-averaged. See preregistration/DEVIATIONS.md
and the module docstrings of phase1_stage_pool.py / phase1_train_lite.py / phase1_predict_lite.py
for the full chain this sits at the end of.

Reuses the same frozen metric definitions (dice, nsd2mm, hd95, detected, fp_count) as the real
Tier A analysis pipeline (scripts/analysis/metrics.py / config/analysis_config.yaml) so numbers
here are directly comparable in *definition*, even though the cohort and compute budget are not.

Input layout expected (matches phase1_predict_lite.py's kernel output):
  <preds-root>/<arm_label>/<case_id>.nii[.gz]
  <refs>/<case_id>.nii[.gz]                  (labelsTr from the staged raw dataset)

Usage:
  python scripts/analysis/phase1_eval_lite.py \
      --preds-root /path/to/phase1_predict/predictions \
      --refs /path/to/phase1_train/nnUNet_raw/Dataset601_PDACTierALite/labelsTr \
      --val-cases 100009_00001 100101_00001 ... \
      --out results/phase1_lite_eval.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from metrics import all_metrics


def _find(dir_: Path, case_id: str) -> Path:
    matches = sorted(dir_.glob(f"{case_id}.nii*"))
    if not matches:
        raise SystemExit(f"no file for case {case_id} in {dir_} (looked for {case_id}.nii*)")
    if len(matches) > 1:
        raise SystemExit(f"ambiguous files for case {case_id} in {dir_}: {matches}")
    return matches[0]


def score_arm(arm_label: str, pred_dir: Path, refs: Path, case_ids: list[str]) -> pd.DataFrame:
    rows = []
    for cid in case_ids:
        pred_path = _find(pred_dir, cid)
        ref_path = _find(refs, cid)
        m = all_metrics(pred_path, ref_path)
        rows.append({"case_id": cid, "arm": arm_label, **m})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-root", required=True, type=Path,
                    help="directory containing one subdirectory per arm, e.g. cnn_resenc_m/, "
                         "transformer_primusv2s/ -- as shipped by phase1_predict_lite.py")
    ap.add_argument("--refs", required=True, type=Path, help="labelsTr directory")
    ap.add_argument("--val-cases", nargs="+", default=None,
                    help="case ids to score; defaults to every arm subdirectory's case ids "
                         "intersected together, so a missing prediction fails loudly instead of "
                         "silently narrowing the cohort")
    ap.add_argument("--out", default=Path("results/phase1_lite_eval.csv"), type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    arm_dirs = sorted(d for d in args.preds_root.iterdir() if d.is_dir())
    if not arm_dirs:
        raise SystemExit(f"no arm subdirectories found under {args.preds_root}")

    if args.val_cases is None:
        per_arm_ids = [{p.name.split(".nii")[0] for p in d.glob("*.nii*")} for d in arm_dirs]
        case_ids = sorted(set.intersection(*per_arm_ids))
        if not case_ids:
            raise SystemExit(f"arms have no case ids in common: {[sorted(s) for s in per_arm_ids]}")
    else:
        case_ids = args.val_cases

    dfs = [score_arm(d.name, d, args.refs, case_ids) for d in arm_dirs]
    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(args.out, index=False)

    print(f"Wrote {args.out}: {len(df)} rows = {len(case_ids)} cases x {len(arm_dirs)} arms")
    print(f"case ids scored: {case_ids}")
    summary = df.groupby("arm").agg(
        n=("dice", "size"), dice=("dice", "mean"), nsd2mm=("nsd2mm", "mean"),
        hd95=("hd95", "mean"), detection_sensitivity=("detected", "mean"),
        fp_per_case=("fp_count", "mean"))
    print(summary.round(4).to_string())
    print("\nTier-A-lite preview only: 7 held-out cases, single fold, single P100 -- not "
          "statistically powered, not comparable in magnitude to a Tier A result. See "
          "preregistration/DEVIATIONS.md.")


if __name__ == "__main__":
    main()
