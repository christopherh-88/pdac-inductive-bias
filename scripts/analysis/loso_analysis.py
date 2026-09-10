#!/usr/bin/env python
"""Deliverable 5: leave-one-source-out results for both arms, and the table H3 consumes.

Pairs each case's held-out-source Dice (from the LOSO model that never saw that source)
against its in-domain out-of-fold Dice (from the Tier A cross-validation model, which did),
per arm. The difference is the cross-source Dice loss — the quantity H3 then asks how much of
sits inside the boundary-tolerance floor.

Prediction layout, one directory per arm containing one subdirectory per held-out source:
    <pred_dir>/loso_<Source>/<case_id>.nii.gz
matching what run_tier_b.sh writes and what make_loso_splits.py's fold mapping names.

Outputs
  loso_per_case.csv   case_id, arm, source, dice/nsd/hd95/detection/FP in domain and LOSO
  loso_dice.csv       the exact 4-column table boundary_tolerance.py --loso expects
  by_source.csv       per (arm, source) mean drop with patient-level bootstrap CIs
  summary.yaml

Usage:
  python scripts/analysis/loso_analysis.py --metrics-table results/per_case_metrics.csv \
      --pred cnn:/preds/loso/cnn --pred tf:/preds/loso/tf --refs /labels/manual \
      --cohort splits/cohort.csv --out results/loso
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from metrics import CFG, all_metrics, bootstrap_ci, find_case_file


def parse_pred(spec: str):
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"--pred expects NAME:DIR, got {spec!r}")
    name, _, path = spec.partition(":")
    return name, Path(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-table", required=True, type=Path,
                    help="in-domain per-case metrics from build_per_case_table.py")
    ap.add_argument("--pred", action="append", required=True, type=parse_pred, metavar="NAME:DIR",
                    help="repeatable; DIR contains loso_<Source>/ subdirectories")
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--loso-folds", default="splits/loso_folds.csv", type=Path)
    ap.add_argument("--out", default=Path("results/loso"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    st = CFG["statistics"]
    in_domain = pd.read_csv(args.metrics_table)
    if "source" not in in_domain.columns:
        raise SystemExit(f"{args.metrics_table} has no 'source' column — rebuild it with "
                         "build_per_case_table.py against a cohort that carries the source.")
    sources = sorted(in_domain["source"].dropna().unique())
    if args.loso_folds.exists():
        sources = pd.read_csv(args.loso_folds)["held_out_source"].tolist()

    rows = []
    for arm, root in args.pred:
        for source in sources:
            pdir = root / f"loso_{source}"
            if not pdir.exists():
                print(f"SKIP {arm} / {source}: no directory {pdir}")
                continue
            cases = in_domain[(in_domain["arm"] == arm) & (in_domain["source"] == source)]
            if cases.empty:
                print(f"SKIP {arm} / {source}: no in-domain rows for this arm+source")
                continue
            for r in tqdm(cases.itertuples(), total=len(cases), desc=f"{arm}/{source}"):
                pred = find_case_file(pdir, r.case_id)
                ref = find_case_file(args.refs, r.case_id)
                if pred is None or ref is None:
                    continue
                m = all_metrics(pred, ref)
                rows.append({
                    "case_id": r.case_id, "arm": arm, "source": source,
                    "volume_tertile": r.volume_tertile, "cnr_tertile": r.cnr_tertile,
                    "dice_in_domain": r.dice, "dice_loso": m["dice"],
                    "nsd_in_domain": r.nsd2mm, "nsd_loso": m["nsd2mm"],
                    "hd95_in_domain": getattr(r, "hd95", np.nan), "hd95_loso": m["hd95"],
                    "detected_in_domain": getattr(r, "detected", np.nan),
                    "detected_loso": m["detected"],
                    "fp_in_domain": getattr(r, "fp_count", np.nan), "fp_loso": m["fp_count"],
                })
    if not rows:
        raise SystemExit("No LOSO predictions matched — check --pred layout "
                         "(<dir>/loso_<Source>/<case_id>.nii.gz)")

    df = pd.DataFrame(rows)
    cohort = pd.read_csv(args.cohort)[["case_id", "patient_id"]].drop_duplicates("case_id")
    df = df.merge(cohort, on="case_id", how="left")
    df["dice_drop"] = df["dice_in_domain"] - df["dice_loso"]
    df["nsd_drop"] = df["nsd_in_domain"] - df["nsd_loso"]
    df.to_csv(args.out / "loso_per_case.csv", index=False)

    # The exact contract boundary_tolerance.py --loso reads.
    df[["case_id", "arm", "dice_in_domain", "dice_loso"]].to_csv(
        args.out / "loso_dice.csv", index=False)

    by_source = []
    for (arm, source), g in df.groupby(["arm", "source"]):
        lo, hi = bootstrap_ci(g, lambda b: b["dice_drop"].mean(),
                              st["bootstrap_resamples"], st["bootstrap_seed"])
        by_source.append({
            "arm": arm, "held_out_source": source, "n": len(g),
            "dice_in_domain": g["dice_in_domain"].mean(), "dice_loso": g["dice_loso"].mean(),
            "dice_drop": g["dice_drop"].mean(), "dice_drop_ci_lo": lo, "dice_drop_ci_hi": hi,
            "nsd_in_domain": g["nsd_in_domain"].mean(), "nsd_loso": g["nsd_loso"].mean(),
            "detection_in_domain": g["detected_in_domain"].mean(),
            "detection_loso": g["detected_loso"].mean(),
            "fp_per_case_in_domain": g["fp_in_domain"].mean(),
            "fp_per_case_loso": g["fp_loso"].mean(),
        })
    by_source = pd.DataFrame(by_source)
    by_source.to_csv(args.out / "by_source.csv", index=False)

    overall = {}
    for arm, g in df.groupby("arm"):
        lo, hi = bootstrap_ci(g, lambda b: b["dice_drop"].mean(),
                              st["bootstrap_resamples"], st["bootstrap_seed"] + 1)
        overall[arm] = {"n": int(len(g)), "mean_dice_drop": float(g["dice_drop"].mean()),
                        "mean_dice_drop_ci": [lo, hi],
                        "mean_dice_in_domain": float(g["dice_in_domain"].mean()),
                        "mean_dice_loso": float(g["dice_loso"].mean())}

    yaml.safe_dump({"overall": overall,
                    "sources": sorted(df["source"].unique().tolist()),
                    "note": "cross-source Dice loss only; H3's share-inside-floor verdict "
                            "comes from boundary_tolerance.py --loso results/loso/loso_dice.csv"},
                   open(args.out / "summary.yaml", "w"), sort_keys=False)

    print("\n" + by_source.round(4).to_string(index=False))
    for arm, v in overall.items():
        print(f"\n{arm}: mean cross-source Dice drop {v['mean_dice_drop']:.4f} "
              f"CI [{v['mean_dice_drop_ci'][0]:.4f}, {v['mean_dice_drop_ci'][1]:.4f}] (n={v['n']})")
    print(f"\nWrote {args.out}/ — feed loso_dice.csv to boundary_tolerance.py --loso for H3.")


if __name__ == "__main__":
    main()
