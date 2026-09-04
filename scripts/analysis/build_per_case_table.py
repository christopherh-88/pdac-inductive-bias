#!/usr/bin/env python
"""Deliverable 1: the paired per-case metrics table, both arms, all folds, no missing cells.

Every downstream analysis (H1 regression, stratified heatmaps, identity comparison, LOSO,
figures, failure gallery) reads Dice/NSD/HD95/detection/FP from *this* table rather than
recomputing surface distances five times over. Metrics are the frozen definitions in
scripts/analysis/metrics.py, which mirror config/analysis_config.yaml.

Long format: one row per (case_id, arm). Arms are given as repeatable --arm NAME:DIR pairs,
so the same script builds the Tier A table (cnn + tf), the H2b table (tf + identity_control),
and any Tier B table (an LOSO or seed-replicate arm) without special-casing.

Every case in --strata must have a prediction from every arm, or the script fails: a paired
analysis with silently dropped cases is not a paired analysis. Use --allow-missing only for
a deliberate partial run (it then reports exactly which cases were dropped, and drops them
from *all* arms so the table stays balanced).

Usage:
  python scripts/analysis/build_per_case_table.py \
      --arm cnn:/preds/cnn --arm tf:/preds/tf --refs /labels/manual \
      --strata splits/strata.csv --cohort splits/cohort.csv \
      --out results/per_case_metrics.csv
"""
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from metrics import CFG, all_metrics


def parse_arm(spec: str):
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"--arm expects NAME:DIR, got {spec!r}")
    name, _, path = spec.partition(":")
    return name, Path(path)


def build(arms, refs: Path, strata: pd.DataFrame, cohort: pd.DataFrame, source_col: str,
          allow_missing: bool) -> pd.DataFrame:
    ring = CFG["strata"]["cnr"]["ring_width_mm_primary"]
    cnr_cols = [c for c in strata.columns if c.startswith("cnr_ring")]

    # A case is usable only if the reference and *every* arm's prediction exist.
    usable, dropped = [], {}
    for cid in strata["case_id"]:
        missing = []
        if not (refs / f"{cid}.nii.gz").exists():
            missing.append("reference")
        missing += [name for name, d in arms if not (d / f"{cid}.nii.gz").exists()]
        (usable.append(cid) if not missing else dropped.setdefault(cid, missing))
    if dropped:
        msg = (f"{len(dropped)} of {len(strata)} cases lack a reference or an arm's prediction, "
               f"e.g. {list(dropped.items())[:5]}")
        if not allow_missing:
            raise SystemExit("ERROR: " + msg + "\nA paired table must be complete. Fix the "
                             "predictions, or pass --allow-missing to drop these from all arms.")
        print("WARNING: " + msg)

    strata = strata[strata["case_id"].isin(usable)]
    rows = []
    for name, pred_dir in arms:
        for r in tqdm(strata.itertuples(), total=len(strata), desc=f"metrics [{name}]"):
            m = all_metrics(pred_dir / f"{r.case_id}.nii.gz", refs / f"{r.case_id}.nii.gz")
            row = {"case_id": r.case_id, "arm": name, **m,
                   "volume_mm3": r.volume_mm3, "volume_tertile": r.volume_tertile,
                   "cnr_tertile": r.cnr_tertile,
                   "max_inplane_diam_mm": getattr(r, "max_inplane_diam_mm", float("nan")),
                   "above_2cm": getattr(r, "above_2cm", None)}
            for c in cnr_cols:
                row[c] = getattr(r, c)
            row["cnr"] = row.get(f"cnr_ring{ring}mm")
            rows.append(row)

    df = pd.DataFrame(rows)
    meta_cols = ["case_id", "patient_id"] + ([source_col] if source_col in cohort.columns else [])
    df = df.merge(cohort[meta_cols].drop_duplicates("case_id"), on="case_id", how="left")
    if source_col in df.columns and source_col != "source":
        df = df.rename(columns={source_col: "source"})
    if "source" not in df.columns:
        df["source"] = "unknown"

    fold_path = Path("splits/fold_assignment.csv")
    if fold_path.exists():
        df = df.merge(pd.read_csv(fold_path)[["case_id", "fold"]].drop_duplicates("case_id"),
                      on="case_id", how="left")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, type=parse_arm, metavar="NAME:DIR",
                    help="repeatable; e.g. --arm cnn:/preds/cnn --arm tf:/preds/tf")
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--strata", default="splits/strata.csv", type=Path)
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--out", default=Path("results/per_case_metrics.csv"), type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = build(args.arm, args.refs, pd.read_csv(args.strata), pd.read_csv(args.cohort),
               args.source_col, args.allow_missing)
    df.to_csv(args.out, index=False)

    print(f"\nWrote {args.out}: {len(df)} rows = {df['case_id'].nunique()} cases x "
          f"{df['arm'].nunique()} arms")
    summary = df.groupby("arm").agg(
        n=("dice", "size"), dice=("dice", "mean"), nsd2mm=("nsd2mm", "mean"),
        hd95=("hd95", "mean"), detection_sensitivity=("detected", "mean"),
        fp_per_case=("fp_count", "mean"))
    print(summary.round(4).to_string())
    print("\n(aggregate means only — no claim without the per-stratum intervals from "
          "paired_analysis.py / make_figures.py)")


if __name__ == "__main__":
    main()
