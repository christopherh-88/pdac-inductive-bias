#!/usr/bin/env python
"""Freeze the patient-level 5-fold splits, stratified by source x volume tertile.

Greedy balanced assignment: patients are shuffled with the frozen seed, then each patient
(with all their scans) is assigned to the fold that currently has the smallest count in that
patient's (source, volume_tertile) cell. Splits are written once to splits/splits_final.json
(nnU-Net format) plus a per-fold case list, and the script refuses to overwrite them.

Usage:
  python scripts/data/make_splits.py --cohort splits/cohort.csv --strata splits/strata.csv
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "config" / "analysis_config.yaml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--strata", default="splits/strata.csv", type=Path)
    ap.add_argument("--source-col", default="source",
                    help="column in cohort.csv holding the acquisition source (verify after dedup)")
    args = ap.parse_args()

    out_json = Path("splits/splits_final.json")
    if out_json.exists():
        raise SystemExit(f"{out_json} exists — splits are frozen once and never regenerated.")

    n_folds = CFG["splits"]["n_folds"]
    rng = np.random.default_rng(CFG["splits"]["split_seed"])

    cohort = pd.read_csv(args.cohort)
    strata = pd.read_csv(args.strata)
    df = cohort[cohort["has_manual_lesion"]].merge(
        strata[["case_id", "volume_tertile"]], on="case_id", how="inner")
    if args.source_col not in df.columns:
        raise SystemExit(f"Column '{args.source_col}' not in cohort.csv — set --source-col to the "
                         f"verified source column. Available: {list(df.columns)}")

    # One stratum per patient: source of their scans + max volume tertile across their lesions
    pat = df.groupby("patient_id").agg(
        source=(args.source_col, "first"),
        volume_tertile=("volume_tertile", "max")).reset_index()
    pat = pat.sample(frac=1.0, random_state=int(rng.integers(0, 2**31)))

    fold_of = {}
    cell_counts = defaultdict(lambda: np.zeros(n_folds, dtype=int))
    fold_totals = np.zeros(n_folds, dtype=int)
    for _, p in pat.iterrows():
        cell = (p["source"], p["volume_tertile"])
        counts = cell_counts[cell]
        candidates = np.flatnonzero(counts == counts.min())
        f = int(candidates[np.argmin(fold_totals[candidates])])
        fold_of[p["patient_id"]] = f
        counts[f] += 1
        fold_totals[f] += 1

    df["fold"] = df["patient_id"].map(fold_of)

    # nnU-Net splits_final.json: list of {"train": [...], "val": [...]} per fold.
    # The "val" fold serves as the held-out fold for out-of-fold predictions.
    splits = []
    all_cases = df["case_id"].tolist()
    for f in range(n_folds):
        val = df.loc[df["fold"] == f, "case_id"].tolist()
        train = [c for c in all_cases if c not in set(val)]
        splits.append({"train": sorted(train), "val": sorted(val)})

    out_json.parent.mkdir(exist_ok=True)
    json.dump(splits, open(out_json, "w"), indent=1)
    df[["case_id", "patient_id", args.source_col, "volume_tertile", "fold"]].to_csv(
        "splits/fold_assignment.csv", index=False)

    print(f"Froze {n_folds}-fold splits over {len(df)} cases / {len(pat)} patients (seed "
          f"{CFG['splits']['split_seed']}).")
    print(df.groupby(["fold", args.source_col]).size().unstack(fill_value=0))
    print("Copy splits_final.json into nnUNet_preprocessed/<Dataset>/ after preprocessing, "
          "then commit splits/ and tag prereg-v1.")


if __name__ == "__main__":
    main()
