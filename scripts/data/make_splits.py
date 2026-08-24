#!/usr/bin/env python
"""Freeze the patient-level 5-fold splits, stratified by source x volume tertile.

Greedy balanced assignment: patients are shuffled with the frozen seed, then each patient
(with all their scans) is assigned to the fold that currently has the smallest count in that
patient's (source, volume_tertile) cell. Splits are written once to splits/splits_final.json
(nnU-Net format) plus a per-fold case list, and the script refuses to overwrite them.

Source grouping: the PANORAMA public set lists five contributing institutions (RUMC, UMCG,
ZGT, Karolinska, Haukeland) plus MSD and NIH, but the pre-registration's leave-one-source-out
axis groups these into a small number of study sources (e.g. Radboud / UMCG / MSKCC / NIH).
If the raw source column needs remapping to those groups, pass --source-mapping pointing to a
JSON file of {raw_value: canonical_group}; every distinct raw value must be covered or the
script fails loudly rather than guessing. Whatever grouping is actually used (identity if no
mapping is given) is frozen into config/frozen_thresholds.yaml so the decision is recorded,
not left as an unwritten manual step.

Usage:
  python scripts/data/make_splits.py --cohort splits/cohort.csv --strata splits/strata.csv \
      [--source-mapping splits/source_mapping.json]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "config" / "analysis_config.yaml"))
FROZEN_PATH = Path(__file__).parents[2] / "config" / "frozen_thresholds.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--strata", default="splits/strata.csv", type=Path)
    ap.add_argument("--source-col", default="source",
                    help="column in cohort.csv holding the acquisition source (verify after dedup)")
    ap.add_argument("--source-mapping", default=None, type=Path,
                    help="optional JSON file mapping raw source values to canonical LOSO groups "
                         "(e.g. {\"RUMC\": \"Radboud\", \"UMCG\": \"UMCG\", ...}); every raw "
                         "value in the cohort must be covered")
    args = ap.parse_args()

    out_json = Path("splits/splits_final.json")
    if out_json.exists():
        raise SystemExit(f"{out_json} exists — splits are frozen once and never regenerated.")
    frozen = yaml.safe_load(open(FROZEN_PATH)) if FROZEN_PATH.exists() else {}
    if "splits" in (frozen or {}):
        raise SystemExit(f"{FROZEN_PATH} already contains a frozen 'splits' section — refusing "
                         "to overwrite. Source grouping is frozen once, same as the splits.")

    n_folds = CFG["splits"]["n_folds"]
    rng = np.random.default_rng(CFG["splits"]["split_seed"])

    cohort = pd.read_csv(args.cohort)
    strata = pd.read_csv(args.strata)
    df = cohort[cohort["has_manual_lesion"]].merge(
        strata[["case_id", "volume_tertile"]], on="case_id", how="inner")
    if args.source_col not in df.columns:
        raise SystemExit(f"Column '{args.source_col}' not in cohort.csv — set --source-col to the "
                         f"verified source column. Available: {list(df.columns)}")

    raw_value_counts = df[args.source_col].value_counts().to_dict()
    source_mapping = None
    if args.source_mapping:
        source_mapping = json.load(open(args.source_mapping))
        uncovered = set(df[args.source_col].unique()) - set(source_mapping.keys())
        if uncovered:
            raise SystemExit(f"--source-mapping does not cover raw values {uncovered} found in "
                             f"'{args.source_col}' — every value must be mapped explicitly.")
        df[args.source_col] = df[args.source_col].map(source_mapping)

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

    # Freeze the source grouping actually used — closes the gap where this decision was
    # documented as an open item but nothing wrote it down.
    frozen = frozen or {}
    frozen["splits"] = {
        "n_folds": n_folds,
        "split_seed": int(CFG["splits"]["split_seed"]),
        "source_column": args.source_col,
        "source_raw_value_counts": {str(k): int(v) for k, v in raw_value_counts.items()},
        "source_mapping": source_mapping,
        "canonical_source_groups": sorted(df[args.source_col].unique().tolist()),
        "n_cases": int(len(df)),
        "n_patients": int(len(pat)),
    }
    yaml.safe_dump(frozen, open(FROZEN_PATH, "w"), sort_keys=False)

    print(f"Froze {n_folds}-fold splits over {len(df)} cases / {len(pat)} patients (seed "
          f"{CFG['splits']['split_seed']}).")
    print(df.groupby(["fold", args.source_col]).size().unstack(fill_value=0))
    print(f"Source grouping frozen into {FROZEN_PATH}.")
    print("Copy splits_final.json into nnUNet_preprocessed/<Dataset>/ after preprocessing, "
          "then commit splits/ and tag prereg-v1.")


if __name__ == "__main__":
    main()
