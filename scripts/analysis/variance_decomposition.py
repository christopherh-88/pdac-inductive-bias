#!/usr/bin/env python
"""Separate seed (initialization) variance from fold variance, for one arm at a time.

Per the pre-registration, three seed replicates run on fold 0 exist specifically so
initialization variance can be told apart from fold-to-fold variance -- otherwise a large
spread across the 5 CV folds could be initialization noise, real fold-composition effects, or
both, and nothing in the 5-fold run alone can tell you which.

Two variance estimates, from two different tables:
  - seed_variance: fold 0's held-out cases evaluated under 3 seeds (default + 2 replicates).
    For each case, the variance of its Dice across seeds is computed, then averaged over
    cases -- a per-case estimate of pure initialization noise, with case difficulty held fixed.
  - fold_variance: the 5 folds' (default-seed) mean out-of-fold Dice, variance taken across
    those 5 numbers. This conflates initialization noise with real across-fold differences
    (different held-out cases, no case overlap by nnU-Net's CV design), which is exactly what
    seed_variance is subtracted out of below.
  - fold_variance_net_of_seed: max(0, fold_variance - seed_variance) -- a simple, transparent
    heuristic decomposition (not a fitted mixed-effects model), assuming the two variance
    sources are independent and additive. Reported alongside the raw numbers, not in place of
    them, so the assumption is visible rather than hidden inside a single statistic.

Usage:
  python scripts/analysis/variance_decomposition.py --arm cnn \
      --fold-dice results/h1/per_case.csv --fold-dice-col dice_cnn \
      --seed-dice results/seed_replicates_cnn.csv --out results/variance_cnn.yaml

  --fold-dice: one row per case with its out-of-fold Dice and its fold (columns: case_id,
    fold, <fold-dice-col>). paired_analysis.py's per_case.csv works directly if you add a
    'fold' column (from splits/fold_assignment.csv) -- or point --fold-dice straight at
    fold_assignment.csv merged with per-case Dice.
  --seed-dice: fold-0 cases only, one row per (case_id, seed) with a 'dice' column.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["cnn", "tf", "identity_control"])
    ap.add_argument("--fold-dice", required=True, type=Path)
    ap.add_argument("--fold-dice-col", default="dice")
    ap.add_argument("--fold-col", default="fold")
    ap.add_argument("--seed-dice", required=True, type=Path)
    ap.add_argument("--seed-dice-col", default="dice")
    ap.add_argument("--seed-col", default="seed")
    ap.add_argument("--out", default=Path("results/variance_decomposition.yaml"), type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fold_df = pd.read_csv(args.fold_dice)
    seed_df = pd.read_csv(args.seed_dice)

    fold_means = fold_df.groupby(args.fold_col)[args.fold_dice_col].mean()
    if len(fold_means) < 2:
        raise SystemExit(f"Need Dice from >=2 folds in {args.fold_dice}, got {len(fold_means)}")
    fold_variance = float(fold_means.var(ddof=1))

    seed_counts = seed_df.groupby("case_id")[args.seed_col].nunique()
    if (seed_counts < 2).any():
        bad = seed_counts[seed_counts < 2].index.tolist()
        raise SystemExit(f"Cases with fewer than 2 seeds in {args.seed_dice}: {bad[:5]}...")
    per_case_seed_var = seed_df.groupby("case_id")[args.seed_dice_col].var(ddof=1)
    seed_variance = float(per_case_seed_var.mean())

    fold_variance_net_of_seed = max(0.0, fold_variance - seed_variance)

    summary = {
        "arm": args.arm,
        "n_folds": int(len(fold_means)),
        "fold_mean_dice_by_fold": {str(k): float(v) for k, v in fold_means.items()},
        "fold_variance": fold_variance,
        "fold_sd": float(np.sqrt(fold_variance)),
        "n_seed_replicate_cases": int(len(per_case_seed_var)),
        "seed_variance": seed_variance,
        "seed_sd": float(np.sqrt(seed_variance)),
        "fold_variance_net_of_seed_estimate": fold_variance_net_of_seed,
        "fold_sd_net_of_seed_estimate": float(np.sqrt(fold_variance_net_of_seed)),
        "note": "net-of-seed estimate assumes independent, additive variance sources — a "
                "transparent heuristic, not a fitted mixed-effects model",
    }
    yaml.safe_dump(summary, open(args.out, "w"), sort_keys=False)
    print(f"[{args.arm}] fold SD = {summary['fold_sd']:.4f} (across {summary['n_folds']} folds), "
          f"seed SD = {summary['seed_sd']:.4f} (across {summary['n_seed_replicate_cases']} fold-0 "
          f"cases x seeds), fold SD net of seed ~= {summary['fold_sd_net_of_seed_estimate']:.4f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
