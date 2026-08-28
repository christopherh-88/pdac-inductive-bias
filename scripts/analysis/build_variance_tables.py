#!/usr/bin/env python
"""Build the two input tables variance_decomposition.py needs, for one arm.

  fold_dice.csv   case_id, fold, dice   — every case's out-of-fold Dice plus the fold that
                  held it out, from the Tier A per-case metrics table + splits/fold_assignment.csv
  seed_dice.csv   case_id, seed, dice   — fold 0's held-out cases under the default seed and
                  each replicate seed, from the replicate prediction directories

The default-seed column of seed_dice.csv is taken from the same Tier A metrics table, so the
default run is not re-inferred: seed replicates are extra seeds *in addition to* the default
run, exactly as config/analysis_config.yaml's training.seed_replicates states.

Usage:
  python scripts/analysis/build_variance_tables.py --arm cnn \
      --metrics-table results/per_case_metrics.csv \
      --seed-pred 1:/preds/cnn_fold0_seed1 --seed-pred 2:/preds/cnn_fold0_seed2 \
      --refs /labels/manual --out results/variance_inputs_cnn
  python scripts/analysis/variance_decomposition.py --arm cnn \
      --fold-dice results/variance_inputs_cnn/fold_dice.csv \
      --seed-dice results/variance_inputs_cnn/seed_dice.csv \
      --out results/variance_cnn.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from metrics import CFG, all_metrics


def parse_seed_pred(spec: str):
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"--seed-pred expects SEED:DIR, got {spec!r}")
    seed, _, path = spec.partition(":")
    return seed, Path(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="arm name as it appears in --metrics-table")
    ap.add_argument("--metrics-table", required=True, type=Path)
    ap.add_argument("--seed-pred", action="append", default=[], type=parse_seed_pred,
                    metavar="SEED:DIR", help="repeatable; replicate seeds on the replicate fold")
    ap.add_argument("--refs", type=Path)
    ap.add_argument("--fold-assignment", default="splits/fold_assignment.csv", type=Path)
    ap.add_argument("--replicate-fold", type=int, default=None,
                    help="default: training.seed_replicates.fold from the frozen config")
    ap.add_argument("--default-seed-label", default="default")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rep_fold = (args.replicate_fold if args.replicate_fold is not None
                else CFG["training"]["seed_replicates"]["fold"])

    long = pd.read_csv(args.metrics_table)
    arm = long[long["arm"] == args.arm]
    if arm.empty:
        raise SystemExit(f"arm {args.arm!r} not in {args.metrics_table} "
                         f"(has: {sorted(set(long['arm']))})")
    if "fold" not in arm.columns:
        fa = pd.read_csv(args.fold_assignment)[["case_id", "fold"]].drop_duplicates("case_id")
        arm = arm.merge(fa, on="case_id", how="left")
    fold_df = arm[["case_id", "fold", "dice"]].dropna(subset=["fold"])
    fold_df["fold"] = fold_df["fold"].astype(int)
    fold_df.to_csv(args.out / "fold_dice.csv", index=False)

    rep_cases = fold_df.loc[fold_df["fold"] == rep_fold, "case_id"].tolist()
    if not rep_cases:
        raise SystemExit(f"No cases held out by fold {rep_fold} in {args.metrics_table}")

    seed_rows = [{"case_id": r.case_id, "seed": args.default_seed_label, "dice": r.dice}
                 for r in fold_df[fold_df["fold"] == rep_fold].itertuples()]
    for seed, pred_dir in args.seed_pred:
        if not args.refs:
            raise SystemExit("--refs is required when --seed-pred is given")
        for cid in tqdm(rep_cases, desc=f"seed {seed}"):
            pred, ref = pred_dir / f"{cid}.nii.gz", args.refs / f"{cid}.nii.gz"
            if not (pred.exists() and ref.exists()):
                continue
            seed_rows.append({"case_id": cid, "seed": seed, "dice": all_metrics(pred, ref)["dice"]})

    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(args.out / "seed_dice.csv", index=False)

    counts = seed_df.groupby("case_id")["seed"].nunique()
    incomplete = int((counts < 2).sum())
    yaml.safe_dump({"arm": args.arm, "replicate_fold": int(rep_fold),
                    "n_cases_all_folds": int(len(fold_df)),
                    "n_replicate_fold_cases": int(len(rep_cases)),
                    "seeds": sorted(seed_df["seed"].astype(str).unique().tolist()),
                    "n_cases_with_fewer_than_2_seeds": incomplete},
                   open(args.out / "summary.yaml", "w"), sort_keys=False)

    print(f"fold_dice.csv: {len(fold_df)} cases across {fold_df['fold'].nunique()} folds")
    print(f"seed_dice.csv: {len(rep_cases)} fold-{rep_fold} cases x "
          f"{seed_df['seed'].nunique()} seeds ({sorted(seed_df['seed'].astype(str).unique())})")
    if incomplete:
        print(f"WARNING: {incomplete} case(s) have <2 seeds — variance_decomposition.py will "
              "reject the table until every replicate-fold case has every seed.")
    print(f"Wrote {args.out}/")


if __name__ == "__main__":
    main()
