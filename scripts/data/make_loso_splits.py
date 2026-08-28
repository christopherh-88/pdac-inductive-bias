#!/usr/bin/env python
"""Tier B: leave-one-source-out folds, derived from the frozen cohort without touching it.

The 5 cross-validation folds in splits/splits_final.json are frozen at prereg-v1 and never
regenerated. LOSO folds are a *separate*, deterministic artifact: for each canonical source
group S (as frozen by make_splits.py), train on every manual-lesion case not from S and hold
out all of S. Nothing here is a new random draw — the assignment is fully determined by the
frozen source grouping, so there is no seed and no freedom to re-roll.

nnU-Net addresses folds by integer index into splits_final.json, so `--emit-combined` writes
the file that actually goes into nnUNet_preprocessed/: the five frozen CV folds first (indices
0-4, byte-identical to the frozen file), then the LOSO folds appended at indices 5, 6, ...
Training LOSO for source S is then just `nnUNetv2_train <D> 3d_fullres <5+i>`. The frozen file
in splits/ is never modified.

The pre-registration's escalation rule — "deduplication drops a source below a usable case
count, which would make leave-one-source-out meaningless for that source" — is enforced here
with --min-cases rather than left to be noticed later.

Usage:
  python scripts/data/make_loso_splits.py --emit-combined splits/splits_with_loso.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).parents[2]
FROZEN_PATH = REPO / "config" / "frozen_thresholds.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-assignment", default="splits/fold_assignment.csv", type=Path)
    ap.add_argument("--cv-splits", default="splits/splits_final.json", type=Path)
    ap.add_argument("--source-col", default=None,
                    help="default: the source column frozen by make_splits.py")
    ap.add_argument("--exclude-source", action="append", default=[],
                    help="source groups that are never a LOSO holdout (e.g. NIH negatives)")
    ap.add_argument("--min-cases", type=int, default=25,
                    help="escalate rather than silently produce a meaningless LOSO fold")
    ap.add_argument("--out", default="splits/splits_loso.json", type=Path)
    ap.add_argument("--emit-combined", type=Path, default=None,
                    help="write frozen CV folds + LOSO folds appended, for nnUNet_preprocessed/")
    args = ap.parse_args()

    frozen = yaml.safe_load(open(FROZEN_PATH)) if FROZEN_PATH.exists() else {}
    frozen = frozen or {}
    source_col = args.source_col or frozen.get("splits", {}).get("source_column", "source")

    fa = pd.read_csv(args.fold_assignment)
    if source_col not in fa.columns:
        raise SystemExit(f"'{source_col}' not in {args.fold_assignment} (has {list(fa.columns)})")
    fa = fa[~fa[source_col].isin(args.exclude_source)]

    cv = json.load(open(args.cv_splits))
    all_cases = sorted(set(fa["case_id"]))
    counts = fa.groupby(source_col)["case_id"].nunique().sort_index()
    print("Manual-lesion cases per canonical source group:")
    print(counts.to_string())

    too_small = counts[counts < args.min_cases]
    if len(too_small):
        raise SystemExit(
            f"\nESCALATE (same day, per the pre-registration): source group(s) "
            f"{dict(too_small)} have fewer than --min-cases={args.min_cases} cases after "
            "deduplication. Leave-one-source-out on that source would be meaningless. Decide "
            "and record whether to merge, drop, or lower the bar BEFORE training — then re-run "
            "with --exclude-source or an explicit --min-cases.")

    loso, records = [], []
    for i, source in enumerate(counts.index):
        val = sorted(fa.loc[fa[source_col] == source, "case_id"])
        train = [c for c in all_cases if c not in set(val)]
        loso.append({"train": train, "val": val})
        records.append({"loso_index": i, "nnunet_fold": len(cv) + i, "held_out_source": source,
                        "n_val": len(val), "n_train": len(train)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(loso, open(args.out, "w"), indent=1)
    pd.DataFrame(records).to_csv(args.out.parent / "loso_folds.csv", index=False)

    if args.emit_combined:
        json.dump(cv + loso, open(args.emit_combined, "w"), indent=1)
        print(f"\nWrote {args.emit_combined}: {len(cv)} frozen CV folds (indices 0-{len(cv)-1}) "
              f"+ {len(loso)} LOSO folds (indices {len(cv)}-{len(cv)+len(loso)-1}). "
              "Copy THIS file into nnUNet_preprocessed/<Dataset>/splits_final.json.")

    if "loso" in frozen:
        print("\n(config/frozen_thresholds.yaml already records the LOSO fold mapping — "
              "left untouched; the mapping is deterministic, so a re-run reproduces it.)")
    else:
        frozen["loso"] = {"source_column": source_col,
                          "excluded_sources": list(args.exclude_source),
                          "min_cases_rule": args.min_cases,
                          "folds": records}
        yaml.safe_dump(frozen, open(FROZEN_PATH, "w"), sort_keys=False)
        print(f"\nRecorded the LOSO fold mapping in {FROZEN_PATH}.")

    print("\n" + pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
