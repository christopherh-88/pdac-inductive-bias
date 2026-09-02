#!/usr/bin/env python
"""Turn the streamed Kaggle output (lesion_stats_raw.csv) into splits/strata.csv and freeze
config/frozen_thresholds.yaml — the same tail end scripts/analysis/compute_strata.py runs,
but starting from lesion volume/diameter/CNR values already computed on Kaggle instead of
recomputing them from images that no longer exist locally.

Must run AFTER scripts/data/finalize_cohort_from_stream.py, so cross-batch duplicates already
found by content hash are excluded here too (lesion_stats_raw.csv is keyed by case_id only and
knows nothing about dedup).

Usage:
  python scripts/data/finalize_strata_from_stream.py \\
      --lesion-stats /path/to/lesion_stats_raw.csv --cohort splits/cohort.csv \\
      --out splits/strata.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "config" / "analysis_config.yaml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesion-stats", required=True, type=Path)
    ap.add_argument("--cohort", default=Path("splits/cohort.csv"), type=Path)
    ap.add_argument("--out", default=Path("splits/strata.csv"), type=Path)
    ap.add_argument("--frozen-out", default=None, type=Path,
                     help="where to write the frozen strata thresholds; defaults to "
                          "config/frozen_thresholds.yaml. Override for a dry run/smoke test so "
                          "it doesn't write (or refuse to overwrite) the real frozen config.")
    args = ap.parse_args()

    raw = pd.read_csv(args.lesion_stats)
    cohort = pd.read_csv(args.cohort)
    kept_ids = set(cohort.loc[cohort["has_manual_lesion"], "case_id"].astype(str))
    df = raw[raw["case_id"].astype(str).isin(kept_ids)].copy()
    n_dropped_as_dupe = len(raw) - len(df)
    if n_dropped_as_dupe:
        print(f"Dropped {n_dropped_as_dupe} lesion_stats rows for cases removed as cross-batch "
              f"duplicates in {args.cohort}")

    # kaggle_stream_cohort.py's process_case() writes no lesion_stats row at all when
    # lesion_stats() finds zero voxels matching the PDAC lesion label (e.g. a manual label file
    # that delineates a non-PDAC finding instead) -- unlike scripts/analysis/compute_strata.py's
    # equivalent local path, which prints an explicit "dropped from strata.csv: [ids]" warning.
    # Reproduce that same visibility here so a has_manual_lesion case missing from strata.csv is
    # never a silent gap.
    missing_ids = kept_ids - set(raw["case_id"].astype(str))
    if missing_ids:
        reasons = cohort.loc[cohort["case_id"].astype(str).isin(missing_ids),
                              ["case_id", "is_pdac", "label"]]
        print(f"WARNING: {len(missing_ids)} case(s) flagged has_manual_lesion in {args.cohort} "
              f"never produced a lesion_stats row on Kaggle (no lesion voxels in the label file) "
              f"-- excluded from strata.csv: {sorted(missing_ids)}")
        print(reasons.to_string(index=False))

    ring_widths = [CFG["strata"]["cnr"]["ring_width_mm_primary"]] + CFG["strata"]["cnr"]["ring_width_mm_sensitivity"]

    vol_cuts = df["volume_mm3"].quantile([1 / 3, 2 / 3]).tolist()
    cnr_col = f"cnr_ring{CFG['strata']['cnr']['ring_width_mm_primary']}mm"
    cnr_cuts = df[cnr_col].quantile([1 / 3, 2 / 3]).tolist()
    df["volume_tertile"] = np.digitize(df["volume_mm3"], vol_cuts)
    df["cnr_tertile"] = np.digitize(df[cnr_col], cnr_cuts)
    df["above_2cm"] = df["max_inplane_diam_mm"] >= CFG["strata"]["volume"]["secondary_cut_diameter_mm"]

    rhos = {}
    cols = [f"cnr_ring{w}mm" for w in ring_widths]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho, _ = spearmanr(df[cols[i]], df[cols[j]], nan_policy="omit")
            rhos[f"{cols[i]}__vs__{cols[j]}"] = float(rho)
    exploratory = any(v < CFG["strata"]["cnr"]["rank_stability_min_spearman"] for v in rhos.values())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    frozen_path = args.frozen_out or Path(__file__).parents[2] / "config" / "frozen_thresholds.yaml"
    frozen = yaml.safe_load(open(frozen_path)) if frozen_path.exists() else {}
    if "strata" in (frozen or {}):
        raise SystemExit(f"{frozen_path} already contains frozen strata — refusing to overwrite. "
                         "Thresholds are computed once; delete manually only before prereg tag.")
    frozen = frozen or {}
    frozen["strata"] = {
        "n_cases": int(len(df)),
        "volume_tertile_cuts_mm3": [float(x) for x in vol_cuts],
        "cnr_tertile_cuts": [float(x) for x in cnr_cuts],
        "cnr_ring_rank_stability_spearman": rhos,
        "contrast_axis_exploratory": bool(exploratory),
        "computed_via": "kaggle_stream_cohort.py (streaming, disk-constrained)",
    }
    yaml.safe_dump(frozen, open(frozen_path, "w"), sort_keys=False)
    print(f"Wrote {args.out} and froze thresholds in {frozen_path}")
    if exploratory:
        print("NOTE: CNR ranks unstable across ring widths — per pre-registration, "
              "the contrast axis is reported as exploratory.")


if __name__ == "__main__":
    main()
