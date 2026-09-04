#!/usr/bin/env python
"""H2b: the identity control's stratified map against the full transformer's, same layout.

Pre-registered rule: "If the identity control reproduces the full model's stratified map
within CI, attention is not the cause, and the paper says so." That rule is qualitative, so
this script makes it explicit without adding anything to it:

  per (volume_tertile x cnr_tertile) cell, the *paired* difference
  Dice(full transformer) - Dice(identity control) on the same cases, with a patient-level
  bootstrap CI. A cell "agrees" when its CI contains zero. The map is reproduced within CI
  when every populated cell agrees.

Reported alongside, because a single boolean over nine cells is thin evidence on its own:
  - the overall paired difference and its CI (is the full model better *anywhere*)
  - Spearman rho between the two arms' cell-mean Dice across cells (does the failure map have
    the same shape, even if shifted)
  - per-cell n, so an "agreement" driven by an empty cell is visible rather than hidden

Deliverable 4 is the CSV this writes: laid out identically to paired_analysis.py's
stratum_heatmap.csv so the two panels can be drawn on one axis pair by make_figures.py.

Usage:
  python scripts/analysis/identity_comparison.py \
      --metrics-table results/per_case_metrics_h2b.csv \
      --arm-full tf --arm-identity identity_control --out results/h2b
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, wilcoxon

from metrics import CFG, bootstrap_ci


def paired(long: pd.DataFrame, arm_full: str, arm_identity: str) -> pd.DataFrame:
    for arm in (arm_full, arm_identity):
        if arm not in set(long["arm"]):
            raise SystemExit(f"arm {arm!r} not in the metrics table "
                             f"(has: {sorted(set(long['arm']))})")
    a = long[long["arm"] == arm_full].set_index("case_id")
    b = long[long["arm"] == arm_identity].set_index("case_id")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise SystemExit("No cases shared by the two arms — H2b is a paired comparison.")
    keep = [c for c in ("patient_id", "source", "volume_mm3", "volume_tertile", "cnr_tertile")
            if c in a.columns]
    df = a.loc[common, keep].copy()
    df["dice_full"] = a.loc[common, "dice"]
    df["dice_identity"] = b.loc[common, "dice"]
    df["nsd_full"] = a.loc[common, "nsd2mm"]
    df["nsd_identity"] = b.loc[common, "nsd2mm"]
    df["delta_dice"] = df["dice_full"] - df["dice_identity"]
    return df.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-table", required=True, type=Path,
                    help="build_per_case_table.py output containing both arms")
    ap.add_argument("--arm-full", default="tf")
    ap.add_argument("--arm-identity", default="identity_control")
    ap.add_argument("--out", default=Path("results/h2b"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    st = CFG["statistics"]
    df = paired(pd.read_csv(args.metrics_table), args.arm_full, args.arm_identity)
    df.to_csv(args.out / "per_case.csv", index=False)

    # Stratified map: one row per cell per arm, matching stratum_heatmap.csv's columns, plus
    # the paired within-cell difference the rule is actually read against.
    cells = []
    for (vt, ct), g in df.groupby(["volume_tertile", "cnr_tertile"]):
        d_lo, d_hi = bootstrap_ci(g, lambda b: b["delta_dice"].mean(),
                                  st["bootstrap_resamples"], st["bootstrap_seed"])
        row = {"volume_tertile": vt, "cnr_tertile": ct, "n": len(g),
               "dice_full": g["dice_full"].mean(), "dice_identity": g["dice_identity"].mean(),
               "nsd_full": g["nsd_full"].mean(), "nsd_identity": g["nsd_identity"].mean(),
               "delta_dice": g["delta_dice"].mean(), "ci_lo": d_lo, "ci_hi": d_hi}
        row["agrees_within_ci"] = bool(d_lo <= 0 <= d_hi)
        cells.append(row)
    cells = pd.DataFrame(cells)
    cells.to_csv(args.out / "identity_vs_full_heatmap.csv", index=False)

    # Long-format twin of paired_analysis.py's stratum_heatmap.csv, so make_figures.py can draw
    # the identity panel with exactly the same code path as the two Tier A panels.
    long_cells = []
    for r in cells.itertuples():
        for arm, col in ((args.arm_full, "dice_full"), (args.arm_identity, "dice_identity")):
            long_cells.append({"volume_tertile": r.volume_tertile, "cnr_tertile": r.cnr_tertile,
                               "arm": arm, "n": r.n, "dice_mean": getattr(r, col),
                               "ci_lo": np.nan, "ci_hi": np.nan})
    pd.DataFrame(long_cells).to_csv(args.out / "stratum_heatmap.csv", index=False)

    lo, hi = bootstrap_ci(df, lambda b: b["delta_dice"].mean(),
                          st["bootstrap_resamples"], st["bootstrap_seed"] + 1)
    rho, rho_p = spearmanr(cells["dice_full"], cells["dice_identity"])
    try:
        _, w_p = wilcoxon(df["dice_full"], df["dice_identity"])
    except ValueError:      # every paired difference is zero
        w_p = 1.0

    populated = cells[cells["n"] > 0]
    n_agree = int(populated["agrees_within_ci"].sum())
    reproduced = bool(n_agree == len(populated))

    summary = {
        "arm_full": args.arm_full,
        "arm_identity": args.arm_identity,
        "n_cases": int(len(df)),
        "n_cells": int(len(populated)),
        "n_cells_agreeing_within_ci": n_agree,
        "map_reproduced_within_ci": reproduced,
        "overall_delta_dice_full_minus_identity": float(df["delta_dice"].mean()),
        "overall_delta_ci": [lo, hi],
        "overall_wilcoxon_p": float(w_p),
        "cell_mean_dice_spearman_rho": float(rho),
        "cell_mean_dice_spearman_p": float(rho_p),
        "h2b_verdict": ("attention is NOT the cause of the stratified failure map — the "
                        "identity control reproduces it within CI"
                        if reproduced else
                        "the identity control does NOT reproduce the full model's map within "
                        "CI in every cell — attention contributes where the CIs exclude zero"),
    }
    yaml.safe_dump(summary, open(args.out / "summary.yaml", "w"), sort_keys=False)

    print(cells.round(4).to_string(index=False))
    print(f"\nOverall Dice({args.arm_full}) - Dice({args.arm_identity}) = "
          f"{df['delta_dice'].mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  (Wilcoxon p={w_p:.4g})")
    print(f"Cell-mean Dice rank agreement across cells: Spearman rho = {rho:.3f}")
    print(f"\nH2b: {n_agree}/{len(populated)} populated cells agree within CI -> "
          f"{'MAP REPRODUCED' if reproduced else 'MAP NOT REPRODUCED'}")
    print(summary["h2b_verdict"])
    print(f"Wrote {args.out}/")


if __name__ == "__main__":
    main()
