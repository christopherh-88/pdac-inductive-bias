#!/usr/bin/env python
"""Pre-specified contrast sensitivity analysis: repeat every contrast-stratified result at
5 mm and 15 mm rings, and apply the frozen rank-stability rule.

The pre-registration fixes CNR at a 10 mm parenchyma ring as primary, and states that the
whole contrast axis is downgraded to *exploratory* if CNR ranks are unstable across ring
widths (Spearman rho < 0.8 between any pair). compute_strata.py already computes all three
ring widths in one pass and records the rho matrix; this script does the second half — it
re-runs the contrast-stratified analysis under each ring width and reports whether the
substantive conclusions move, not just whether the ranks do.

For each ring width w in {5, 10, 15}:
  - re-tertile CNR using that ring's values (tertiles are within-ring by construction, since
    a fixed cut point in CNR units is not comparable across ring widths)
  - per-(volume x cnr) cell mean Dice per arm, with patient-level bootstrap CIs
  - the H1 regression's CNR coefficient under that ring width
Then: the Spearman rho matrix between ring widths, the frozen rule's verdict, and a
cell-by-cell check of whether any (volume x cnr) cell changes its sign of mean delta-Dice
across ring widths.

Because the ranking verdict is a pre-registered gate rather than a result, this script writes
`contrast_axis_exploratory` into its summary and the paper reports whichever the rule says.

Usage:
  python scripts/analysis/contrast_sensitivity.py --metrics-table results/per_case_metrics.csv \
      --strata splits/strata.csv --out results/contrast_sensitivity
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from scipy.stats import spearmanr

from metrics import CFG, bootstrap_ci


def ring_widths():
    c = CFG["strata"]["cnr"]
    return [c["ring_width_mm_primary"]] + list(c["ring_width_mm_sensitivity"])


def paired_frame(metrics_table: Path, arm_cnn: str, arm_tf: str) -> pd.DataFrame:
    long = pd.read_csv(metrics_table)
    a = long[long["arm"] == arm_cnn].set_index("case_id")
    b = long[long["arm"] == arm_tf].set_index("case_id")
    common = a.index.intersection(b.index)
    keep = [c for c in ("patient_id", "source", "volume_mm3", "volume_tertile") if c in a.columns]
    df = a.loc[common, keep].copy()
    df["dice_cnn"] = a.loc[common, "dice"]
    df["dice_tf"] = b.loc[common, "dice"]
    df["delta_dice"] = df["dice_cnn"] - df["dice_tf"]
    df["log_volume"] = np.log(df["volume_mm3"])
    return df.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-table", required=True, type=Path)
    ap.add_argument("--strata", default="splits/strata.csv", type=Path)
    ap.add_argument("--arm-cnn", default="cnn")
    ap.add_argument("--arm-tf", default="tf")
    ap.add_argument("--out", default=Path("results/contrast_sensitivity"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    st = CFG["statistics"]
    min_rho = CFG["strata"]["cnr"]["rank_stability_min_spearman"]
    widths = ring_widths()
    cols = [f"cnr_ring{w}mm" for w in widths]

    strata = pd.read_csv(args.strata)
    missing = [c for c in cols if c not in strata.columns]
    if missing:
        raise SystemExit(f"{args.strata} lacks {missing} — compute_strata.py writes all three "
                         "ring widths in one pass; re-run it rather than back-filling here.")

    df = paired_frame(args.metrics_table, args.arm_cnn, args.arm_tf)
    df = df.merge(strata[["case_id"] + cols], on="case_id", how="left")

    # 1. Rank stability across ring widths (the frozen gate)
    rhos = {}
    for a, b in itertools.combinations(cols, 2):
        rho, _ = spearmanr(df[a], df[b], nan_policy="omit")
        rhos[f"{a}__vs__{b}"] = float(rho)
    exploratory = any(v < min_rho for v in rhos.values())

    # 2. Re-tertile within each ring width and redo the stratified table + CNR coefficient
    cell_rows, coef_rows = [], []
    for w, col in zip(widths, cols):
        cuts = df[col].quantile([1 / 3, 2 / 3]).tolist()
        df[f"cnr_tertile_ring{w}"] = np.digitize(df[col], cuts)
        sub = df.dropna(subset=[col]).copy()
        sub["cnr_w"] = sub[col]

        model = smf.ols("delta_dice ~ log_volume + cnr_w + C(source)", data=sub).fit()
        lo, hi = bootstrap_ci(
            sub, lambda b: smf.ols("delta_dice ~ log_volume + cnr_w + C(source)",
                                   data=b).fit().params["cnr_w"],
            st["bootstrap_resamples"], st["bootstrap_seed"])
        coef_rows.append({"ring_mm": w, "n": len(sub), "cnr_coef": float(model.params["cnr_w"]),
                          "cnr_coef_ci_lo": lo, "cnr_coef_ci_hi": hi,
                          "log_volume_coef": float(model.params["log_volume"]),
                          "cnr_tertile_cuts": [float(c) for c in cuts]})

        for (vt, ct), g in sub.groupby(["volume_tertile", f"cnr_tertile_ring{w}"]):
            d_lo, d_hi = bootstrap_ci(g, lambda b: b["delta_dice"].mean(),
                                      st["bootstrap_resamples"], st["bootstrap_seed"] + 3)
            cell_rows.append({"ring_mm": w, "volume_tertile": vt, "cnr_tertile": ct, "n": len(g),
                              "dice_cnn": g["dice_cnn"].mean(), "dice_tf": g["dice_tf"].mean(),
                              "delta_dice": g["delta_dice"].mean(),
                              "delta_ci_lo": d_lo, "delta_ci_hi": d_hi})

    cells = pd.DataFrame(cell_rows)
    coefs = pd.DataFrame(coef_rows)
    cells.to_csv(args.out / "stratified_by_ring_width.csv", index=False)
    coefs.to_csv(args.out / "cnr_coefficient_by_ring_width.csv", index=False)

    # 3. Do any cells flip the sign of mean delta-Dice when the ring width changes?
    signs = cells.pivot_table(index=["volume_tertile", "cnr_tertile"], columns="ring_mm",
                              values="delta_dice")
    flipped = signs[(np.sign(signs).nunique(axis=1) > 1)]
    n_flipped = int(len(flipped))

    summary = {
        "ring_widths_mm": [int(w) for w in widths],
        "primary_ring_mm": int(CFG["strata"]["cnr"]["ring_width_mm_primary"]),
        "spearman_rank_stability": rhos,
        "min_spearman_rule": float(min_rho),
        "contrast_axis_exploratory": bool(exploratory),
        "cnr_coefficient_by_ring": coefs.set_index("ring_mm")[
            ["cnr_coef", "cnr_coef_ci_lo", "cnr_coef_ci_hi"]].to_dict("index"),
        "n_cells_changing_delta_sign_across_rings": n_flipped,
        "n_cells": int(signs.shape[0]),
    }
    yaml.safe_dump(summary, open(args.out / "summary.yaml", "w"), sort_keys=False, default_flow_style=False)

    print("Spearman rho between ring widths:")
    for k, v in rhos.items():
        print(f"  {k}: {v:.3f}{'   < RULE' if v < min_rho else ''}")
    print(f"\nFrozen rule (rho >= {min_rho} for every pair): contrast axis is "
          f"{'EXPLORATORY' if exploratory else 'CONFIRMATORY'}")
    print(f"\nCNR coefficient on delta-Dice by ring width:\n{coefs.round(5).to_string(index=False)}")
    print(f"\n{n_flipped} of {signs.shape[0]} (volume x cnr) cells change the sign of mean "
          f"delta-Dice across ring widths")
    print(f"Wrote {args.out}/")


if __name__ == "__main__":
    main()
