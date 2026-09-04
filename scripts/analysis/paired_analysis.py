#!/usr/bin/env python
"""H1 paired analysis over out-of-fold predictions of both arms.

Inputs: two prediction directories (one per arm, out-of-fold predictions for every cohort
case), reference labels, and splits/strata.csv. Implements exactly the pre-registered plan:

  1. delta_dice = dice(CNN) - dice(transformer) per case
  2. OLS of delta_dice on log(volume) + CNR + source, patient-level bootstrap CIs
     (2,000 resamples, seed frozen) on the log-volume slope  -> H1 decision rule
  3. Wilcoxon signed-rank within each volume tertile
  4. TOST at +/-3 Dice points on the large-lesion tertile
  5. Per-stratum (volume x CNR tertile) tables with per-cell n and bootstrap CIs

Metrics come from build_per_case_table.py's long-format table when --metrics-table is given
(the normal path: surface distances are computed once for every downstream analysis, not
recomputed here). Passing --pred-cnn/--pred-tf/--refs instead makes this script compute Dice
and NSD itself, which is kept for standalone use.

Usage:
  python scripts/analysis/build_per_case_table.py --arm cnn:DIR --arm tf:DIR --refs DIR \
      --out results/per_case_metrics.csv
  python scripts/analysis/paired_analysis.py --metrics-table results/per_case_metrics.csv \
      --cohort splits/cohort.csv --out results/h1

  # or, standalone (recomputes metrics):
  python scripts/analysis/paired_analysis.py --pred-cnn DIR --pred-tf DIR --refs DIR \
      --strata splits/strata.csv --cohort splits/cohort.csv --out results/h1
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from scipy.stats import wilcoxon
from tqdm import tqdm

from metrics import all_metrics, bootstrap_ci, CFG


def wide_from_metrics_table(path: Path, arm_cnn: str, arm_tf: str, cnr_col: str):
    """Pivot build_per_case_table.py's long table to the wide, paired frame this analysis
    expects. Only cases present for BOTH arms survive — the pairing is the whole design."""
    long = pd.read_csv(path)
    for arm in (arm_cnn, arm_tf):
        if arm not in set(long["arm"]):
            raise SystemExit(f"arm {arm!r} not in {path} (has: {sorted(set(long['arm']))})")
    keep = ["case_id", "patient_id", "source", "volume_mm3", "volume_tertile", "cnr_tertile",
            cnr_col]
    keep = [c for c in keep if c in long.columns]
    per_arm = {}
    for arm in (arm_cnn, arm_tf):
        g = long[long["arm"] == arm].set_index("case_id")
        per_arm[arm] = g
    common = per_arm[arm_cnn].index.intersection(per_arm[arm_tf].index)
    base = per_arm[arm_cnn].loc[common, [c for c in keep if c != "case_id"]].copy()
    for tag, arm in (("cnn", arm_cnn), ("tf", arm_tf)):
        for metric, out_col in (("dice", "dice"), ("nsd2mm", "nsd"), ("hd95", "hd95"),
                                ("detected", "detected"), ("fp_count", "fp")):
            if metric in per_arm[arm].columns:
                base[f"{out_col}_{tag}"] = per_arm[arm].loc[common, metric]
    df = base.reset_index().rename(columns={cnr_col: "cnr"})
    return df, int(long["case_id"].nunique())


def wide_from_predictions(args, cnr_col: str):
    """Standalone path: compute Dice/NSD directly from two prediction directories."""
    strata = pd.read_csv(args.strata)
    cohort_full = pd.read_csv(args.cohort).drop_duplicates("case_id")
    cols = ["case_id", "patient_id"] + ([args.source_col]
                                        if args.source_col in cohort_full.columns else [])
    cohort = cohort_full[cols].rename(columns={args.source_col: "source"}).copy()
    if "source" not in cohort.columns:
        cohort["source"] = "unknown"
    rows = []
    for r in tqdm(strata.itertuples(), total=len(strata)):
        ref = args.refs / f"{r.case_id}.nii.gz"
        pc, pt = args.pred_cnn / f"{r.case_id}.nii.gz", args.pred_tf / f"{r.case_id}.nii.gz"
        if not (ref.exists() and pc.exists() and pt.exists()):
            continue
        mc, mt = all_metrics(pc, ref), all_metrics(pt, ref)
        rows.append({"case_id": r.case_id, "dice_cnn": mc["dice"], "dice_tf": mt["dice"],
                     "nsd_cnn": mc["nsd2mm"], "nsd_tf": mt["nsd2mm"],
                     "hd95_cnn": mc["hd95"], "hd95_tf": mt["hd95"],
                     "detected_cnn": mc["detected"], "detected_tf": mt["detected"],
                     "fp_cnn": mc["fp_count"], "fp_tf": mt["fp_count"],
                     "volume_mm3": r.volume_mm3, "volume_tertile": r.volume_tertile,
                     "cnr_tertile": r.cnr_tertile, "cnr": getattr(r, cnr_col)})
    return pd.DataFrame(rows).merge(cohort, on="case_id", how="left"), len(strata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-table", type=Path, default=None,
                    help="long-format per-case table from build_per_case_table.py "
                         "(preferred; avoids recomputing surface metrics)")
    ap.add_argument("--arm-cnn", default="cnn", help="arm name in --metrics-table")
    ap.add_argument("--arm-tf", default="tf", help="arm name in --metrics-table")
    ap.add_argument("--pred-cnn", type=Path)
    ap.add_argument("--pred-tf", type=Path)
    ap.add_argument("--refs", type=Path)
    ap.add_argument("--strata", default="splits/strata.csv", type=Path)
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--cnr-col", default=None,
                    help="CNR column to use as the contrast covariate; default is the "
                         "primary ring width from the frozen config (contrast_sensitivity.py "
                         "passes the 5 mm / 15 mm columns here)")
    ap.add_argument("--out", default="results/h1", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.metrics_table is None and not (args.pred_cnn and args.pred_tf and args.refs):
        ap.error("give --metrics-table, or all of --pred-cnn/--pred-tf/--refs")

    cnr_col = args.cnr_col or f"cnr_ring{CFG['strata']['cnr']['ring_width_mm_primary']}mm"

    if args.metrics_table:
        df, n_expected = wide_from_metrics_table(args.metrics_table, args.arm_cnn, args.arm_tf,
                                                 cnr_col)
    else:
        df, n_expected = wide_from_predictions(args, cnr_col)

    df["delta_dice"] = df["dice_cnn"] - df["dice_tf"]
    df["log_volume"] = np.log(df["volume_mm3"])
    df.to_csv(args.out / "per_case.csv", index=False)
    missing = n_expected - len(df)
    if missing:
        print(f"WARNING: {missing} cohort cases missing predictions — paired analysis requires all")

    st = CFG["statistics"]

    # 2. Regression + bootstrap CI on log-volume slope
    model = smf.ols("delta_dice ~ log_volume + cnr + C(source)", data=df).fit()
    slope = model.params["log_volume"]
    lo, hi = bootstrap_ci(
        df, lambda b: smf.ols("delta_dice ~ log_volume + cnr + C(source)", data=b).fit().params["log_volume"],
        st["bootstrap_resamples"], st["bootstrap_seed"])
    h1_supported = hi < 0
    print(f"H1 slope of delta_dice on log volume: {slope:.5f}  95% CI [{lo:.5f}, {hi:.5f}]"
          f"  -> {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'} (rule: CI below zero)")

    # 3. Wilcoxon within volume tertiles
    wil = []
    for t, g in df.groupby("volume_tertile"):
        s, p = wilcoxon(g["dice_cnn"], g["dice_tf"])
        wil.append({"volume_tertile": t, "n": len(g), "median_delta": g["delta_dice"].median(),
                    "wilcoxon_stat": s, "p": p})
    pd.DataFrame(wil).to_csv(args.out / "wilcoxon_by_tertile.csv", index=False)

    # 4. TOST at +/-3 Dice points on the large-lesion tertile (bootstrap CI vs margins)
    margin = CFG["hypotheses"]["h1"]["equivalence_margin_dice_points"] / 100.0
    large = df[df["volume_tertile"] == df["volume_tertile"].max()]
    lo_m, hi_m = bootstrap_ci(large, lambda b: b["delta_dice"].mean(),
                                st["bootstrap_resamples"], st["bootstrap_seed"] + 1)
    equivalent = (lo_m > -margin) and (hi_m < margin)
    print(f"Large-tertile mean delta_dice CI [{lo_m:.4f}, {hi_m:.4f}] vs +/-{margin:.2f} "
          f"-> {'EQUIVALENT' if equivalent else 'NOT SHOWN EQUIVALENT'}")

    # 5. Stratum tables with per-cell n and CIs
    cells = []
    for (vt, ct), g in df.groupby(["volume_tertile", "cnr_tertile"]):
        for arm in ("cnn", "tf"):
            lo_c, hi_c = bootstrap_ci(g, lambda b, a=arm: b[f"dice_{a}"].mean(),
                                        st["bootstrap_resamples"], st["bootstrap_seed"] + 2)
            cell = {"volume_tertile": vt, "cnr_tertile": ct, "arm": arm, "n": len(g),
                    "dice_mean": g[f"dice_{arm}"].mean(), "ci_lo": lo_c, "ci_hi": hi_c,
                    "nsd_mean": g[f"nsd_{arm}"].mean()}
            # HD95 / detection / FP are present whenever the table came from
            # build_per_case_table.py; the standalone path fills them too.
            for src, dst in (("hd95", "hd95_mean"), ("detected", "detection_sensitivity"),
                             ("fp", "fp_per_case")):
                col = f"{src}_{arm}"
                if col in g.columns:
                    cell[dst] = g[col].mean()
            cells.append(cell)
    pd.DataFrame(cells).to_csv(args.out / "stratum_heatmap.csv", index=False)

    summary = {"h1_slope": float(slope), "h1_ci": [lo, hi], "h1_supported": bool(h1_supported),
               "large_tertile_tost_ci": [lo_m, hi_m], "large_tertile_equivalent": bool(equivalent),
               "n_cases": int(len(df))}
    yaml.safe_dump(summary, open(args.out / "summary.yaml", "w"))
    print(f"Wrote {args.out}/")


if __name__ == "__main__":
    main()
