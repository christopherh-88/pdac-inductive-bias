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

Usage:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-cnn", required=True, type=Path)
    ap.add_argument("--pred-tf", required=True, type=Path)
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--strata", default="splits/strata.csv", type=Path)
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--source-col", default="source")
    ap.add_argument("--out", default="results/h1", type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    strata = pd.read_csv(args.strata)
    cohort_full = pd.read_csv(args.cohort)
    cohort = cohort_full[["case_id", "patient_id"]].copy()
    cohort["source"] = (cohort_full[args.source_col] if args.source_col in cohort_full.columns
                        else "unknown")

    rows = []
    for r in tqdm(strata.itertuples(), total=len(strata)):
        ref = args.refs / f"{r.case_id}.nii.gz"
        pc, pt = args.pred_cnn / f"{r.case_id}.nii.gz", args.pred_tf / f"{r.case_id}.nii.gz"
        if not (ref.exists() and pc.exists() and pt.exists()):
            continue
        mc, mt = all_metrics(pc, ref), all_metrics(pt, ref)
        rows.append({"case_id": r.case_id, "dice_cnn": mc["dice"], "dice_tf": mt["dice"],
                     "nsd_cnn": mc["nsd2mm"], "nsd_tf": mt["nsd2mm"],
                     "volume_mm3": r.volume_mm3, "volume_tertile": r.volume_tertile,
                     "cnr_tertile": r.cnr_tertile,
                     "cnr": getattr(r, f"cnr_ring{CFG['strata']['cnr']['ring_width_mm_primary']}mm")})
    df = pd.DataFrame(rows).merge(cohort, on="case_id", how="left")
    df["delta_dice"] = df["dice_cnn"] - df["dice_tf"]
    df["log_volume"] = np.log(df["volume_mm3"])
    df.to_csv(args.out / "per_case.csv", index=False)
    missing = len(strata) - len(df)
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
            cells.append({"volume_tertile": vt, "cnr_tertile": ct, "arm": arm, "n": len(g),
                          "dice_mean": g[f"dice_{arm}"].mean(), "ci_lo": lo_c, "ci_hi": hi_c,
                          "nsd_mean": g[f"nsd_{arm}"].mean()})
    pd.DataFrame(cells).to_csv(args.out / "stratum_heatmap.csv", index=False)

    summary = {"h1_slope": float(slope), "h1_ci": [lo, hi], "h1_supported": bool(h1_supported),
               "large_tertile_tost_ci": [lo_m, hi_m], "large_tertile_equivalent": bool(equivalent),
               "n_cases": int(len(df))}
    yaml.safe_dump(summary, open(args.out / "summary.yaml", "w"))
    print(f"Wrote {args.out}/")


if __name__ == "__main__":
    main()
