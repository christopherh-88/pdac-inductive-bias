#!/usr/bin/env python
"""H3 boundary-tolerance floor.

For each reference lesion mask: Dice between the mask and its 1- and 2-voxel eroded and
dilated versions, plus a random boundary-jitter version (seeded), gives the Dice ceiling
attributable to boundary ambiguity at that tolerance. The per-case floor is the minimum
Dice over the 1..2-voxel perturbations (the pre-registered tolerance band).

Then, given per-arm in-domain and leave-one-source-out Dice tables, reports the share of
each arm's cross-source Dice loss that falls inside the floor (H3 rule: supported if >50%
for at least one arm).

Usage:
  python scripts/analysis/boundary_tolerance.py --refs DIR --cohort splits/cohort.csv \
      [--loso results/loso_dice.csv] --out results/h3
  # loso_dice.csv columns: case_id, arm, dice_in_domain, dice_loso
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage
from tqdm import tqdm

from metrics import CFG, load_mask, dice, bootstrap_ci, find_case_file


def jitter_mask(mask, rng, magnitude=1):
    """Random boundary jitter: independently erode/dilate random boundary patches."""
    dil = ndimage.binary_dilation(mask, iterations=magnitude)
    ero = ndimage.binary_erosion(mask, iterations=magnitude)
    band_out = dil & ~mask
    band_in = mask & ~ero
    out = mask.copy()
    out[band_out] = rng.random(band_out.sum()) < 0.5   # add ~half the outer band
    out[band_in] = rng.random(band_in.sum()) >= 0.5    # drop ~half the inner band
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--loso", type=Path, default=None)
    ap.add_argument("--out", default=Path("results/h3"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    h3 = CFG["hypotheses"]["h3"]
    rng = np.random.default_rng(h3["jitter"]["seed"])
    cohort = pd.read_csv(args.cohort)
    cases = cohort[cohort["has_manual_lesion"]]["case_id"].tolist()

    rows = []
    for cid in tqdm(cases):
        ref_p = find_case_file(args.refs, cid)
        if ref_p is None:
            continue
        ref, _ = load_mask(ref_p)
        row = {"case_id": cid}
        floor_vals = []
        for v in h3["boundary_perturbations_voxels"]:
            d_ero = dice(ndimage.binary_erosion(ref, iterations=v), ref)
            d_dil = dice(ndimage.binary_dilation(ref, iterations=v), ref)
            row[f"dice_erode{v}"], row[f"dice_dilate{v}"] = d_ero, d_dil
            floor_vals += [d_ero, d_dil]
        if h3["jitter"]["enabled"]:
            row["dice_jitter"] = dice(jitter_mask(ref, rng, h3["jitter"]["magnitude_voxels"]), ref)
            floor_vals.append(row["dice_jitter"])
        row["tolerance_floor_dice"] = min(floor_vals)   # 1 - floor = tolerable Dice loss
        rows.append(row)
    floor = pd.DataFrame(rows)
    floor.to_csv(args.out / "boundary_floor_per_case.csv", index=False)
    print(f"Boundary-tolerance floor over {len(floor)} cases: "
          f"median tolerable Dice loss = {(1 - floor['tolerance_floor_dice']).median():.4f}")

    if args.loso and args.loso.exists():
        loso = pd.read_csv(args.loso).merge(floor[["case_id", "tolerance_floor_dice"]], on="case_id")
        loso = loso.merge(cohort[["case_id", "patient_id"]], on="case_id")
        loso["dice_loss"] = loso["dice_in_domain"] - loso["dice_loso"]
        loso["tolerable_loss"] = 1 - loso["tolerance_floor_dice"]
        loso["loss_inside_floor"] = np.minimum(loso["dice_loss"].clip(lower=0), loso["tolerable_loss"])

        def share_stat(g):
            total = g["dice_loss"].clip(lower=0).sum()
            if total <= 0:
                return np.nan
            return g["loss_inside_floor"].sum() / total

        st = CFG["statistics"]
        summary = {}
        for arm, g in loso.groupby("arm"):
            share = float(share_stat(g))
            lo, hi = bootstrap_ci(g, share_stat, st["bootstrap_resamples"], st["bootstrap_seed"])
            summary[arm] = {"share_inside_floor": share, "share_inside_floor_ci": [lo, hi],
                            "n": int(len(g))}
            print(f"{arm}: {share:.1%} of cross-source Dice loss inside the boundary floor "
                  f"(CI [{lo:.1%}, {hi:.1%}])")
        supported = any(v["share_inside_floor"] > h3["share_threshold"] for v in summary.values()
                        if not np.isnan(v["share_inside_floor"]))
        summary["h3_supported"] = bool(supported)
        yaml.safe_dump(summary, open(args.out / "summary.yaml", "w"))
        print(f"H3 (> {h3['share_threshold']:.0%} for at least one arm): "
              f"{'SUPPORTED' if supported else 'NOT SUPPORTED'}")
    else:
        print("No --loso table given: floor computed only. Provide loso_dice.csv after the "
              "leave-one-source-out runs (Tier B).")


if __name__ == "__main__":
    main()
