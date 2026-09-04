#!/usr/bin/env python
"""Compute the pre-registered stratification variables and freeze the thresholds.

Per case (manual-lesion PDAC cases only):
  - lesion volume (mm^3) and max in-plane diameter (mm) from the manual label (label 1)
  - CNR = |mean HU(lesion) - mean HU(ring)| / SD(ring) for ring widths 5, 10, 15 mm,
    ring = parenchyma (label 4) within the ring distance of the lesion surface, excluding
    lesion, duct, vessels, CBD (labels 1,2,3,5,6) — all read from the SAME manual-side file,
    since manual_labels/automatic_labels are mutually exclusive per case in the real repo and
    each is one multi-class file (lesion + organs together), not two files to combine.

Then computes volume and CNR tertile cut points over the cohort ONCE and appends them to
config/frozen_thresholds.yaml. Also runs the pre-specified ring-width rank-stability check
(Spearman rho >= 0.8, else the contrast axis is flagged exploratory).

Usage:
  python scripts/analysis/compute_strata.py --data-root /path/to/data --cohort splits/cohort.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from scipy.stats import spearmanr
from tqdm import tqdm

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "config" / "analysis_config.yaml"))
L = CFG["labels"]["panorama"]


def lesion_stats(img: sitk.Image, manual: sitk.Image, auto: sitk.Image, ring_widths):
    """Volume, max in-plane diameter, and CNR at EVERY ring width, from one pass.

    The distance transform dominates the cost of this script, and it does not depend on the
    ring width — only the threshold applied to it does. Computing it once and thresholding it
    three times is the difference between one distance transform per case and three, on the
    single most expensive step of cohort preparation.

    Returns (volume_mm3, max_inplane_diameter_mm, {ring_mm: cnr}), or None when the case has
    no manual lesion.
    """
    sp = np.array(manual.GetSpacing())          # (x, y, z)
    vox_mm3 = float(np.prod(sp))
    m_full = sitk.GetArrayViewFromImage(manual)  # (z, y, x)
    a_full = sitk.GetArrayViewFromImage(auto)
    hu_full = sitk.GetArrayViewFromImage(img)

    lesion_full = m_full == L["pdac_lesion"]
    if not lesion_full.any():
        return None

    # Crop to the lesion's bounding box + the largest ring margin before the distance-map step:
    # a full-res PANORAMA CT is ~512x512x150-300 (40M+ voxels), and running
    # SignedMaurerDistanceMap over the whole volume reliably segfaulted ITK's native code in
    # practice (confirmed on Kaggle against real cases — a synthetic 64x64x48 test volume never
    # surfaces this). The margin must cover every ring width computed below, not just one.
    zs, ys, xs = np.where(lesion_full)
    max_ring_mm = max(ring_widths)
    mz, my, mx = (int(np.ceil(max_ring_mm / sp[i])) + 2 for i in (2, 1, 0))
    z0, z1 = max(zs.min() - mz, 0), min(zs.max() + mz + 1, m_full.shape[0])
    y0, y1 = max(ys.min() - my, 0), min(ys.max() + my + 1, m_full.shape[1])
    x0, x1 = max(xs.min() - mx, 0), min(xs.max() + mx + 1, m_full.shape[2])
    m, a, hu = m_full[z0:z1, y0:y1, x0:x1], a_full[z0:z1, y0:y1, x0:x1], hu_full[z0:z1, y0:y1, x0:x1]
    lesion = m == L["pdac_lesion"]

    volume = float(lesion.sum()) * vox_mm3
    # max in-plane diameter: largest Feret-ish extent per axial slice (bounding-box diagonal proxy)
    zs, ys, xs = np.where(lesion)
    diam = 0.0
    for z in np.unique(zs):
        sel = zs == z
        dy = (ys[sel].max() - ys[sel].min() + 1) * sp[1]
        dx = (xs[sel].max() - xs[sel].min() + 1) * sp[0]
        diam = max(diam, float(np.hypot(dx, dy)))

    # signed distance (mm) from lesion surface, positive outside
    lesion_img = sitk.GetImageFromArray(lesion.astype(np.uint8))
    lesion_img.SetSpacing(tuple(sp))
    # SignedMaurerDistanceMap's return is an unnamed temporary; GetArrayViewFromImage on it
    # directly is a zero-copy view that goes stale as soon as the temporary is garbage-collected.
    # Name it and copy the array instead.
    dmap_img = sitk.SignedMaurerDistanceMap(lesion_img, insideIsPositive=False, squaredDistance=False,
                                             useImageSpacing=True)
    dmap = sitk.GetArrayFromImage(dmap_img)

    exclude = np.isin(m, [L["pdac_lesion"]]) | np.isin(
        a, [L["pdac_lesion"], L["veins"], L["arteries"], L["pancreatic_duct"], L["common_bile_duct"]])
    parenchyma = (a == L["pancreas_parenchyma"]) & (dmap > 0) & ~exclude
    lesion_mean = float(hu[lesion].mean())

    cnr_by_ring = {}
    for ring_mm in ring_widths:
        ring = parenchyma & (dmap <= ring_mm)
        if ring.sum() < 10:
            cnr_by_ring[ring_mm] = np.nan
            continue
        ring_hu = hu[ring].astype(np.float64)
        cnr_by_ring[ring_mm] = float(abs(lesion_mean - ring_hu.mean())
                                     / (ring_hu.std() + 1e-8))
    return volume, diam, cnr_by_ring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--out", default="splits/strata.csv", type=Path)
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort)
    cases = cohort[cohort["has_manual_lesion"]].copy()
    ring_widths = [CFG["strata"]["cnr"]["ring_width_mm_primary"]] + CFG["strata"]["cnr"]["ring_width_mm_sensitivity"]

    rows = []
    dropped = []
    for _, r in tqdm(cases.iterrows(), total=len(cases)):
        img = sitk.ReadImage(r["path"])
        # manual_labels/automatic_labels are mutually exclusive per case in the real repo (482 +
        # 1756 = 2238 total, zero overlap) — the single manual-side file already contains lesion
        # + vessels + parenchyma + duct + CBD together, so it doubles for both roles below.
        manual = sitk.ReadImage(r["manual_label"])
        auto = manual
        res = lesion_stats(img, manual, auto, ring_widths)
        if res is None:
            dropped.append(r["case_id"])
            continue
        volume, diam, cnr_by_ring = res
        row = {"case_id": r["case_id"], "volume_mm3": volume, "max_inplane_diam_mm": diam}
        for w in ring_widths:
            row[f"cnr_ring{w}mm"] = cnr_by_ring[w]
        rows.append(row)
    if dropped:
        print(f"WARNING: {len(dropped)} case(s) flagged has_manual_lesion but contain no lesion "
              f"voxels in the label file — dropped from strata.csv: {dropped}")
    df = pd.DataFrame(rows)

    # Frozen tertile cut points over the cohort (computed once)
    vol_cuts = df["volume_mm3"].quantile([1 / 3, 2 / 3]).tolist()
    cnr_col = f"cnr_ring{CFG['strata']['cnr']['ring_width_mm_primary']}mm"
    cnr_cuts = df[cnr_col].quantile([1 / 3, 2 / 3]).tolist()
    df["volume_tertile"] = np.digitize(df["volume_mm3"], vol_cuts)
    df["cnr_tertile"] = np.digitize(df[cnr_col], cnr_cuts)
    df["above_2cm"] = df["max_inplane_diam_mm"] >= CFG["strata"]["volume"]["secondary_cut_diameter_mm"]

    # Ring-width rank stability (pre-specified sensitivity gate)
    rhos = {}
    cols = [f"cnr_ring{w}mm" for w in ring_widths]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rho, _ = spearmanr(df[cols[i]], df[cols[j]], nan_policy="omit")
            rhos[f"{cols[i]}__vs__{cols[j]}"] = float(rho)
    exploratory = any(v < CFG["strata"]["cnr"]["rank_stability_min_spearman"] for v in rhos.values())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    frozen_path = Path(__file__).parents[2] / "config" / "frozen_thresholds.yaml"
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
    }
    yaml.safe_dump(frozen, open(frozen_path, "w"), sort_keys=False)
    print(f"Wrote {args.out} and froze thresholds in {frozen_path}")
    if exploratory:
        print("NOTE: CNR ranks unstable across ring widths — per pre-registration, "
              "the contrast axis is reported as exploratory.")


if __name__ == "__main__":
    main()
