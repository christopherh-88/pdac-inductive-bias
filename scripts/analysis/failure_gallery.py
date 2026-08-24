#!/usr/bin/env python
"""Failure gallery (figure 6 / analysis-plan step 5): rule-selected matched failure cases,
both predictions and the reference shown on the same slice.

Selection rule (frozen, config/analysis_config.yaml -> failure_gallery): within each
(volume_tertile x cnr_tertile) stratum cell, the case with the largest |delta_dice| is
selected. Never selected by hand.

For each selected case, renders the axial slice through the reference lesion's centroid with
three panels: image + reference contour, image + CNN prediction contour, image + transformer
prediction contour — all on the same slice, same window.

Inputs: paired_analysis.py's per_case.csv (needs case_id, delta_dice, dice_cnn, dice_tf,
volume_tertile, cnr_tertile), the image root, and the same prediction/reference directories
used there.

Usage:
  python scripts/analysis/failure_gallery.py --per-case results/h1/per_case.csv \
      --pred-cnn DIR --pred-tf DIR --refs DIR --images DIR --cohort splits/cohort.csv \
      --out results/failure_gallery
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "config" / "analysis_config.yaml"))


def load_arr(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def centroid_slice(mask: np.ndarray) -> int:
    """Axial (z) slice with the largest lesion cross-section; falls back to the mask's
    z-centroid if the mask is empty (nothing to show)."""
    if not mask.any():
        return mask.shape[0] // 2
    counts = mask.reshape(mask.shape[0], -1).sum(axis=1)
    return int(np.argmax(counts))


def select_gallery(per_case: pd.DataFrame) -> pd.DataFrame:
    rule = CFG["failure_gallery"]["selection_rule"]
    per_stratum = CFG["failure_gallery"]["per_stratum"]
    assert rule == "largest_abs_delta_dice_within_each_stratum", f"unknown selection rule: {rule}"
    per_case = per_case.copy()
    per_case["abs_delta_dice"] = per_case["delta_dice"].abs()
    selected = (
        per_case.sort_values("abs_delta_dice", ascending=False)
        .groupby(["volume_tertile", "cnr_tertile"], as_index=False)
        .head(per_stratum)
        .sort_values(["volume_tertile", "cnr_tertile"])
    )
    return selected


def render_case(case_id, delta_dice, img_path, ref_path, pred_cnn_path, pred_tf_path, out_path):
    img = load_arr(img_path)
    ref = load_arr(ref_path) > 0
    pred_cnn = load_arr(pred_cnn_path) > 0
    pred_tf = load_arr(pred_tf_path) > 0

    z = centroid_slice(ref)
    wl, ww = 40, 400  # soft-tissue CT window
    vmin, vmax = wl - ww / 2, wl + ww / 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    panels = [("Reference", ref), ("CNN prediction", pred_cnn), ("Transformer prediction", pred_tf)]
    for ax, (title, mask) in zip(axes, panels):
        ax.imshow(img[z], cmap="gray", vmin=vmin, vmax=vmax)
        if mask[z].any():
            ax.contour(mask[z], levels=[0.5], colors="red", linewidths=1.5)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{case_id}  |  ΔDice(CNN−TF) = {delta_dice:+.3f}  |  slice z={z}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-case", required=True, type=Path,
                    help="per_case.csv from paired_analysis.py")
    ap.add_argument("--pred-cnn", required=True, type=Path)
    ap.add_argument("--pred-tf", required=True, type=Path)
    ap.add_argument("--refs", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path,
                    help="directory of *_0000.nii.gz raw CT images (nnU-Net imagesTr/imagesTs)")
    ap.add_argument("--out", default=Path("results/failure_gallery"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_case = pd.read_csv(args.per_case)
    required = {"case_id", "delta_dice", "volume_tertile", "cnr_tertile"}
    missing = required - set(per_case.columns)
    if missing:
        raise SystemExit(f"{args.per_case} is missing columns: {missing}")

    selected = select_gallery(per_case)
    manifest = []
    for r in selected.itertuples():
        img_path = args.images / f"{r.case_id}_0000.nii.gz"
        ref_path = args.refs / f"{r.case_id}.nii.gz"
        pred_cnn_path = args.pred_cnn / f"{r.case_id}.nii.gz"
        pred_tf_path = args.pred_tf / f"{r.case_id}.nii.gz"
        paths = {"image": img_path, "reference": ref_path, "pred_cnn": pred_cnn_path,
                 "pred_tf": pred_tf_path}
        missing_files = [k for k, p in paths.items() if not p.exists()]
        if missing_files:
            print(f"SKIP {r.case_id}: missing {missing_files}")
            continue
        out_png = args.out / f"vol{r.volume_tertile}_cnr{r.cnr_tertile}_{r.case_id}.png"
        render_case(r.case_id, r.delta_dice, img_path, ref_path, pred_cnn_path, pred_tf_path, out_png)
        manifest.append({"case_id": r.case_id, "volume_tertile": r.volume_tertile,
                         "cnr_tertile": r.cnr_tertile, "delta_dice": r.delta_dice,
                         "abs_delta_dice": r.abs_delta_dice, "figure": str(out_png)})

    pd.DataFrame(manifest).to_csv(args.out / "manifest.csv", index=False)
    print(f"Wrote {len(manifest)} failure-gallery figures to {args.out}/ "
          f"(selection rule: {CFG['failure_gallery']['selection_rule']})")


if __name__ == "__main__":
    main()
