#!/usr/bin/env python
"""Assemble the nnU-Net raw dataset for the PDAC lesion-segmentation task.

Dataset501_PDAC: images from the deduplicated cohort, segmentation = manual PDAC lesion
(label 1 -> foreground 1, everything else background). Only manual-lesion cases are included
in the main dataset; the 194 model-generated delineations are staged into a separate optional
Dataset502_PDACauto (training-ablation only, never evaluation).

Usage:
  export nnUNet_raw=/path/to/data/nnUNet_raw
  python scripts/data/convert_to_nnunet.py --data-root /path/to/data --cohort splits/cohort.csv
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import yaml
from tqdm import tqdm
import pandas as pd

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "config" / "analysis_config.yaml"))
LESION = CFG["labels"]["panorama"]["pdac_lesion"]


def write_binary_lesion_label(src: str, dst: Path, img: sitk.Image):
    seg = sitk.ReadImage(src)
    # nnU-Net's verify_dataset_integrity requires the label to share the image's exact
    # geometry. Real PANORAMA cases can differ here -- confirmed in practice: both a tiny
    # float-precision spacing mismatch (a re-serialization rounding artifact) and a genuine
    # one (8.0mm vs ~4.8mm z-spacing for a case whose manual label came from a differently
    # reconstructed series of the same study). Resample onto the image's own grid rather than
    # assume they already match.
    if (seg.GetSize() != img.GetSize() or seg.GetSpacing() != img.GetSpacing()
            or seg.GetDirection() != img.GetDirection() or seg.GetOrigin() != img.GetOrigin()):
        seg = sitk.Resample(seg, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, seg.GetPixelID())
    arr = (sitk.GetArrayFromImage(seg) == LESION).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(dst), useCompression=True)


def build(dataset_dir: Path, rows, desc: str):
    (dataset_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    for r in tqdm(rows.itertuples(), total=len(rows), desc=desc):
        img_dst = dataset_dir / "imagesTr" / f"{r.case_id}_0000.nii.gz"
        lab_dst = dataset_dir / "labelsTr" / f"{r.case_id}.nii.gz"
        if not img_dst.exists():
            if r.path.endswith(".nii.gz"):
                shutil.copy(r.path, img_dst)
            else:  # convert .mha etc. to nii.gz
                sitk.WriteImage(sitk.ReadImage(r.path), str(img_dst), useCompression=True)
        if not lab_dst.exists():
            write_binary_lesion_label(r.label_src, lab_dst, sitk.ReadImage(str(img_dst)))
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "pdac_lesion": 1},
        "numTraining": len(rows),
        "file_ending": ".nii.gz",
        "licence": "images CC BY-NC 4.0 (PANORAMA); this arrangement MIT",
        "description": desc,
    }
    json.dump(dataset_json, open(dataset_dir / "dataset.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--include-auto-ablation", action="store_true",
                    help="also build Dataset502 with model-generated delineations (training ablation)")
    args = ap.parse_args()

    raw = Path(os.environ.get("nnUNet_raw", args.data_root / "nnUNet_raw"))
    cohort = pd.read_csv(args.cohort)

    manual = cohort[cohort["has_manual_lesion"]].copy()
    manual["label_src"] = manual["manual_label"]
    build(raw / "Dataset501_PDAC", manual, "manual-lesion PDAC cohort")
    print(f"Dataset501_PDAC: {len(manual)} cases at {raw / 'Dataset501_PDAC'}")

    if args.include_auto_ablation:
        auto = cohort[(~cohort["has_manual_lesion"]) & (cohort["automatic_label"] != "")].copy()
        auto["label_src"] = auto["automatic_label"]
        build(raw / "Dataset502_PDACauto", auto, "model-generated delineations — training ablation only, never evaluation")
        print(f"Dataset502_PDACauto: {len(auto)} cases (ablation only)")

    print("Next: nnUNetv2_plan_and_preprocess -d 501 -pl nnUNetPlannerResEnc{M|L|XL} --verify_dataset_integrity")
    print("Then copy splits/splits_final.json into nnUNet_preprocessed/Dataset501_PDAC/")


if __name__ == "__main__":
    main()
