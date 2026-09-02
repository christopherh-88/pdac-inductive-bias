"""Synthetic PANORAMA-like dataset generator, used only by tests/run_smoke_test.py.

Not real data and never used for the study itself — this exists so the pipeline scripts can
be exercised end to end (dedup -> strata -> splits -> analysis) without downloading PANORAMA
or running a GPU training job. Mirrors the real layout exactly:

  data_root/panorama/images/<case_id>_0000.nii.gz
  data_root/panorama/panorama_labels/manual_labels/<case_id>.nii.gz    (ONE file, labels 1-6
                                                                         combined: confirmed-lesion
                                                                         cases only)
  data_root/panorama/panorama_labels/automatic_labels/<case_id>.nii.gz (ONE file, labels 2-6
                                                                         combined: the rest)
  data_root/panorama/panorama_labels/clinical_information.xlsx

manual_labels/ and automatic_labels/ are mutually exclusive per case — confirmed against the
real DIAGNijmegen/panorama_labels repo (482 + 1756 = 2238 total cases, zero overlap). Each
directory holds a single multi-class segmentation (lesion + veins + arteries + parenchyma +
duct + CBD all in one file), not two complementary partial-label files to be combined.

Also writes one exact-content duplicate case (mimicking MSD/NIH cases redistributed inside
PANORAMA) so deduplicate.py's content-hash stage has something real to catch.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

SPACING = (1.0, 1.0, 1.5)  # x, y, z mm -- nontrivial anisotropic spacing to catch axis-order bugs
SHAPE = (64, 64, 48)       # x, y, z
SOURCES = ["Radboud", "UMCG", "MSKCC"]
LESION_RADII = [3, 6, 10]                  # small / medium / large -> volume tertiles
CONTRASTS = [(30, 8), (55, 8), (90, 8)]    # (lesion HU delta, std) -> low/med/high CNR


def _make_case(img_dir, manual_dir, auto_dir, case_id, lesion_radius, lesion_hu, seed):
    r = np.random.default_rng(seed)
    zc, yc, xc = SHAPE[2] // 2, SHAPE[1] // 2, SHAPE[0] // 2
    zz, yy, xx = np.meshgrid(np.arange(SHAPE[2]), np.arange(SHAPE[1]), np.arange(SHAPE[0]),
                              indexing="ij")

    hu = r.normal(0, 15, size=(SHAPE[2], SHAPE[1], SHAPE[0])).astype(np.float32)

    parenchyma_r = 18
    parenchyma_mask = ((xx - xc) ** 2 + (yy - yc) ** 2 + ((zz - zc) * 1.5) ** 2) <= parenchyma_r ** 2
    hu[parenchyma_mask] = r.normal(5, 10, size=parenchyma_mask.sum())

    lesion_mask = np.zeros_like(parenchyma_mask)
    if lesion_radius > 0:
        ox, oy, oz = xc + 4, yc - 3, zc
        lesion_mask = ((xx - ox) ** 2 + (yy - oy) ** 2 + ((zz - oz) * 1.5) ** 2) <= lesion_radius ** 2
        hu[lesion_mask] = r.normal(lesion_hu, 8, size=lesion_mask.sum())

    veins_mask = ((xx - (xc - 20)) ** 2 + (yy - yc) ** 2 + (zz - zc) ** 2) <= 4 ** 2
    arteries_mask = ((xx - (xc + 20)) ** 2 + (yy - yc) ** 2 + (zz - zc) ** 2) <= 4 ** 2
    duct_mask = ((xx - xc) ** 2 + (yy - (yc + 20)) ** 2 + (zz - zc) ** 2) <= 3 ** 2
    cbd_mask = ((xx - xc) ** 2 + (yy - (yc - 20)) ** 2 + (zz - zc) ** 2) <= 3 ** 2

    seg = np.zeros(SHAPE[::-1], dtype=np.uint8)
    seg[parenchyma_mask] = 4
    seg[veins_mask] = 2
    seg[arteries_mask] = 3
    seg[duct_mask] = 5
    seg[cbd_mask] = 6
    if lesion_radius > 0:
        seg[lesion_mask] = 1

    def to_img(arr):
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(SPACING)
        img.SetOrigin((0.0, 0.0, 0.0))
        img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        return img

    sitk.WriteImage(to_img(hu), str(img_dir / f"{case_id}_0000.nii.gz"), useCompression=True)
    # One combined multi-class file, in manual_labels/ if this case has a confirmed lesion,
    # else automatic_labels/ — mirrors the real repo's mutually-exclusive layout.
    dest_dir = manual_dir if lesion_radius > 0 else auto_dir
    sitk.WriteImage(to_img(seg), str(dest_dir / f"{case_id}.nii.gz"), useCompression=True)
    return int(lesion_mask.sum())


def generate(data_root: Path, n_patients=30, seed=20260823):
    """Returns (info_df, img_dir, manual_dir, auto_dir)."""
    img_dir = data_root / "panorama" / "images"
    labels_root = data_root / "panorama" / "panorama_labels"
    manual_dir = labels_root / "manual_labels"
    auto_dir = labels_root / "automatic_labels"
    for d in (img_dir, manual_dir, auto_dir):
        d.mkdir(parents=True, exist_ok=True)

    info_rows = []
    for p in range(n_patients):
        patient_id = f"{100000 + p}"
        case_id = f"{patient_id}_00001"
        source = SOURCES[p % len(SOURCES)]
        radius = LESION_RADII[p % len(LESION_RADII)]
        lesion_hu, _ = CONTRASTS[(p // len(LESION_RADII)) % len(CONTRASTS)]
        _make_case(img_dir, manual_dir, auto_dir, case_id, radius, lesion_hu, seed=1000 + p)
        info_rows.append({"case_id": case_id, "source": source})

    for k in range(4):  # NIH-like negatives: no lesion, no manual label
        patient_id = f"{200000 + k}"
        case_id = f"{patient_id}_00001"
        _make_case(img_dir, manual_dir, auto_dir, case_id, 0, 0, seed=2000 + k)
        info_rows.append({"case_id": case_id, "source": "NIH"})

    # One exact-content duplicate, no manual label of its own (as if redistributed under another
    # ID and only auto-segmented there). dup_src (p=0) has a lesion, so its combined seg file
    # lives in manual_dir — mirror that content into auto_dir under the new id.
    dup_src = info_rows[0]["case_id"]
    dup_id = "999001_00001"
    sitk.WriteImage(sitk.ReadImage(str(img_dir / f"{dup_src}_0000.nii.gz")),
                    str(img_dir / f"{dup_id}_0000.nii.gz"), useCompression=True)
    sitk.WriteImage(sitk.ReadImage(str(manual_dir / f"{dup_src}.nii.gz")),
                    str(auto_dir / f"{dup_id}.nii.gz"), useCompression=True)
    info_rows.append({"case_id": dup_id, "source": "MSKCC"})

    pd.DataFrame(info_rows).rename(columns={"case_id": "Case ID", "source": "Source"}).to_excel(
        labels_root / "clinical_information.xlsx", index=False)
    return pd.DataFrame(info_rows)


def make_predictions(cohort_csv: Path, manual_dir: Path, out_dir: Path, seed=42):
    """Refs + CNN/TF predictions with a known pattern: CNN erodes little on small lesions and a
    lot on large ones; the transformer does the opposite. Lets tests assert paired_analysis.py
    recovers a specific sign/magnitude, not just "runs without crashing"."""
    refs_dir, cnn_dir, tf_dir = out_dir / "refs", out_dir / "pred_cnn", out_dir / "pred_tf"
    for d in (refs_dir, cnn_dir, tf_dir):
        d.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(cohort_csv)
    manual = cohort[cohort["has_manual_lesion"]]
    rng = np.random.default_rng(seed)

    for _, r in manual.iterrows():
        cid = r["case_id"]
        seg = sitk.ReadImage(str(manual_dir / f"{cid}.nii.gz"))
        arr = (sitk.GetArrayFromImage(seg) == 1).astype(np.uint8)
        vol = arr.sum()

        ref_img = sitk.GetImageFromArray(arr)
        ref_img.CopyInformation(seg)
        sitk.WriteImage(ref_img, str(refs_dir / f"{cid}.nii.gz"), useCompression=True)

        if vol < 300:
            cnn_iter, tf_iter = 0, 2
        elif vol < 1500:
            cnn_iter, tf_iter = 1, 1
        else:
            cnn_iter, tf_iter = 3, 0

        pred_cnn = ndimage.binary_erosion(arr, iterations=cnn_iter).astype(np.uint8) if cnn_iter else arr.copy()
        pred_tf = ndimage.binary_erosion(arr, iterations=tf_iter).astype(np.uint8) if tf_iter else arr.copy()

        for pred, d in ((pred_cnn, cnn_dir), (pred_tf, tf_dir)):
            noisy = pred.copy()
            flip = rng.random(noisy.shape) < 0.0005
            noisy[flip] = 1 - noisy[flip]
            out_img = sitk.GetImageFromArray(noisy.astype(np.uint8))
            out_img.CopyInformation(seg)
            sitk.WriteImage(out_img, str(d / f"{cid}.nii.gz"), useCompression=True)

    return refs_dir, cnn_dir, tf_dir
