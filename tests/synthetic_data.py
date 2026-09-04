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

Also writes two duplicate cases, mimicking MSD/NIH scans redistributed inside PANORAMA, so
deduplicate.py has something real to catch:

  - a byte-for-byte file copy, and
  - a re-encoded copy holding the same values at a different dtype.

Both are made with `shutil.copy` or an explicit dtype cast rather than by round-tripping the
image through SimpleITK. An earlier version did round-trip it, on the assumption that
ReadImage -> WriteImage preserves voxel data. It does on some platforms and not others: on
Kaggle the round-trip changed the data, so the "duplicate" was not one, and the test failed
for a reason that had nothing to do with the code under test. A fixture must not depend on
library behaviour that varies by platform — that is the thing it is supposed to be testing
around.
"""
import shutil
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

    # Two duplicates, no manual label of their own (as if redistributed under another ID and
    # only auto-segmented there). dup_src (p=0) has a lesion, so its combined seg file lives in
    # manual_dir — source from there and file the duplicates' copies under auto_dir instead.
    dup_src = info_rows[0]["case_id"]
    src_img = img_dir / f"{dup_src}_0000.nii.gz"
    src_seg = manual_dir / f"{dup_src}.nii.gz"

    # (a) A byte-for-byte copy. Unambiguously a duplicate on every platform.
    shutil.copy(src_img, img_dir / "999001_00001_0000.nii.gz")
    shutil.copy(src_seg, auto_dir / "999001_00001.nii.gz")
    info_rows.append({"case_id": "999001_00001", "source": "MSKCC"})

    # (b) A re-encoded copy: same values, stored as float64 instead of float32. This is what a
    # redistribution actually looks like, and it is the case a byte-level hash misses.
    reencoded = sitk.ReadImage(str(src_img))
    arr64 = sitk.GetArrayFromImage(reencoded).astype(np.float64)
    out = sitk.GetImageFromArray(arr64)
    out.CopyInformation(reencoded)
    sitk.WriteImage(out, str(img_dir / "999002_00001_0000.nii.gz"), useCompression=True)
    shutil.copy(src_seg, auto_dir / "999002_00001.nii.gz")
    info_rows.append({"case_id": "999002_00001", "source": "MSKCC"})

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


def _degrade(arr, iterations, rng, flip_p=0.0005):
    """Erode a mask by `iterations` and add a little speckle, the same way make_predictions
    builds its designed patterns — so every synthetic prediction set in the tests is produced
    by one mechanism with one knob."""
    out = ndimage.binary_erosion(arr, iterations=iterations).astype(np.uint8) if iterations else arr.copy()
    flip = rng.random(out.shape) < flip_p
    out[flip] = 1 - out[flip]
    return out.astype(np.uint8)


def _write_like(arr, ref_img, path):
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref_img)
    sitk.WriteImage(img, str(path), useCompression=True)


def make_identity_predictions(tf_dir: Path, out_dir: Path, extra_erosion=0):
    """H2b identity-control predictions, derived from the transformer arm's own predictions so
    the two fixtures differ in exactly one way.

    extra_erosion=0 copies them: every per-case difference is exactly zero, so the stratified
    map is trivially the same map and identity_comparison.py must read "attention is not the
    cause". extra_erosion>0 erodes every prediction instead, degrading every cell, and the same
    script must read the opposite. Both branches are exercised, so a hard-wired verdict fails.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(tf_dir.glob("*.nii.gz")):
        if not extra_erosion:
            # A byte copy, not a SimpleITK round-trip: the "reproduced" fixture asserts an
            # exactly zero paired difference, and a round-trip is not guaranteed to preserve
            # data on every platform. Same reason the duplicate fixtures are file copies.
            shutil.copy(f, out_dir / f.name)
            continue
        img = sitk.ReadImage(str(f))
        arr = sitk.GetArrayFromImage(img).astype(np.uint8)
        arr = ndimage.binary_erosion(arr, iterations=extra_erosion).astype(np.uint8)
        _write_like(arr, img, out_dir / f.name)
    return out_dir


def make_loso_predictions(cohort_csv: Path, manual_dir: Path, out_dir: Path,
                          source_col="source", seed=11):
    """Leave-one-source-out predictions laid out as loso_analysis.py expects:
    <out_dir>/<arm>/loso_<Source>/<case>.nii.gz. Designed pattern: the transformer degrades
    much more under source shift than the CNN, so the H3 share-inside-floor statistic has a
    real contrast to find."""
    cohort = pd.read_csv(cohort_csv)
    manual = cohort[cohort["has_manual_lesion"]]
    rng = np.random.default_rng(seed)
    extra = {"cnn": 1, "tf": 3}      # extra erosion iterations under source shift
    for arm, it in extra.items():
        for source, g in manual.groupby(source_col):
            d = out_dir / arm / f"loso_{source}"
            d.mkdir(parents=True, exist_ok=True)
            for cid in g["case_id"]:
                seg = sitk.ReadImage(str(manual_dir / f"{cid}.nii.gz"))
                arr = (sitk.GetArrayFromImage(seg) == 1).astype(np.uint8)
                vol = arr.sum()
                base = ({"cnn": 0, "tf": 2} if vol < 300 else
                        {"cnn": 1, "tf": 1} if vol < 1500 else {"cnn": 3, "tf": 0})[arm]
                _write_like(_degrade(arr, base + it, rng), seg, d / f"{cid}.nii.gz")
    return out_dir


def make_seed_predictions(fold_assignment_csv: Path, manual_dir: Path, out_dir: Path,
                          fold=0, seeds=(1, 2), base_seed=23):
    """Seed replicates on one fold: the same designed CNN pattern, re-speckled per seed, so
    the only variation across seeds is initialization-like noise. Returns {seed: dir}."""
    fa = pd.read_csv(fold_assignment_csv)
    cases = fa.loc[fa["fold"] == fold, "case_id"].tolist()
    dirs = {}
    for s in seeds:
        d = out_dir / f"seed{s}"
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(base_seed + int(s))
        for cid in cases:
            seg = sitk.ReadImage(str(manual_dir / f"{cid}.nii.gz"))
            arr = (sitk.GetArrayFromImage(seg) == 1).astype(np.uint8)
            vol = arr.sum()
            cnn_iter = 0 if vol < 300 else (1 if vol < 1500 else 3)
            _write_like(_degrade(arr, cnn_iter, rng, flip_p=0.0008), seg, d / f"{cid}.nii.gz")
        dirs[s] = d
    return dirs
