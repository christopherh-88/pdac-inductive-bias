"""Frozen metric implementations shared by all analysis scripts.

Definitions mirror config/analysis_config.yaml exactly:
Dice, NSD at 2 mm, HD95, detection (prediction covers >=10% of reference lesion),
false positives (zero-overlap components >= 100 mm^3).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from scipy import ndimage
from surface_distance import (compute_surface_distances,
                              compute_robust_hausdorff,
                              compute_surface_dice_at_tolerance)

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "config" / "analysis_config.yaml"))


def find_case_file(dir_, cid):
    """Resolve <dir_>/<cid>.nii or .nii.gz — nnU-Net's file_ending varies by dataset config."""
    matches = sorted(Path(dir_).glob(f"{cid}.nii*"))
    return matches[0] if matches else None


def load_mask(path, label=None):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayViewFromImage(img)
    mask = (arr == label) if label is not None else (arr > 0)
    spacing_zyx = tuple(reversed(img.GetSpacing()))
    return mask, spacing_zyx


def dice(pred: np.ndarray, ref: np.ndarray) -> float:
    inter = np.logical_and(pred, ref).sum()
    denom = pred.sum() + ref.sum()
    return 1.0 if denom == 0 else 2.0 * inter / denom


def nsd(pred: np.ndarray, ref: np.ndarray, spacing_zyx) -> float:
    if not pred.any() and not ref.any():
        return 1.0
    if not pred.any() or not ref.any():
        return 0.0
    sd = compute_surface_distances(ref, pred, spacing_zyx)
    return compute_surface_dice_at_tolerance(sd, CFG["metrics"]["nsd_tolerance_mm"])


def hd95(pred: np.ndarray, ref: np.ndarray, spacing_zyx) -> float:
    if not pred.any() or not ref.any():
        return np.nan
    sd = compute_surface_distances(ref, pred, spacing_zyx)
    return compute_robust_hausdorff(sd, 95)


def detected(pred: np.ndarray, ref: np.ndarray) -> bool:
    if not ref.any():
        return False
    cov = np.logical_and(pred, ref).sum() / ref.sum()
    return cov >= CFG["metrics"]["detection"]["min_reference_coverage"]


def false_positives(pred: np.ndarray, ref: np.ndarray, spacing_zyx) -> int:
    vox_mm3 = float(np.prod(spacing_zyx))
    labeled, n = ndimage.label(pred)
    fp = 0
    for i in range(1, n + 1):
        comp = labeled == i
        if np.logical_and(comp, ref).sum() == 0 and comp.sum() * vox_mm3 >= \
                CFG["metrics"]["false_positive"]["min_volume_mm3"]:
            fp += 1
    return fp


def bootstrap_ci(df: pd.DataFrame, stat_fn, n_boot: int, seed: int, patient_col="patient_id"):
    """Patient-level cluster bootstrap CI, shared by every script that needs one.

    Resamples whole patients (with replacement) rather than rows, so a patient's rows always
    move together. Precomputes each patient's positional row-indices once, then per resample
    concatenates small integer-index arrays and takes a single vectorized .iloc — avoids the
    O(n_boot * n_patients) boolean-mask filtering that dominates a naive per-patient filter loop.
    """
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True)
    groups = df.groupby(patient_col).indices  # {patient_id: ndarray of positional indices}
    patients = np.array(list(groups.keys()))
    out = []
    last_exc = None
    for _ in range(n_boot):
        sample = rng.choice(patients, size=len(patients), replace=True)
        pos = np.concatenate([groups[p] for p in sample])
        boot = df.iloc[pos]
        try:
            out.append(stat_fn(boot))
        except Exception as exc:
            out.append(np.nan)
            last_exc = exc
    if last_exc is not None and np.isnan(out).all():
        raise RuntimeError(
            f"stat_fn failed on every one of {n_boot} bootstrap resamples — "
            "this is a bug in stat_fn, not expected resample degeneracy"
        ) from last_exc
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def all_metrics(pred_path, ref_path):
    pred, sp = load_mask(pred_path)
    ref, ref_sp = load_mask(ref_path)
    if sp != ref_sp:
        raise ValueError(
            f"spacing mismatch: pred {pred_path} has {sp}, ref {ref_path} has {ref_sp} — "
            "distance-based metrics (NSD, HD95, false-positive volume) require a common grid"
        )
    return {
        "dice": dice(pred, ref),
        "nsd2mm": nsd(pred, ref, sp),
        "hd95": hd95(pred, ref, sp),
        "detected": detected(pred, ref),
        "fp_count": false_positives(pred, ref, sp),
    }
