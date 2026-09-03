#!/usr/bin/env python
"""Tier-A-lite pool staging, run on a Kaggle CPU-only kernel (no GPU quota used).

This is NOT Tier A training and NOT the pre-registered Tier A cohort. Tier A (5 folds x 2
arms, full 478-case cohort, matched compute budget on a dedicated 24-48GB GPU for ~3 weeks) is
still blocked on confirmed NYU/USC GPU access -- see preregistration/DEVIATIONS.md. Kaggle's
free tier (~30 GPU-hrs/week, ~9-12hr session cap) cannot supply the 10-20 real GPU-days Tier A
needs, so this is explicitly a further diagnostic step in the same spirit as
scripts/verify/phase0_gpu_verify.py: an interim, clearly-labeled substitute that produces a
larger (but still small, still not fold-averaged, still not statistically powered) real-data
preview, so *something* real trains on *more* real cases while formal GPU access is pending.

Purpose of this specific kernel: stream a larger real subset of manually-delineated PANORAMA
PDAC cases (target: N_CASES) from batch 1, build a proper nnU-Net raw dataset
(Dataset601_PDACTierALite) with a real (not smoke-tiny) train/val split for "fold 0", and save
it as kernel output so it can be pushed as a persistent Kaggle Dataset. A later GPU kernel
attaches that dataset as input (no re-download needed) and does the actual longer, real
training across multiple weekly GPU-quota resets.

Ships back: raw_images/, panorama_labels manual labels actually used, the built
nnUNet_raw/Dataset601_PDACTierALite/ tree (imagesTr, labelsTr, dataset.json, splits_final.json),
and a small provenance JSON (case ids kept, split assignment, byte sizes).
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working/phase1_pool")
OUT.mkdir(parents=True, exist_ok=True)
N_CASES = 32
VAL_FRACTION = 0.25
SPLIT_SEED = 20260823  # same split seed already frozen in config/frozen_thresholds.yaml
LABELS_REPO_ZIP_URL = "https://codeload.github.com/DIAGNijmegen/panorama_labels/zip/refs/heads/main"
BATCH1_RECORD_ID = 13715870
IMG_EXT = (".mha", ".nii.gz", ".nrrd")
LESION_LABEL = 1


def log(msg):
    print(f"[phase1-pool] {msg}", flush=True)


def pip_install(args, timeout=300, retries=3, backoff=15):
    import subprocess
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                            check=True, timeout=timeout)
            return
        except Exception as e:
            last_err = e
            if attempt < retries:
                log(f"WARNING: pip install {args} failed ({e!r}), attempt {attempt}/{retries} "
                    f"-- retrying after {backoff}s")
                time.sleep(backoff)
    raise last_err


def step0_env():
    log("=== step 0: environment (CPU-only, no GPU/torch needed for staging) ===")
    pip_install(["stream-unzip"])
    import SimpleITK  # noqa: F401  -- fail fast if unavailable
    log("SimpleITK import OK")


def step1_download_subset():
    log(f"=== step 1: download up to {N_CASES} real manual-lesion cases from batch 1 ===")
    import requests
    import stream_unzip as stream_unzip_mod
    stream_unzip = stream_unzip_mod.stream_unzip

    labels_dir = OUT / "panorama_labels"
    if not labels_dir.exists():
        zip_path = OUT / "panorama_labels.zip"
        with requests.get(LABELS_REPO_ZIP_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        import zipfile
        extract_tmp = OUT / "_labels_extract"
        shutil.rmtree(extract_tmp, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_tmp)
        zip_path.unlink()
        inner = next(p for p in extract_tmp.iterdir() if p.is_dir())
        inner.rename(labels_dir)
        shutil.rmtree(extract_tmp, ignore_errors=True)
    manual_dir = labels_dir / "manual_labels"
    log(f"panorama_labels ready, {len(list(manual_dir.glob('*')))} manual labels available")

    for attempt in range(1, 4):
        try:
            r = requests.get(f"https://zenodo.org/api/records/{BATCH1_RECORD_ID}", timeout=60)
            r.raise_for_status()
            break
        except (requests.exceptions.RequestException, OSError) as e:
            if attempt == 3:
                raise
            log(f"WARNING: Zenodo metadata fetch failed ({e!r}), attempt {attempt}/3 -- retrying after 20s")
            time.sleep(20)
    files = r.json()["files"]
    zip_meta = next(f for f in files if f["key"].endswith(".zip"))
    url = zip_meta["links"]["self"]

    images_dir = OUT / "raw_images"
    images_dir.mkdir(exist_ok=True)
    kept = []

    def http_chunks():
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            yield from resp.iter_content(chunk_size=1 << 20)

    kept_ids = set()
    MAX_RETRIES = 40
    attempt = 0
    while len(kept) < N_CASES:
        attempt += 1
        try:
            for name, _size, unzipped_chunks in stream_unzip(http_chunks()):
                name = name.decode() if isinstance(name, bytes) else name
                if not name.endswith(IMG_EXT):
                    for _ in unzipped_chunks:
                        pass
                    continue
                cid = Path(name).name
                for ext in IMG_EXT:
                    if cid.endswith(ext):
                        cid = cid[: -len(ext)]
                        break
                bare_cid = cid[:-5] if cid.endswith("_0000") else cid
                if bare_cid in kept_ids:
                    for _ in unzipped_chunks:
                        pass
                    continue
                label_path = None
                for ext in IMG_EXT:
                    cand = manual_dir / f"{bare_cid}{ext}"
                    if cand.exists():
                        label_path = cand
                        break
                if label_path is None:
                    for _ in unzipped_chunks:
                        pass
                    continue  # only want manually-labeled cases for this pool
                dst = images_dir / f"{bare_cid}_0000.nii.gz"
                with open(dst, "wb") as f:
                    for chunk in unzipped_chunks:
                        f.write(chunk)
                kept.append((bare_cid, dst, label_path))
                kept_ids.add(bare_cid)
                log(f"  kept {bare_cid} ({len(kept)}/{N_CASES})")
                if len(kept) >= N_CASES:
                    break
            break
        except (requests.exceptions.RequestException, OSError, EOFError, IndexError,
                stream_unzip_mod.DataError, stream_unzip_mod.UncompressError) as e:
            if attempt >= MAX_RETRIES or len(kept) >= N_CASES:
                raise
            log(f"WARNING: download connection broke ({e!r}) after {len(kept)}/{N_CASES} cases "
                f"-- retrying whole batch from the start (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(15)

    if len(kept) < N_CASES:
        log(f"WARNING: only found {len(kept)}/{N_CASES} manual-lesion cases before batch ended")
    if len(kept) < 8:
        raise RuntimeError(f"only {len(kept)} usable cases -- too few for a meaningful lite pool")
    return kept


def write_binary_lesion_label(src, dst, img):
    """Verbatim logic from scripts/data/convert_to_nnunet.py::write_binary_lesion_label,
    inlined here so this kernel stays self-contained (no repo checkout on Kaggle)."""
    import SimpleITK as sitk
    import numpy as np
    seg = sitk.ReadImage(str(src))
    if (seg.GetSize() != img.GetSize() or seg.GetSpacing() != img.GetSpacing()
            or seg.GetDirection() != img.GetDirection() or seg.GetOrigin() != img.GetOrigin()):
        seg = sitk.Resample(seg, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, seg.GetPixelID())
    arr = (sitk.GetArrayFromImage(seg) == LESION_LABEL).astype(np.uint8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(dst), useCompression=True)


def step2_build_dataset(kept):
    log("=== step 2: build nnU-Net raw dataset (Dataset601_PDACTierALite) ===")
    import SimpleITK as sitk
    import random

    nnunet_raw = OUT / "nnUNet_raw"
    ds_dir = nnunet_raw / "Dataset601_PDACTierALite"
    (ds_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    case_ids = []
    for cid, img_path, label_path in kept:
        img_dst = ds_dir / "imagesTr" / f"{cid}_0000.nii.gz"
        lab_dst = ds_dir / "labelsTr" / f"{cid}.nii.gz"
        shutil.copy(img_path, img_dst)
        img = sitk.ReadImage(str(img_dst))
        write_binary_lesion_label(label_path, lab_dst, img)
        case_ids.append(cid)

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "pdac_lesion": 1},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
        "licence": "images CC BY-NC 4.0 (PANORAMA); this arrangement MIT",
        "description": "Tier-A-lite diagnostic pool -- NOT the pre-registered Tier A cohort "
                        "(478 manual-lesion cases). A larger-than-smoke-test but still small "
                        "real subset used only because confirmed NYU/USC GPU access is still "
                        "pending; see preregistration/DEVIATIONS.md.",
    }
    json.dump(dataset_json, open(ds_dir / "dataset.json", "w"), indent=2)

    rng = random.Random(SPLIT_SEED)
    shuffled = case_ids[:]
    rng.shuffle(shuffled)
    n_val = max(2, round(len(shuffled) * VAL_FRACTION))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    splits_final = [{"train": train_ids, "val": val_ids}]
    json.dump(splits_final, open(OUT / "splits_final.json", "w"), indent=2)
    log(f"fold-0 split: {len(train_ids)} train / {len(val_ids)} val (seed {SPLIT_SEED})")

    provenance = {
        "n_cases": len(case_ids),
        "case_ids": case_ids,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "split_seed": SPLIT_SEED,
        "val_fraction": VAL_FRACTION,
        "note": "Tier-A-lite diagnostic pool. NOT Tier A. See preregistration/DEVIATIONS.md.",
    }
    json.dump(provenance, open(OUT / "provenance.json", "w"), indent=2)
    return ds_dir


def main():
    step0_env()
    kept = step1_download_subset()
    ds_dir = step2_build_dataset(kept)
    log(f"DONE. Dataset staged at {ds_dir}, {len(kept)} cases.")
    # Drop the raw pre-conversion copies to keep kernel output small; the built dataset
    # under nnUNet_raw/ is what a later training kernel actually needs.
    shutil.rmtree(OUT / "raw_images", ignore_errors=True)
    shutil.rmtree(OUT / "panorama_labels", ignore_errors=True)


if __name__ == "__main__":
    main()
