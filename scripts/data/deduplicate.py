#!/usr/bin/env python
"""Build the cohort table and remove duplicate scans across PANORAMA / MSD / NIH.

MSD Task07 and NIH Pancreas-CT are redistributed *inside* PANORAMA, and AbdomenCT-1K-style
leakage is exactly what the pre-registration excludes, so duplicates must be identified by
image content, not by case ID.

Two-stage duplicate detection:
  1. metadata fingerprint (voxel counts, spacing, origin, direction) — cheap, groups candidates
  2. full-array SHA-256 within candidate groups — decisive

Outputs
  splits/cohort.csv       one row per retained case: case_id, patient_id, path, source,
                          is_pdac, has_manual_lesion, label paths
  splits/duplicates.csv   every removed scan and the scan it duplicates

Usage:
  python scripts/data/deduplicate.py --data-root /path/to/data --out splits/cohort.csv
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

IMG_EXT = (".mha", ".nii.gz", ".nrrd")


def find_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.name.endswith(IMG_EXT))


def meta_fingerprint(path: Path) -> str:
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    parts = (
        tuple(r.GetSize()),
        tuple(round(s, 4) for s in r.GetSpacing()),
        tuple(round(o, 2) for o in r.GetOrigin()),
        tuple(round(d, 4) for d in r.GetDirection()),
    )
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def content_hash(path: Path) -> str:
    arr = sitk.GetArrayViewFromImage(sitk.ReadImage(str(path)))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def case_id_from_path(path: Path) -> str:
    name = path.name
    for ext in IMG_EXT:
        if name.endswith(ext):
            name = name[: -len(ext)]
    # PANORAMA convention: <patient>_<study>[_0000]; strip a trailing channel suffix
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", default="splits/cohort.csv", type=Path)
    args = ap.parse_args()

    img_root = args.data_root / "panorama" / "images"
    labels_root = args.data_root / "panorama" / "panorama_labels"
    images = find_images(img_root)
    if not images:
        raise SystemExit(f"No images found under {img_root} — run download_panorama.sh first")

    print(f"Found {len(images)} images; computing metadata fingerprints ...")
    rows = []
    for p in tqdm(images):
        rows.append({"path": str(p), "case_id": case_id_from_path(p), "meta_fp": meta_fingerprint(p)})
    df = pd.DataFrame(rows)

    # Stage 2: content hash only where metadata collides
    dupes = []
    keep = set(df.index)
    for _, group in df.groupby("meta_fp"):
        if len(group) < 2:
            continue
        hashes = {}
        for idx in group.index:
            h = content_hash(Path(df.loc[idx, "path"]))
            if h in hashes:
                dupes.append({"removed": df.loc[idx, "case_id"], "removed_path": df.loc[idx, "path"],
                              "duplicate_of": df.loc[hashes[h], "case_id"]})
                keep.discard(idx)
            else:
                hashes[h] = idx
    df = df.loc[sorted(keep)].drop(columns="meta_fp")
    print(f"Removed {len(dupes)} exact duplicates; {len(df)} scans retained")

    # Merge clinical metadata (source, diagnosis). Column names are verified at freeze time;
    # fail loudly rather than guess silently.
    clin_path = labels_root / "clinical_information.xlsx"
    if clin_path.exists():
        clin = pd.read_excel(clin_path)
        clin.columns = [c.strip().lower().replace(" ", "_") for c in clin.columns]
        print(f"clinical_information.xlsx columns: {list(clin.columns)}")
        id_cols = [c for c in clin.columns if "case" in c or "patient" in c or "id" in c]
        if not id_cols:
            raise SystemExit("Could not find a case-id column in clinical_information.xlsx — inspect and adjust")
        clin = clin.rename(columns={id_cols[0]: "case_id"})
        clin["case_id"] = clin["case_id"].astype(str)
        df = df.merge(clin, on="case_id", how="left")
    else:
        print(f"WARNING: {clin_path} not found — cohort will lack source/diagnosis columns")

    # Attach label paths
    def label_path(sub: str, cid: str):
        for ext in IMG_EXT:
            cand = labels_root / sub / f"{cid}{ext}"
            if cand.exists():
                return str(cand)
        return ""

    df["manual_label"] = [label_path("manual_labels", c) for c in df["case_id"]]
    df["automatic_label"] = [label_path("automatic_labels", c) for c in df["case_id"]]
    df["has_manual_lesion"] = df["manual_label"] != ""
    df["patient_id"] = df["case_id"].str.split("_").str[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.DataFrame(dupes).to_csv(args.out.parent / "duplicates.csv", index=False)
    print(f"Wrote {args.out} ({len(df)} rows, {df['has_manual_lesion'].sum()} with manual lesion labels)")
    print("NEXT: verify source / diagnosis columns, then run compute_strata.py")


if __name__ == "__main__":
    main()
