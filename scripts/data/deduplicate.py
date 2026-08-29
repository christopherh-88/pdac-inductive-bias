#!/usr/bin/env python
"""Build the cohort table and remove duplicate scans across PANORAMA / MSD / NIH.

MSD Task07 and NIH Pancreas-CT are redistributed *inside* PANORAMA, and AbdomenCT-1K-style
leakage is exactly what the pre-registration excludes, so duplicates must be identified by
image content, not by case ID.

Two-stage duplicate detection:
  1. metadata fingerprint (voxel counts + spacing) — cheap, groups candidates
  2. full-array SHA-256 within candidate groups — decisive

Stage 1 is deliberately coarse. It used to include origin and direction, which made it
*formatting*-sensitive: SimpleITK returns direction cosines containing -0.0, `repr(-0.0)` is
not `repr(0.0)`, and which of the two you get for a given axis varies with the writer and the
library version. A scan and its redistributed copy could therefore land in different candidate
groups, never be content-compared, and survive as a duplicate. That is data leakage across
folds — exactly what the pre-registration excludes — traded for saving a few hashes.

The asymmetry decides the design: over-grouping costs time, under-grouping costs correctness.
So stage 1 keeps only what is exact (integer voxel counts) and near-exact (spacing, rounded
and canonically formatted), and stage 2 decides.

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

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

IMG_EXT = (".mha", ".nii.gz", ".nrrd")
DUPLICATE_COLUMNS = ["removed", "removed_path", "duplicate_of", "rule"]


def find_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.name.endswith(IMG_EXT))


def _canon(value: float, decimals: int) -> str:
    """Format a float so that equal numbers always give equal strings.

    `repr()` does not: -0.0 and 0.0 are numerically equal but print differently, and header
    round-trips flip the sign of zero components freely. Adding 0.0 normalizes -0.0 to 0.0,
    and fixed-point formatting removes every other repr difference.
    """
    return f"{round(float(value), decimals) + 0.0:.{decimals}f}"


def meta_fingerprint(path: Path) -> str:
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    parts = (tuple(int(v) for v in r.GetSize()),
             tuple(_canon(s, 4) for s in r.GetSpacing()))
    return hashlib.sha256(repr(parts).encode()).hexdigest()


def content_hash(path: Path) -> str:
    """Decisive comparison: the voxel VALUES, not the bytes that happen to encode them.

    Hashing raw bytes makes the answer depend on storage dtype, and a redistributed scan is
    frequently re-encoded on the way — int16 to float32, or promoted by a reader on one
    platform and not another. Those files hold the same scan; calling them distinct is exactly
    the miss that lets a duplicate through into a different fold.

    So values are canonicalized to float64 before hashing. That is exact for every dtype CT
    data realistically uses (int16, uint16, int32, float32), and it makes the hash a property
    of the image rather than of the writer. Hashing proceeds in chunks so a large volume costs
    bounded memory rather than 8 bytes per voxel all at once.
    """
    arr = sitk.GetArrayViewFromImage(sitk.ReadImage(str(path)))
    flat = np.ascontiguousarray(arr).reshape(-1)
    h = hashlib.sha256()
    h.update(repr(arr.shape).encode())          # same values, different shape => not the same
    for start in range(0, flat.size, 8_000_000):
        h.update(flat[start:start + 8_000_000].astype(np.float64, copy=False).tobytes())
    return h.hexdigest()


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

    # Stage 2: content hash only where the metadata already matches
    dupes = []
    keep = set(df.index)
    n_groups = n_hashed = 0
    for _, group in df.groupby("meta_fp"):
        if len(group) < 2:
            continue
        n_groups += 1
        hashes = {}
        for idx in group.index:
            n_hashed += 1
            h = content_hash(Path(df.loc[idx, "path"]))
            if h in hashes:
                dupes.append({"removed": df.loc[idx, "case_id"], "removed_path": df.loc[idx, "path"],
                              "duplicate_of": df.loc[hashes[h], "case_id"],
                              "rule": "identical voxel data and dtype within a "
                                      "size+spacing candidate group"})
                keep.discard(idx)
            else:
                hashes[h] = idx
    singletons = int((df.groupby("meta_fp")["path"].transform("size") == 1).sum())
    df = df.loc[sorted(keep)].drop(columns="meta_fp")

    # Stage 1 is a filter, so its selectivity is worth stating: if it grouped almost nothing,
    # stage 2 never got the chance to decide and "no duplicates" means "nothing was compared".
    print(f"Stage 1: {n_groups} candidate group(s) covering {n_hashed} scans; "
          f"{singletons} scans had a unique size+spacing and were not compared")
    print(f"Removed {len(dupes)} exact duplicates; {len(df)} scans retained")
    if n_hashed == 0:
        print("WARNING: no two scans shared a size and spacing, so no content comparison "
              "happened at all. On a cohort that redistributes MSD Task07 and NIH "
              "Pancreas-CT inside PANORAMA that is not plausible — check the inputs before "
              "trusting a zero-duplicate result.")

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
    # Always with a header, even when nothing was removed: a zero-duplicate cohort is a
    # perfectly good outcome, and pandas cannot read back a file with no columns at all, so
    # writing a bare empty file turns a clean result into a crash for every reader downstream.
    pd.DataFrame(dupes, columns=DUPLICATE_COLUMNS).to_csv(
        args.out.parent / "duplicates.csv", index=False)
    print(f"Wrote {args.out} ({len(df)} rows, {df['has_manual_lesion'].sum()} with manual lesion labels)")
    print("NEXT: verify source / diagnosis columns, then run compute_strata.py")


if __name__ == "__main__":
    main()
