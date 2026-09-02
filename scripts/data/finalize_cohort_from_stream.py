#!/usr/bin/env python
"""Turn the streamed Kaggle output (case_records.csv, clinical_information.xlsx) into
splits/cohort.csv + splits/duplicates.csv — the same schema scripts/data/deduplicate.py
produces from live files, but built from precomputed per-case hashes instead.

This is the "finalize" half of the disk-constrained cohort pipeline in
scripts/download/kaggle_stream_cohort.py: that script streams raw CT through a
disk-constrained environment and discards it after recording a metadata fingerprint and a
full-array content hash per case; this script does the actual dedup decision (content hash
collision -> drop) and the clinical-metadata merge, entirely from those small CSVs.

`path` in the output cohort.csv is NOT a currently-existing file — it is
"<data-root>/panorama/images/<member_path>", i.e. where that case will land once the real
Zenodo batches are (re-)extracted into --data-root, exactly matching
scripts/download/download_panorama.sh's layout. Anything that reads cohort.csv["path"]
(convert_to_nnunet.py, compute_strata.py) needs the real images physically present at
--data-root before it can run — this script only settles which cases exist and are unique.

Usage:
  python scripts/data/finalize_cohort_from_stream.py \\
      --records /path/to/case_records.csv --clinical /path/to/clinical_information.xlsx \\
      --data-root /path/to/data --out splits/cohort.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--clinical", type=Path)
    ap.add_argument("--data-root", required=True, type=Path,
                     help="where the real PANORAMA images will live (need not exist yet)")
    ap.add_argument("--out", default=Path("splits/cohort.csv"), type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.records)
    n_total = len(df)

    # Cross-batch dedup by content hash: keep the first occurrence (by batch, then case_id).
    df = df.sort_values(["batch", "case_id"]).reset_index(drop=True)
    dupe_mask = df.duplicated(subset="content_hash", keep="first")
    dupes = df[dupe_mask].copy()
    first_of = df[~dupe_mask].set_index("content_hash")["case_id"]
    dupes["duplicate_of"] = dupes["content_hash"].map(first_of)
    df = df[~dupe_mask].copy()
    print(f"Removed {dupe_mask.sum()} exact duplicates out of {n_total}; {len(df)} scans retained")

    images_root = args.data_root / "panorama" / "images"
    labels_root = args.data_root / "panorama" / "panorama_labels"
    df["path"] = df["member_path"].apply(lambda m: str(images_root / Path(m).name))
    df["manual_label"] = df.apply(
        lambda r: str(labels_root / "manual_labels" / f"{r.case_id}.nii.gz") if r.has_manual_label else "", axis=1)
    df["automatic_label"] = df.apply(
        lambda r: str(labels_root / "automatic_labels" / f"{r.case_id}.nii.gz") if r.has_automatic_label else "", axis=1)
    df["has_manual_lesion"] = df["has_manual_label"]
    df["patient_id"] = df["case_id"].str.split("_").str[0]
    df = df.drop(columns=["member_path", "meta_fp", "content_hash", "has_manual_label", "has_automatic_label", "batch"])

    if args.clinical and args.clinical.exists():
        clin = pd.read_excel(args.clinical)
        clin.columns = [c.strip().lower().replace(" ", "_") for c in clin.columns]
        print(f"clinical_information.xlsx columns: {list(clin.columns)}")
        # case_id everywhere in this repo is the STUDY id ("100000_00001"); prefer a study_id
        # column explicitly over a naive substring match, which would pick the bare patient id
        # first and silently no-op every merge. See deduplicate.py for the verified schema.
        study_id_cols = [c for c in clin.columns if "study_id" in c]
        case_id_cols = [c for c in clin.columns if "case_id" in c or c == "caseid"]
        id_cols = study_id_cols or case_id_cols or [c for c in clin.columns if "case" in c or "patient" in c or "id" in c]
        if not id_cols:
            raise SystemExit("Could not find a case-id column in clinical_information.xlsx — inspect and adjust")
        print(f"Using '{id_cols[0]}' as the case-id join key")
        clin = clin.rename(columns={id_cols[0]: "case_id"})
        clin["case_id"] = clin["case_id"].astype(str)
        df = df.merge(clin, on="case_id", how="left")
        if df["case_id"].isin(clin["case_id"]).mean() < 0.99:
            raise SystemExit(f"Fewer than 99% of cohort case_ids matched '{id_cols[0]}' in "
                             "clinical_information.xlsx — wrong join key or a real data-quality gap; inspect before proceeding")
        if "label" in df.columns:
            df["is_pdac"] = df["label"].astype(str).str.strip().str.upper() == "PDAC"
        if "level" in df.columns:
            df["source"] = df["level"].map({"MSD_dataset": "MSKCC", "NIH_dataset": "NIH"}).fillna("PANORAMA")
    else:
        print("WARNING: no clinical_information.xlsx given — cohort will lack source/diagnosis columns")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    dupes.to_csv(args.out.parent / "duplicates.csv", index=False)
    print(f"Wrote {args.out} ({len(df)} rows, {int(df['has_manual_lesion'].sum())} with manual lesion labels)")
    print(f"NOTE: 'path'/'manual_label'/'automatic_label' point at files under {args.data_root} "
          "that must still be (re-)downloaded before compute_strata.py / convert_to_nnunet.py can run — "
          "the lesion-derived strata numbers themselves were already computed on Kaggle; "
          "see finalize_strata_from_stream.py.")


if __name__ == "__main__":
    main()
