#!/usr/bin/env python
"""Extract whole patients from a PANORAMA batch zip, without unpacking the whole thing.

A Zenodo batch is one ~50 GB zip (batch 1 is 49.3 GB), and the images inside are already
compressed, so unpacking one costs about as much again. On a machine with tens of GB rather
than hundreds — a Kaggle session, say — downloading a batch and extracting all of it does not
fit, whatever the time budget says. The download cannot be made selective (a zip is fetched
whole), but the extraction can, which bounds the peak at one zip plus the subset.

Selection is by *patient*, never by file: PANORAMA names entries
`<patient>_<study>_<channel>.<ext>`, and a patient whose scans were split across a subset
boundary would break the patient-level splits the whole study depends on. Patients are taken
in sorted order so the subset is deterministic and re-running extends rather than reshuffles.

  python scripts/download/extract_subset.py --zip batch_1.zip --dest images/ --max-cases 60
  python scripts/download/extract_subset.py --zip batch_1.zip --dest images/   # everything

Prints the patients it took, so a subset cohort is traceable to the command that made it.
"""
import argparse
import subprocess
import zipfile
from pathlib import Path

IMG_EXT = (".nii.gz", ".mha", ".nrrd")


def patient_of(entry: str) -> str:
    return Path(entry).name.split("_")[0]


def list_images(zip_path: Path):
    with zipfile.ZipFile(zip_path) as z:
        return [n for n in z.namelist() if n.endswith(IMG_EXT)]


def select_entries(zip_path: Path, max_cases=None):
    """(entries_to_extract, patients). entries is None when everything is wanted."""
    images = list_images(zip_path)
    patients = sorted({patient_of(e) for e in images})
    if max_cases is None or max_cases >= len(patients):
        return None, patients
    keep = set(patients[:max_cases])
    return [e for e in images if patient_of(e) in keep], sorted(keep)


def extract(zip_path: Path, dest: Path, max_cases=None, chunk=40):
    dest.mkdir(parents=True, exist_ok=True)
    entries, patients = select_entries(zip_path, max_cases)
    if entries is None:
        print(f"{zip_path.name}: extracting all {len(patients)} patients")
        subprocess.run(["unzip", "-n", "-q", str(zip_path), "-d", str(dest)], check=True)
        return patients

    print(f"{zip_path.name}: extracting {len(patients)} of "
          f"{len(set(patient_of(e) for e in list_images(zip_path)))} patients "
          f"({len(entries)} files)")
    # Chunked so the argument list stays well inside any shell limit, and -n so a re-run
    # after an interrupted extraction resumes instead of redoing the work.
    for i in range(0, len(entries), chunk):
        subprocess.run(["unzip", "-n", "-q", str(zip_path), *entries[i:i + chunk],
                        "-d", str(dest)], check=True)
    return patients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--max-cases", type=int, default=None,
                    help="number of patients to extract; omit for all of them")
    ap.add_argument("--chunk", type=int, default=40)
    args = ap.parse_args()

    patients = extract(args.zip, args.dest, args.max_cases, args.chunk)
    n_files = sum(1 for p in args.dest.rglob("*") if p.name.endswith(IMG_EXT))
    print(f"{len(patients)} patients, {n_files} image files now under {args.dest}")
    print("patients: " + ", ".join(patients[:10]) + (" ..." if len(patients) > 10 else ""))


if __name__ == "__main__":
    main()
