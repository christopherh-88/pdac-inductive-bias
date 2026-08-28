#!/usr/bin/env python
"""Inter-rater proxy: does any case carry a second, independent delineation?

Phase 1 task whose "done when" is deliberately two-sided — either a list of paired cases or a
written statement that there are none. This script produces exactly that, so the answer is a
committed artifact rather than something someone remembers checking.

Why it matters for H3: the boundary-tolerance floor in boundary_tolerance.py is *synthetic*
(morphological perturbation of one reader's mask). A real second delineation of the same case
gives an empirical floor instead, and the two can be compared. If none exist, the paper has to
say the floor is synthetic — which is a limitation to state, not to discover at review time.

What counts as a second delineation: the same case_id appearing in more than one delineation
directory under the label root. PANORAMA's automatic_labels/ is model-generated, so it is NOT
an independent reader; it is only compared when --include-automatic is passed, and then it is
labelled `model_generated` in the output and excluded from the inter-rater summary.

Usage:
  python scripts/analysis/paired_delineation_check.py \
      --labels-root /data/panorama/panorama_labels --cohort splits/cohort.csv \
      --out results/inter_rater
"""
import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

from metrics import CFG, dice, load_mask

LESION = CFG["labels"]["panorama"]["pdac_lesion"]
IMG_EXT = (".mha", ".nii.gz", ".nrrd")
AUTOMATIC_DIRS = {"automatic_labels"}


def case_id_of(path: Path) -> str:
    name = path.name
    for ext in IMG_EXT:
        if name.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def index_delineations(labels_root: Path, include_automatic: bool):
    """{case_id: [(delineation_source, path), ...]} over every label directory found."""
    index = defaultdict(list)
    for d in sorted(p for p in labels_root.rglob("*") if p.is_dir()):
        if ".git" in d.parts:
            continue
        if d.name in AUTOMATIC_DIRS and not include_automatic:
            continue
        files = [p for p in d.iterdir() if p.is_file() and p.name.endswith(IMG_EXT)]
        rel = str(d.relative_to(labels_root))
        for f in files:
            index[case_id_of(f)].append((rel, f))
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", required=True, type=Path)
    ap.add_argument("--cohort", default="splits/cohort.csv", type=Path)
    ap.add_argument("--include-automatic", action="store_true",
                    help="also compare against automatic_labels/ (model-generated, NOT an "
                         "independent reader; reported separately and never as inter-rater)")
    ap.add_argument("--out", default=Path("results/inter_rater"), type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort)
    cohort_cases = set(cohort.loc[cohort["has_manual_lesion"], "case_id"].astype(str))
    index = index_delineations(args.labels_root, args.include_automatic)
    dirs_found = sorted({src for v in index.values() for src, _ in v})

    rows = []
    for cid, entries in sorted(index.items()):
        if cid not in cohort_cases or len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                (src_a, pa), (src_b, pb) = entries[i], entries[j]
                a, _ = load_mask(pa, label=LESION)
                b, _ = load_mask(pb, label=LESION)
                if not (a.any() or b.any()):
                    continue
                model_generated = (Path(src_a).name in AUTOMATIC_DIRS
                                   or Path(src_b).name in AUTOMATIC_DIRS)
                rows.append({"case_id": cid, "source_a": src_a, "source_b": src_b,
                             "dice": dice(a, b), "model_generated": model_generated})

    paired = pd.DataFrame(rows, columns=["case_id", "source_a", "source_b", "dice",
                                         "model_generated"])
    paired.to_csv(args.out / "paired_delineations.csv", index=False)
    independent = paired[~paired["model_generated"]] if len(paired) else paired

    n_ind = int(independent["case_id"].nunique()) if len(independent) else 0
    summary = {
        "labels_root": str(args.labels_root),
        "delineation_directories_scanned": dirs_found,
        "cohort_cases_checked": len(cohort_cases),
        "n_cases_with_independent_second_delineation": n_ind,
        "n_cases_with_any_second_delineation": int(paired["case_id"].nunique()) if len(paired) else 0,
    }
    if n_ind:
        summary["inter_rater_dice_mean"] = float(independent["dice"].mean())
        summary["inter_rater_dice_median"] = float(independent["dice"].median())
        summary["inter_rater_dice_min"] = float(independent["dice"].min())
        statement = (
            f"{n_ind} cohort case(s) carry a second independent delineation "
            f"({', '.join(dirs_found)}). Empirical inter-rater Dice: mean "
            f"{summary['inter_rater_dice_mean']:.4f}, median "
            f"{summary['inter_rater_dice_median']:.4f}, min {summary['inter_rater_dice_min']:.4f} "
            f"(paired_delineations.csv). The H3 boundary-tolerance floor is reported against "
            f"this empirical floor as well as the synthetic morphological one.")
    else:
        statement = (
            "No cohort case carries a second independent delineation. Directories scanned: "
            f"{', '.join(dirs_found) if dirs_found else '(none found)'}. The H3 "
            "boundary-tolerance floor is therefore SYNTHETIC ONLY — 1- and 2-voxel "
            "morphological perturbation plus seeded jitter of a single reader's mask — with no "
            "empirical inter-rater estimate available to validate it. This is a stated "
            "limitation of the study, not an omission.")
    summary["statement"] = statement
    yaml.safe_dump(summary, open(args.out / "summary.yaml", "w"), sort_keys=False)

    (args.out / "STATEMENT.md").write_text(
        "# Inter-rater proxy: paired independent delineations\n\n"
        f"Generated by `scripts/analysis/paired_delineation_check.py` against "
        f"`{args.labels_root}`.\n\n{statement}\n")

    print(statement)
    print(f"\nWrote {args.out}/ (paired_delineations.csv, summary.yaml, STATEMENT.md)")


if __name__ == "__main__":
    main()
