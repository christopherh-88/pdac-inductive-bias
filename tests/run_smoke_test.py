#!/usr/bin/env python
"""End-to-end smoke test of the whole pipeline against synthetic data.

Not a study run -- no real data, no GPU, no nnU-Net training. This exercises every script in
scripts/data and scripts/analysis against a small synthetic PANORAMA-like dataset (see
synthetic_data.py) so regressions are caught by running the pipeline, not just by reading it.
Predictions are built with a known, designed pattern (CNN better on small lesions, worse on
large; transformer degrades more in the far occlusion shell) so the checks below assert on the
actual statistical outcome, not merely "the script exited zero".

Runs entirely inside a temp directory copy of scripts/src/config, so it never touches (or
pollutes) the real repo's config/frozen_thresholds.yaml or splits/.

Usage:
  python tests/run_smoke_test.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import synthetic_data as sd

failures = []
passed = []


def check(name, condition, detail=""):
    if condition:
        passed.append(name)
        print(f"  PASS  {name}")
    else:
        failures.append(f"{name}  {detail}")
        print(f"  FAIL  {name}  {detail}")


def run(args, cwd, **kw):
    r = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
    return r


def main():
    # --keep leaves the working directory in place (figures, tables, memo) so the outputs can
    # be inspected by eye, which is the one check an assertion cannot make.
    keep = "--keep" in sys.argv
    tmp = Path(tempfile.mkdtemp(prefix="pdac_smoke_"))
    repo = tmp / "repo"
    data_root = tmp / "data"
    for d in ("scripts", "src", "config"):
        shutil.copytree(REPO / d, repo / d)
    # The real repo's config/frozen_thresholds.yaml now holds real, frozen PANORAMA strata/
    # splits (frozen 2026-09-01). Copied verbatim, compute_strata.py/make_splits.py would
    # correctly refuse to overwrite it -- but this test needs a clean slate to freeze fresh
    # synthetic-data thresholds into, matching the "never touches the real frozen_thresholds"
    # promise above.
    (repo / "config" / "frozen_thresholds.yaml").unlink(missing_ok=True)
    (repo / "splits").mkdir()
    print(f"Working dir: {tmp}")

    try:
        # 1. Synthetic data + full data pipeline -----------------------------------------
        print("\n== data generation ==")
        sd.generate(data_root, n_patients=30)
        check("synthetic images written", (data_root / "panorama" / "images").glob("*_0000.nii.gz").__next__() is not None)

        # Verify the fixture is what it claims to be, BEFORE any code under test looks at it.
        # A previous version built its "duplicates" by round-tripping an image through
        # SimpleITK, assuming that preserves voxel data. On some platforms it does not — so
        # the duplicates were not duplicates, and dedup got blamed for correctly declining to
        # merge two different images. A fixture that depends on library behaviour varying by
        # platform tests the platform, not the code.
        import SimpleITK as _sitk
        _img = data_root / "panorama" / "images"
        _src = _img / "100000_00001_0000.nii.gz"
        check("fixture: the plain duplicate is byte-identical to its source",
              (_img / "999001_00001_0000.nii.gz").read_bytes() == _src.read_bytes())
        _a = _sitk.GetArrayFromImage(_sitk.ReadImage(str(_src)))
        _b = _sitk.GetArrayFromImage(_sitk.ReadImage(str(_img / "999002_00001_0000.nii.gz")))
        check("fixture: the re-encoded duplicate holds identical values at a different dtype",
              _a.shape == _b.shape and np.array_equal(_a.astype(np.float64),
                                                      _b.astype(np.float64))
              and _a.dtype != _b.dtype,
              f"{_a.dtype} vs {_b.dtype}, equal={np.array_equal(_a.astype(np.float64), _b.astype(np.float64))}")

        print("\n== deduplicate.py ==")
        r = run(["scripts/data/deduplicate.py", "--data-root", str(data_root),
                "--out", "splits/cohort.csv"], cwd=repo)
        check("deduplicate.py exits 0", r.returncode == 0, r.stderr[-500:])
        cohort = pd.read_csv(repo / "splits" / "cohort.csv")
        dupes_csv = pd.read_csv(repo / "splits" / "duplicates.csv")
        check("dedup drops exactly the 2 injected duplicates", len(cohort) == 34,
              f"got {len(cohort)} rows{_dedup_diagnosis(repo, data_root)}")
        check("both duplicates are named in duplicates.csv with the rule that removed them",
              len(dupes_csv) == 2 and "rule" in dupes_csv.columns,
              f"got {len(dupes_csv)} rows: {list(dupes_csv.columns)}")
        check("both the byte-identical copy and the re-encoded copy are caught "
              "(a redistributed scan is often stored at a different dtype, and a byte-level "
              "hash calls that a different image)",
              set(dupes_csv["removed"]) == {"999001_00001", "999002_00001"},
              f"removed: {sorted(dupes_csv['removed'])}")

        # The same property, asserted directly rather than through the filesystem, so it holds
        # on every platform regardless of what the local SimpleITK writes into a header.
        sys.path.insert(0, str(repo / "scripts" / "data"))
        import importlib
        import deduplicate as dedup_mod
        importlib.reload(dedup_mod)
        check("equal numbers give equal fingerprint text, including -0.0 and 0.0",
              dedup_mod._canon(-0.0, 4) == dedup_mod._canon(0.0, 4)
              and dedup_mod._canon(1.5, 4) == dedup_mod._canon(1.5000001, 4)
              and dedup_mod._canon(1.5, 4) != dedup_mod._canon(1.6, 4),
              f"{dedup_mod._canon(-0.0, 4)!r} vs {dedup_mod._canon(0.0, 4)!r}")
        check("30 manual-lesion cases retained", cohort["has_manual_lesion"].sum() == 30,
              f"got {cohort['has_manual_lesion'].sum()}")
        check("4 NIH negatives retained with no manual label",
              (cohort.loc[cohort['source'] == 'NIH', 'has_manual_lesion'] == False).all())

        print("\n== compute_strata.py ==")
        r = run(["scripts/analysis/compute_strata.py", "--data-root", str(data_root),
                "--cohort", "splits/cohort.csv", "--out", "splits/strata.csv"], cwd=repo)
        check("compute_strata.py exits 0", r.returncode == 0, r.stderr[-500:])
        strata = pd.read_csv(repo / "splits" / "strata.csv")
        expected_vol = {109.5, 877.5, 4165.5}  # voxel counts (73/585/2777) * 1.5mm3 spacing
        check("lesion volumes match known synthetic ground truth",
              set(np.round(strata["volume_mm3"], 1).unique()) == expected_vol,
              f"got {sorted(strata['volume_mm3'].unique())}")
        check("volume tertiles assigned 0/1/2 in equal thirds",
              (strata["volume_tertile"].value_counts() == 10).all())
        # The distance map is computed once and thresholded per ring width. The failure mode
        # of that optimization is computing one ring three times, which would look perfectly
        # healthy in every downstream table.
        cnr_cols = [f"cnr_ring{w}mm" for w in (5, 10, 15)]
        check("all three ring widths are present and finite", 
              all(c in strata.columns for c in cnr_cols)
              and strata[cnr_cols].notna().all().all(), str(list(strata.columns)))
        differing = (strata[cnr_cols].nunique(axis=1) > 1).sum()
        check("the three ring widths give genuinely different CNRs (one distance map, three "
              "thresholds — not the same ring computed three times)",
              differing >= len(strata) * 0.5,
              f"only {differing}/{len(strata)} cases differ across ring widths")
        frozen = yaml.safe_load(open(repo / "config" / "frozen_thresholds.yaml"))
        check("frozen_thresholds.yaml strata section written", "strata" in frozen)

        print("\n== make_splits.py ==")
        r = run(["scripts/data/make_splits.py", "--cohort", "splits/cohort.csv",
                "--strata", "splits/strata.csv"], cwd=repo)
        check("make_splits.py exits 0", r.returncode == 0, r.stderr[-500:])
        splits = json.load(open(repo / "splits" / "splits_final.json"))
        check("5 folds written", len(splits) == 5)
        val_union = set()
        overlap = False
        for f in splits:
            if set(f["train"]) & set(f["val"]):
                overlap = True
            val_union |= set(f["val"])
        check("no train/val overlap in any fold", not overlap)
        check("val-fold union covers exactly the manual-lesion cohort",
              val_union == set(cohort.loc[cohort["has_manual_lesion"], "case_id"]))
        frozen = yaml.safe_load(open(repo / "config" / "frozen_thresholds.yaml"))
        check("frozen_thresholds.yaml splits section written", "splits" in frozen)
        r2 = run(["scripts/data/make_splits.py", "--cohort", "splits/cohort.csv",
                 "--strata", "splits/strata.csv"], cwd=repo)
        check("make_splits.py refuses to re-run once frozen", r2.returncode != 0)

        print("\n== record_arch_stats.py ==")
        r0 = run(["scripts/training/record_arch_stats.py", "--patch-size", "97", "160", "160",
                 "--nnunet-commit", "x", "--gpu", "x", "--cnn-plans", "x",
                 "--primus-trainer", "x"], cwd=repo)
        frozen_before = yaml.safe_load(open(repo / "config" / "frozen_thresholds.yaml"))
        check("record_arch_stats.py rejects patch size not divisible by tokenizer stride",
              r0.returncode != 0 and "architecture" not in frozen_before)
        r = run(["scripts/training/record_arch_stats.py", "--patch-size", "96", "160", "160",
                "--nnunet-commit", "deadbeef", "--gpu", "A100-40GB",
                "--cnn-plans", "nnUNetResEncUNetLPlans", "--cnn-params", "102000000",
                "--cnn-vram-gb", "22.4",
                "--primus-trainer", "nnUNet_PrimusV2M_Trainer", "--tf-params", "98000000",
                "--tf-vram-gb", "23.1"], cwd=repo)
        check("record_arch_stats.py exits 0", r.returncode == 0, r.stderr[-500:])
        frozen = yaml.safe_load(open(repo / "config" / "frozen_thresholds.yaml"))
        arch = frozen.get("architecture", {})
        check("frozen_thresholds.yaml architecture section written", "architecture" in frozen)
        check("token grid computed correctly (96/8, 160/8, 160/8)",
              arch.get("transformer", {}).get("token_grid_zyx") == [12, 20, 20])
        check("n_tokens computed correctly", arch.get("transformer", {}).get("n_tokens") == 4800)
        r2 = run(["scripts/training/record_arch_stats.py", "--patch-size", "96", "160", "160",
                 "--nnunet-commit", "x", "--gpu", "x", "--cnn-plans", "x",
                 "--primus-trainer", "x"], cwd=repo)
        check("record_arch_stats.py refuses to re-run once frozen", r2.returncode != 0)

        # 2. Predictions + H1 paired analysis --------------------------------------------
        print("\n== synthetic predictions (designed pattern) ==")
        manual_dir = data_root / "panorama" / "panorama_labels" / "manual_labels"
        refs_dir, cnn_dir, tf_dir = sd.make_predictions(
            repo / "splits" / "cohort.csv", manual_dir, tmp / "preds")

        print("\n== paired_analysis.py (H1) ==")
        r = run(["scripts/analysis/paired_analysis.py",
                "--pred-cnn", str(cnn_dir), "--pred-tf", str(tf_dir), "--refs", str(refs_dir),
                "--strata", "splits/strata.csv", "--cohort", "splits/cohort.csv",
                "--out", str(tmp / "results" / "h1")], cwd=repo)
        check("paired_analysis.py exits 0", r.returncode == 0, r.stderr[-500:])
        h1 = yaml.safe_load(open(tmp / "results" / "h1" / "summary.yaml"))
        check("H1 recovers the designed negative slope (CNN advantage shrinks with volume)",
              h1["h1_slope"] < -0.05, f"slope={h1['h1_slope']}")
        check("H1 CI excludes zero (SUPPORTED)", h1["h1_supported"] is True)
        check("large-tertile gap correctly NOT shown equivalent (designed TF win is large)",
              h1["large_tertile_equivalent"] is False)
        per_case = pd.read_csv(tmp / "results" / "h1" / "per_case.csv")
        check("per_case.csv has all 30 cases, no missing predictions", len(per_case) == 30,
              f"got {len(per_case)}")

        # 3. Occlusion test (regression test for the case-ID mangling bug) ---------------
        print("\n== occlusion_test.py (occlude stage) ==")
        occ_dir = tmp / "occlusion"
        r = run(["scripts/analysis/occlusion_test.py", "--stage", "occlude",
                "--cohort", "splits/cohort.csv", "--strata", "splits/strata.csv",
                "--workdir", str(occ_dir)], cwd=repo)
        check("occlusion occlude stage exits 0", r.returncode == 0, r.stderr[-500:])
        occ_files = {p.name for p in (occ_dir / "occluded" / "shell_10_20" / "imagesTs").glob("*.nii.gz")}
        expected_names = {f"{cid}_0000.nii.gz" for cid in
                          cohort.loc[cohort["has_manual_lesion"], "case_id"]}
        check("occluded filenames preserve the full case_id (regression test: study number "
              "'_00001' must not collide with the '_0000' channel-suffix strip)",
              occ_files == expected_names,
              f"missing={expected_names - occ_files}, extra={occ_files - expected_names}")

        print("\n== synthetic occlusion predictions (designed shell/arm pattern) ==")
        _make_occlusion_predictions(occ_dir, cnn_dir, tf_dir)

        print("\n== occlusion_test.py (score stage) ==")
        r = run(["scripts/analysis/occlusion_test.py", "--stage", "score",
                "--workdir", str(occ_dir), "--pred-base-cnn", str(cnn_dir),
                "--pred-base-tf", str(tf_dir), "--refs", str(refs_dir),
                "--cohort", "splits/cohort.csv", "--out", str(tmp / "results" / "h2")], cwd=repo)
        check("occlusion score stage exits 0", r.returncode == 0, r.stderr[-500:])
        h2 = yaml.safe_load(open(tmp / "results" / "h2" / "summary.yaml"))
        check("H2 recovers all 30 cases (no silent case-ID mismatch drops)", h2["n_cases"] == 30,
              f"got {h2['n_cases']}")
        check("H2 decisive-shell contrast SUPPORTED (matches designed pattern)",
              h2["h2_supported"] is True)

        # 4. Boundary tolerance ------------------------------------------------------------
        print("\n== boundary_tolerance.py ==")
        r = run(["scripts/analysis/boundary_tolerance.py", "--refs", str(refs_dir),
                "--cohort", "splits/cohort.csv", "--out", str(tmp / "results" / "h3")], cwd=repo)
        check("boundary_tolerance.py exits 0", r.returncode == 0, r.stderr[-500:])
        floor = pd.read_csv(tmp / "results" / "h3" / "boundary_floor_per_case.csv")
        check("boundary floor computed for all 30 cases", len(floor) == 30, f"got {len(floor)}")

        print("\n== boundary_tolerance.py (--loso, H3 share-of-loss) ==")
        # Designed pattern: cnn's cross-source Dice loss stays mostly inside the boundary
        # floor (arm robust to source shift); tf's loss swamps it many times over, so most of
        # tf's loss is the *excess* above the floor rather than the floor-covered portion (the
        # share statistic gives partial credit up to the floor even when loss exceeds it, so
        # the excess has to dominate, not just barely exceed the floor, to pull the share down).
        tolerable = 1 - floor.set_index("case_id")["tolerance_floor_dice"]
        loso_rows = []
        for cid, tol in tolerable.items():
            loso_rows.append({"case_id": cid, "arm": "cnn", "dice_in_domain": 0.8,
                              "dice_loso": 0.8 - 0.5 * tol})   # loss = 0.5*tol < tol: inside
            loso_rows.append({"case_id": cid, "arm": "tf", "dice_in_domain": 0.8,
                              "dice_loso": 0.8 - (20 * tol + 0.5)})  # loss >> tol: mostly excess
        loso_path = tmp / "loso_dice.csv"
        pd.DataFrame(loso_rows).to_csv(loso_path, index=False)
        r = run(["scripts/analysis/boundary_tolerance.py", "--refs", str(refs_dir),
                "--cohort", "splits/cohort.csv", "--loso", str(loso_path),
                "--out", str(tmp / "results" / "h3_loso")], cwd=repo)
        check("boundary_tolerance.py --loso exits 0", r.returncode == 0, r.stderr[-500:])
        h3_summary = yaml.safe_load(open(tmp / "results" / "h3_loso" / "summary.yaml"))
        check("H3 CNN share_inside_floor near 1.0 (designed: always inside)",
              h3_summary["cnn"]["share_inside_floor"] > 0.95,
              f"got {h3_summary['cnn']['share_inside_floor']}")
        check("H3 transformer share_inside_floor much lower than CNN's (designed: loss "
              "dominated by the excess above the floor)",
              h3_summary["tf"]["share_inside_floor"] < 0.3,
              f"got {h3_summary['tf']['share_inside_floor']}")
        check("H3 share_inside_floor_ci present for both arms",
              "share_inside_floor_ci" in h3_summary["cnn"] and "share_inside_floor_ci" in h3_summary["tf"])
        check("H3 supported (CNN share exceeds 50% threshold)", h3_summary["h3_supported"] is True)

        # 5. Failure gallery -----------------------------------------------------------------
        print("\n== failure_gallery.py ==")
        r = run(["scripts/analysis/failure_gallery.py",
                "--per-case", str(tmp / "results" / "h1" / "per_case.csv"),
                "--pred-cnn", str(cnn_dir), "--pred-tf", str(tf_dir), "--refs", str(refs_dir),
                "--images", str(data_root / "panorama" / "images"),
                "--out", str(tmp / "results" / "gallery")], cwd=repo)
        check("failure_gallery.py exits 0", r.returncode == 0, r.stderr[-500:])
        pngs = list((tmp / "results" / "gallery").glob("*.png"))
        check("one figure per (volume x cnr) stratum cell (9 cells)", len(pngs) == 9,
              f"got {len(pngs)}")

        # 6. Effective receptive field self-test (no nnU-Net/GPU needed) -------------------
        print("\n== effective_receptive_field.py --self-test ==")
        r = run(["scripts/analysis/effective_receptive_field.py", "--self-test"], cwd=repo)
        check("ERF self-test passes (monotonic growth with depth)", r.returncode == 0, r.stderr[-500:])

        # 7. NIH false-positive harness -----------------------------------------------------
        print("\n== nih_false_positives.py ==")
        nih_preds_cnn, nih_preds_tf = _make_nih_predictions(data_root, tmp / "nih_preds")
        r = run(["scripts/analysis/nih_false_positives.py",
                "--pred-cnn", str(nih_preds_cnn), "--pred-tf", str(nih_preds_tf),
                "--cohort", "splits/cohort.csv", "--out", str(tmp / "results" / "nih_fp")], cwd=repo)
        check("nih_false_positives.py exits 0", r.returncode == 0, r.stderr[-500:])
        nih_summary = yaml.safe_load(open(tmp / "results" / "nih_fp" / "summary.yaml"))
        check("NIH FP harness covers all 4 negative-control cases per arm",
              nih_summary["cnn"]["n_cases"] == 4 and nih_summary["tf"]["n_cases"] == 4,
              f"got {nih_summary}")
        check("NIH FP harness detects the designed higher FP rate for the noisier arm",
              nih_summary["tf"]["mean_fp_per_case"] > nih_summary["cnn"]["mean_fp_per_case"],
              f"cnn={nih_summary['cnn']['mean_fp_per_case']}, tf={nih_summary['tf']['mean_fp_per_case']}")

        # 8. Variance decomposition (seed vs. fold), synthetic numbers with a known answer -----
        print("\n== variance_decomposition.py ==")
        fold_means = [0.70, 0.75, 0.72, 0.78, 0.74]
        fold_rows = [{"case_id": f"f{f}c{i}", "fold": f, "dice": fold_means[f]}
                     for f in range(5) for i in range(4)]
        pd.DataFrame(fold_rows).to_csv(tmp / "fold_dice.csv", index=False)
        seed_rng = np.random.default_rng(3)
        seed_rows = []
        seed_noise_sd = 0.02
        for i in range(6):  # 6 fold-0 cases
            base = 0.70
            for seed in (0, 1, 2):
                seed_rows.append({"case_id": f"f0c{i}", "seed": seed,
                                  "dice": base + seed_rng.normal(0, seed_noise_sd)})
        pd.DataFrame(seed_rows).to_csv(tmp / "seed_dice.csv", index=False)

        r = run(["scripts/analysis/variance_decomposition.py", "--arm", "cnn",
                "--fold-dice", str(tmp / "fold_dice.csv"),
                "--seed-dice", str(tmp / "seed_dice.csv"),
                "--out", str(tmp / "results" / "variance_cnn.yaml")], cwd=repo)
        check("variance_decomposition.py exits 0", r.returncode == 0, r.stderr[-500:])
        var_summary = yaml.safe_load(open(tmp / "results" / "variance_cnn.yaml"))
        expected_fold_var = float(np.var(fold_means, ddof=1))
        check("fold_variance matches the known synthetic fold means exactly",
              abs(var_summary["fold_variance"] - expected_fold_var) < 1e-9,
              f"got {var_summary['fold_variance']}, expected {expected_fold_var}")
        check("seed_variance is small and close to the injected noise variance "
              f"({seed_noise_sd**2:.5f})",
              abs(var_summary["seed_variance"] - seed_noise_sd ** 2) < 0.002,
              f"got {var_summary['seed_variance']}")
        check("fold_variance_net_of_seed_estimate is non-negative and <= raw fold_variance",
              0 <= var_summary["fold_variance_net_of_seed_estimate"] <= var_summary["fold_variance"] + 1e-9)

        # 7. log_run.py (GPU run-log enforcement used by verify_pipeline.sh / run_tier_a.sh) --
        print("\n== log_run.py ==")
        r = run(["scripts/training/log_run.py", "start", "--arm", "cnn", "--fold", "0",
                "--seed", "0", "--nnunet-commit", "nnunetv2==test", "--trainer-or-plans",
                "nnUNetResEncUNetLPlans", "--gpu", "TEST-GPU"], cwd=repo)
        check("log_run.py start exits 0", r.returncode == 0, r.stderr[-500:])
        run_id = r.stdout.strip()
        log_csv = repo / "scripts" / "training" / "run_log.csv"
        check("run_log.csv created with a running row", log_csv.exists() and run_id in log_csv.read_text())
        row_running = pd.read_csv(log_csv).set_index("run_id").loc[run_id]
        check("started row has status=running and empty end_time",
              row_running["status"] == "running" and pd.isna(row_running["end_time"]))
        r = run(["scripts/training/log_run.py", "finish", "--run-id", run_id, "--status",
                "completed", "--checkpoint-path", "/fake/ckpt.pth"], cwd=repo)
        check("log_run.py finish exits 0", r.returncode == 0, r.stderr[-500:])
        row_done = pd.read_csv(log_csv).set_index("run_id").loc[run_id]
        check("finished row has status=completed and a checkpoint path",
              row_done["status"] == "completed" and row_done["checkpoint_path"] == "/fake/ckpt.pth")
        r = run(["scripts/training/log_run.py", "finish", "--run-id", "does-not-exist",
                "--status", "completed"], cwd=repo)
        check("log_run.py finish rejects an unknown run_id", r.returncode != 0)

        # 9. Full per-case metrics table (deliverable 1) -----------------------------------
        print("\n== build_per_case_table.py ==")
        metrics_csv = tmp / "results" / "per_case_metrics.csv"
        r = run(["scripts/analysis/build_per_case_table.py",
                "--arm", f"cnn:{cnn_dir}", "--arm", f"tf:{tf_dir}", "--refs", str(refs_dir),
                "--strata", "splits/strata.csv", "--cohort", "splits/cohort.csv",
                "--out", str(metrics_csv)], cwd=repo)
        check("build_per_case_table.py exits 0", r.returncode == 0, r.stderr[-800:])
        mt = pd.read_csv(metrics_csv)
        check("metrics table is 30 cases x 2 arms with no missing cells",
              len(mt) == 60 and mt["arm"].nunique() == 2 and mt["dice"].notna().all(),
              f"got {len(mt)} rows")
        check("metrics table carries every frozen metric (Dice, NSD, HD95, detection, FP)",
              {"dice", "nsd2mm", "hd95", "detected", "fp_count"} <= set(mt.columns))
        check("metrics table carries source and fold for stratified/LOSO analysis",
              {"source", "fold"} <= set(mt.columns))
        r_missing = run(["scripts/analysis/build_per_case_table.py",
                        "--arm", f"cnn:{cnn_dir}", "--arm", f"tf:{tmp / 'nope'}",
                        "--refs", str(refs_dir), "--strata", "splits/strata.csv",
                        "--cohort", "splits/cohort.csv", "--out", str(tmp / "x.csv")], cwd=repo)
        check("build_per_case_table.py refuses an incomplete pairing by default",
              r_missing.returncode != 0)

        print("\n== paired_analysis.py --metrics-table (same answer as the direct path) ==")
        r = run(["scripts/analysis/paired_analysis.py", "--metrics-table", str(metrics_csv),
                "--cohort", "splits/cohort.csv", "--out", str(tmp / "results" / "h1_tbl")],
                cwd=repo)
        check("paired_analysis.py --metrics-table exits 0", r.returncode == 0, r.stderr[-800:])
        h1b = yaml.safe_load(open(tmp / "results" / "h1_tbl" / "summary.yaml"))
        check("metrics-table path reproduces the direct path's H1 slope exactly",
              abs(h1b["h1_slope"] - h1["h1_slope"]) < 1e-9,
              f"{h1b['h1_slope']} vs {h1['h1_slope']}")
        heat = pd.read_csv(tmp / "results" / "h1_tbl" / "stratum_heatmap.csv")
        check("stratum heatmap carries HD95 / detection / FP per cell",
              {"hd95_mean", "detection_sensitivity", "fp_per_case"} <= set(heat.columns))

        # 10. Contrast sensitivity at 5 / 10 / 15 mm rings -----------------------------------
        print("\n== contrast_sensitivity.py ==")
        r = run(["scripts/analysis/contrast_sensitivity.py", "--metrics-table", str(metrics_csv),
                "--strata", "splits/strata.csv",
                "--out", str(tmp / "results" / "contrast_sensitivity")], cwd=repo)
        check("contrast_sensitivity.py exits 0", r.returncode == 0, r.stderr[-800:])
        cs = yaml.safe_load(open(tmp / "results" / "contrast_sensitivity" / "summary.yaml"))
        check("contrast sensitivity covers all three frozen ring widths",
              sorted(cs["ring_widths_mm"]) == [5, 10, 15], f"got {cs['ring_widths_mm']}")
        check("contrast sensitivity applies the frozen rank-stability gate",
              isinstance(cs["contrast_axis_exploratory"], bool)
              and len(cs["spearman_rank_stability"]) == 3)
        by_ring = pd.read_csv(tmp / "results" / "contrast_sensitivity"
                              / "stratified_by_ring_width.csv")
        check("every stratified cell is recomputed under each ring width",
              set(by_ring["ring_mm"]) == {5, 10, 15})

        # 11. Leave-one-source-out (deliverables 5 and 6) -------------------------------------
        print("\n== make_loso_splits.py ==")
        r_small = run(["scripts/data/make_loso_splits.py", "--exclude-source", "NIH",
                      "--out", str(tmp / "unused_loso.json")], cwd=repo)
        check("make_loso_splits.py escalates when a source is below the usable case count",
              r_small.returncode != 0 and "ESCALATE" in (r_small.stdout + r_small.stderr))
        r = run(["scripts/data/make_loso_splits.py", "--exclude-source", "NIH", "--min-cases", "5",
                "--emit-combined", "splits/splits_with_loso.json"], cwd=repo)
        check("make_loso_splits.py exits 0", r.returncode == 0, r.stderr[-800:])
        loso_folds = pd.read_csv(repo / "splits" / "loso_folds.csv")
        check("one LOSO fold per canonical source, NIH excluded",
              sorted(loso_folds["held_out_source"]) == ["MSKCC", "Radboud", "UMCG"],
              f"got {loso_folds['held_out_source'].tolist()}")
        combined = json.load(open(repo / "splits" / "splits_with_loso.json"))
        check("combined splits keep the frozen 5 CV folds byte-identical at indices 0-4",
              combined[:5] == splits)
        check("LOSO folds appended at indices 5+ with the right nnU-Net fold numbers",
              len(combined) == 8 and loso_folds["nnunet_fold"].tolist() == [5, 6, 7])
        check("each LOSO fold holds out exactly its own source and trains on the rest",
              all(len(set(combined[5 + i]["train"]) & set(combined[5 + i]["val"])) == 0
                  for i in range(3))
              and sum(len(combined[5 + i]["val"]) for i in range(3)) == 30)

        print("\n== loso_analysis.py ==")
        loso_pred_root = sd.make_loso_predictions(repo / "splits" / "cohort.csv", manual_dir,
                                                  tmp / "loso_preds")
        r = run(["scripts/analysis/loso_analysis.py", "--metrics-table", str(metrics_csv),
                "--pred", f"cnn:{loso_pred_root / 'cnn'}", "--pred", f"tf:{loso_pred_root / 'tf'}",
                "--refs", str(refs_dir), "--cohort", "splits/cohort.csv",
                "--out", str(tmp / "results" / "loso")], cwd=repo)
        check("loso_analysis.py exits 0", r.returncode == 0, r.stderr[-800:])
        loso_pc = pd.read_csv(tmp / "results" / "loso" / "loso_per_case.csv")
        check("LOSO covers every case in every arm (30 x 2)", len(loso_pc) == 60,
              f"got {len(loso_pc)}")
        loso_sum = yaml.safe_load(open(tmp / "results" / "loso" / "summary.yaml"))
        check("LOSO recovers the designed larger cross-source drop for the transformer",
              loso_sum["overall"]["tf"]["mean_dice_drop"] >
              loso_sum["overall"]["cnn"]["mean_dice_drop"],
              f"cnn={loso_sum['overall']['cnn']['mean_dice_drop']}, "
              f"tf={loso_sum['overall']['tf']['mean_dice_drop']}")
        loso_dice = pd.read_csv(tmp / "results" / "loso" / "loso_dice.csv")
        check("loso_dice.csv matches the exact contract boundary_tolerance.py --loso reads",
              list(loso_dice.columns) == ["case_id", "arm", "dice_in_domain", "dice_loso"])
        r = run(["scripts/analysis/boundary_tolerance.py", "--refs", str(refs_dir),
                "--cohort", "splits/cohort.csv",
                "--loso", str(tmp / "results" / "loso" / "loso_dice.csv"),
                "--out", str(tmp / "results" / "h3")], cwd=repo)
        check("H3 runs end to end on the real LOSO table (not a hand-built one)",
              r.returncode == 0, r.stderr[-800:])
        h3_real = yaml.safe_load(open(tmp / "results" / "h3" / "summary.yaml"))
        check("H3 verdict computed from the real LOSO table", "h3_supported" in h3_real)

        # 12. Identity control (H2b, deliverable 4) -------------------------------------------
        print("\n== identity_comparison.py ==")
        ident_dir = sd.make_identity_predictions(tf_dir, tmp / "preds" / "pred_identity")
        metrics_h2b = tmp / "results" / "per_case_metrics_h2b.csv"
        r = run(["scripts/analysis/build_per_case_table.py",
                "--arm", f"tf:{tf_dir}", "--arm", f"identity_control:{ident_dir}",
                "--refs", str(refs_dir), "--strata", "splits/strata.csv",
                "--cohort", "splits/cohort.csv", "--out", str(metrics_h2b)], cwd=repo)
        check("H2b metrics table builds", r.returncode == 0, r.stderr[-800:])
        r = run(["scripts/analysis/identity_comparison.py", "--metrics-table", str(metrics_h2b),
                "--out", str(tmp / "results" / "h2b")], cwd=repo)
        check("identity_comparison.py exits 0", r.returncode == 0, r.stderr[-800:])
        h2b = yaml.safe_load(open(tmp / "results" / "h2b" / "summary.yaml"))
        check("H2b compares all 9 stratum cells in the same layout as the Tier A map",
              h2b["n_cells"] == 9, f"got {h2b['n_cells']}")
        check("H2b reads an identity control that reproduces the map as reproducing it",
              h2b["map_reproduced_within_ci"] is True
              and abs(h2b["overall_delta_dice_full_minus_identity"]) < 1e-12,
              f"agreeing cells: {h2b['n_cells_agreeing_within_ci']}/{h2b['n_cells']}, "
              f"delta={h2b['overall_delta_dice_full_minus_identity']}")
        check("H2b heatmap CSV written for the figure",
              (tmp / "results" / "h2b" / "identity_vs_full_heatmap.csv").exists())

        # ... and the opposite fixture: an identity control that really is worse everywhere
        # must NOT be read as reproducing the map.
        ident_bad = sd.make_identity_predictions(tf_dir, tmp / "preds" / "pred_identity_bad",
                                                 extra_erosion=1)
        metrics_bad = tmp / "results" / "per_case_metrics_h2b_bad.csv"
        run(["scripts/analysis/build_per_case_table.py", "--arm", f"tf:{tf_dir}",
             "--arm", f"identity_control:{ident_bad}", "--refs", str(refs_dir),
             "--strata", "splits/strata.csv", "--cohort", "splits/cohort.csv",
             "--out", str(metrics_bad)], cwd=repo)
        r = run(["scripts/analysis/identity_comparison.py", "--metrics-table", str(metrics_bad),
                "--out", str(tmp / "results" / "h2b_bad")], cwd=repo)
        h2b_bad = yaml.safe_load(open(tmp / "results" / "h2b_bad" / "summary.yaml"))
        check("H2b reads a genuinely degraded identity control as NOT reproducing the map",
              r.returncode == 0 and h2b_bad["map_reproduced_within_ci"] is False
              and h2b_bad["overall_delta_dice_full_minus_identity"] > 0,
              f"{h2b_bad.get('n_cells_agreeing_within_ci')}/{h2b_bad.get('n_cells')} agree, "
              f"delta={h2b_bad.get('overall_delta_dice_full_minus_identity')}")

        # 13. Seed vs fold variance input tables ---------------------------------------------
        print("\n== build_variance_tables.py ==")
        seed_dirs = sd.make_seed_predictions(repo / "splits" / "fold_assignment.csv", manual_dir,
                                             tmp / "seed_preds")
        r = run(["scripts/analysis/build_variance_tables.py", "--arm", "cnn",
                "--metrics-table", str(metrics_csv), "--refs", str(refs_dir),
                *sum([["--seed-pred", f"{s}:{d}"] for s, d in seed_dirs.items()], []),
                "--out", str(tmp / "results" / "variance_inputs_cnn")], cwd=repo)
        check("build_variance_tables.py exits 0", r.returncode == 0, r.stderr[-800:])
        sdf = pd.read_csv(tmp / "results" / "variance_inputs_cnn" / "seed_dice.csv")
        check("seed table holds the default seed plus both replicates for every fold-0 case",
              sdf.groupby("case_id")["seed"].nunique().eq(3).all(),
              f"seeds per case: {sdf.groupby('case_id')['seed'].nunique().unique()}")
        r = run(["scripts/analysis/variance_decomposition.py", "--arm", "cnn",
                "--fold-dice", str(tmp / "results" / "variance_inputs_cnn" / "fold_dice.csv"),
                "--seed-dice", str(tmp / "results" / "variance_inputs_cnn" / "seed_dice.csv"),
                "--out", str(tmp / "results" / "variance_cnn.yaml")], cwd=repo)
        check("variance_decomposition.py consumes build_variance_tables.py output directly",
              r.returncode == 0, r.stderr[-800:])

        # 14. Inter-rater proxy statement ------------------------------------------------------
        print("\n== paired_delineation_check.py ==")
        r = run(["scripts/analysis/paired_delineation_check.py",
                "--labels-root", str(data_root / "panorama" / "panorama_labels"),
                "--cohort", "splits/cohort.csv",
                "--out", str(tmp / "results" / "inter_rater")], cwd=repo)
        check("paired_delineation_check.py exits 0", r.returncode == 0, r.stderr[-800:])
        ir = yaml.safe_load(open(tmp / "results" / "inter_rater" / "summary.yaml"))
        check("no second independent delineation found in the single-reader fixture",
              ir["n_cases_with_independent_second_delineation"] == 0)
        check("a written statement is produced either way (the task's 'done when')",
              (tmp / "results" / "inter_rater" / "STATEMENT.md").exists()
              and "SYNTHETIC ONLY" in (tmp / "results" / "inter_rater" / "STATEMENT.md").read_text())
        r = run(["scripts/analysis/paired_delineation_check.py",
                "--labels-root", str(data_root / "panorama" / "panorama_labels"),
                "--cohort", "splits/cohort.csv", "--include-automatic",
                "--out", str(tmp / "results" / "inter_rater_auto")], cwd=repo)
        ir2 = yaml.safe_load(open(tmp / "results" / "inter_rater_auto" / "summary.yaml"))
        # manual_labels/ and automatic_labels/ are mutually exclusive per case in the real repo
        # (see synthetic_data.py's docstring), so no case can ever appear in both directories --
        # --include-automatic has nothing to find here even in principle, not just in this
        # fixture. Confirm it still exits cleanly and scans automatic_labels/ rather than
        # silently no-op'ing on a bad path.
        check("--include-automatic scans automatic_labels/ and finds no case in both directories "
              "(manual/automatic are mutually exclusive per case, so there is nothing to find)",
              r.returncode == 0 and ir2["n_cases_with_independent_second_delineation"] == 0
              and ir2["n_cases_with_any_second_delineation"] == 0
              and "automatic_labels" in ir2["delineation_directories_scanned"],
              f"any={ir2.get('n_cases_with_any_second_delineation')} "
              f"dirs={ir2.get('delineation_directories_scanned')}")

        # 15. NIH negative control under source shift (deliverable 8) -------------------------
        print("\n== nih_false_positives.py (in domain vs source shift) ==")
        # Under source shift the noisier arm hallucinates more, which is the whole point of
        # running the negative control in both conditions.
        nih_shift_cnn, nih_shift_tf = _make_nih_predictions(data_root, tmp / "nih_preds_shift",
                                                            n_blobs=6, seed=123)
        r = run(["scripts/analysis/nih_false_positives.py",
                "--pred", f"cnn:in_domain:{nih_preds_cnn}",
                "--pred", f"tf:in_domain:{nih_preds_tf}",
                "--pred", f"cnn:source_shift:{nih_shift_cnn}",
                "--pred", f"tf:source_shift:{nih_shift_tf}",
                "--cohort", "splits/cohort.csv",
                "--out", str(tmp / "results" / "nih_fp_shift")], cwd=repo)
        check("nih_false_positives.py runs both conditions", r.returncode == 0, r.stderr[-800:])
        nih2 = yaml.safe_load(open(tmp / "results" / "nih_fp_shift" / "summary.yaml"))
        check("both conditions reported per arm",
              set(nih2["conditions"]) == {"in_domain", "source_shift"})
        check("paired in-domain -> shifted change reported per arm",
              "delta_source_shift_minus_in_domain" in nih2["cnn"]
              and "delta_source_shift_minus_in_domain" in nih2["tf"])
        check("in-domain rate still available at the top level (unchanged contract)",
              abs(nih2["tf"]["mean_fp_per_case"] - nih_summary["tf"]["mean_fp_per_case"]) < 1e-9)
        check("negative control detects the designed rise in false positives under source shift",
              nih2["tf"]["delta_source_shift_minus_in_domain"]["mean_delta_fp_per_case"] > 0,
              f"got {nih2['tf']['delta_source_shift_minus_in_domain']}")

        # 16. Figures and verdict memo ---------------------------------------------------------
        print("\n== make_figures.py ==")
        results_root = tmp / "results"
        shutil.copytree(results_root / "h1_tbl", results_root / "h1", dirs_exist_ok=True)
        r = run(["scripts/analysis/make_figures.py", "--results", str(results_root),
                "--out", str(results_root / "figures")], cwd=repo)
        check("make_figures.py exits 0", r.returncode == 0, r.stderr[-1500:])
        figs = {p.name for p in (results_root / "figures").glob("*.png")}
        check("all six figures render from the analysis outputs", len(figs) == 6,
              f"got {sorted(figs)}")
        check("figures include the deliverable panels (stratified maps, occlusion, identity, LOSO)",
              {"fig2_stratified_maps.png", "fig4_occlusion_curves.png",
               "fig5_identity_control_map.png", "fig6_loso.png"} <= figs, f"got {sorted(figs)}")

        print("\n== verdict_memo.py ==")
        r = run(["scripts/analysis/verdict_memo.py", "--results", str(results_root),
                "--out", str(results_root / "VERDICT.md")], cwd=repo)
        check("verdict_memo.py exits 0", r.returncode == 0, r.stderr[-800:])
        memo = (results_root / "VERDICT.md").read_text()
        check("verdict memo reports all four pre-registered hypotheses",
              all(h in memo for h in ("H1 —", "H2 —", "H2b —", "H3 —")))
        check("verdict memo quotes the frozen margins, not invented ones",
              "±3.0 Dice points" in memo and "2.0 Dice points" in memo and "50%" in memo)
        check("verdict memo reads H1 and H2 as SUPPORTED on the designed fixture",
              memo.count("**SUPPORTED**") >= 2, memo[:400])

        # 17. The task registry — the unit of concurrency ---------------------------------
        print("\n== tasks.py ==")
        r = run(["scripts/training/tasks.py"], cwd=repo)
        check("tasks.py lists the runs", r.returncode == 0, r.stderr[-500:])
        r_json = run(["scripts/training/tasks.py", "--json"], cwd=repo)
        names = json.loads(r_json.stdout)
        check("25 tasks with LOSO folds present (10 Tier A + 6 LOSO + 5 identity + 4 seeds)",
              len(names) == 25, f"got {len(names)}: {names}")
        check("task names are unique", len(set(names)) == len(names))

        sys.path.insert(0, str(repo / "scripts" / "training"))
        import importlib
        import tasks as task_mod
        importlib.reload(task_mod)
        parsed = [task_mod.parse(n, repo=repo, budget_suffix="_budget") for n in names]
        check("every task resolves to a trainer and an nnU-Net fold index",
              all(t["trainer"] and isinstance(t["nnunet_fold"], int) for t in parsed))
        check("the budget suffix reaches both the trainer and the results directory",
              all(t["trainer"].endswith("_budget") and "_budget__" in t["results_subdir"]
                  for t in parsed))
        # The collision that would silently corrupt a concurrent fan-out: two runs writing the
        # same results directory, or two runs writing the same prediction directory.
        subdirs = [t["results_subdir"] + f"/fold_{t['nnunet_fold']}" for t in parsed]
        check("no two tasks share an nnU-Net results directory",
              len(set(subdirs)) == len(subdirs),
              f"{len(subdirs) - len(set(subdirs))} collision(s)")
        pred_paths = [(t["arm_key"], t["pred_key"]) for t in parsed]
        check("no two tasks share a prediction directory (seed replicates stay separate)",
              len(set(pred_paths)) == len(pred_paths),
              f"{len(pred_paths) - len(set(pred_paths))} collision(s)")
        reps = [t for t in parsed if t["is_replicate"]]
        check("seed replicates are flagged and keyed by seed", len(reps) == 4
              and all("-seed" in t["pred_key"] for t in reps), f"got {len(reps)}")
        check("leave-one-source-out tasks map onto the appended fold indices",
              sorted({t["nnunet_fold"] for t in parsed if t["fold_label"].startswith("loso_")})
              == [5, 6, 7])
        r_bad = run(["scripts/training/tasks.py", "--describe", "identity-fold0-seed1"], cwd=repo)
        check("a seed replicate of the identity control is refused, not silently run",
              r_bad.returncode != 0)

        # 18. Per-fold occlusion (the per-worker path) -------------------------------------
        print("\n== occlusion_test.py --fold ==")
        occ_fold = tmp / "occlusion_fold0"
        r = run(["scripts/analysis/occlusion_test.py", "--stage", "occlude",
                "--cohort", "splits/cohort.csv", "--strata", "splits/strata.csv",
                "--workdir", str(occ_fold), "--fold", "0"], cwd=repo)
        check("per-fold occlusion exits 0", r.returncode == 0, r.stderr[-500:])
        fold0 = set(pd.read_csv(repo / "splits" / "fold_assignment.csv")
                    .query("fold == 0")["case_id"])
        got = {p.name[: -len("_0000.nii.gz")]
               for p in (occ_fold / "occluded" / "shell_10_20" / "imagesTs").glob("*.nii.gz")}
        check("per-fold occlusion covers exactly the cases that fold held out, and no others",
              got == fold0, f"missing={fold0 - got}, extra={got - fold0}")
        check("per-fold occlusion is a strict subset of the whole-cohort run",
              0 < len(got) < len(occ_files))

        # 19. Subset extraction from a batch zip ------------------------------------------
        print("\n== extract_subset.py ==")
        import zipfile
        zip_path = tmp / "fake_batch.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for pid in range(10):
                for study in (1, 2):        # two scans per patient
                    zf.writestr(f"batch_1/{100000 + pid}_0000{study}_0000.nii.gz",
                                b"not-a-real-volume")
            zf.writestr("batch_1/README.txt", b"ignored: not an image")

        sys.path.insert(0, str(repo / "scripts" / "download"))
        import extract_subset as xs
        importlib.reload(xs)
        entries, patients = xs.select_entries(zip_path, max_cases=3)
        check("subset selection takes whole patients, not loose files",
              len(patients) == 3 and len(entries) == 6,
              f"{len(patients)} patients / {len(entries)} entries")
        check("subset selection is deterministic and sorted, so re-running extends rather "
              "than reshuffles the cohort",
              patients == ["100000", "100001", "100002"], str(patients))
        check("non-image entries are never selected",
              all(e.endswith(".nii.gz") for e in entries))
        _, all_patients = xs.select_entries(zip_path, max_cases=None)
        check("no cap selects every patient", len(all_patients) == 10, str(len(all_patients)))
        _, capped = xs.select_entries(zip_path, max_cases=999)
        check("a cap larger than the archive is not an error", len(capped) == 10)

        dest = tmp / "extracted"
        got = xs.extract(zip_path, dest, max_cases=3)
        files = sorted(p.name for p in dest.rglob("*.nii.gz"))
        check("extraction writes exactly the selected patients' files",
              len(files) == 6 and {f.split("_")[0] for f in files} == set(got),
              f"{files}")
        check("nothing outside the selection is unpacked",
              not list(dest.rglob("README.txt")))

        # 20. Custom trainers, against a stubbed nnU-Net ----------------------------------
        print("\n== trainers (stubbed nnU-Net) ==")
        r = run(["tests/test_trainers.py"], cwd=REPO)
        n_trainer_checks = r.stdout.count("  PASS  ")
        check("trainer checks pass (seed replicates, wall-clock budget, epoch override)",
              r.returncode == 0 and n_trainer_checks >= 18,
              f"{n_trainer_checks} passed; {r.stdout[-1200:]}")

        # 21. Generated Kaggle notebooks --------------------------------------------------
        print("\n== notebooks ==")
        nb_dir = REPO / "notebooks"
        nbs = sorted(nb_dir.glob("*.ipynb"))
        check("the three Kaggle notebooks are committed", len(nbs) == 3,
              f"got {[p.name for p in nbs]}")
        check("notebooks are prepare / run / analyze",
              [p.stem for p in nbs] == ["00_prepare", "01_run", "02_analyze"],
              f"got {[p.stem for p in nbs]}")
        syntax_errors, cell_counts = [], {}
        for nb_path in nbs:
            nb = json.load(open(nb_path))
            cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
            cell_counts[nb_path.name] = len(cells)
            for i, c in enumerate(cells):
                try:
                    compile("".join(c["source"]), f"{nb_path.name}#{i}", "exec")
                except SyntaxError as e:
                    syntax_errors.append(f"{nb_path.name} cell {i}: {e}")
        check("every notebook code cell is valid Python", not syntax_errors,
              "; ".join(syntax_errors[:3]))
        check("no notebook is empty of code", all(v > 0 for v in cell_counts.values()),
              str(cell_counts))

        # The .ipynb files are generated; if the generator and the committed files disagree,
        # editing one of them silently does nothing, which is worse than either being wrong.
        gen_dir = tmp / "nb_regen"
        # Without ignoring __pycache__ the copy can carry a compiled nb_common from before the
        # last edit, and the comparison then reports drift that does not exist.
        shutil.copytree(nb_dir, gen_dir, ignore=shutil.ignore_patterns("__pycache__"))
        r = run(["build_notebooks.py"], cwd=gen_dir)
        check("build_notebooks.py runs", r.returncode == 0, r.stderr[-500:])

        def cell_content(path):
            """What the notebook is: the ordered cell types and sources. Deliberately not the
            raw bytes — opening a notebook in an editor rewrites JSON formatting and cell ids
            without changing anything that runs, and a drift check that fires on that trains
            people to ignore it."""
            nb = json.load(open(path))
            return [(c["cell_type"], "".join(c["source"])) for c in nb["cells"]]

        drift = [p.name for p in nbs if cell_content(gen_dir / p.name) != cell_content(p)]
        check("committed notebooks match what build_notebooks.py generates", not drift,
              f"stale: {drift} — re-run python notebooks/build_notebooks.py and commit")

    finally:
        if keep:
            print(f"\n--keep: working directory left at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 60}\n{len(passed)} passed, {len(failures)} failed\n{'=' * 60}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


def _dedup_diagnosis(repo, data_root):
    """Explain a dedup failure in the log itself.

    Duplicate detection is two stages, and the two fail for completely different reasons:
    stage 1 groups by header metadata, stage 2 compares voxel data. Knowing which one let a
    duplicate through is the whole diagnosis, and the feedback loop for this test on a remote
    runner is minutes per attempt — long enough that a bare count is not worth reporting on
    its own.
    """
    try:
        sys.path.insert(0, str(repo / "scripts" / "data"))
        import importlib
        import deduplicate as dd
        importlib.reload(dd)
        img_dir = data_root / "panorama" / "images"
        names = ["100000_00001", "999001_00001", "999002_00001"]
        lines = ["", "  dedup diagnosis (original, plain copy, header-jittered copy):"]
        for cid in names:
            f = img_dir / f"{cid}_0000.nii.gz"
            if not f.exists():
                lines.append(f"    {cid}: MISSING"); continue
            lines.append(f"    {cid}: meta={dd.meta_fingerprint(f)[:12]} "
                         f"content={dd.content_hash(f)[:12]}")
        metas = {dd.meta_fingerprint(img_dir / f"{c}_0000.nii.gz") for c in names
                 if (img_dir / f"{c}_0000.nii.gz").exists()}
        contents = {dd.content_hash(img_dir / f"{c}_0000.nii.gz") for c in names
                    if (img_dir / f"{c}_0000.nii.gz").exists()}
        if len(metas) > 1:
            lines.append("    -> STAGE 1: the copies do not share a candidate group, so they "
                         "were never content-compared. The metadata fingerprint is picking up "
                         "header differences it should ignore.")
        elif len(contents) > 1:
            lines.append("    -> STAGE 2: same candidate group, different voxel data. The "
                         "copies are not byte-identical on this platform — the writer is "
                         "changing the data, not just the header.")
        else:
            lines.append("    -> both stages agree the files are identical; the failure is in "
                         "the bookkeeping, not the detection.")
        return "\n".join(lines)
    except Exception as e:  # a diagnosis must never mask the failure it explains
        return f"  (diagnosis unavailable: {type(e).__name__}: {e})"


def _make_nih_predictions(data_root, out_dir, n_blobs=3, seed=99):
    """All-background CNN predictions vs. transformer predictions seeded with a few small
    speckle blobs (>= 100mm^3 each, per the frozen FP volume threshold) on the NIH negatives --
    a designed higher-FP-rate pattern for the harness to detect."""
    import SimpleITK as sitk

    img_dir = data_root / "panorama" / "images"
    cnn_dir, tf_dir = out_dir / "pred_cnn", out_dir / "pred_tf"
    cnn_dir.mkdir(parents=True, exist_ok=True)
    tf_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    nih_ids = [f"20000{k}_00001" for k in range(4)]
    for cid in nih_ids:
        img = sitk.ReadImage(str(img_dir / f"{cid}_0000.nii.gz"))
        shape = sitk.GetArrayFromImage(img).shape

        empty = np.zeros(shape, dtype=np.uint8)
        out = sitk.GetImageFromArray(empty)
        out.CopyInformation(img)
        sitk.WriteImage(out, str(cnn_dir / f"{cid}.nii.gz"), useCompression=True)

        noisy = np.zeros(shape, dtype=np.uint8)
        for _ in range(n_blobs):  # small foreground blobs -> false positives (no reference lesion)
            cz, cy, cx = (rng.integers(10, shape[0] - 10), rng.integers(10, shape[1] - 10),
                         rng.integers(10, shape[2] - 10))
            zz, yy, xx = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]),
                                     indexing="ij")
            blob = ((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2) <= 6 ** 2  # r=6vox, spacing (1,1,1.5) -> >100mm3
            noisy[blob] = 1
        out2 = sitk.GetImageFromArray(noisy)
        out2.CopyInformation(img)
        sitk.WriteImage(out2, str(tf_dir / f"{cid}.nii.gz"), useCompression=True)

    return cnn_dir, tf_dir


def _make_occlusion_predictions(workdir, base_cnn, base_tf):
    """Degrade base predictions per shell/arm with a designed pattern: transformer degrades
    much more than CNN in the far (40-80mm) shell -- the exact contrast H2's decisive-shell
    test should detect."""
    import SimpleITK as sitk
    from scipy import ndimage

    shells = [(10, 20), (20, 40), (40, 80)]
    erosion = {
        ("cnn", (10, 20)): 0, ("cnn", (20, 40)): 0, ("cnn", (40, 80)): 1,
        ("tf", (10, 20)): 0, ("tf", (20, 40)): 1, ("tf", (40, 80)): 3,
    }
    for arm, base_dir in (("cnn", base_cnn), ("tf", base_tf)):
        for lo, hi in shells:
            out_dir = workdir / f"pred_{arm}" / f"shell_{lo}_{hi}"
            out_dir.mkdir(parents=True, exist_ok=True)
            it = erosion[(arm, (lo, hi))]
            for base_f in sorted(base_dir.glob("*.nii.gz")):
                img = sitk.ReadImage(str(base_f))
                arr = sitk.GetArrayFromImage(img)
                deg = ndimage.binary_erosion(arr, iterations=it).astype(np.uint8) if it else arr
                out = sitk.GetImageFromArray(deg)
                out.CopyInformation(img)
                sitk.WriteImage(out, str(out_dir / base_f.name), useCompression=True)


if __name__ == "__main__":
    main()
