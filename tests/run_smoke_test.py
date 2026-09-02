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

        print("\n== deduplicate.py ==")
        r = run(["scripts/data/deduplicate.py", "--data-root", str(data_root),
                "--out", "splits/cohort.csv"], cwd=repo)
        check("deduplicate.py exits 0", r.returncode == 0, r.stderr[-500:])
        cohort = pd.read_csv(repo / "splits" / "cohort.csv")
        check("dedup drops exactly the 1 injected duplicate", len(cohort) == 34, f"got {len(cohort)} rows")
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
        pred_cnn_shells = tmp / "occ_pred_cnn"
        pred_tf_shells = tmp / "occ_pred_tf"
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

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'=' * 60}\n{len(passed)} passed, {len(failures)} failed\n{'=' * 60}")
    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


def _make_nih_predictions(data_root, out_dir):
    """All-background CNN predictions vs. transformer predictions seeded with a few small
    speckle blobs (>= 100mm^3 each, per the frozen FP volume threshold) on the NIH negatives --
    a designed higher-FP-rate pattern for the harness to detect."""
    import SimpleITK as sitk
    from scipy import ndimage

    img_dir = data_root / "panorama" / "images"
    cnn_dir, tf_dir = out_dir / "pred_cnn", out_dir / "pred_tf"
    cnn_dir.mkdir(parents=True, exist_ok=True)
    tf_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(99)
    nih_ids = [f"20000{k}_00001" for k in range(4)]
    for cid in nih_ids:
        img = sitk.ReadImage(str(img_dir / f"{cid}_0000.nii.gz"))
        shape = sitk.GetArrayFromImage(img).shape

        empty = np.zeros(shape, dtype=np.uint8)
        out = sitk.GetImageFromArray(empty)
        out.CopyInformation(img)
        sitk.WriteImage(out, str(cnn_dir / f"{cid}.nii.gz"), useCompression=True)

        noisy = np.zeros(shape, dtype=np.uint8)
        for _ in range(3):  # a few small foreground blobs -> false positives (no reference lesion)
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
