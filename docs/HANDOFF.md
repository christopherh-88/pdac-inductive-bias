# Handoff: GPU / Kaggle training phase

**State as of 2026-09-13:** Phase 0 (freeze) is done. `prereg-v1.1` is tagged and pushed to both
remotes. All pipeline code is written and pipeline-tested end-to-end against a 7-case CPU-lite
dataset (`Dataset601_PDACTierALite`) — every script in the deliverables table runs without
error. **Nothing has been run against the real ~481-case cohort.** That's the entire remaining
gap: no code to write, only GPU time to spend and data to assemble.

Read [README.md](../../../README.md) sections "Where each publication deliverable comes from"
and "Running it" first — this document is a sequenced checklist against that, not a
replacement for it.

## 0. Before anything else

- [ ] Read `notebooks/README.md`. Kaggle mechanics (20GB output cap per session, 12-hour kill,
      weekly GPU quota) shape the whole plan below — don't start `run_tier_a.sh` on Kaggle
      without reading it first.
- [ ] Confirm `prereg-v1.1` is still intact: `bash scripts/check_prereg_tag.sh`. If it fails,
      stop and fix it before training — a silently-drifted frozen config invalidates every
      downstream result.

## 1. Assemble the real cohort (not done yet — this machine only has placeholder paths)

`splits/cohort.csv` currently points at `/data/panorama_placeholder/...`, which doesn't exist.

- [ ] `bash scripts/download/download_panorama.sh /path/to/data` — needs ~400GB free (180GB
      zipped, all 4 batches). **If that disk isn't available (likely on Kaggle),** use the
      streaming alternative instead:
      `scripts/download/kaggle_stream_cohort.py` (one ~50GB Zenodo batch at a time, deletes as
      it goes) → `scripts/data/finalize_cohort_from_stream.py` +
      `finalize_strata_from_stream.py` to produce the same `cohort.csv`/`strata.csv`.
- [ ] `python scripts/data/deduplicate.py --data-root ... --out splits/cohort.csv` (dedupes
      PANORAMA/MSD/NIH by image content)
- [ ] `python scripts/analysis/compute_strata.py --data-root ... --cohort splits/cohort.csv`
      (volume/CNR strata — **this freezes `config/frozen_thresholds.yaml`**)
- [ ] `python scripts/data/make_splits.py --cohort ... --strata ...` then
      `python scripts/data/make_loso_splits.py --exclude-source NIH --emit-combined splits/splits_with_loso.json`
- [ ] **Freeze here:** commit, tag (new tag if `frozen_thresholds.yaml` actually changed vs.
      `prereg-v1.1` — see `scripts/check_prereg_tag.sh` header for the re-freeze workflow),
      push to both `origin` and `upstream`. Nothing below this line runs before the tag exists
      on the remote.
- [ ] `python scripts/data/convert_to_nnunet.py --data-root ... --cohort splits/cohort.csv` —
      builds `Dataset501_PDAC` with `file_ending: ".nii.gz"`.
- [ ] Pack whatever's large and reusable (the preprocessed dataset) for Kaggle upload:
      `scripts/kaggle/pack_for_kaggle.py --src ... --out ... --slug pdac-preprocessed`, then
      `kaggle datasets create -d -r skip`. This has to go in as a Kaggle **Dataset**, not
      notebook output — datasets aren't capped at 20GB, notebook outputs are.

## 2. Week-one verification (smoke test, not a study run)

- [ ] `bash scripts/training/verify_pipeline.sh` — proves both arms train and checkpoint on
      Kaggle's actual GPU/VRAM tier before committing to the full schedule.
- [ ] If `nnUNet_PrimusV2*_Trainer` won't run in the installed nnU-Net commit, fall back to
      Primus v1 per the script's documented rule, and record the gap in
      `preregistration/DEVIATIONS.md` — don't silently substitute.

## 3. Tier A — 10 runs (the main H1 comparison)

- [ ] `python scripts/training/tasks.py` to see the 25 named tasks `01_run` launches against
      (Tier A is the first 10 of these).
- [ ] `bash scripts/training/run_tier_a.sh` — 5 folds × {CNN, transformer}, ~10-20 GPU-days on
      a 24-48GB card. Writes rows to `scripts/training/run_log.csv` automatically via
      `log_run.py` — never hand-edit that file.
- [ ] **Decide and document the reduced schedule now if quota won't cover all 25 runs.** This
      is the single most important open decision for the next person — it's a pre-registration
      deviation, not an implementation detail, so it needs a `DEVIATIONS.md` entry before
      training starts, not after.
- [ ] `python scripts/training/collect_oof.py --results-dir ... --out /preds/cnn` (and `/preds/tf`)
- [ ] `build_per_case_table.py` → `paired_analysis.py` (H1) → `contrast_sensitivity.py`

## 4. Tier B — 15 runs (on top of Tier A)

- [ ] Copy `splits/splits_with_loso.json` into
      `$nnUNet_preprocessed/<Dataset>/splits_final.json` first — this is what makes LOSO folds
      addressable as nnU-Net fold indices 5+.
- [ ] `bash scripts/training/run_tier_b.sh`:
      - LOSO × both arms × 3 sources = 6 runs → `loso_analysis.py` (H3 input)
      - Identity control × 5 folds = 5 runs → `identity_comparison.py` (H2b) — needs its own
        `per_case_metrics_h2b.csv`, separate from the main metrics table
      - Seed replicates × both arms × 2 extra seeds = 4 runs → `build_variance_tables.py` /
        `variance_decomposition.py`

## 5. Occlusion test (H2) on real predictions

- [ ] `occlusion_test.py --stage occlude` per fold → `predict_occlusion.py` (runs
      `nnUNetv2_predict` per shell per arm) → `occlusion_test.py --stage score`. This exact
      chain was pipeline-tested this session on lite/fake data and the `.nii`/`.nii.gz`
      extension-handling bug was fixed centrally in `metrics.py`'s `find_case_file()` — it
      should run against real Tier A checkpoints without further changes.

## 6. NIH negative controls (Deliverable 8)

- [ ] Predict on NIH Pancreas-CT cases with the Tier A CV models (in-domain) and a LOSO model
      (source-shift), then `nih_false_positives.py --pred cnn:in_domain:DIR --pred cnn:source_shift:DIR ...`

## 7. Boundary tolerance, figures, gallery, verdict

- [ ] `boundary_tolerance.py --loso results/loso/loso_dice.csv --out results/h3`
- [ ] `make_figures.py --results results`
- [ ] `failure_gallery.py --per-case results/h1/per_case.csv ...`
- [ ] `verdict_memo.py --results results` — assembles the final per-hypothesis verdict against
      the frozen rules. No `_lite`-directory symlink workaround needed here once real `results/h1`,
      `h2`, `h3`, `loso` exist — that was only a workaround for this session's tiny preview run.

## Standing discipline, don't skip

- Every GPU-touching command needs a row in `scripts/training/run_log.csv` before it starts and
  an update when it ends (`log_run.py start`/`finish`) — a result with no run ID doesn't count.
  `run_tier_a.sh`/`run_tier_b.sh`/`verify_pipeline.sh` do this automatically; a manual LOSO or
  seed-replicate run outside those scripts needs the same two calls by hand.
- Any edit to a frozen file (`config/analysis_config.yaml`, `config/frozen_thresholds.yaml`,
  `preregistration/PREREGISTRATION.md`, `splits/splits_final.json`,
  `splits/fold_assignment.csv`) after a tag is cut is a deviation: record it in
  `preregistration/DEVIATIONS.md` and cut a new tag. Never a silent edit — `check_prereg_tag.sh`
  exists specifically to catch this and should be run periodically, not just once.
- Commits to this repo omit the `Co-Authored-By: Claude` trailer — it caused a full revert once
  before (see repo history / commit norms if picking this convention up mid-stream).
