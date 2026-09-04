# Deviations from prereg-v1

`PREREGISTRATION.md` is frozen at tag `prereg-v1` and is never edited after that tag. Nothing
in the pre-registration or in `config/analysis_config.yaml` is ever changed in place after the
tag; if a frozen value genuinely has to change, it becomes a new tag (`prereg-v2`, …) with the
reason recorded here, and the paper reports both the original and the revised value. Any
clarification, correction, or deviation discovered after the freeze goes here, with a date and
a pointer to the section it concerns.

`scripts/check_prereg_tag.sh` fails if any frozen file differs between the tag and HEAD, so a
deviation cannot pass unnoticed — it either appears here or it breaks the check.

## What counts as a deviation

- Any change to `config/analysis_config.yaml`, `config/frozen_thresholds.yaml`,
  `preregistration/PREREGISTRATION.md`, `splits/splits_final.json`, or
  `splits/fold_assignment.csv` after the tag.
- Training a schedule other than the frozen 1000 × 250 (including any reduced schedule run
  for compute reasons — it is a deviation even when the reason is good, and it is reported
  as one).
- Switching the transformer arm away from PrimusV2, the CNN preset away from the recorded
  one, or the identity control away from the block-replacement defined in
  `src/trainers/primus_identity_trainer.py`.
- Any analysis run under a rule, margin, or threshold not already in the frozen config.
- Any case entering or leaving the cohort after the splits were frozen.

## What is *not* a deviation

- Bug fixes to analysis code that leave every frozen number unchanged (record the commit in
  the run log; the frozen config is what is pre-registered, not the implementation).
- Additional exploratory analyses, as long as they are labelled exploratory in the paper and
  do not replace a pre-registered test.
- Anything the pre-registration itself already provides for: the Primus v1 fallback, the
  contrast axis being downgraded to exploratory by its own rank-stability rule, or the
  optional model-generated-delineation training ablation.

## Log

| Date | Frozen item | Change | Reason | New tag |
| --- | --- | --- | --- | --- |
| — | — | none to date | — | — |

Nothing below has changed a frozen value — each is a clarification, or a diagnostic-only
GPU-access substitute explicitly not used to inform any frozen number. They are recorded here
per the transparency principle above, not because they moved anything in the log table.

## H3 — no inter-rater proxy available (2026-09-02)

Section "H3 — Source shift vs boundary tolerance" names "paired independent delineations" as an
optional input to the boundary-tolerance floor, alongside 1- and 2-voxel morphological
perturbation of the reference. No such pairs exist in this dataset: the panorama_labels repo's
own README states manual PDAC lesion segmentations were each made by **one of two** trained
investigators (single annotator per case, supervised by one expert radiologist), not by both.

This does not change the test or decision rule — morphological perturbation alone was always
the primary method: it just means the "plus any paired independent delineations" clause has no
data to act on, so the boundary-tolerance floor rests on morphological perturbation of the
single reference alone. See `docs/project_description_v2.md` point 4 for the same note in the
supporting narrative (that document is not frozen and was updated directly).

## Architecture record is preliminary, from a Kaggle GPU substitute (2026-09-02)

`config/frozen_thresholds.yaml`'s `architecture` section was frozen from a Kaggle free-tier
P100 kernel (`scripts/verify/phase0_gpu_verify.py`), run as an interim substitute for confirmed
NYU/USC GPU access, solely to clear Phase 0's two-arm smoke-test verification (real end-to-end
train + predict + Dice for both arms and the identity control, on 8 real manual-lesion cases —
`Dataset999_PDACSmoke`, not the full 481-case cohort). It is not the matched-budget Tier A
configuration.

Two specific numbers in that record are diagnostic-only and will be superseded once real Tier A
planning runs on the full cohort:

- **Patch size mismatch between arms:** the CNN (`nnUNetResEncUNetMPlans`) auto-planned a 3D
  fullres patch of `(64, 160, 160)` on this tiny dataset, while the transformer's default-plans
  fingerprint (used for `nnUNet_PrimusV2S_Trainer`) auto-planned `(64, 192, 160)` — the frozen
  record uses the transformer's patch since that is the one PrimusV2's 8×8×8 tokenizer stride
  must divide evenly. Section 5's matched-budget control requires the two arms share one patch
  size; that alignment happens during real Tier A fingerprinting on the full cohort, not here.
- **CNN parameter count not captured:** the verification kernel measured the transformer's
  parameter count directly (25,477,406, incidental to the identity-control ablation's own
  parameter-drop assertion) but did not instrument the CNN arm's parameter count, so
  `architecture.cnn.n_params` is `null`. Filled in at real Tier A standup, not fabricated here.

Both `peak_vram_gb` figures (CNN 7.149 GB, transformer 6.341 GB) and the transformer's
`n_tokens` = 3,840 (token grid 8×24×20) are real measurements from that kernel run and are not
placeholders — they are simply from a 10-iteration smoke run on 8 cases on a P100, not the real
compute tier, and are recorded as such.

## 481 manual delineations vs. 478-case working strata/split cohort (2026-09-02)

`splits/cohort.csv` flags 481 cases `has_manual_lesion=True` (matching README's stated count),
but `splits/strata.csv` and `config/frozen_thresholds.yaml`'s `strata.n_cases`/`splits.n_cases`
both report **478**. This is not a bug: 3 of the 481 (`100598_00001`, `100667_00001`,
`101632_00001`) are non-PDAC cases (`is_pdac=False`, `label=non-PDAC` in `clinical_information.xlsx`)
whose manual label file delineates that non-PDAC finding rather than a PDAC lesion, so they
contain zero voxels matching the PDAC lesion label. Section 4 of the pre-registration already
requires "non-PDAC masses excluded from all PDAC analyses" — these 3 are that exclusion working
as intended, not a data-loss bug.

The exclusion previously happened silently: `kaggle_stream_cohort.py`'s streaming lesion-stats
computation writes no `lesion_stats_raw.csv` row at all when a case's manual label has zero
PDAC-lesion voxels, and `scripts/data/finalize_strata_from_stream.py` only logged rows dropped
as cross-batch duplicates, not cases present in `cohort.csv` but absent from the raw lesion
stats. `finalize_strata_from_stream.py` now detects and prints this reconciliation explicitly
(`kept_ids - set(raw.case_id)`), matching the equivalent warning
`scripts/analysis/compute_strata.py` already gave for its non-streaming path. The frozen
`n_cases: 478` in `config/frozen_thresholds.yaml` was already computed correctly before this
fix — the fix adds visibility, it does not change the frozen numbers.

Separately: `manual_labels/` and `automatic_labels/` are mutually exclusive per case in the real
panorama_labels repo (481 + 1756 = 2,237 in the deduplicated cohort, zero overlap, zero gaps —
verified directly against `splits/cohort.csv`). Each is a single multi-class file (lesion +
vessels + parenchyma + duct + CBD together), not two complementary files to combine — so
`compute_strata.py` and `kaggle_stream_cohort.py` both read the manual-side file for both the
"manual" and "automatic" roles in the CNR ring computation. A version of `compute_strata.py`
that reads `automatic_label` as a separate file crashes on every manually-delineated case, since
that column is empty by construction whenever `manual_label` is populated.

## Phase 1 real-data training preview, from the same Kaggle GPU substitute (2026-09-03)

`scripts/verify/phase1_stage_pool.py` and `scripts/verify/phase1_train_lite.py` extend the
Phase 0 Kaggle P100 substitute (see above) one step further: not just a smoke test, but a real
"fold 0" training curve for both arms (CNN ResEncM, PrimusV2S) on `Dataset601_PDACTierALite` — a
32-case real manual-lesion PANORAMA pool, staged and split 24/8 train/val by
`phase1_stage_pool.py`. This is still not Tier A: 32 cases vs. the pre-registered 478-case
cohort, single P100 vs. the matched-budget dedicated 24-48GB card, and only fold 0 vs. 5 folds.
Nothing here supersedes or informs any frozen threshold; it exists solely as a further
diagnostic preview while confirmed NYU/USC GPU access is pending.

Kaggle's free tier (~30 GPU-hrs/week, ~9-12hr session cap) cannot finish real training in one
sitting, so `phase1_train_lite.py` is written to run incrementally across multiple weekly quota
resets: it checkpoints and exits cleanly partway through training (via a per-arm wall-clock
budget checked at each epoch boundary), ships `nnUNet_preprocessed/`/`nnUNet_results/` back out
as kernel output for repackaging into a persistent `pdac-tier-a-lite-checkpoints` Kaggle Dataset,
and resumes from that checkpoint (`nnUNetv2_train --c`) on a later run.

Two real bugs surfaced and were fixed during the first two multi-hour runs on this kernel,
recorded here since they reflect real infrastructure findings, not preview results:

- Every training-launch attempt silently crash-looped (whole container reset, zero output ever
  printed) at the exact same point. The actual cause: `nnUNetv2_train` is itself a Python
  entry-point process, and CPython fully block-buffers a child process's stdout when it's
  redirected to a pipe rather than a tty — so anything it printed was trapped in its own
  unflushed buffer and lost the moment the container was killed, regardless of how fast the
  parent read from the pipe. Fixed by setting `PYTHONUNBUFFERED=1` in the subprocess environment,
  which finally surfaced real output/tracebacks instead of silence.
- With real output visible, the transformer arm (`nnUNet_PrimusV2_TierALite`) was then observed
  to be genuinely training (decreasing loss) but got hard-killed mid-epoch by the *outer*
  subprocess timeout, because the *inner* clean-exit budget check only runs once per epoch, and a
  full 250-iteration epoch (with `nnUNet_n_proc_DA=0`, chosen earlier to avoid a suspected
  CUDA-after-fork crash) took longer than the budget-check interval. Fixed by capping
  `num_iterations_per_epoch`/`num_val_iterations_per_epoch` to 50/10 in the time-boxed trainer
  mixin, so the clean-exit check fires often enough to guarantee a checkpointed stop regardless
  of per-iteration speed.

The CNN arm (`cnn_resenc_m`) completed 19 real epochs on the first successful run (pseudo-dice
improving to ~0.39) and exited cleanly; the transformer arm's first clean run is pending the
next Kaggle GPU-quota reset.

A third bug surfaced once that first session's output was inspected: nnU-Net only writes
`checkpoint_latest.pth` every `save_every` (default 50) epochs and `checkpoint_final.pth` only
at the true end of training, so a session stopping at epoch 19 left only `checkpoint_best.pth`
on disk. `nnUNetv2_train --c` (via `maybe_load_checkpoint`) only ever looks for
`checkpoint_final.pth`/`checkpoint_latest.pth`, never `checkpoint_best.pth` -- so the next
session's `--c` would have silently found nothing and restarted from scratch, discarding all 19
epochs. Fixed by having `_TimeBoxedMixin.on_epoch_end` explicitly save `checkpoint_final.pth`
before exiting. For the already-completed first session (which predates this fix), the fold_0
`checkpoint_best.pth` for both arms was copied to `checkpoint_final.pth` by hand before
repackaging as the `pdac-tier-a-lite-checkpoints` Dataset -- the underlying saved state dict is
identical regardless of which filename `save_checkpoint()` is given, so this recovers the same
progress the code-level fix would have produced automatically.
