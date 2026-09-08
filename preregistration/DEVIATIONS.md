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

## Phase 1 training: checkpoint off-by-one crashed the first real resume (2026-09-05)

Progress since the previous entry: the fix above (explicit `checkpoint_final.pth` save) let a
full session complete cleanly on both arms via `--c` resume -- CNN (`cnn_resenc_m`) reached
epoch 117/300 (pseudo-dice climbing from ~0.24 at epoch 0 to a ~0.42-0.45 plateau from roughly
epoch 40 onward), and the transformer (`transformer_primusv2s`) reached epoch 67/300 (pseudo-dice
far noisier and lower, ~0.00-0.17, consistent with a heavier architecture learning much more
slowly on only 25 real training cases). Both were genuine time-budget stops (`returncode: 0`),
not crashes.

The *next* session's `--c` resume then crashed immediately on both arms (`IndexError: list index
out of range` inside nnU-Net's own `nnunet_logger.get_value`, within seconds of starting the
first post-resume epoch). Root cause: `_TimeBoxedMixin.on_epoch_end` calls
`super().on_epoch_end()` (which increments `self.current_epoch`) and *then* calls
`self.save_checkpoint(...)`. nnU-Net's `save_checkpoint()` itself stores
`'current_epoch': self.current_epoch + 1`, because in nnU-Net's own vanilla `on_epoch_end()` it
is always called *before* that method's trailing increment -- the `+1` is a compensation for
that ordering. Calling it *after* the increment (as the mixin does) double-counts, saving an
epoch number one past what `self.logger`'s per-epoch lists actually contain. On resume, nnU-Net
starts an epoch that was never logged and crashes the moment it tries to read the previous
epoch's value for the EMA pseudo-dice update.

This means the *very first* checkpoint written by the (then-new) explicit-save fix above was
already corrupted this way -- session 2's checkpoint_final.pth for both arms had
`current_epoch` inflated by exactly one epoch (117 saved vs. 116 logged for CNN; 68 saved vs. 67
logged for the transformer). Fixed in `scripts/verify/phase1_train_lite.py` two ways: (1) the
mixin now decrements `self.current_epoch` immediately before the save call, undoing the
already-applied increment so `save_checkpoint()`'s own `+1` lines up correctly; (2) a new
`step4b_repair_inflated_checkpoints` runs at the start of every session and self-heals any
already-shipped checkpoint whose saved `current_epoch` does not match its logger's actual
entry count (rather than requiring the already-corrupted `pdac-tier-a-lite-checkpoints` Dataset
to be manually patched) -- so the existing session-2 checkpoints resume correctly with no data
loss, no manual `.pth` surgery, and no epochs re-trained.

## Phase 1 training complete: both arms reached 300/300 epochs (2026-09-08)

Both arms finished the full `TARGET_EPOCHS_PER_ARM = 300` on Dataset601_PDACTierALite fold 0.
Final epoch-averaged (last 20 epochs) pseudo-dice: CNN (`cnn_resenc_m`, `nnUNetResEncUNetMPlans`)
~0.50 (best single-epoch 0.67), transformer (`transformer_primusv2s`, `nnUNetPlans`/PrimusV2S)
~0.40 (best single-epoch 0.56) -- consistent with the earlier-noted pattern of the heavier
transformer architecture learning more slowly on only 25 real training cases. Both
`checkpoint_final.pth` files were verified directly (`torch.load`, `current_epoch` compared
against the logger's actual entry count) to show exactly `current_epoch=300` with no off-by-one
inflation, confirming the fix above held across every subsequent session.

Getting there required handing the last ~10 epochs of the transformer arm off to a collaborator
(Spreal) once the owning account exhausted its weekly Kaggle GPU quota at 290/300. This surfaced
a new, unrelated issue: the collaborator's first handoff run resumed from epoch 182 instead of
290 and re-trained ground already covered, burning a full ~7h session for zero net progress.
Root cause: Kaggle's kernel input snapshotting can lag behind a just-published dataset version by
more than the few minutes that elapsed between the last checkpoint push and the collaborator
starting their run, so the kernel silently grabbed a stale (pre-session-6) version of the
`pdac-tier-a-lite-checkpoints` input instead of the latest one. There is no code-level fix for
this (it is a property of Kaggle's own dataset-versioning propagation, not of
`phase1_train_lite.py`); the workaround was to have the collaborator explicitly detach and
re-attach the checkpoints dataset input immediately before each run, and to verify the very first
logged epoch number in `log_train_transformer_primusv2s.txt` matched the expected resume point
before letting a session run to completion. The corrected re-run resumed cleanly from epoch 291
and completed the remaining 9 epochs in one short (~36 min) session.

Next: `scripts/verify/phase1_predict_lite.py` (Kaggle GPU inference on the 7 fold-0 held-out
cases for both arms) and `scripts/analysis/phase1_eval_lite.py` (local CPU scoring against ground
truth with the same frozen metrics used elsewhere in this repo) are ready to run against these
final checkpoints -- neither has been run against real data yet, only verified against synthetic
data and a source-level CLI cross-check.
