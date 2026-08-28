# Kaggle notebooks

Three notebooks. One prepares, one runs a single task and is launched many times
**concurrently**, one analyses.

```
00_prepare    no GPU   once      download, dedup, strata, folds, freeze, preprocess
01_run        GPU      ×25       ONE task: train (resumable), predict, occlude, negative control
02_analyze    no GPU   any time  every hypothesis, six figures, the verdict memo
```

They are **generated**. Edit `nb_common.py` (bootstrap, disk guard, environment, trainers) or
`build_notebooks.py` (contents), then `python notebooks/build_notebooks.py`, and commit both
the source and the regenerated `.ipynb`. The smoke test fails if they drift apart.

## Concurrency

`01_run` is the unit of parallelism. Each copy:

- reads only the preparation output and its own previous session,
- writes only its own paths — `preds/<arm>/<pred_key>/`, its own results directory, its own
  run-log file named after the task,
- needs no lock, no shared counter, and no ordering.

So you can launch as many at once as your Kaggle quota allows. `python
scripts/training/tasks.py` lists all 25 task names; fork the notebook once per task and change
the single `TASK` line.

Two collisions would break a fan-out silently, and both are tested in
`tests/run_smoke_test.py`: no two tasks may share an nnU-Net results directory, and no two may
share a prediction directory. That is why seed replicates carry the seed in their path — a
replicate of fold 0 and the default run of fold 0 are different runs of the same fold, and
merging them would give some cases two predictions from one arm and report a seed variance of
zero.

## The 20 GB limit

Kaggle refuses to save a notebook output over 20 GB, and it does not warn — the session just
ends without saving. The design stays under it structurally, not by luck:

**The cap is per notebook output, and the worker is launched once per run.** Twenty-five runs
are twenty-five separate ~2 GB outputs, not one 50 GB pile. Fanning out for concurrency is the
same move that fixes storage.

On top of that:

| | Location | Counts against the cap | Holds |
| --- | --- | --- | --- |
| Output | `/kaggle/working` | **yes** | frozen config, splits, one checkpoint, predictions, results |
| Scratch | `/kaggle/temp` | no (wiped at session end) | raw images, `nnUNet_preprocessed`, occluded volumes |

- `nnUNet_preprocessed` and the raw images never enter the output: they are large and exactly
  regenerable, which is the definition of what belongs in scratch.
- `prune_checkpoints()` deletes `checkpoint_best.pth` always (this study evaluates
  `checkpoint_final.pth`; the frozen schedule has no early stopping) and `checkpoint_latest.pth`
  once final exists. Left alone that is three checkpoints per run for no gain.
- Occlusion is done **per fold, by the worker that owns that fold** — occluding the whole
  cohort centrally would write three more copies of every image.
- `enforce_output_budget()` runs after everything that writes. Over 17 GB it *frees* space in
  increasing order of regret — checkpoints the study never reads, then softmax dumps, then
  resume checkpoints for runs that already finished — and only raises if that is not enough.
  Raising alone would not help: Kaggle refuses the save regardless of what the notebook thinks,
  so a guard that only complains still costs you the session.
- `disk_report()` prints the split so you can see which limit a directory counts against.

For a cohort whose preprocessed data does not fit — which is most real cohorts — `00_prepare`
says so with the measured number and points at `scripts/kaggle/pack_for_kaggle.py`. That packs
a directory into size-bounded parts with a manifest and the Kaggle dataset metadata, uploaded
once from a machine that has the disk. **An attached dataset is not subject to the 20 GB cap**,
so this is the supported route for anything large: the worker attaches it, `restore_chunks()`
unpacks it, and preprocessing is skipped entirely.

## The 12-hour limit

A session killed at the hard cap **never commits its output** — you lose everything since the
last save, not just the tail. That asymmetry drives the whole design:

- `src/trainers/budget_trainers.py` stops after `PDAC_MAX_HOURS` (the worker sets this to
  whatever is left minus `RESERVE_HOURS`).
- The check is **"would one more epoch fit?"**, not "have I spent the budget?", using the
  longest recent epoch times a 1.25 margin. Checking the latter would overshoot by up to a
  full epoch, which is exactly the thing that costs a session.
- It checkpoints every 5 epochs rather than nnU-Net's default 50, so a session that is killed
  anyway loses minutes, not hours.
- It **raises** rather than returning, because returning normally would make nnU-Net run
  `on_train_end()` and write `checkpoint_final.pth` — marking an unfinished run as finished.
- Preprocessing refuses to start with under 4 h left: it is not interruptible, and its output
  lives in scratch, so a session that dies inside it has nothing to show.
- Inference passes use `--continue_prediction`, so a shell interrupted halfway resumes instead
  of redoing an hour of work.

Re-launch the same `TASK` with the previous session attached and it resumes from
`checkpoint_latest.pth` via `--c`. Re-launching a task that already finished skips training
entirely and does inference only — that is the intended way to handle a task whose occlusion
pass does not fit in the same session as its training.

## What is actually guaranteed, and what is an estimate

Worth being precise about, because "there is a guard" and "it cannot happen" are different
claims.

**Structurally bounded — these hold regardless of the data or the card:**

- Preprocessed data and raw images are in scratch by construction and cannot enter the capped
  output.
- Each worker writes only its own results directory and its own predictions. Nothing
  accumulates across tasks in one output, because each task is its own output.
- nnU-Net overwrites `checkpoint_latest`/`best` rather than accumulating, so checkpoints per
  run are bounded at three during training and one after pruning.
- No training command requests `--npz`, so no softmax dumps are written at all.
- `enforce_output_budget()` frees space rather than only reporting, so the common overrun is
  self-correcting.

**Estimated, and worth checking on your first real session:**

- **Checkpoint size.** Weights plus SGD momentum is roughly 8 bytes per parameter, so ResEnc M
  lands near 250 MB and a ~150 M-parameter Primus near 1.2 GB. A worker should peak around
  3–5 GB against a 17 GB budget. That margin is large, but it is arithmetic, not a measurement.
- **Epoch duration.** The early-stop projection needs at least one completed epoch before it
  has anything to project from, so the very first epoch of a fresh run is unprotected. If a
  single epoch is longer than `RESERVE_HOURS`, no projection saves you — lower the batch or
  the preset instead.
- **`RESERVE_HOURS = 2.0`.** It has to cover nnU-Net's end-of-training validation over the
  held-out fold (which runs *inside* training, after the last epoch, and the wall-clock stop
  does not apply to it), plus the occlusion inference, plus saving. Two hours is a starting
  estimate. The first worker session tells you the real number; the right response to running
  short is a second session, not a bigger reserve.

The honest summary: **the 20 GB cap has a wide margin and a self-correcting guard; the 12-hour
cap degrades safely rather than being guaranteed.** Worst realistic case there is a killed
session that loses up to five epochs of training and any in-flight inference, both of which
resume. Attaching a preprocessed pack removes the largest single time risk.

## What still does not fit

Notebook 03 of the old layout printed this and it has not changed: at ~250,000 steps per run
and 25 runs, against roughly 30 GPU-hours a week, **the full study does not fit on Kaggle**.
What fits is every analysis notebook for real, and a complete proof of the training path —
both arms, the identity ablation, the seed variants, the run log, resume — on a subset or a
short schedule.

Any schedule other than the frozen 1000 × 250 is a deviation. Setting `EPOCHS` in `01_run`
prints it, appends it to `preregistration/DEVIATIONS.md`, and routes the run to a results
directory suffixed `_budget<N>ep`, so a short run can never be mistaken for or overwrite a
full one.

Nothing changes off Kaggle: `ON_KAGGLE` is False, paths come from `PDAC_WORK`, `PDAC_SCRATCH`
and `PDAC_DATA`, and the same three notebooks run the frozen schedule on a cluster.

## Order of freezing

Nothing downstream is meaningful if this order breaks:

1. `00_prepare` writes the strata, folds, and source grouping into
   `config/frozen_thresholds.yaml` and `splits/`. Each script refuses to overwrite its own
   section, so a second run errors rather than re-rolling.
2. You tag `prereg-v1` **on your own machine** and push it — Kaggle has no push credentials.
3. The first `01_run` worker measures both arms on one card and freezes the architecture
   record; `record_arch_stats.py` refuses to run twice, so whichever worker arrives first wins
   and the rest read it back.
4. Every later notebook reads those numbers rather than re-deriving them.
5. `bash scripts/check_prereg_tag.sh` fails if any frozen file moved since the tag.

Run step 5 before every analysis session. It catches the one failure mode that would
invalidate everything downstream of it.

## The notebooks run against a *clone*, not your working copy

This is the one thing that catches people out. The notebooks are generated from this repo, but
on Kaggle they `git clone` it — so **an uncommitted or unpushed change is invisible there**,
however current the notebook you uploaded is. The symptom lands far from the cause: an old
test asserting old counts, or a script missing a flag the notebook passes it.

The bootstrap checks for a handful of files that must exist and fails with one clear message
if the clone predates the notebook. Either push, or upload the repo as a dataset — the
bootstrap prefers any attached input containing `config/analysis_config.yaml` over cloning,
which is also how you pin a version or run with Internet off.
