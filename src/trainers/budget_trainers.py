"""Trainers that fit a bounded session: a wall-clock stop, frequent checkpoints, and an
optional (deviating) epoch override.

A Kaggle session is killed at 12 hours with no warning and no chance to write anything. A
pre-registered run is 1000 x 250 steps, so it spans many sessions, and the only thing that
makes that survivable is stopping *before* the kill with a checkpoint on disk. Three knobs,
all from the environment so nothing has to be edited to change them:

  PDAC_MAX_HOURS    stop cleanly after this many hours of training (default 10.5)
  PDAC_SAVE_EVERY   checkpoint every N epochs (default 5; nnU-Net's own default is 50, which
                    on a 12-hour session can throw away hours of work)
  PDAC_EPOCHS       run this many epochs instead of the frozen schedule — a DEVIATION

The budget is checked at epoch boundaries, which is the only place a checkpoint is consistent.
Checking "have I already spent the budget?" would therefore overshoot by up to a full epoch —
and on Kaggle an overshoot is not a small cost, because a session killed at the hard cap never
commits its output and loses everything since the last save. So the check is instead "would
one more epoch take me past the budget?", using the longest recent epoch as the estimate. It
stops up to one epoch early rather than up to one epoch late.

Stopping is a raised `TimeBudgetReached`, deliberately *not* a clean return: nnU-Net writes
`checkpoint_final.pth` and runs validation in `on_train_end()`, so returning normally would
mark an unfinished run as finished. The exception skips that, leaves `checkpoint_latest.pth`
as the resume point, and the caller catches it and reports a pause. `nnUNetv2_train --c` then
picks up exactly where it stopped.

Class names carry what actually differs, because nnU-Net keys its results directory off the
trainer class name and two runs that differ must never share a directory:

  <Base>_budget          time-budgeted only — same schedule, same model, NOT a deviation
  <Base>_budget<N>ep     epoch count overridden to N — a deviation, its own directory

Generated for every arm and control: the CNN trainer, the Primus base named by
PRIMUS_BASE_TRAINER, the H2b identity trainer, and each seed replicate variant.

Usage:
  export PDAC_MAX_HOURS=10.5 PRIMUS_BASE_TRAINER=nnUNet_PrimusV2M_Trainer
  nnUNetv2_train 501 3d_fullres 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_budget --c
"""
import os
import time
from pathlib import Path

import yaml
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2

from src.trainers.primus_identity_trainer import nnUNet_PrimusV2_Identity_Trainer
from src.trainers import seed_variant_trainers

_CFG = yaml.safe_load(
    open(Path(__file__).parents[2] / "config" / "analysis_config.yaml"))["training"]
_FROZEN_EPOCHS = _CFG["epochs"]

MAX_HOURS = float(os.environ.get("PDAC_MAX_HOURS", 10.5))
# Epoch durations drift (validation, disk contention), so the projection carries a margin.
EPOCH_MARGIN = float(os.environ.get("PDAC_EPOCH_MARGIN", 1.25))
SAVE_EVERY = int(os.environ.get("PDAC_SAVE_EVERY", 5))
EPOCHS = os.environ.get("PDAC_EPOCHS")
EPOCHS = int(EPOCHS) if EPOCHS else None

SUFFIX = "_budget" + (f"{EPOCHS}ep" if EPOCHS else "")


class TimeBudgetReached(RuntimeError):
    """Raised to stop training before the session is killed, with a checkpoint on disk.

    Not an error: the run is paused, not failed. Re-launching the same command with --c
    resumes from `checkpoint_latest.pth`.
    """


def _make(base, name):
    def on_train_start(self):
        base.on_train_start(self)
        self._budget_t0 = time.time()
        self._epoch_t0 = None
        self._epoch_seconds = []
        self.save_every = SAVE_EVERY
        self.print_to_log_file(
            f"[budget] {name}: stopping after {MAX_HOURS} h, checkpoint every {SAVE_EVERY} "
            f"epochs, {self.num_epochs} epochs total"
            + (f" (DEVIATION: frozen schedule is {_FROZEN_EPOCHS})" if EPOCHS else ""))

    def on_epoch_start(self):
        base.on_epoch_start(self)
        self._epoch_t0 = time.time()

    def on_epoch_end(self):
        # super() first: it writes the epoch's logs and nnU-Net's own periodic checkpoint, and
        # advances current_epoch, so the checkpoint written below resumes at the right place.
        base.on_epoch_end(self)
        now = time.time()
        if getattr(self, "_epoch_t0", None):
            self._epoch_seconds.append(now - self._epoch_t0)
            self._epoch_seconds = self._epoch_seconds[-5:]

        elapsed_h = (now - self._budget_t0) / 3600.0
        # The longest recent epoch, not the mean: the question is whether the NEXT one fits,
        # and being wrong in that direction costs the whole session.
        next_epoch_h = (max(self._epoch_seconds) * EPOCH_MARGIN / 3600.0
                        if self._epoch_seconds else 0.0)
        if self.current_epoch >= self.num_epochs:
            return                      # finished: let on_train_end run and mark it final
        if elapsed_h + next_epoch_h >= MAX_HOURS:
            self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_latest.pth"))
            self.print_to_log_file(
                f"[budget] stopping at epoch {self.current_epoch} of {self.num_epochs}: "
                f"{elapsed_h:.2f} h spent, next epoch projected at {next_epoch_h:.2f} h, "
                f"budget {MAX_HOURS} h. Checkpoint written.")
            raise TimeBudgetReached(
                f"{name}: paused at epoch {self.current_epoch}/{self.num_epochs} after "
                f"{elapsed_h:.2f} h (next epoch would not fit). Resume with --c.")

    attrs = {"on_train_start": on_train_start, "on_epoch_start": on_epoch_start,
             "on_epoch_end": on_epoch_end, "pdac_max_hours": MAX_HOURS}

    if EPOCHS is not None:
        def initialize(self):
            # BEFORE base.initialize(): configure_optimizers() builds the poly LR schedule
            # against self.num_epochs, so setting it afterwards would leave a short run
            # traversing only the first few percent of the intended schedule — a different
            # experiment again, and a silent one.
            self.num_epochs = EPOCHS
            base.initialize(self)
            self.print_to_log_file(
                f"DEVIATION: {name} runs {EPOCHS} epochs, not the pre-registered "
                f"{_FROZEN_EPOCHS}. Not a study run unless recorded in "
                f"preregistration/DEVIATIONS.md.")
        attrs["initialize"] = initialize
        attrs["num_epochs"] = EPOCHS
        attrs["pdac_epoch_override"] = EPOCHS

    return type(name, (base,), attrs)


_bases = [(nnUNetTrainer, "nnUNetTrainer"),
          (nnUNet_PrimusV2_Identity_Trainer, "nnUNet_PrimusV2_Identity_Trainer")]

_primus_name = os.environ.get("PRIMUS_BASE_TRAINER", "nnUNet_PrimusV2M_Trainer")
_primus = recursive_find_python_class(
    os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), _primus_name,
    current_module="nnunetv2.training.nnUNetTrainer")
if _primus is not None:
    _bases.append((_primus, _primus_name))

for _seed_name in seed_variant_trainers.__all__:
    _bases.append((getattr(seed_variant_trainers, _seed_name), _seed_name))

__all__ = ["TimeBudgetReached", "SUFFIX", "MAX_HOURS", "SAVE_EVERY", "EPOCHS"]
for _base, _name in _bases:
    _cls_name = f"{_name}{SUFFIX}"
    globals()[_cls_name] = _make(_base, _cls_name)
    __all__.append(_cls_name)

if EPOCHS:
    print(f"[DEVIATION] epoch override active: {EPOCHS} instead of the pre-registered "
          f"{_FROZEN_EPOCHS}. These runs write to their own results directory "
          f"(...{SUFFIX}__...) and are not study results until the deviation is recorded.")
