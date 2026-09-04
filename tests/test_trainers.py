#!/usr/bin/env python
"""Test the custom trainers against a stubbed nnU-Net.

The three trainer modules in src/trainers/ gate all 25 GPU runs, and every bug in them is
expensive: an identity ablation that removed the wrong blocks, a seed replicate that silently
reproduced the default run, or a wall-clock stop that let nnU-Net mark an unfinished run as
finished would each produce results that look fine and are wrong.

Installing nnU-Net to test them needs a GPU-class environment, so instead this stubs the
handful of nnU-Net symbols they touch — `nnUNetTrainer`, `recursive_find_python_class`, and
the package layout the class finder walks — and exercises the study-specific logic on top:
class generation, the epoch override landing *before* the LR schedule is built, the time
budget raising rather than returning, and the checkpoint being on disk when it does.

Run standalone, or as the last section of tests/run_smoke_test.py:

    python tests/test_trainers.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parents[1]

STUB = '''
"""Minimal stand-in for the parts of nnunetv2 that src/trainers touches."""
import os, sys, types

pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class nnUNetTrainer:
    """Records what the real trainer would do, so the tests can assert on the order."""

    def __init__(self):
        self.num_epochs = 1000
        self.current_epoch = 0
        self.save_every = 50
        self.output_folder = os.environ["STUB_OUTPUT_FOLDER"]
        self.log = []
        self.network = None
        self.optimizer = None
        self.lr_scheduler = None
        self.epochs_at_optimizer_build = None

    def initialize(self):
        self.log.append("base.initialize")
        # The real configure_optimizers() builds a poly LR schedule against num_epochs, so
        # whatever num_epochs is at THIS moment is what the schedule spans.
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

    def configure_optimizers(self):
        self.epochs_at_optimizer_build = self.num_epochs
        return ("optimizer", "scheduler")

    def on_train_start(self):
        self.log.append("base.on_train_start")

    def on_epoch_start(self):
        self.log.append("base.on_epoch_start")

    def on_epoch_end(self):
        self.log.append("base.on_epoch_end")
        self.current_epoch += 1

    def save_checkpoint(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write(f"epoch {self.current_epoch}")
        self.log.append(f"save:{os.path.basename(filename)}")

    def print_to_log_file(self, *a, **k):
        self.log.append("log:" + " ".join(str(x) for x in a))


class _PrimusLike(nnUNetTrainer):
    """Stands in for a PrimusV2 trainer: same interface, different class."""


def recursive_find_python_class(folder, class_name, current_module=None):
    if "Primus" in class_name and "Identity" not in class_name:
        return type(class_name, (_PrimusLike,), {})
    return None
'''


def build_stub(root: Path):
    """A package tree matching what recursive_find_python_class walks."""
    pkg = root / "nnunetv2"
    (pkg / "training" / "nnUNetTrainer" / "variants").mkdir(parents=True)
    (pkg / "utilities").mkdir(parents=True)
    for p in (pkg, pkg / "training", pkg / "training" / "nnUNetTrainer",
              pkg / "training" / "nnUNetTrainer" / "variants", pkg / "utilities"):
        (p / "__init__.py").write_text("")
    (pkg / "training" / "nnUNetTrainer" / "nnUNetTrainer.py").write_text(STUB)
    (pkg / "utilities" / "find_class_by_name.py").write_text(
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import "
        "recursive_find_python_class  # noqa: F401\n")
    return root


CHECKS = r'''
import os, sys, time
from pathlib import Path

failures, passed = [], []
def check(name, cond, detail=""):
    (passed if cond else failures).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if not cond else ""))

import src.trainers.seed_variant_trainers as seedmod
import src.trainers.budget_trainers as budget

# --- seed replicates ------------------------------------------------------------------
check("a trainer class is generated for every configured replicate seed",
      {"nnUNetTrainer_seed1", "nnUNetTrainer_seed2"} <= set(seedmod.__all__),
      str(seedmod.__all__))
check("the transformer arm gets its own seed variants",
      {"nnUNet_Primus_seed1_Trainer", "nnUNet_Primus_seed2_Trainer"} <= set(seedmod.__all__),
      str(seedmod.__all__))
s1, s2 = seedmod.nnUNetTrainer_seed1, seedmod.nnUNetTrainer_seed2
check("each replicate is a distinct class, so nnU-Net writes it to its own results directory",
      s1 is not s2 and s1.__name__ != s2.__name__)
check("each replicate carries the seed it was generated for",
      (s1.replicate_seed, s2.replicate_seed) == (1, 2))

# The property that matters is that the seed is set BEFORE the base builds the network, so
# it is tested by its consequence: identical seeds must reproduce a draw, different ones must
# not. Asserting on call order would pass even if the seeding had no effect.
import numpy as np
a = s1(); a.initialize(); draw_a = np.random.rand()
b = s1(); b.initialize(); draw_b = np.random.rand()
c = s2(); c.initialize(); draw_c = np.random.rand()
check("the same replicate seed reproduces the same initialization", draw_a == draw_b)
check("different replicate seeds give different initializations", draw_a != draw_c)

# --- the wall-clock budget ------------------------------------------------------------
check("the budget suffix is _budget with no epoch override", budget.SUFFIX == "_budget")
check("a budgeted variant exists for every arm and control",
      {"nnUNetTrainer_budget", "nnUNet_PrimusV2_Identity_Trainer_budget",
       "nnUNetTrainer_seed1_budget"} <= set(budget.__all__), str(budget.__all__))

Timed = budget.nnUNetTrainer_budget
tt = Timed(); tt.on_train_start()
check("the budget trainer tightens the checkpoint interval",
      tt.save_every == int(os.environ["PDAC_SAVE_EVERY"]), str(tt.save_every))

tt.on_epoch_start(); tt.on_epoch_end()
check("a normal epoch end does not stop training", tt.current_epoch == 1)

# The projection: an epoch that took a big share of the budget must stop the run BEFORE the
# budget is spent, not after. A session killed at the hard cap never commits its output, so
# overshooting by one epoch costs everything since the last save.
proj = budget.nnUNetTrainer_budget(); proj.on_train_start()
budget_s = budget.MAX_HOURS * 3600
proj.on_epoch_start()
proj._epoch_t0 = time.time() - budget_s * 0.5      # that epoch took half the budget
proj._budget_t0 = time.time() - budget_s * 0.6     # and 60% of the budget is gone
stopped_early = False
try:
    proj.on_epoch_end()
except budget.TimeBudgetReached:
    stopped_early = True
check("it stops one epoch EARLY when the next epoch would not fit, rather than one epoch late",
      stopped_early)

# Pretend the session started long ago: the next epoch end must stop.
tt._budget_t0 = time.time() - 3600 * 24
raised = None
try:
    tt.on_epoch_start(); tt.on_epoch_end()
except budget.TimeBudgetReached as e:
    raised = e
check("the trainer stops once the wall-clock budget is spent", raised is not None)
check("it raises rather than returning, so nnU-Net never runs on_train_end and never writes "
      "checkpoint_final for an unfinished run", isinstance(raised, RuntimeError))
ckpt = Path(os.environ["STUB_OUTPUT_FOLDER"]) / "checkpoint_latest.pth"
check("a resume checkpoint is on disk when it stops", ckpt.exists())
check("the checkpoint records the epoch reached, so --c resumes there",
      ckpt.read_text().strip() == f"epoch {tt.current_epoch}", ckpt.read_text())

# A finished run must not be stopped: current_epoch == num_epochs means on_train_end should run.
done = Timed(); done.on_train_start()
done.current_epoch, done.num_epochs = 999, 1000
done._budget_t0 = time.time() - 3600 * 24
try:
    done.on_epoch_start(); done.on_epoch_end()    # advances to 1000 == num_epochs
    stopped = False
except budget.TimeBudgetReached:
    stopped = True
check("the final epoch is allowed to complete so the run can be marked finished", not stopped)

print(f"\n{len(passed)} passed, {len(failures)} failed")
sys.exit(1 if failures else 0)
'''

OVERRIDE_CHECKS = r'''
import os, sys
import src.trainers.budget_trainers as budget

failures, passed = [], []
def check(name, cond, detail=""):
    (passed if cond else failures).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if not cond else ""))

n = int(os.environ["PDAC_EPOCHS"])
check("an epoch override changes the class name, so a short run cannot overwrite a full one",
      budget.SUFFIX == f"_budget{n}ep", budget.SUFFIX)
check("the override reaches every arm",
      f"nnUNetTrainer_budget{n}ep" in budget.__all__, str(budget.__all__))

t = getattr(budget, f"nnUNetTrainer_budget{n}ep")()
t.initialize()
check("the shortened epoch count is applied", t.num_epochs == n, str(t.num_epochs))
check("it is applied BEFORE the optimizer and LR schedule are built, so the poly schedule "
      "spans the run that is actually happening", t.epochs_at_optimizer_build == n,
      f"schedule built for {t.epochs_at_optimizer_build} epochs")

print(f"\n{len(passed)} passed, {len(failures)} failed")
sys.exit(1 if failures else 0)
'''


def main():
    tmp = Path(tempfile.mkdtemp(prefix="pdac_trainers_"))
    try:
        build_stub(tmp)
        out_folder = tmp / "results" / "fold_0"
        out_folder.mkdir(parents=True)

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(tmp), str(REPO)])
        env["STUB_OUTPUT_FOLDER"] = str(out_folder)
        env["PDAC_MAX_HOURS"] = "1.0"
        env["PDAC_SAVE_EVERY"] = "5"
        env["PRIMUS_BASE_TRAINER"] = "nnUNet_PrimusV2M_Trainer"
        env.pop("PDAC_EPOCHS", None)

        print("== trainers: seed replicates and the wall-clock budget ==")
        r1 = subprocess.run([sys.executable, "-c", CHECKS], env=env, cwd=str(REPO))

        print("\n== trainers: the epoch override (a deviation) ==")
        env["PDAC_EPOCHS"] = "50"
        r2 = subprocess.run([sys.executable, "-c", OVERRIDE_CHECKS], env=env, cwd=str(REPO))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if r1.returncode or r2.returncode:
        print("\nTRAINER CHECKS FAILED")
        sys.exit(1)
    print("\nALL TRAINER CHECKS PASSED")


if __name__ == "__main__":
    main()
