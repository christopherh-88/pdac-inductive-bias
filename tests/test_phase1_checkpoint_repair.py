#!/usr/bin/env python
"""Test step4b_repair_inflated_checkpoints in scripts/verify/phase1_train_lite.py.

Real incident this guards against: _TimeBoxedMixin.on_epoch_end() used to call
self.save_checkpoint() AFTER super().on_epoch_end() had already incremented self.current_epoch.
nnU-Net's own save_checkpoint() stores 'current_epoch': self.current_epoch + 1 -- correct only
when called (as nnU-Net itself does) BEFORE that increment. Calling it after double-counted the
increment, so every checkpoint_final.pth this kernel wrote had current_epoch one epoch ahead of
what self.logger actually had entries for. The first --c resume from such a checkpoint crashed
immediately with IndexError inside nnU-Net's own logger. See preregistration/DEVIATIONS.md,
"Phase 1 training: checkpoint off-by-one crashed the first real resume".

step4b_repair_inflated_checkpoints() self-heals this: on every session start, for each arm, if
a checkpoint_final.pth's current_epoch does not match its logger's actual entry count, it is
corrected in place. This test exercises that function directly against real checkpoint files
(real torch.save/torch.load, not stubbed) so a regression here cannot silently ship again.

phase1_train_lite.py is a Kaggle-kernel script, not an importable package: its module level
does OUT.mkdir(parents=True) against a hardcoded /kaggle/working path, which is neither valid
nor desirable to execute on a dev machine. This test execs the file's source with that one path
patched to a temp directory (the only line touched) rather than importing it directly, so the
real function body under test is unmodified.

Needs real torch (to build/read genuine checkpoint dicts) -- unlike test_trainers.py, which
avoids that need by stubbing nnU-Net entirely, there is no way to exercise a real torch.save /
torch.load round trip without it. If this interpreter has no torch, re-execs under
.venv_kaggle_run2 (if present); otherwise skips with an explicit message rather than failing.

Run standalone:
    python tests/test_phase1_checkpoint_repair.py
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).parents[1]
SCRIPT = REPO / "scripts" / "verify" / "phase1_train_lite.py"


def _ensure_torch():
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    venv_python = REPO / ".venv_kaggle_run2" / "bin" / "python3"
    if venv_python.exists() and Path(sys.executable) != venv_python:
        os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])
    print("SKIPPED: no torch available in this interpreter and no "
          f".venv_kaggle_run2 found at {venv_python} -- cannot round-trip a real checkpoint "
          "without it.")
    sys.exit(0)


def _load_repair_fn():
    """exec phase1_train_lite.py's source with its hardcoded /kaggle/working OUT path patched
    to a temp dir, so module-level `OUT.mkdir(parents=True)` doesn't touch the real filesystem
    root. Everything else in the file runs unmodified."""
    src = SCRIPT.read_text()
    marker = 'OUT = Path("/kaggle/working/phase1_train")'
    if marker not in src:
        raise RuntimeError(
            f"expected line {marker!r} not found in {SCRIPT} -- it may have moved or been "
            "reworded; update this test's patch target to match.")
    tmp_root = tempfile.mkdtemp(prefix="phase1_train_lite_test_")
    src = src.replace(marker, f'OUT = Path({tmp_root!r})', 1)
    ns = {"__name__": "phase1_train_lite_under_test", "__file__": str(SCRIPT)}
    exec(compile(src, str(SCRIPT), "exec"), ns)
    return ns["step4b_repair_inflated_checkpoints"]


def _write_checkpoint(path: Path, current_epoch: int, n_logged_epochs: int):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "network_weights": {},
        "optimizer_state": {},
        "logging": {"mean_fg_dice": [0.1 * i for i in range(n_logged_epochs)],
                    "ema_fg_dice": [0.1 * i for i in range(n_logged_epochs)]},
        "_best_ema": None,
        "current_epoch": current_epoch,
        "trainer_name": "nnUNetTrainer_TierALite",
    }
    torch.save(ckpt, path)


def main():
    _ensure_torch()
    import torch

    repair = _load_repair_fn()
    dataset_name = "Dataset601_PDACTierALite"
    arm_specs = [
        ("nnUNetTrainer_TierALite", "nnUNetResEncUNetMPlans"),   # inflated (the real incident)
        ("nnUNet_PrimusV2_TierALite", "nnUNetPlans"),             # already correct
        ("nnUNetTrainer_NoCheckpointYet", "nnUNetPlans"),         # no checkpoint_final.pth at all
    ]

    failures, passed = [], []

    def check(name, cond, detail=""):
        (passed if cond else failures).append(name)
        print(("  PASS  " if cond else "  FAIL  ") + name + (f"  {detail}" if not cond else ""))

    with tempfile.TemporaryDirectory(prefix="phase1_checkpoint_repair_") as results_dir:
        results_dir = Path(results_dir)

        def ckpt_path(trainer, plans):
            return (results_dir / dataset_name / f"{trainer}__{plans}__3d_fullres" /
                    "fold_0" / "checkpoint_final.pth")

        # Real incident: 117 saved (double-counted) vs. 116 actually logged.
        _write_checkpoint(ckpt_path(*arm_specs[0]), current_epoch=117, n_logged_epochs=116)
        # Correct: nothing to repair.
        _write_checkpoint(ckpt_path(*arm_specs[1]), current_epoch=67, n_logged_epochs=67)
        # arm_specs[2]: deliberately no file written.

        repair(str(results_dir), dataset_name, arm_specs)

        fixed = torch.load(ckpt_path(*arm_specs[0]), map_location="cpu", weights_only=False)
        check("an inflated current_epoch (saved 117, logged 116) is corrected to match the "
              "logger's actual entry count", fixed["current_epoch"] == 116,
              f"got {fixed['current_epoch']}")

        untouched = torch.load(ckpt_path(*arm_specs[1]), map_location="cpu", weights_only=False)
        check("an already-correct checkpoint (saved 67, logged 67) is left unchanged",
              untouched["current_epoch"] == 67, f"got {untouched['current_epoch']}")

        # No exception means step4b_repair_inflated_checkpoints() correctly skipped the missing
        # checkpoint instead of crashing on a session-1-style attach with no prior training.
        check("a missing checkpoint_final.pth (nothing to repair yet) is silently skipped",
              not ckpt_path(*arm_specs[2]).exists())

    print(f"\n{len(passed)} passed, {len(failures)} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
