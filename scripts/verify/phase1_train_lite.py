#!/usr/bin/env python
"""Tier-A-lite GPU training, run on a Kaggle P100 kernel across multiple weekly GPU-quota
resets. NOT Tier A: this trains on Dataset601_PDACTierALite (32 real manual-lesion PANORAMA
cases staged by phase1_stage_pool.py), not the pre-registered 478-case cohort, and on a single
P100 rather than the matched-budget dedicated 24-48GB card Tier A requires. See
preregistration/DEVIATIONS.md. Real Tier A (5 folds x 2 arms, full cohort) remains blocked on
confirmed NYU/USC GPU access.

Purpose: produce a real (not 10-iteration-smoke) "fold 0" training curve for both arms (CNN
ResEncM, PrimusV2S) on this small real pool -- a further diagnostic preview, still not
statistically powered or fold-averaged, while formal GPU access is pending.

Session/quota mechanics: Kaggle GPU sessions cap out around 9-12 wall-clock hours and the free
tier gives ~30 GPU-hrs/week. A single run cannot finish real training in one sitting, so this
kernel:
  1. Attaches the persistent `pdac-tier-a-lite-pool` Dataset as input (no re-download).
  2. On first run, runs real preprocessing (fingerprint + plans + resample) once and ships the
     preprocessed cache back out as kernel output.
  3. On every run, trains both arms for as many epochs as fit in the time budget, stopping
     cleanly with a checkpoint before nnU-Net's own training loop would be killed by the kernel
     time limit, and ships nnUNet_results/ back out as a persistent Dataset.
  4. On a LATER run (once weekly quota resets), that persistent Dataset is attached as input too,
     and training resumes via `nnUNetv2_train --c` (nnU-Net's own continue-from-checkpoint flag)
     picking up mid-training rather than restarting from scratch.

This kernel is idempotent per invocation: it inspects what state it was handed (has
nnUNet_preprocessed/ been done? does a checkpoint already exist?) and does the next incremental
step, rather than assuming a fixed global iteration count.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

OUT = Path("/kaggle/working/phase1_train")
OUT.mkdir(parents=True, exist_ok=True)
NNUNET_COMMIT = "0e495086eb108ff79afe106291e8c15bd2f2bc3a"
DATASET_ID = "601"
DATASET_NAME = "Dataset601_PDACTierALite"

# Kaggle GPU sessions are capped around 9-12 wall-clock hours; stop well short of that so
# nnU-Net's own checkpoint-then-exit path (via --c-compatible epoch boundaries) completes cleanly
# rather than being killed mid-epoch by the kernel's hard timeout. Each arm gets half the budget.
# NOTE: this budget only bounds step5's *training* time (the timer starts inside the trainer, not
# at container boot) -- step0-3 (env setup + preprocessing) add ~20-25min on top, uncounted here.
# With only ~8hrs of real quota left, default to 7hrs so that overhead plus the worst-case
# one-epoch overshoot past each arm's end-of-epoch budget check still lands safely under 8hrs.
SESSION_BUDGET_S = int(os.environ.get("SESSION_BUDGET_S", 7 * 3600))
PER_ARM_BUDGET_S = SESSION_BUDGET_S // 2

# The CNN arm (~130s/epoch) reaches any given epoch count far faster than the transformer arm
# (~230s/epoch). Splitting the session budget evenly wastes CNN time once it's already past this
# target -- that time is better spent on the slower, more diagnostically important transformer arm.
# Once CNN hits TARGET_EPOCHS_PER_ARM it checkpoints and exits early (see _TimeBoxedMixin), and
# main() hands whatever budget it didn't use to the transformer arm for this same session.
TARGET_EPOCHS_PER_ARM = int(os.environ.get("TARGET_EPOCHS_PER_ARM", 300))

def _find_input_dataset(slug):
    """Kaggle's kernel input mount layout has varied across environments (seen: flat
    /kaggle/input/<slug>, and nested /kaggle/input/datasets/<owner>/<slug>); search rather than
    hardcode a single layout so this doesn't silently break again if it changes once more."""
    root = Path("/kaggle/input")
    if (root / slug).exists():
        return root / slug
    for candidate in root.glob(f"**/{slug}"):
        if candidate.is_dir():
            return candidate
    return root / slug  # fall back to the naive guess; callers check .exists() themselves


POOL_INPUT = _find_input_dataset("pdac-tier-a-lite-pool")
CHECKPOINT_INPUT = _find_input_dataset("pdac-tier-a-lite-checkpoints")

PASCAL_COMPAT_TORCH = ("torch==2.4.1", "torchvision==0.19.1",
                       "--index-url", "https://download.pytorch.org/whl/cu121")
CUDA_SANITY_CHECK = (
    "import torch, json\n"
    "assert torch.cuda.is_available(), 'cuda not available'\n"
    "x = (torch.randn(64, 64, device='cuda') @ torch.randn(64, 64, device='cuda')).sum().item()\n"
    "assert x == x, 'nan from cuda matmul'\n"
    "print(json.dumps({'torch_version': torch.__version__, 'device_name': torch.cuda.get_device_name(0),\n"
    "                   'arch_list': torch.cuda.get_arch_list(), 'matmul_ok': True}))\n"
)


def log(msg):
    print(f"[phase1-train] {msg}", flush=True)


def pip_install(args, timeout=900, retries=3, backoff=20):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                            check=True, timeout=timeout)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            last_err = e
            if attempt < retries:
                log(f"WARNING: pip install {args} failed ({e!r}), attempt {attempt}/{retries} "
                    f"-- retrying after {backoff}s")
                time.sleep(backoff)
    raise last_err


def _cuda_sanity():
    r = subprocess.run([sys.executable, "-c", CUDA_SANITY_CHECK], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"CUDA sanity check failed:\n{r.stdout}\n{r.stderr}")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    log(f"CUDA sanity OK: {info}")
    return info


def step0_env():
    log("=== step 0: environment ===")
    # Same install-order fix as scripts/verify/phase0_gpu_verify.py: everything else first, the
    # Pascal-compatible (sm_60) torch pin LAST with --no-deps, so nothing downstream can silently
    # reintroduce Kaggle's default torch (which dropped sm_60 support).
    pip_install([f"git+https://github.com/MIC-DKFZ/nnUNet.git@{NNUNET_COMMIT}"])
    pip_install(["-U", "numpy<2.3", "scipy<1.15"], timeout=600)
    subprocess.run([sys.executable, "-c", "import scipy.ndimage"], check=True, timeout=120)
    pip_install(["--force-reinstall", "--no-deps", *PASCAL_COMPAT_TORCH])
    gpu_info = _cuda_sanity()
    json.dump(gpu_info, open(OUT / "gpu_info.json", "w"), indent=2)
    return gpu_info


def step1_prepare_raw():
    """Copy (not symlink -- nnU-Net's integrity check follows symlinks fine, but Kaggle's
    /kaggle/input is read-only and nnUNetv2_plan_and_preprocess writes cache files alongside
    nnUNet_raw's dataset.json in some code paths) the attached pool dataset into a writable
    nnUNet_raw/ location."""
    log("=== step 1: stage nnUNet_raw from attached pool dataset ===")
    input_root = Path("/kaggle/input")
    tree = [str(p.relative_to(input_root)) for p in input_root.glob("**/*") if p.is_dir()] if input_root.exists() else []
    log(f"DEBUG /kaggle/input dir tree: {tree}")
    log(f"DEBUG resolved POOL_INPUT={POOL_INPUT} exists={POOL_INPUT.exists()}")
    if not POOL_INPUT.exists():
        raise RuntimeError(
            f"{POOL_INPUT} not found -- attach the 'pdac-tier-a-lite-pool' Dataset as kernel "
            f"input before running this kernel. /kaggle/input dir tree: {tree}")
    nnunet_raw = OUT / "nnUNet_raw"
    ds_dst = nnunet_raw / DATASET_NAME
    if not ds_dst.exists():
        ds_src = POOL_INPUT / "nnUNet_raw" / DATASET_NAME
        shutil.copytree(ds_src, ds_dst)
        log(f"copied {ds_src} -> {ds_dst}")
    else:
        log(f"{ds_dst} already present, skipping copy")

    # Kaggle auto-decompresses single-file .gz members on dataset ingestion, so the mounted
    # copy has case_0000.nii / case.nii, not the .nii.gz the staging kernel wrote into
    # dataset.json. Detect the real extension on disk and patch dataset.json to match --
    # nnU-Net only needs file_ending to be internally consistent with what's actually there.
    images_dir = ds_dst / "imagesTr"
    sample = next(images_dir.iterdir())
    real_ext = ".nii.gz" if sample.name.endswith(".nii.gz") else "".join(sample.suffixes[-1:])
    dataset_json_path = ds_dst / "dataset.json"
    dj = json.load(open(dataset_json_path))
    if dj.get("file_ending") != real_ext:
        log(f"patching dataset.json file_ending: {dj.get('file_ending')!r} -> {real_ext!r} "
            f"(actual files on disk, e.g. {sample.name})")
        dj["file_ending"] = real_ext
        json.dump(dj, open(dataset_json_path, "w"), indent=2)
    return nnunet_raw


def step2_restore_or_init_preprocessed_and_results(env):
    """If a prior session's checkpoint Dataset is attached, restore nnUNet_preprocessed/ and
    nnUNet_results/ from it so training resumes instead of restarting. Otherwise this is session
    1: leave both empty for step3/step4 to populate fresh."""
    log("=== step 2: restore prior state (if any) or start fresh ===")
    preproc_dir = Path(env["nnUNet_preprocessed"])
    results_dir = Path(env["nnUNet_results"])
    preproc_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT_INPUT.exists():
        prior_preproc = CHECKPOINT_INPUT / "nnUNet_preprocessed"
        prior_results = CHECKPOINT_INPUT / "nnUNet_results"
        if prior_preproc.exists() and not any(preproc_dir.iterdir()):
            shutil.copytree(prior_preproc, preproc_dir, dirs_exist_ok=True)
            log(f"restored nnUNet_preprocessed from {prior_preproc}")
        if prior_results.exists() and not any(results_dir.iterdir()):
            shutil.copytree(prior_results, results_dir, dirs_exist_ok=True)
            log(f"restored nnUNet_results from {prior_results}")
        session_log = CHECKPOINT_INPUT / "session_log.json"
        prior_sessions = json.load(open(session_log)) if session_log.exists() else []
        log(f"resuming: {len(prior_sessions)} prior session(s) recorded")
        return prior_sessions
    log("no prior checkpoint Dataset attached -- this is session 1 (fresh preprocessing + training)")
    return []


def step3_preprocess_if_needed(nnunet_raw, env):
    preproc_dir = Path(env["nnUNet_preprocessed"]) / DATASET_NAME
    if (preproc_dir / "nnUNetResEncUNetMPlans.json").exists() and (preproc_dir / "nnUNetPlans.json").exists():
        log("=== step 3: preprocessing already done (restored from checkpoint), skipping ===")
        return
    log("=== step 3: fingerprint + preprocess (ResEnc M and default plans, real, not smoke) ===")
    # Kaggle's P100 kernels have limited host RAM, and these are full-size real CT volumes (up
    # to ~500MB uncompressed each) rather than phase0_gpu_verify.py's tiny smoke subset --
    # nnU-Net's default multiprocessing pool size (usually ~8-12 workers) OOM-killed a worker
    # ("Some background worker is 6 feet under") on the first attempt at this dataset's real
    # scale. -npfp/-np cap worker count; 2 is a conservative fit for 32 real-size volumes.
    run(["nnUNetv2_plan_and_preprocess", "-d", DATASET_ID, "-c", "3d_fullres",
         "--verify_dataset_integrity", "-pl", "nnUNetPlannerResEncM", "-npfp", "2", "-np", "2"],
        OUT / "log_preprocess_resenc.txt", env=env)
    run(["nnUNetv2_extract_fingerprint", "-d", DATASET_ID, "-np", "2"],
        OUT / "log_fingerprint_default.txt", env=env, check=False)
    run(["nnUNetv2_plan_experiment", "-d", DATASET_ID], OUT / "log_plan_default.txt",
        env=env, check=False)
    run(["nnUNetv2_preprocess", "-d", DATASET_ID, "-c", "3d_fullres", "-np", "2"],
        OUT / "log_preprocess_default.txt", env=env, check=False)


def run(cmd, log_path, env=None, check=True, timeout=None):
    """Stream subprocess output live (flushed per line) instead of capturing it all and printing
    only after the subprocess returns. Every prior training-launch crash on this kernel has shown
    ZERO output, ever, before the whole container silently reset -- but that's an artifact of how
    this function used to work (subprocess.run(..., stdout=PIPE) blocks until the child returns,
    and it never did), not necessarily evidence that nnU-Net itself never printed anything. If the
    container dies again, this at least gets whatever nnU-Net wrote up to that point into both the
    log file and Kaggle's own kernel log stream in real time, instead of losing it entirely."""
    log(f"RUN: {' '.join(cmd)}")
    start = time.time()
    tail = []
    timed_out = False
    with open(log_path, "a") as lf:
        lf.write(f"\n\n>>> {time.ctime()}: {' '.join(cmd)}\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 bufsize=1, env=env or os.environ.copy())
        try:
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                print(line.rstrip("\n"), flush=True)
                tail.append(line.rstrip("\n"))
                if len(tail) > 40:
                    tail.pop(0)
                if timeout is not None and time.time() - start > timeout:
                    timed_out = True
                    proc.kill()
                    break
        finally:
            proc.wait()
    log(f"... tail:\n{chr(10).join(tail)}")
    if timed_out:
        raise subprocess.TimeoutExpired(cmd, timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed (rc={proc.returncode}): {' '.join(cmd)}\nsee {log_path}")
    return proc.returncode


TRAINER_VARIANT_SRC = '''
import os
import time
import torch
import nnunetv2
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

PER_ARM_BUDGET_S = int(os.environ["PER_ARM_BUDGET_S"])
TARGET_EPOCHS_PER_ARM = int(os.environ.get("TARGET_EPOCHS_PER_ARM", 10 ** 9))


class _TimeBoxedMixin:
    def initialize(self):
        super().initialize()
        self._arm_start_time = time.time()
        # v9 proved (via the now-working PYTHONUNBUFFERED fix) that the transformer arm hit the
        # OUTER subprocess timeout (run()'s hard proc.kill()) instead of exiting cleanly here --
        # because this check only runs once per epoch, and with nnUNet_n_proc_DA=0 (single-
        # threaded augmentation) a full 250-iteration epoch on the heavier transformer apparently
        # takes longer than PER_ARM_BUDGET_S + the 900s safety margin. Shrinking the epoch length
        # makes this check fire far more often, so a clean, checkpointed SystemExit(0) is
        # guaranteed regardless of how slow a single iteration turns out to be -- it doesn't
        # change total training math (iterations still accumulate across "epochs"), just the
        # granularity at which we get to check the clock and stop safely.
        self.num_iterations_per_epoch = min(self.num_iterations_per_epoch, 50)
        self.num_val_iterations_per_epoch = min(self.num_val_iterations_per_epoch, 10)

    def on_epoch_end(self):
        super().on_epoch_end()
        elapsed = time.time() - self._arm_start_time
        budget_hit = elapsed > PER_ARM_BUDGET_S
        # current_epoch is post-increment here (base on_epoch_end() already bumped it), so it
        # equals the number of epochs completed so far -- comparable directly to a target count.
        target_hit = self.current_epoch >= TARGET_EPOCHS_PER_ARM
        if budget_hit or target_hit:
            # nnU-Net's own on_epoch_end() only writes checkpoint_latest.pth every
            # self.save_every (default 50) epochs, and checkpoint_final.pth only at the true end
            # of training -- so a session that stops early (as this one always does) can exit
            # with ONLY checkpoint_best.pth on disk. nnUNetv2_train --c (via maybe_load_checkpoint)
            # only ever looks for checkpoint_final.pth or checkpoint_latest.pth, never
            # checkpoint_best.pth -- so without this explicit save, the next session's --c would
            # silently find nothing and restart from scratch, discarding this session's progress.
            # Writing checkpoint_final.pth here (the name --c checks first) makes this session's
            # stop point look like a normal completed run to nnU-Net's own resume logic.
            #
            # save_checkpoint() itself stores 'current_epoch': self.current_epoch + 1, because in
            # nnU-Net's OWN on_epoch_end() it is always called BEFORE that method's trailing
            # `self.current_epoch += 1`. We just ran super().on_epoch_end() above, which already
            # did that increment -- so calling save_checkpoint() now double-counts it, saving an
            # epoch number one past what self.logger actually has entries for. On resume, nnU-Net
            # reads that inflated current_epoch, starts an epoch whose index was never logged, and
            # immediately crashes in on_validation_epoch_end() with IndexError (get_value() indexes
            # my_fantastic_logging[key] by raw list position). Undo the increment just for this
            # save call so the compensation inside save_checkpoint() lines back up; the process
            # exits right after via SystemExit so current_epoch never needs restoring.
            self.current_epoch -= 1
            self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_final.pth"))
            reason = (f"TARGET EPOCHS REACHED ({self.current_epoch} >= {TARGET_EPOCHS_PER_ARM})"
                      if target_hit else f"TIME BUDGET REACHED ({elapsed:.0f}s > {PER_ARM_BUDGET_S}s)")
            self.print_to_log_file(
                f"{reason} after epoch {self.current_epoch} -- stopping cleanly for this "
                "session; checkpoint_final.pth written, resume with --c next session.")
            raise SystemExit(0)


class nnUNetTrainer_TierALite(_TimeBoxedMixin, nnUNetTrainer):
    pass


_PRIMUS_BASE_NAME = os.environ.get("PRIMUS_BASE_TRAINER", "nnUNet_PrimusV2S_Trainer")
_PrimusBase = recursive_find_python_class(
    os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), _PRIMUS_BASE_NAME,
    current_module="nnunetv2.training.nnUNetTrainer")


class nnUNet_PrimusV2_TierALite(_TimeBoxedMixin, _PrimusBase):
    pass
'''


def step4_install_trainer_variant():
    log("=== step 4: install time-boxed trainer variant into nnunetv2 ===")
    import nnunetv2
    variants_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "tieralite"
    variants_dir.mkdir(parents=True, exist_ok=True)
    (variants_dir / "__init__.py").write_text("")
    (variants_dir / "tieralite_trainers.py").write_text(TRAINER_VARIANT_SRC)


def _checkpoint_exists(results_dir, dataset_name, trainer_name, plans_id):
    run_dir = (Path(results_dir) / dataset_name /
               f"{trainer_name}__{plans_id}__3d_fullres" / "fold_0")
    return (run_dir / "checkpoint_latest.pth").exists() or (run_dir / "checkpoint_final.pth").exists()


def _arm_already_at_target(results_dir, dataset_name, trainer_name, plans_id, target_epochs):
    """True once an arm's checkpoint_final.pth already records current_epoch >= target_epochs.
    Without this check, a session launches nnUNetv2_train --c for an already-finished arm anyway
    -- it immediately re-triggers _TimeBoxedMixin's target_hit exit after one wasted epoch (plus
    subprocess/checkpoint-load startup), a few minutes every session for the rest of this study
    that would otherwise go to whichever arm still has epochs left."""
    ckpt_path = (Path(results_dir) / dataset_name /
                 f"{trainer_name}__{plans_id}__3d_fullres" / "fold_0" / "checkpoint_final.pth")
    if not ckpt_path.exists():
        return False
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["current_epoch"] >= target_epochs


def step4b_repair_inflated_checkpoints(results_dir, dataset_name, arm_specs):
    """One-time self-heal for checkpoints written by the pre-fix _TimeBoxedMixin.on_epoch_end():
    it called self.save_checkpoint() AFTER super().on_epoch_end() had already incremented
    self.current_epoch, so save_checkpoint()'s own '+1' compensation (correct only when called
    from inside nnU-Net's vanilla on_epoch_end, before that increment) double-counted, inflating
    the saved current_epoch one epoch past what self.logger's per-epoch lists actually contain.
    Loading such a checkpoint makes nnU-Net start an epoch that was never logged and crash in
    on_validation_epoch_end() with IndexError the moment it tries to read the previous epoch's
    value. Detect and fix that here (current_epoch != logged-epoch-count) rather than relying on
    every already-shipped checkpoint having been produced by the fixed code."""
    import torch
    for trainer_name, plans_id in arm_specs:
        ckpt_path = (Path(results_dir) / dataset_name /
                     f"{trainer_name}__{plans_id}__3d_fullres" / "fold_0" / "checkpoint_final.pth")
        if not ckpt_path.exists():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        logged_epochs = len(ckpt["logging"]["mean_fg_dice"])
        if ckpt["current_epoch"] != logged_epochs:
            log(f"repairing inflated checkpoint {ckpt_path}: "
                f"current_epoch={ckpt['current_epoch']} != logged_epochs={logged_epochs}, "
                f"correcting current_epoch -> {logged_epochs}")
            ckpt["current_epoch"] = logged_epochs
            torch.save(ckpt, ckpt_path)


def step5_train_arm(env, plans_id, trainer_name, arm_label, budget_s, primus_base=None):
    log(f"=== step 5: train {arm_label} ({trainer_name} / {plans_id}), budget={budget_s:.0f}s ===")
    run_env = env.copy()
    run_env["PER_ARM_BUDGET_S"] = str(int(budget_s))
    run_env["TARGET_EPOCHS_PER_ARM"] = str(TARGET_EPOCHS_PER_ARM)
    run_env["nnUNet_compile"] = "False"  # same triton/torch-2.4.1 mismatch as phase0_gpu_verify.py
    # v6 attempt: capped nnUNet_n_proc_DA=1 (one augmentation worker process instead of the
    # default ~12). Still crashed identically -- whole container reset, zero output, at the exact
    # same point. Verified against nnU-Net's actual source at the pinned commit
    # (nnUNetTrainer.get_dataloaders(), on_train_start()): even the FIRST call ever made to a
    # multi-process augmenter is `next(mt_gen_train)` inside get_dataloaders(), which happens
    # before initialize()/get_dataloaders() print anything -- so a crash at that exact call
    # explains the "zero output, ever" pattern regardless of worker count (1 worker still forks/
    # spawns a child process). The network is already moved to CUDA (.to(self.device)) earlier in
    # initialize(), and the augmenter uses pin_memory=True on GPU -- forking (or a small
    # containerized /dev/shm) after a CUDA context already exists in the parent is a well-known
    # crash source, independent of raw worker *count*. get_dataloaders() has a dedicated
    # subprocess-free code path for exactly this: nnUNet_n_proc_DA=0 selects SingleThreadedAugmenter
    # instead of NonDetMultiThreadedAugmenter (see get_allowed_n_proc_DA() docstring + the
    # `if allowed_num_processes == 0` branch) -- no forked/spawned child, no shared memory, no
    # CUDA-after-fork hazard, for the exact step where every crash so far has happened. Slower per
    # iteration (no augmentation/compute overlap), but the alternative is another silent
    # crash-restart loop burning the last of the remaining quota for zero progress.
    run_env["nnUNet_n_proc_DA"] = "0"
    run_env["OMP_NUM_THREADS"] = "1"
    run_env["MKL_NUM_THREADS"] = "1"
    if primus_base:
        run_env["PRIMUS_BASE_TRAINER"] = primus_base

    resume = _checkpoint_exists(env["nnUNet_results"], DATASET_NAME, trainer_name, plans_id)
    cmd = ["nnUNetv2_train", DATASET_ID, "3d_fullres", "0", "-tr", trainer_name, "-p", plans_id, "--npz"]
    if resume:
        cmd.append("--c")
        log(f"{arm_label}: found existing checkpoint, resuming with --c")
    else:
        log(f"{arm_label}: no checkpoint found, starting fresh")

    t0 = time.time()
    # Backstop only: with epochs now capped at 50 iterations (see _TimeBoxedMixin), the epoch-end
    # budget check should exit cleanly long before this ever fires. Kept generous (not tight) in
    # case a single iteration is still unexpectedly slow, so we don't re-create the exact race that
    # killed the transformer arm in v9.
    rc = run(cmd, OUT / f"log_train_{arm_label}.txt", env=run_env, check=False,
              timeout=budget_s + 3600)
    wall_s = time.time() - t0
    result = {"arm": arm_label, "trainer": trainer_name, "plans": plans_id, "resumed": resume,
              "returncode": rc, "wall_seconds": wall_s}
    log(f"{arm_label} session result: {result}")
    return result


def main():
    session_started = time.time()
    results = {"session_started_utc": time.ctime(session_started)}
    gpu_info = step0_env()
    results["gpu_info"] = gpu_info

    nnunet_raw = step1_prepare_raw()
    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(OUT / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(OUT / "nnUNet_results")
    # v8's Popen+streaming fix only changed how WE read the child's stdout; it didn't touch how
    # the child itself buffers. A Python process whose stdout is a pipe (not a tty) -- exactly
    # what subprocess.Popen(..., stdout=PIPE) creates -- defaults to full block buffering, not
    # line buffering. nnUNetv2_train is itself a Python entry-point script, so anything it prints
    # sits in ITS OWN internal buffer until that buffer fills or it exits cleanly. If the
    # container gets OOM-killed within the first second or two of launch (consistent with the
    # observed "zero output, ever, then full container restart" signature persisting even after
    # the read-side fix), whatever nnU-Net had printed is lost before it ever reaches our pipe --
    # no amount of read-side speed fixes that. PYTHONUNBUFFERED=1 forces CPython to flush every
    # write immediately regardless of invocation method (works for console-script entry points
    # too, since they still run under the python interpreter).
    env["PYTHONUNBUFFERED"] = "1"

    prior_sessions = step2_restore_or_init_preprocessed_and_results(env)
    step3_preprocess_if_needed(nnunet_raw, env)
    step4_install_trainer_variant()

    tf_plans = "nnUNetPlans" if (Path(env["nnUNet_preprocessed"]) / DATASET_NAME / "nnUNetPlans.json").exists() \
        else "nnUNetResEncUNetMPlans"

    step4b_repair_inflated_checkpoints(env["nnUNet_results"], DATASET_NAME, [
        ("nnUNetTrainer_TierALite", "nnUNetResEncUNetMPlans"),
        ("nnUNet_PrimusV2_TierALite", tf_plans),
    ])

    def _skipped_result(arm_label, trainer_name, plans_id):
        log(f"{arm_label} already reached TARGET_EPOCHS_PER_ARM ({TARGET_EPOCHS_PER_ARM}) -- "
            "skipping training this session, handing its full budget to the other arm")
        return {"arm": arm_label, "trainer": trainer_name, "plans": plans_id, "resumed": True,
                "returncode": 0, "wall_seconds": 0.0, "skipped_already_at_target": True}

    arms = []
    if _arm_already_at_target(env["nnUNet_results"], DATASET_NAME, "nnUNetTrainer_TierALite",
                              "nnUNetResEncUNetMPlans", TARGET_EPOCHS_PER_ARM):
        cnn_result = _skipped_result("cnn_resenc_m", "nnUNetTrainer_TierALite", "nnUNetResEncUNetMPlans")
    else:
        cnn_result = step5_train_arm(env, "nnUNetResEncUNetMPlans", "nnUNetTrainer_TierALite",
                                      "cnn_resenc_m", budget_s=PER_ARM_BUDGET_S)
    arms.append(cnn_result)
    # If CNN finished under budget (hit TARGET_EPOCHS_PER_ARM early, or genuinely completed),
    # hand its unused time to the transformer arm this same session instead of leaving it idle.
    cnn_leftover = max(0.0, PER_ARM_BUDGET_S - cnn_result["wall_seconds"])
    transformer_budget = PER_ARM_BUDGET_S + cnn_leftover
    if cnn_leftover > 0:
        log(f"CNN arm used {cnn_result['wall_seconds']:.0f}s of its {PER_ARM_BUDGET_S}s budget; "
            f"handing {cnn_leftover:.0f}s leftover to the transformer arm "
            f"(budget now {transformer_budget:.0f}s)")

    if _arm_already_at_target(env["nnUNet_results"], DATASET_NAME, "nnUNet_PrimusV2_TierALite",
                              tf_plans, TARGET_EPOCHS_PER_ARM):
        transformer_result = _skipped_result("transformer_primusv2s", "nnUNet_PrimusV2_TierALite", tf_plans)
    else:
        transformer_result = step5_train_arm(env, tf_plans, "nnUNet_PrimusV2_TierALite",
                                             "transformer_primusv2s", budget_s=transformer_budget,
                                             primus_base="nnUNet_PrimusV2S_Trainer")
    arms.append(transformer_result)
    results["arms"] = arms

    session_record = {
        "session_ended_utc": time.ctime(),
        "wall_seconds": time.time() - session_started,
        "arms": arms,
    }
    all_sessions = prior_sessions + [session_record]
    json.dump(all_sessions, open(OUT / "session_log.json", "w"), indent=2, default=str)
    json.dump(results, open(OUT / "this_session_results.json", "w"), indent=2, default=str)

    # Ship the state a future session needs to resume, as kernel output for re-packaging into
    # the 'pdac-tier-a-lite-checkpoints' persistent Dataset -- NOT the raw images (those live in
    # the already-persistent pool Dataset and don't need re-shipping every session).
    log(">> ALL DONE for this session. Pull back /kaggle/working/phase1_train/ "
        "(nnUNet_preprocessed/, nnUNet_results/, session_log.json) and push it as/into the "
        "'pdac-tier-a-lite-checkpoints' Dataset for next session's --c resume.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(">> FATAL ERROR:", flush=True)
        traceback.print_exc()
        raise
