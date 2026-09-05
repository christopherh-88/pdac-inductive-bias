#!/usr/bin/env python
"""Tier-A-lite fold-0 held-out inference, run on a Kaggle GPU kernel once phase1_train_lite.py
has produced a usable checkpoint for both arms.

NOT Tier A: same caveat as phase1_train_lite.py and phase1_stage_pool.py -- this predicts on
Dataset601_PDACTierALite's 7-case fold-0 validation split, not the pre-registered 478-case
cohort, and is a further diagnostic preview only. See preregistration/DEVIATIONS.md.

Why this exists (rather than reusing nnU-Net's own auto-validation): phase1_train_lite.py's
_TimeBoxedMixin always exits training via `raise SystemExit(0)` mid-loop -- on a time-budget hit
AND on a target-epoch hit -- so nnU-Net's normal end-of-run `perform_actual_validation()` (which
would otherwise populate fold_0/validation/ with held-out predictions automatically) never runs.
Without this script, there is no held-out prediction to evaluate against, ever, for this
lite pipeline.

Ships back: predictions/<arm>/<case_id>.nii.gz for both arms' 7 fold-0 validation cases, as
kernel output -- small enough (7 cases x 2 arms of label maps) to download directly rather than
needing another persistent Dataset round-trip. Run scripts/analysis/phase1_eval_lite.py locally
afterward (CPU-only) to score them against ground truth with the same frozen metrics used
elsewhere in this repo.

Usage (as a Kaggle kernel, same input Datasets as phase1_train_lite.py):
  kaggle kernels push -p .
with kernel-metadata.json's dataset_sources = [pdac-tier-a-lite-pool, pdac-tier-a-lite-checkpoints]
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path("/kaggle/working/phase1_predict")
OUT.mkdir(parents=True, exist_ok=True)
NNUNET_COMMIT = "0e495086eb108ff79afe106291e8c15bd2f2bc3a"
DATASET_ID = "601"
DATASET_NAME = "Dataset601_PDACTierALite"
FOLD = "0"

# Which checkpoint to predict from: "best" (highest EMA pseudo-dice seen so far) is the more
# meaningful diagnostic snapshot; "final" is whatever epoch training happened to stop at (a
# mid-training snapshot for this lite pipeline, since sessions always stop early -- see the
# _TimeBoxedMixin note above). Overridable per-run without editing the kernel.
import os
CHECKPOINT_NAME = os.environ.get("PREDICT_CHECKPOINT", "checkpoint_best.pth")

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

ARMS = [
    {"label": "cnn_resenc_m", "trainer": "nnUNetTrainer_TierALite", "plans": "nnUNetResEncUNetMPlans"},
    {"label": "transformer_primusv2s", "trainer": "nnUNet_PrimusV2_TierALite", "plans": None},  # plans resolved at runtime
]


def log(msg):
    print(f"[phase1-predict] {msg}", flush=True)


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


def _find_input_dataset(slug):
    root = Path("/kaggle/input")
    if (root / slug).exists():
        return root / slug
    for candidate in root.glob(f"**/{slug}"):
        if candidate.is_dir():
            return candidate
    return root / slug


POOL_INPUT = _find_input_dataset("pdac-tier-a-lite-pool")
CHECKPOINT_INPUT = _find_input_dataset("pdac-tier-a-lite-checkpoints")

# Same trainer-variant source as phase1_train_lite.py's step4 -- nnUNetv2_predict needs the
# trainer class importable under the exact name used during training (it reads the class name
# out of the checkpoint and looks it up), even though prediction itself never touches the
# time-boxing logic inside _TimeBoxedMixin.
TRAINER_VARIANT_SRC = '''
import os
import time
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2


class _TimeBoxedMixin:
    def initialize(self):
        super().initialize()
        self._arm_start_time = time.time()
        self.num_iterations_per_epoch = min(self.num_iterations_per_epoch, 50)
        self.num_val_iterations_per_epoch = min(self.num_val_iterations_per_epoch, 10)


class nnUNetTrainer_TierALite(_TimeBoxedMixin, nnUNetTrainer):
    pass


_PRIMUS_BASE_NAME = os.environ.get("PRIMUS_BASE_TRAINER", "nnUNet_PrimusV2S_Trainer")
_PrimusBase = recursive_find_python_class(
    os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), _PRIMUS_BASE_NAME,
    current_module="nnunetv2.training.nnUNetTrainer")


class nnUNet_PrimusV2_TierALite(_TimeBoxedMixin, _PrimusBase):
    pass
'''


def step0_env():
    log("=== step 0: environment ===")
    pip_install([f"git+https://github.com/MIC-DKFZ/nnUNet.git@{NNUNET_COMMIT}"])
    pip_install(["-U", "numpy<2.3", "scipy<1.15"], timeout=600)
    subprocess.run([sys.executable, "-c", "import scipy.ndimage"], check=True, timeout=120)
    pip_install(["--force-reinstall", "--no-deps", *PASCAL_COMPAT_TORCH])
    gpu_info = _cuda_sanity()
    json.dump(gpu_info, open(OUT / "gpu_info.json", "w"), indent=2)
    return gpu_info


def step1_install_trainer_variant():
    log("=== step 1: install time-boxed trainer variant into nnunetv2 (name-compat only) ===")
    import nnunetv2
    variants_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "tieralite"
    variants_dir.mkdir(parents=True, exist_ok=True)
    (variants_dir / "__init__.py").write_text("")
    (variants_dir / "tieralite_trainers.py").write_text(TRAINER_VARIANT_SRC)


def step2_stage_inputs():
    log("=== step 2: stage nnUNet_raw / nnUNet_preprocessed / nnUNet_results from attached Datasets ===")
    if not POOL_INPUT.exists():
        raise RuntimeError(f"{POOL_INPUT} not found -- attach 'pdac-tier-a-lite-pool' as kernel input")
    if not CHECKPOINT_INPUT.exists():
        raise RuntimeError(f"{CHECKPOINT_INPUT} not found -- attach 'pdac-tier-a-lite-checkpoints' "
                           "as kernel input (predict needs a trained checkpoint, there is nothing "
                           "to predict from on session 1)")

    nnunet_raw = OUT / "nnUNet_raw"
    ds_dst = nnunet_raw / DATASET_NAME
    if not ds_dst.exists():
        shutil.copytree(POOL_INPUT / "nnUNet_raw" / DATASET_NAME, ds_dst)

    preproc_dst = OUT / "nnUNet_preprocessed"
    if not preproc_dst.exists():
        shutil.copytree(CHECKPOINT_INPUT / "nnUNet_preprocessed", preproc_dst)

    results_dst = OUT / "nnUNet_results"
    if not results_dst.exists():
        shutil.copytree(CHECKPOINT_INPUT / "nnUNet_results", results_dst)

    # Same Kaggle single-file-.gz auto-decompression fix as phase1_train_lite.py step1.
    images_dir = ds_dst / "imagesTr"
    sample = next(images_dir.iterdir())
    real_ext = ".nii.gz" if sample.name.endswith(".nii.gz") else "".join(sample.suffixes[-1:])
    dataset_json_path = ds_dst / "dataset.json"
    dj = json.load(open(dataset_json_path))
    if dj.get("file_ending") != real_ext:
        dj["file_ending"] = real_ext
        json.dump(dj, open(dataset_json_path, "w"), indent=2)

    return nnunet_raw, preproc_dst, results_dst


def _fold0_val_case_ids(preproc_dst):
    splits = json.load(open(preproc_dst / DATASET_NAME / "splits_final.json"))
    return splits[int(FOLD)]["val"]


def _stage_predict_input(nnunet_raw, case_ids, images_dir_ext):
    """nnUNetv2_predict wants a flat input folder of *_0000<ext> images -- build one containing
    only the fold-0 held-out cases, not the full 32-case pool, so we never accidentally predict
    (and silently look great on) a case the model was trained on."""
    src = nnunet_raw / DATASET_NAME / "imagesTr"
    dst = OUT / "predict_input"
    dst.mkdir(parents=True, exist_ok=True)
    for cid in case_ids:
        fname = f"{cid}_0000{images_dir_ext}"
        s = src / fname
        if not s.exists():
            raise RuntimeError(f"expected held-out image {s} not found in imagesTr")
        shutil.copy(s, dst / fname)
    return dst


def step3_predict_arm(arm, results_dst, predict_input_dir, env):
    plans_id = arm["plans"]
    trainer_name = arm["trainer"]
    run_dir = results_dst / DATASET_NAME / f"{trainer_name}__{plans_id}__3d_fullres" / "fold_0"
    ckpt = run_dir / CHECKPOINT_NAME
    if not ckpt.exists():
        fallback = run_dir / "checkpoint_final.pth"
        if not fallback.exists():
            raise RuntimeError(f"neither {ckpt} nor {fallback} exists -- has {arm['label']} trained yet?")
        log(f"{arm['label']}: {CHECKPOINT_NAME} not found, falling back to checkpoint_final.pth")
        ckpt = fallback

    out_dir = OUT / "predictions" / arm["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["nnUNetv2_predict", "-i", str(predict_input_dir), "-o", str(out_dir),
           "-d", DATASET_ID, "-c", "3d_fullres", "-p", plans_id, "-tr", trainer_name,
           "-f", FOLD, "-chk", ckpt.name, "--disable_tta"]
    log(f"=== step 3: predict {arm['label']} ({trainer_name}/{plans_id}, ckpt={ckpt.name}) ===")
    log(f"RUN: {' '.join(cmd)}")
    t0 = time.time()
    subprocess.run(cmd, env=env, check=True)
    log(f"{arm['label']}: predicted {len(list(out_dir.glob('*.nii.gz')))} cases in {time.time()-t0:.0f}s")
    return out_dir


def main():
    gpu_info = step0_env()
    step1_install_trainer_variant()
    nnunet_raw, preproc_dst, results_dst = step2_stage_inputs()

    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(preproc_dst)
    env["nnUNet_results"] = str(results_dst)
    env["nnUNet_compile"] = "False"
    env["PYTHONUNBUFFERED"] = "1"

    tf_plans = "nnUNetPlans" if (preproc_dst / DATASET_NAME / "nnUNetPlans.json").exists() \
        else "nnUNetResEncUNetMPlans"
    for arm in ARMS:
        if arm["plans"] is None:
            arm["plans"] = tf_plans

    case_ids = _fold0_val_case_ids(preproc_dst)
    log(f"fold {FOLD} held-out validation cases ({len(case_ids)}): {case_ids}")

    dataset_json = json.load(open(nnunet_raw / DATASET_NAME / "dataset.json"))
    img_ext = dataset_json["file_ending"]
    predict_input_dir = _stage_predict_input(nnunet_raw, case_ids, img_ext)

    results = {"checkpoint_name": CHECKPOINT_NAME, "fold": FOLD, "val_case_ids": case_ids,
               "gpu_info": gpu_info, "arms": []}
    for arm in ARMS:
        out_dir = step3_predict_arm(arm, results_dst, predict_input_dir, env)
        results["arms"].append({"label": arm["label"], "trainer": arm["trainer"],
                                "plans": arm["plans"], "n_predicted": len(list(out_dir.glob("*.nii.gz")))})

    json.dump(results, open(OUT / "predict_results.json", "w"), indent=2)
    log(f"done: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
