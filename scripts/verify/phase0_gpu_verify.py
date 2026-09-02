#!/usr/bin/env python
"""Phase 0 R2/R3 GPU verification, run on a Kaggle GPU kernel as an interim substitute for
formal NYU/USC GPU access (which is a separate, human, in-writing confirmation this script
cannot provide). Purpose: actually stand up nnU-Net v2 + PrimusV2 on a real GPU, fingerprint/
preprocess a REAL (tiny) subset of the frozen PDAC cohort, measure VRAM and step time for both
arms, run the identity-control trainer for real iterations, and smoke-test both arms through to
a prediction file with a Dice number -- clearing the Phase 0 boxes that need a GPU + real images,
without touching the full ~190GB cohort (disk-constrained, same reasoning as
scripts/download/kaggle_stream_cohort.py).

This is explicitly a SMALL-SCALE verification run (~8 real cases, 2 epochs x 5 iterations),
NOT Tier A training. Results (VRAM, plans, Dice) are diagnostic, not the frozen matched-budget
config -- that still gets chosen once real GPU access (larger VRAM tier) is confirmed.

Ships back only small files: pip freeze, GPU info, dataset fingerprint/plans json, per-arm
VRAM/step-time/Dice json, training log tails, identity-control log. No checkpoints, no images.
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

OUT = Path("/kaggle/working/phase0_verify")
OUT.mkdir(parents=True, exist_ok=True)
NNUNET_COMMIT = "0e495086eb108ff79afe106291e8c15bd2f2bc3a"
# v6 lost its connection at 5.2GB into batch 1's ~46GB zip -- close to the ~4-5GB needed to
# scan enough cases to find 16 manual-lesion ones (~28% manual-lesion rate), so 16 sits right at
# Zenodo's apparent failure window and retries were mostly re-hitting the same wall rather than
# making progress. Lowering the target moves it comfortably under that window.
N_CASES = 8
LABELS_REPO_ZIP_URL = "https://codeload.github.com/DIAGNijmegen/panorama_labels/zip/refs/heads/main"
BATCH1_RECORD_ID = 13715870
IMG_EXT = (".mha", ".nii.gz", ".nrrd")
LESION_LABEL = 1


def log(msg):
    print(f"[phase0] {msg}", flush=True)


def run(cmd, log_path=None, env=None, check=True, timeout=1800):
    log(f"RUN (timeout={timeout}s): {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=env or os.environ.copy(), timeout=timeout)
        out, rc = proc.stdout, proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.output or "") + f"\n\n>>> TIMED OUT after {timeout}s, process killed <<<"
        rc = -1
    with open(log_path, "w") if log_path else open(os.devnull, "w") as lf:
        lf.write(out)
    tail = "\n".join(out.splitlines()[-40:])
    log(f"... tail:\n{tail}")
    if check and rc != 0:
        raise RuntimeError(f"command failed (rc={rc}): {' '.join(cmd)}\nsee {log_path}")
    return rc


class VramSampler:
    """Polls nvidia-smi in a background thread; nnU-Net's CLI training loop gives us no direct
    hook into torch.cuda.max_memory_allocated(), so this is the simplest way to get a real peak
    VRAM number for a subprocess-launched training run."""

    def __init__(self, interval=2.0):
        self.interval = interval
        self.peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                used = max(int(x.strip()) for x in out.splitlines() if x.strip())
                self.peak_mb = max(self.peak_mb, used)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._thread.join(timeout=5)


# torchvision must be pinned alongside torch: Primus's dependency chain (dynamic_network_
# architectures -> timm -> torchvision.models.feature_extraction) imports torchvision at
# module load time, and a torchvision built against a different torch ABI fails registering
# its own meta ops ("operator torchvision::nms does not exist") -- confirmed in practice when
# only torch was pinned with --no-deps, leaving Kaggle's original (mismatched) torchvision.
PASCAL_COMPAT_TORCH = ("torch==2.4.1", "torchvision==0.19.1",
                       "--index-url", "https://download.pytorch.org/whl/cu121")
CUDA_SANITY_CHECK = (
    "import torch, json, sys\n"
    "assert torch.cuda.is_available(), 'cuda not available'\n"
    "x = (torch.randn(64, 64, device='cuda') @ torch.randn(64, 64, device='cuda')).sum().item()\n"
    "assert x == x, 'nan from cuda matmul'\n"
    "print(json.dumps({'torch_version': torch.__version__, 'device_name': torch.cuda.get_device_name(0),\n"
    "                   'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,\n"
    "                   'arch_list': torch.cuda.get_arch_list(), 'matmul_ok': True}))\n"
)


def _cuda_sanity(label):
    """Actually launches a CUDA kernel rather than trusting torch.cuda.is_available() -- that
    returned True on the P100 kernel even though the installed torch build (2.10.0+cu128)
    had dropped Pascal (sm_60) support entirely; only a real kernel launch surfaces that."""
    r = subprocess.run([sys.executable, "-c", CUDA_SANITY_CHECK], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"CUDA sanity check failed ({label}):\n{r.stdout}\n{r.stderr}")
    info = json.loads(r.stdout.strip().splitlines()[-1])
    log(f"CUDA sanity OK ({label}): {info}")
    return info


def pip_install(args, timeout=900, retries=3, backoff=20):
    """v7 lesson: `pip install git+https://...` failed once with a bare git-clone exit code 128
    (no informative stderr captured) -- almost certainly a transient network blip in the kernel's
    container, not a real dependency problem, since the exact same command had worked identically
    in every prior version. Retry rather than treat every install as a hard failure."""
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


def step0_env():
    log("=== step 0: environment ===")
    # v3 lesson: installing nnunetv2 WITHOUT --no-deps silently pulled in a newer torch to
    # satisfy some transitive dependency, even though nnunetv2's own constraint
    # (torch>=2.1.2,!=2.9.*) was already satisfied by the Pascal-compatible pin -- pip's
    # resolver reconciles the WHOLE dependency graph, not just nnunetv2's direct constraint.
    # A sanity check right after pinning torch passed; the same check after installing
    # nnunetv2 failed with the exact "no kernel image for device" error this pin exists to
    # avoid. Fix: do all other installs FIRST, then pin the Pascal-compatible torch with
    # --no-deps as the LAST install, so nothing downstream can silently override it again.
    pip_install([f"git+https://github.com/MIC-DKFZ/nnUNet.git@{NNUNET_COMMIT}"])
    pip_install(["stream-unzip"], timeout=300)
    # nnunetv2's dependency resolution can pull in a scipy/numpy pair where numpy.testing.utils
    # (added in numpy 2.3) calls a _multiarray_umath symbol this platform's wheel doesn't
    # actually expose -- breaks the whole nnUNetTrainer import chain (-> batchgeneratorsv2 ->
    # skimage -> scipy.ndimage -> numpy.testing) with an AttributeError. A first attempt to fix
    # this by upgrading to the newest numpy/scipy made it WORSE (still broken on numpy 2.5.2).
    # The reliable fix is to stay below 2.3, where numpy.testing.utils never calls that symbol.
    pip_install(["-U", "numpy<2.3", "scipy<1.15"], timeout=600)
    subprocess.run([sys.executable, "-c", "import scipy.ndimage"], check=True, timeout=120)

    # Pin the Pascal-compatible torch LAST, with --no-deps so nothing above can reintroduce a
    # torch build that dropped sm_60 support (Kaggle's default image: 2.10.0+cu128, sm_70+ only).
    pip_install(["--force-reinstall", "--no-deps", *PASCAL_COMPAT_TORCH])
    gpu_info = _cuda_sanity("after pinning cu121 torch (final step)")
    json.dump(gpu_info, open(OUT / "gpu_info.json", "w"), indent=2)
    if "sm_60" not in gpu_info["arch_list"] and "compute_60" not in " ".join(gpu_info["arch_list"]):
        log(f"WARNING: pinned torch's arch_list does not obviously list sm_60: {gpu_info['arch_list']} "
            "-- matmul succeeded anyway, continuing, but watch for later kernel-launch errors.")

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    open(OUT / "pip_freeze.txt", "w").write(freeze)
    nnunet_line = next((l for l in freeze.splitlines() if "nnunetv2" in l.lower()), "nnunetv2 (unknown)")
    log(f"nnunetv2: {nnunet_line}")
    return gpu_info, nnunet_line


def step1_verify_primus_trainers():
    log("=== step 1: verify PrimusV2 trainer imports ===")
    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    base_dir = os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
    found = {}
    for size in "SBML":
        name = f"nnUNet_PrimusV2{size}_Trainer"
        cls = recursive_find_python_class(base_dir, name, current_module="nnunetv2.training.nnUNetTrainer")
        found[name] = cls is not None
    log(f"PrimusV2 trainers found: {found}")
    json.dump(found, open(OUT / "primus_trainers_found.json", "w"), indent=2)
    if not all(found.values()):
        raise RuntimeError(f"missing PrimusV2 trainer(s): {found}")
    return base_dir


def step2_download_subset():
    """Stream batch-1 zip, keep only the first N_CASES manual-lesion PANORAMA cases on disk
    (abandon the connection once we have enough -- we don't need the whole ~50GB zip for a
    16-case smoke subset)."""
    log(f"=== step 2: download {N_CASES} real manual-lesion cases from batch 1 ===")
    import requests
    import stream_unzip as stream_unzip_mod
    stream_unzip = stream_unzip_mod.stream_unzip

    labels_dir = OUT / "panorama_labels"
    if not labels_dir.exists():
        zip_path = OUT / "panorama_labels.zip"
        with requests.get(LABELS_REPO_ZIP_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        import zipfile
        extract_tmp = OUT / "_labels_extract"
        shutil.rmtree(extract_tmp, ignore_errors=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_tmp)
        zip_path.unlink()
        inner = next(p for p in extract_tmp.iterdir() if p.is_dir())
        inner.rename(labels_dir)
        shutil.rmtree(extract_tmp, ignore_errors=True)
    manual_dir = labels_dir / "manual_labels"
    log(f"panorama_labels ready, {len(list(manual_dir.glob('*')))} manual labels available")

    # v10 lesson: Zenodo's own API returned a transient 504 Gateway Timeout on this metadata
    # call (before any actual download started) -- retry rather than treat it as fatal.
    for attempt in range(1, 4):
        try:
            r = requests.get(f"https://zenodo.org/api/records/{BATCH1_RECORD_ID}", timeout=60)
            r.raise_for_status()
            break
        except (requests.exceptions.RequestException, OSError) as e:
            if attempt == 3:
                raise
            log(f"WARNING: Zenodo metadata fetch failed ({e!r}), attempt {attempt}/3 -- retrying after 20s")
            time.sleep(20)
    files = r.json()["files"]
    zip_meta = next(f for f in files if f["key"].endswith(".zip"))
    url = zip_meta["links"]["self"]

    images_dir = OUT / "raw_images"
    images_dir.mkdir(exist_ok=True)
    kept = []

    def http_chunks():
        # Reading stops (and this generator + its `with` block get closed/cleaned up via
        # CPython refcounting) as soon as the caller's `for` loop below hits `break` -- no
        # manual abort needed here, and raising StopIteration inside a generator would (PEP
        # 479) surface as a RuntimeError instead of ending iteration cleanly.
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            yield from resp.iter_content(chunk_size=1 << 20)

    # Zenodo has no Range support, so a dropped connection (observed in practice: broke after
    # 5.2/49.3 GB) can't be resumed mid-file -- the only option is retrying the whole batch
    # stream from byte 0. Ported from scripts/download/kaggle_stream_cohort.py, which already
    # hit and solved this exact failure; skip case_ids already kept across retries. stream_unzip
    # surfaces a broken connection as its own IndexError (popping an empty internal queue) with
    # the real requests/urllib3 exception chained as the cause, so catch both.
    kept_ids = set()
    MAX_RETRIES = 25
    attempt = 0
    while len(kept) < N_CASES:
        attempt += 1
        try:
            for name, _size, unzipped_chunks in stream_unzip(http_chunks()):
                name = name.decode() if isinstance(name, bytes) else name
                if not name.endswith(IMG_EXT):
                    for _ in unzipped_chunks:
                        pass
                    continue
                cid = Path(name).name
                for ext in IMG_EXT:
                    if cid.endswith(ext):
                        cid = cid[: -len(ext)]
                        break
                bare_cid = cid[:-5] if cid.endswith("_0000") else cid
                if bare_cid in kept_ids:
                    for _ in unzipped_chunks:
                        pass
                    continue
                label_path = None
                for ext in IMG_EXT:
                    cand = manual_dir / f"{bare_cid}{ext}"
                    if cand.exists():
                        label_path = cand
                        break
                if label_path is None:
                    for _ in unzipped_chunks:
                        pass
                    continue  # only want manually-labeled cases for this smoke subset
                dst = images_dir / f"{bare_cid}_0000.nii.gz"
                with open(dst, "wb") as f:
                    for chunk in unzipped_chunks:
                        f.write(chunk)
                kept.append((bare_cid, dst, label_path))
                kept_ids.add(bare_cid)
                log(f"  kept {bare_cid} ({len(kept)}/{N_CASES})")
                if len(kept) >= N_CASES:
                    break
            break  # finished streaming the whole zip without a network error
        except (requests.exceptions.RequestException, OSError, EOFError, IndexError,
                stream_unzip_mod.DataError, stream_unzip_mod.UncompressError) as e:
            if attempt >= MAX_RETRIES or len(kept) >= N_CASES:
                raise
            log(f"WARNING: download connection broke ({e!r}) after {len(kept)}/{N_CASES} cases "
                f"-- retrying whole batch from the start (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(15)

    if len(kept) < N_CASES:
        log(f"WARNING: only found {len(kept)}/{N_CASES} manual-lesion cases before batch ended")
    if len(kept) < 4:
        raise RuntimeError(f"only {len(kept)} usable cases -- too few for even a smoke fold split")
    return kept


def step3_build_dataset(kept):
    log("=== step 3: build tiny nnU-Net raw dataset (Dataset999_PDACSmoke) ===")
    import SimpleITK as sitk
    import numpy as np

    nnunet_raw = OUT / "nnUNet_raw"
    ds_dir = nnunet_raw / "Dataset999_PDACSmoke"
    (ds_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labelsTr").mkdir(parents=True, exist_ok=True)
    for cid, img_path, label_path in kept:
        shutil.copy(img_path, ds_dir / "imagesTr" / f"{cid}_0000.nii.gz")
        img = sitk.ReadImage(str(img_path))
        seg = sitk.ReadImage(str(label_path))
        # nnU-Net's verify_dataset_integrity requires the label to share the image's exact
        # geometry. Real PANORAMA cases can differ here -- observed both a tiny float-precision
        # spacing mismatch (2.4000000953674316 vs 2.4000244140625, a re-serialization rounding
        # artifact) AND a genuine one (8.0mm vs ~4.8mm z-spacing for a different case, likely
        # the manual label coming from a differently-reconstructed series of the same study).
        # Resampling onto the image's own grid resolves both uniformly rather than assuming
        # they already match (scripts/data/convert_to_nnunet.py's write_binary_lesion_label
        # applies the identical resample-before-CopyInformation fix).
        if (seg.GetSize() != img.GetSize() or seg.GetSpacing() != img.GetSpacing()
                or seg.GetDirection() != img.GetDirection() or seg.GetOrigin() != img.GetOrigin()):
            log(f"  resampling label for {cid} onto image grid (geometry mismatch)")
            seg = sitk.Resample(seg, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, seg.GetPixelID())
        arr = (sitk.GetArrayFromImage(seg) == LESION_LABEL).astype(np.uint8)
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)
        sitk.WriteImage(out, str(ds_dir / "labelsTr" / f"{cid}.nii.gz"), useCompression=True)
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "pdac_lesion": 1},
        "numTraining": len(kept),
        "file_ending": ".nii.gz",
        "description": "Phase 0 GPU-verification smoke subset -- NOT the frozen Dataset501_PDAC cohort",
    }
    json.dump(dataset_json, open(ds_dir / "dataset.json", "w"), indent=2)
    log(f"Dataset999_PDACSmoke built: {len(kept)} cases")
    return nnunet_raw


def step4_preprocess(nnunet_raw, env):
    log("=== step 4: fingerprint + preprocess (ResEnc M and default plans) ===")
    run(["nnUNetv2_plan_and_preprocess", "-d", "999", "-c", "3d_fullres",
         "--verify_dataset_integrity", "-pl", "nnUNetPlannerResEncM"],
        OUT / "log_preprocess_resenc.txt", env=env)
    run(["nnUNetv2_extract_fingerprint", "-d", "999"],
        OUT / "log_fingerprint_default.txt", env=env, check=False)
    run(["nnUNetv2_plan_experiment", "-d", "999"],
        OUT / "log_plan_default.txt", env=env, check=False)
    run(["nnUNetv2_preprocess", "-d", "999", "-c", "3d_fullres"],
        OUT / "log_preprocess_default.txt", env=env, check=False)

    preproc_dir = Path(env["nnUNet_preprocessed"]) / "Dataset999_PDACSmoke"
    for name in ("nnUNetResEncUNetMPlans.json", "nnUNetPlans.json", "dataset_fingerprint.json"):
        src = preproc_dir / name
        if src.exists():
            shutil.copy(src, OUT / name)
            log(f"shipped {name}")
    return preproc_dir


QUICK_TRAINER_SRC = '''
import os
import torch
import nnunetv2
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

# nnUNetTrainer.__init__ auto-captures its init kwargs into self.my_init_kwargs (for checkpoint
# reproducibility) by introspecting the RUNTIME class's __init__ signature (type(self).__init__)
# and doing self.my_init_kwargs[k] = locals()[k] for each parameter name -- inside its OWN frame.
# A generic (*a, **kw) subclass signature makes that loop look for locals named 'a'/'kw' in
# nnUNetTrainer.__init__'s frame, which don't exist -> KeyError. Matching the base signature's
# exact parameter names (as upstream's own nnUNetTrainer_5epochs variant does) is required, not
# just a style choice.

class nnUNetTrainer_Quick10it(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 5
        self.num_val_iterations_per_epoch = 2


_PRIMUS_BASE_NAME = os.environ.get("PRIMUS_BASE_TRAINER", "nnUNet_PrimusV2S_Trainer")
_PrimusBase = recursive_find_python_class(
    os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), _PRIMUS_BASE_NAME,
    current_module="nnunetv2.training.nnUNetTrainer")


class nnUNet_PrimusV2_Quick10it(_PrimusBase):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 5
        self.num_val_iterations_per_epoch = 2


import torch.nn as nn


class _PassThrough(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


def _find_transformer_blocks(network):
    candidates = []
    for name, module in network.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4:
            has_attn = any(
                any("attn" in sub_name.lower() for sub_name, _ in block.named_modules())
                for block in module)
            if has_attn:
                candidates.append((name, module))
    if not candidates:
        raise RuntimeError("Could not locate the transformer block stack in PrimusV2.")
    candidates.sort(key=lambda c: len(c[1]), reverse=True)
    name, blocks = candidates[0]
    print(f"[identity control] replacing {len(blocks)} blocks at '{name}' with identity")
    return blocks


class nnUNet_PrimusV2_Identity_Quick10it(_PrimusBase):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.num_iterations_per_epoch = 10
        self.num_val_iterations_per_epoch = 2

    def initialize(self):
        super().initialize()
        net = self.network.module if hasattr(self.network, "module") else self.network
        params_before = sum(p.numel() for p in net.parameters())
        blocks = _find_transformer_blocks(net)
        n_blocks = len(blocks)
        for i in range(n_blocks):
            blocks[i] = _PassThrough()
        params_after = sum(p.numel() for p in net.parameters())
        assert params_after < params_before * 0.8, (
            f"identity ablation removed too few parameters ({params_before} -> {params_after})")
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        self.print_to_log_file(
            f"IDENTITY CONTROL ACTIVE (base={_PRIMUS_BASE_NAME}): {n_blocks} blocks -> identity; "
            f"params {params_before:,} -> {params_after:,}")
'''


def step5_install_quick_trainers():
    log("=== step 5: install quick-training trainer variants into nnunetv2 ===")
    import nnunetv2
    variants_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants" / "phase0quick"
    variants_dir.mkdir(parents=True, exist_ok=True)
    (variants_dir / "__init__.py").write_text("")
    (variants_dir / "quick_trainers.py").write_text(QUICK_TRAINER_SRC)
    log(f"wrote quick trainers to {variants_dir}")


def step6_train_and_measure(env, plans_id, trainer_name, arm_label, primus_base=None):
    log(f"=== step 6: train+measure {arm_label} ({trainer_name} / {plans_id}) ===")
    run_env = env.copy()
    # torch.compile is on by default in nnUNetTrainer, and Kaggle's pre-baked `triton` package
    # (installed for the original torch 2.10 image) doesn't match our pinned torch 2.4.1 --
    # inductor's lowering fails with "cannot import name 'triton_key'". Not worth chasing a
    # matching triton pin for a tiny 2-epoch smoke run; just disable compile.
    run_env["nnUNet_compile"] = "False"
    if primus_base:
        run_env["PRIMUS_BASE_TRAINER"] = primus_base
    log_path = OUT / f"log_train_{arm_label}.txt"
    t0 = time.time()
    with VramSampler() as sampler:
        rc = run(["nnUNetv2_train", "999", "3d_fullres", "0", "-tr", trainer_name,
                  "-p", plans_id, "--npz"], log_path, env=run_env, check=False)
    wall_s = time.time() - t0
    result = {"arm": arm_label, "trainer": trainer_name, "plans": plans_id,
              "returncode": rc, "wall_seconds": wall_s, "peak_vram_mb": sampler.peak_mb}
    log(f"{arm_label} result: {result}")
    return result


def step7_predict_and_dice(env, plans_id, trainer_name, arm_label):
    log(f"=== step 7: predict + dice for {arm_label} ===")
    import numpy as np
    import SimpleITK as sitk

    imgs_dir = OUT / "nnUNet_raw" / "Dataset999_PDACSmoke" / "imagesTr"
    labels_dir = OUT / "nnUNet_raw" / "Dataset999_PDACSmoke" / "labelsTr"
    splits_json = (Path(env["nnUNet_preprocessed"]) / "Dataset999_PDACSmoke" / "splits_final.json")
    if not splits_json.exists():
        log("no splits_final.json found -- skipping dice (training likely failed)")
        return {"arm": arm_label, "dice_mean": None, "n_cases": 0}
    val_ids = json.load(open(splits_json))[0]["val"]

    pred_in = OUT / f"predict_in_{arm_label}"
    pred_out = OUT / f"predict_out_{arm_label}"
    pred_in.mkdir(exist_ok=True)
    pred_out.mkdir(exist_ok=True)
    for cid in val_ids:
        shutil.copy(imgs_dir / f"{cid}_0000.nii.gz", pred_in / f"{cid}_0000.nii.gz")

    rc = run(["nnUNetv2_predict", "-i", str(pred_in), "-o", str(pred_out), "-d", "999",
              "-c", "3d_fullres", "-tr", trainer_name, "-p", plans_id, "-f", "0",
              "-chk", "checkpoint_final.pth"],
             OUT / f"log_predict_{arm_label}.txt", env=env, check=False)
    if rc != 0:
        return {"arm": arm_label, "dice_mean": None, "n_cases": 0, "predict_returncode": rc}

    dices = []
    for cid in val_ids:
        pred_path = pred_out / f"{cid}.nii.gz"
        gt_path = labels_dir / f"{cid}.nii.gz"
        if not pred_path.exists():
            continue
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))) > 0
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))) > 0
        inter = np.logical_and(pred, gt).sum()
        denom = pred.sum() + gt.sum()
        dice = float(2 * inter / denom) if denom > 0 else float("nan")
        dices.append({"case_id": cid, "dice": dice})
    mean_dice = float(np.nanmean([d["dice"] for d in dices])) if dices else None
    result = {"arm": arm_label, "dice_mean": mean_dice, "n_cases": len(dices), "per_case": dices}
    log(f"{arm_label} dice: {result}")
    return result


def main():
    results = {}
    gpu_info, nnunet_commit_line = step0_env()
    results["gpu_info"] = gpu_info
    results["nnunet_commit_line"] = nnunet_commit_line

    step1_verify_primus_trainers()

    kept = step2_download_subset()
    nnunet_raw = step3_build_dataset(kept)
    # panorama_labels is ~1.2GB of label files across the whole cohort; step3 has already
    # copied out the ~16 labels this subset needs into nnunet_raw, so drop the rest rather than
    # ship it back wholesale as kernel output.
    shutil.rmtree(OUT / "panorama_labels", ignore_errors=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(nnunet_raw)
    env["nnUNet_preprocessed"] = str(OUT / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(OUT / "nnUNet_results")
    for p in (env["nnUNet_preprocessed"], env["nnUNet_results"]):
        Path(p).mkdir(parents=True, exist_ok=True)

    step4_preprocess(nnunet_raw, env)
    step5_install_quick_trainers()

    arms = []
    cnn = step6_train_and_measure(env, "nnUNetResEncUNetMPlans", "nnUNetTrainer_Quick10it", "cnn_resenc_m")
    arms.append(cnn)
    if cnn["returncode"] == 0:
        arms.append(step7_predict_and_dice(env, "nnUNetResEncUNetMPlans", "nnUNetTrainer_Quick10it", "cnn_resenc_m"))

    tf_plans = "nnUNetResEncUNetMPlans"  # fallback if the default-planner Primus path didn't produce nnUNetPlans
    if (Path(env["nnUNet_preprocessed"]) / "Dataset999_PDACSmoke" / "nnUNetPlans.json").exists():
        tf_plans = "nnUNetPlans"
    tf = step6_train_and_measure(env, tf_plans, "nnUNet_PrimusV2_Quick10it", "transformer_primusv2s",
                                  primus_base="nnUNet_PrimusV2S_Trainer")
    arms.append(tf)
    if tf["returncode"] == 0:
        arms.append(step7_predict_and_dice(env, tf_plans, "nnUNet_PrimusV2_Quick10it", "transformer_primusv2s"))

    ident = step6_train_and_measure(env, tf_plans, "nnUNet_PrimusV2_Identity_Quick10it", "identity_control",
                                     primus_base="nnUNet_PrimusV2S_Trainer")
    arms.append(ident)

    results["arms"] = arms
    json.dump(results, open(OUT / "results.json", "w"), indent=2, default=str)
    log(">> ALL DONE. Pull back /kaggle/working/phase0_verify/ (results.json + logs + plans jsons).")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(">> FATAL ERROR:", flush=True)
        traceback.print_exc()
        raise
