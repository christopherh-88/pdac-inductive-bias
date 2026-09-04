#!/usr/bin/env python
"""Shared pieces of the generated Kaggle notebooks: cell constructors, the bootstrap cell,
and the environment/trainer-installation cells.

Three notebooks share one bootstrap, one set of path conventions, one disk-budget guard, and
one cross-session state mechanism. Hand-maintaining that across three .ipynb JSON files means
three places for it to drift, so it lives here once and build_notebooks.py assembles the
notebooks from it.
"""
import hashlib
import json  # re-exported: build_notebooks.py imports it from here
from pathlib import Path

OUT = Path(__file__).parent
REPO_URL = "https://github.com/spraldev/pdac-inductive-bias.git"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def notebook(cells, gpu=False, name=""):
    # nbformat 4.5+ requires an id on every cell, and editors add one on save. Assigning them
    # deterministically here means opening a notebook in Jupyter or VS Code and saving it is a
    # no-op rather than something that shows up as drift from the generator.
    for i, cell in enumerate(cells):
        cell["id"] = hashlib.sha1(f"{name}:{i}".encode()).hexdigest()[:8]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none",
                       "dataSources": [], "isInternetEnabled": True,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------------------
# The bootstrap cell, shared by every notebook.
# --------------------------------------------------------------------------------------
BOOTSTRAP = '''
# === PDAC study bootstrap =============================================================
# Identical in all three notebooks. Paths, environment, repo, disk budget, and the
# cross-session state helpers.
import os, sys, subprocess, shutil, json, tarfile, textwrap, time
from pathlib import Path

ON_KAGGLE = Path("/kaggle").exists()
REPO_URL  = "''' + REPO_URL + '''"

# --- the two hard limits, in one place -------------------------------------------------
# Kaggle kills a session at 12 h and refuses to save an output larger than 20 GB. Both are
# silent failures if you meet them by accident, so both are budgeted with headroom and
# checked rather than hoped for.
SESSION_HOURS   = float(os.environ.get("PDAC_SESSION_HOURS", 11.0))   # of a 12 h cap
OUTPUT_LIMIT_GB = float(os.environ.get("PDAC_OUTPUT_LIMIT_GB", 17.0)) # of a 20 GB cap
SESSION_T0 = time.time()

# --- paths ------------------------------------------------------------------------------
# WORK persists as the notebook's saved output and is what the 20 GB cap applies to: only
# small, precious, or genuinely needed-downstream things go there. SCRATCH is much larger and
# is wiped with the session, so everything regenerable lives there — raw images, nnU-Net
# preprocessed data, occluded volumes.
if ON_KAGGLE:
    WORK    = Path("/kaggle/working")
    SCRATCH = Path("/kaggle/temp/pdac"); SCRATCH.mkdir(parents=True, exist_ok=True)
    REPO    = WORK / "pdac-research"
else:
    WORK    = Path(os.environ.get("PDAC_WORK", Path.cwd() / "pdac_work"))
    SCRATCH = Path(os.environ.get("PDAC_SCRATCH", WORK / "scratch"))
    REPO    = Path(os.environ.get("PDAC_REPO", Path.cwd()))
    WORK.mkdir(parents=True, exist_ok=True); SCRATCH.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(os.environ.get("PDAC_DATA", SCRATCH / "data"))
RESULTS   = WORK / "results";  RESULTS.mkdir(parents=True, exist_ok=True)
STATE     = WORK / "state";    STATE.mkdir(parents=True, exist_ok=True)
PREDS     = WORK / "preds";    PREDS.mkdir(parents=True, exist_ok=True)

# --- repo ---------------------------------------------------------------------------------
def _find_attached(*required):
    """First attached input containing all of the given relative paths."""
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    for p in sorted(root.glob("*")):
        for cand in [p] + sorted(x for x in p.glob("*") if x.is_dir()):
            if all((cand / r).exists() for r in required):
                return cand
    return None

if ON_KAGGLE and not (REPO / "config" / "analysis_config.yaml").exists():
    src = _find_attached("config/analysis_config.yaml")
    if src is not None:
        print(f"Using repo attached as a dataset: {src}")
        shutil.copytree(src, REPO, dirs_exist_ok=True)
    else:
        print("Cloning the repo (Internet must be on in notebook settings) ...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
sys.path.insert(0, str(REPO / "scripts" / "training"))
os.environ["PDAC_REPO"] = str(REPO)

# These notebooks are generated FROM the repo but, on Kaggle, run AGAINST a clone of it — so
# an uncommitted or unpushed change is invisible here no matter how current the notebook is.
# When the clone predates the notebook the symptom lands far from the cause: an old test
# asserting old counts, a script missing a flag this notebook passes. Checking the contract
# up front turns that into one clear message.
_REQUIRED = ["config/analysis_config.yaml", "scripts/training/tasks.py",
             "scripts/analysis/build_per_case_table.py", "scripts/analysis/make_figures.py",
             "src/trainers/budget_trainers.py", "scripts/kaggle/pack_for_kaggle.py"]
_missing = [r for r in _REQUIRED if not (REPO / r).exists()]
if _missing:
    raise RuntimeError(
        "The repo this notebook is running against is older than the notebook itself.\\n"
        f"  missing: {_missing}\\n"
        f"  repo:    {REPO}\\n"
        "These notebooks clone " + REPO_URL + ", so local commits only reach Kaggle "
        "once they are PUSHED. "
        "Either push, or upload the repo as a Kaggle Dataset and attach it "
        "(the bootstrap prefers an attached input containing config/analysis_config.yaml).")

# --- nnU-Net environment --------------------------------------------------------------------
# raw and preprocessed are regenerable and enormous -> SCRATCH.
# results holds checkpoints, which are neither -> WORK, under the budget guard below.
os.environ["nnUNet_raw"]          = str(SCRATCH / "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = str(SCRATCH / "nnUNet_preprocessed")
os.environ["nnUNet_results"]      = str(WORK / "nnUNet_results")
for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    Path(os.environ[k]).mkdir(parents=True, exist_ok=True)

# --- shell / install helpers -------------------------------------------------------------------
def sh(cmd, cwd=None, check=True):
    """Run a shell command from the repo root, streaming output into the notebook."""
    cwd = str(cwd or REPO)
    print(f"$ {cmd}")
    p = subprocess.run(cmd, shell=True, cwd=cwd, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    return p.returncode

def pip_install(pkgs, quiet=True):
    sh(f"{sys.executable} -m pip install {'-q ' if quiet else ''}--no-warn-script-location {pkgs}")

def gpu_info():
    try:
        import torch
    except ImportError:
        print("torch not installed yet"); return None
    if not torch.cuda.is_available():
        print("No CUDA device. Turn on a GPU accelerator in notebook settings."); return None
    name = torch.cuda.get_device_name(0)
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {name}  ({gb:.1f} GB)")
    return {"name": name, "vram_gb": round(gb, 1)}

# --- the 20 GB guard ------------------------------------------------------------------------------
def dir_gb(path):
    path = Path(path)
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9

def disk_report(detail=True):
    """What the session is holding, split by which limit it counts against."""
    out = dir_gb(WORK)
    free_scratch = shutil.disk_usage(SCRATCH).free / 1e9
    print(f"OUTPUT (counts against the {OUTPUT_LIMIT_GB:.0f}/20 GB cap): {out:.2f} GB")
    if detail:
        for sub in sorted(p for p in WORK.iterdir() if p.is_dir()):
            g = dir_gb(sub)
            if g > 0.01:
                print(f"    {g:7.2f} GB  {sub.name}/")
    print(f"SCRATCH (wiped with the session, not capped): {dir_gb(SCRATCH):.2f} GB used, "
          f"{free_scratch:.0f} GB free")
    return out

def enforce_output_budget(limit_gb=None, where=""):
    """Get the output back under budget, and only then complain if it cannot be done.

    Raising alone would not help: Kaggle refuses the save regardless of what the notebook
    thinks, so an over-budget session loses its GPU hours either way. This frees space in
    increasing order of regret and re-measures after each step, so the common case (a
    checkpoint the study never evaluates) is handled silently and only a genuine overrun
    reaches the user.
    """
    limit = OUTPUT_LIMIT_GB if limit_gb is None else limit_gb
    used = dir_gb(WORK)
    if used <= limit:
        print(f"output {used:.2f} / {limit:.0f} GB{' at ' + where if where else ''}  OK")
        return used

    print(f"output {used:.2f} GB is over the {limit:.0f} GB budget — freeing space")

    # 1. Checkpoints this study never reads. Zero regret.
    prune_checkpoints()
    used = dir_gb(WORK)

    # 2. Softmax dumps and validation scratch. Nothing here reads them either; they only
    #    appear if a training command was run with --npz, which this repo no longer does.
    if used > limit:
        for pattern in ("*.npz", "*.pkl"):
            for f in Path(os.environ["nnUNet_results"]).rglob(pattern):
                print(f"  removed {f.name}"); f.unlink()
        used = dir_gb(WORK)

    # 3. Resume checkpoints for runs that finished. Costs the ability to resume a run that
    #    has nothing left to resume.
    if used > limit:
        prune_checkpoints(keep_latest_if_unfinished=False)
        used = dir_gb(WORK)

    if used > limit:
        disk_report()
        raise RuntimeError(
            f"Output is still {used:.1f} GB after pruning, over the {limit:.0f} GB budget"
            f"{' at ' + where if where else ''}. Kaggle will refuse to save this session. "
            "Drop this task's checkpoint (DROP_CHECKPOINT = True) if its predictions are "
            "already written — every analysis except the receptive-field measurement reads "
            "predictions, not weights.")
    print(f"output now {used:.2f} / {limit:.0f} GB  OK")
    return used

def time_left_h():
    return SESSION_HOURS - (time.time() - SESSION_T0) / 3600.0

def check_time(where=""):
    left = time_left_h()
    print(f"{left:.2f} h left of the {SESSION_HOURS:.1f} h session budget"
          f"{' at ' + where if where else ''}")
    return left

def prune_checkpoints(root=None, keep_latest_if_unfinished=True):
    """Keep exactly what the study needs from each run's directory.

    nnU-Net writes checkpoint_best, checkpoint_latest and checkpoint_final. Evaluation in this
    study is on checkpoint_final only (the pre-registered schedule has no early stopping), so
    best is always removable, and latest is removable the moment final exists. Left alone,
    three checkpoints per run is three times the storage for no gain.
    """
    root = Path(root or os.environ["nnUNet_results"])
    freed = 0.0
    for fold_dir in sorted(p for p in root.rglob("fold_*") if p.is_dir()):
        final = fold_dir / "checkpoint_final.pth"
        drop = [fold_dir / "checkpoint_best.pth"]
        if final.exists() or not keep_latest_if_unfinished:
            drop.append(fold_dir / "checkpoint_latest.pth")
        for f in drop:
            if f.exists():
                freed += f.stat().st_size / 1e9
                f.unlink()
                print(f"  removed {f.relative_to(root)}")
    if freed:
        print(f"freed {freed:.2f} GB")
    return freed

# --- cross-session state --------------------------------------------------------------------------
def save_state(*rel_paths):
    """Copy repo-relative paths into WORK/state so they survive as notebook output."""
    for rel in rel_paths:
        src = REPO / rel
        if not src.exists():
            print(f"  (skip, absent) {rel}"); continue
        dst = STATE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(
            src, dst, **({"dirs_exist_ok": True} if src.is_dir() else {}))
        print(f"  saved {rel}")

def restore_state(*rel_paths, required=True):
    """Restore from WORK/state or from any attached dataset holding a state/ directory."""
    sources = [STATE]
    if Path("/kaggle/input").exists():
        sources += sorted(Path("/kaggle/input").rglob("state"))
    missing = []
    for rel in rel_paths:
        for base in sources:
            src = base / rel
            if src.exists():
                dst = REPO / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                (shutil.copytree if src.is_dir() else shutil.copy2)(
                    src, dst, **({"dirs_exist_ok": True} if src.is_dir() else {}))
                print(f"  restored {rel}  <- {base}")
                break
        else:
            missing.append(rel)
    if missing:
        msg = ("Missing state: " + ", ".join(missing) + "\\n  Run the preparation notebook, "
               "then attach its output here (Add Input -> Your Work).")
        if required:
            raise FileNotFoundError(msg)
        print("  " + msg)
    return not missing

def restore_chunks(dest, manifest_name="pack_manifest.json"):
    """Unpack a chunked dataset produced by scripts/kaggle/pack_for_kaggle.py.

    Large reusable data (the preprocessed cohort) cannot travel as notebook output — that is
    what the 20 GB cap forbids — so it travels as an attached dataset in size-bounded parts.
    This finds the manifest in any attached input and extracts every part into dest.
    """
    root = Path("/kaggle/input")
    if not root.exists():
        return False
    for man_path in sorted(root.rglob(manifest_name)):
        man = json.loads(man_path.read_text())
        parts = man.get("parts", [])
        print(f"Found a {len(parts)}-part pack at {man_path.parent} "
              f"({man.get('uncompressed_bytes', 0)/1e9:.1f} GB)")
        dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
        for entry in parts:
            src = man_path.parent / entry["name"]
            if not src.exists():
                print(f"  MISSING {entry['name']} — attach every part, not just some"); continue
            with tarfile.open(src) as tar:
                tar.extractall(dest)
            print(f"  extracted {entry['name']} ({entry['n_files']} files)")
        return True
    return False

print(f"ON_KAGGLE={ON_KAGGLE}\\nREPO={REPO}\\nWORK={WORK}\\nSCRATCH={SCRATCH}\\n"
      f"DATA_ROOT={DATA_ROOT}\\nbudgets: {SESSION_HOURS} h session, {OUTPUT_LIMIT_GB} GB output")
'''

ENV_INSTALL = '''
# Analysis environment. Kaggle already ships numpy/pandas/scipy/matplotlib; these are the rest.
pip_install("SimpleITK nibabel openpyxl pyyaml statsmodels zenodo-get "
            "'surface-distance @ git+https://github.com/google-deepmind/surface-distance.git'")
import importlib
for m in ("SimpleITK", "surface_distance", "statsmodels", "yaml", "pandas", "scipy"):
    importlib.import_module(m)
print("analysis environment OK")
'''

NNUNET_INSTALL = '''
# nnU-Net from master: the PrimusV2 trainers are not guaranteed to be in the PyPI release.
# The commit is pinned into the frozen config the first time this runs, so every later session
# and every collaborator gets the same one.
import yaml
frozen_path = REPO / "config" / "frozen_thresholds.yaml"
frozen = (yaml.safe_load(open(frozen_path)) or {}) if frozen_path.exists() else {}
pinned = (frozen.get("architecture") or {}).get("nnunet_commit")
pinned = pinned.split("@")[-1].strip() if pinned and "@" in str(pinned) else None

spec = "git+https://github.com/MIC-DKFZ/nnUNet.git" + (f"@{pinned}" if pinned else "")
print(f"Installing nnunetv2 from {spec}")
pip_install(f"'nnunetv2 @ {spec}'")
import nnunetv2
sh("pip freeze | grep -i nnunet")
'''

INSTALL_TRAINERS = '''
# Put this study's custom trainers where nnU-Net's class finder looks. A shim module rather
# than a copy, so each trainer's own __file__ still points into the repo — budget_trainers.py
# and seed_variant_trainers.py both read the frozen config relative to it.
import nnunetv2
variants = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants"
(variants / "pdac_custom_trainers.py").write_text(textwrap.dedent(f"""
    import sys
    if {str(REPO)!r} not in sys.path:
        sys.path.insert(0, {str(REPO)!r})
    from src.trainers.primus_identity_trainer import nnUNet_PrimusV2_Identity_Trainer  # noqa: F401
    from src.trainers.seed_variant_trainers import *  # noqa: F401,F403
    from src.trainers.budget_trainers import *  # noqa: F401,F403
"""))

from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
search = str(Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer")
def trainer_exists(name):
    return recursive_find_python_class(
        search, name, current_module="nnunetv2.training.nnUNetTrainer") is not None
for name in (PRIMUS_TRAINER, "nnUNet_PrimusV2_Identity_Trainer",
             "nnUNetTrainer" + BUDGET_SUFFIX, PRIMUS_TRAINER + BUDGET_SUFFIX):
    print(f"  {'OK  ' if trainer_exists(name) else 'MISS'} {name}")
'''
