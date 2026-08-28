#!/usr/bin/env python
"""Generate the three Kaggle notebooks from one source of truth.

    python notebooks/build_notebooks.py

Three notebooks, not ten: one preparation pass, one worker that is launched many times
concurrently, and one analysis pass. They share a bootstrap, a disk-budget guard, and a
cross-session state mechanism, all of which live in nb_common.py. Edit this file or that one,
re-run, and commit both the source and the .ipynb files: Kaggle needs something to upload, and
a reviewer should be able to read the notebooks without running anything.
"""
from nb_common import (BOOTSTRAP, ENV_INSTALL, INSTALL_TRAINERS, NNUNET_INSTALL, OUT, code,
                       json, md, notebook)

NOTEBOOKS = {}

# ======================================================================================
NOTEBOOKS["00_prepare"] = (False, [
    md("""
# 00 · Prepare — cohort, strata, folds, freeze, preprocessing

**Run once, on CPU.** Everything downstream reads what this notebook freezes, and nothing
downstream is meaningful if this is re-run afterwards: each script here refuses to overwrite
its own frozen section, so a second run errors rather than quietly re-rolling the folds.

In order: the pipeline smoke test, the PANORAMA download, content-hash deduplication, the
volume and CNR strata, the patient-level folds, the leave-one-source-out folds, the nnU-Net
dataset, preprocessing, and the matched-budget architecture record.

## The two limits, and how this design stays under them

Kaggle kills a session at 12 hours and refuses to save an output over 20 GB. Both fail
silently — the session simply ends, or the output simply does not save — so both are budgeted
with headroom (11 h, 17 GB) and checked by `enforce_output_budget()` and `check_time()` rather
than hoped for.

The important structural point: **the 20 GB cap is per notebook output, and the worker
notebook is launched once per training run.** Twenty-five runs are twenty-five separate
outputs of ~2 GB each, not one 50 GB pile. Fanning out for concurrency is the same move that
fixes storage.

What goes where:

| | Location | Capped? | Holds |
| --- | --- | --- | --- |
| Output | `/kaggle/working` | **yes, 20 GB** | frozen config, splits, one checkpoint per run, predictions, results |
| Scratch | `/kaggle/temp` | no (wiped) | raw images, `nnUNet_preprocessed`, occluded volumes |

Preprocessed data is deliberately *not* in the output: it is large and exactly regenerable.
For a cohort small enough it is regenerated per session; for the full cohort it is staged as
an attached dataset (last section).
"""),
    code(BOOTSTRAP),
    code(ENV_INSTALL),
    code(NNUNET_INSTALL),
    code("""
# --- the pipeline smoke test ---------------------------------------------------------
# 123 checks over every data, analysis, and training script, against synthetic data with a designed
# answer, so it asserts on the statistical outcome rather than on exit codes. If this is
# green the analysis works, and everything after this is data and compute.
sh(f"{sys.executable} tests/run_smoke_test.py")
check_time("after smoke test")
"""),
    code("""
# --- configuration --------------------------------------------------------------------
SUBSET = True     # batch 1 only. Set False only where ~400 GB of disk actually exists.
BATCHES = ["batch_1"] if SUBSET else ["batch_1", "batch_2", "batch_3", "batch_4"]

# If PANORAMA is already attached as a dataset, use it and skip the download entirely.
# That is the intended path once you have the data once: uploading it as a dataset is not
# subject to the 20 GB output cap.
attached = _find_attached("panorama/panorama_labels")
if attached:
    print(f"Using attached PANORAMA at {attached}")
    DATA_ROOT = attached
else:
    print(f"Will download: {BATCHES}")
print(f"free scratch: {shutil.disk_usage(SCRATCH).free/1e9:.0f} GB")
"""),
    code("""
# --- download ---------------------------------------------------------------------------
# zenodo_get resolves each record, downloads every file, and verifies md5 checksums.
if not attached:
    (DATA_ROOT / "panorama" / "images").mkdir(parents=True, exist_ok=True)
    zips = DATA_ROOT / "panorama" / "zips"
    RECORDS = {"batch_1": 13715870, "batch_2": 13742336, "batch_3": 11034011, "batch_4": 10999754}
    for b in BATCHES:
        d = zips / b
        if (d / ".done").exists():
            print(f"{b} already downloaded"); continue
        d.mkdir(parents=True, exist_ok=True)
        sh(f"zenodo_get -o {d} {RECORDS[b]}")
        (d / ".done").touch()
        check_time(f"after {b}")
    for b in BATCHES:
        for z in sorted((zips / b).glob("*.zip")):
            sh(f"unzip -n -q {z} -d {DATA_ROOT / 'panorama' / 'images'}")
    # The zips are dead weight once extracted, and scratch is finite too.
    shutil.rmtree(zips, ignore_errors=True)

    labels = DATA_ROOT / "panorama" / "panorama_labels"
    if not (labels / ".git").exists():
        sh("git lfs install || true", cwd=WORK, check=False)
        sh(f"git clone https://github.com/DIAGNijmegen/panorama_labels {labels}", cwd=WORK)
    commit = subprocess.run(["git", "-C", str(labels), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    (DATA_ROOT / "panorama" / "labels_commit.txt").write_text(commit + chr(10))
    print("labels commit:", commit)

print(f"{len(list((DATA_ROOT / 'panorama' / 'images').rglob('*')))} files under images/")
disk_report()

# Everything below writes to scratch, which is wiped when this session ends. A session that
# runs out of time mid-preprocessing therefore loses the download too — there is no partial
# credit here, so the check is up front rather than after the fact.
left = check_time("after download")
if left < 3.0:
    print("*** Less than 3 h left and preprocessing has not started. Scratch does not survive "
          "the session, so finishing this notebook in a later one is not possible: it would "
          "re-download from scratch.")
    print("    Upload the data once as a Kaggle Dataset from a machine with disk, attach it, "
          "and re-run — the download is then skipped entirely and this notebook fits easily. "
          "That is also the only route that works for the full ~190 GB cohort. ***")
"""),
    code("""
# --- cohort table + content-hash deduplication --------------------------------------------
# Duplicates are found by image content, not case ID: MSD Task07 and NIH Pancreas-CT are
# redistributed inside PANORAMA under different IDs, and ID-based dedup would miss them.
sh(f"{sys.executable} scripts/data/deduplicate.py --data-root {DATA_ROOT} --out splits/cohort.csv")

import pandas as pd, yaml, datetime
cohort = pd.read_csv(REPO / "splits" / "cohort.csv")
dupes  = pd.read_csv(REPO / "splits" / "duplicates.csv")
print(f"{len(cohort)} scans retained, {len(dupes)} exact duplicates removed, "
      f"{cohort['has_manual_lesion'].sum()} with a manual lesion delineation")
if "source" in cohort.columns:
    print(cohort.groupby("source")["case_id"].nunique().to_string())
else:
    print("NO 'source' COLUMN — inspect the clinical_information.xlsx column names printed "
          "above. The source axis, the splits, and leave-one-source-out all depend on it.")
"""),
    code("""
# Record what this cohort came from, so a subset can never be mistaken for the study cohort.
prov = {
    "built": datetime.date.today().isoformat(),
    "data_root": str(DATA_ROOT),
    "batches": "attached-dataset" if attached else BATCHES,
    "subset_of_full_panorama": bool(SUBSET and not attached),
    "labels_commit": ((DATA_ROOT / "panorama" / "labels_commit.txt").read_text().strip()
                      if (DATA_ROOT / "panorama" / "labels_commit.txt").exists() else None),
    "n_scans_retained": int(len(cohort)),
    "n_duplicates_removed": int(len(dupes)),
    "n_manual_lesion_cases": int(cohort["has_manual_lesion"].sum()),
}
(REPO / "splits" / "cohort_provenance.yaml").write_text(yaml.safe_dump(prov, sort_keys=False))
print(yaml.safe_dump(prov, sort_keys=False))
if prov["subset_of_full_panorama"]:
    print("*** SUBSET cohort. Nothing computed from it is a study result. ***")
"""),
    code("""
# --- strata: volume, diameter, CNR at 5/10/15 mm in one pass --------------------------------
# One pass over the images gives the primary ring and both sensitivity rings, so the
# pre-specified 5 mm / 15 mm analysis costs no second read.
sh(f"{sys.executable} scripts/analysis/compute_strata.py --data-root {DATA_ROOT} "
   f"--cohort splits/cohort.csv --out splits/strata.csv")

frozen = yaml.safe_load(open(REPO / "config" / "frozen_thresholds.yaml"))
print(yaml.safe_dump(frozen["strata"], sort_keys=False))
if frozen["strata"]["contrast_axis_exploratory"]:
    print("Pre-registered rule fired: CNR ranks are unstable across ring widths, so the "
          "contrast axis is reported as EXPLORATORY. Pre-specified — not a retreat, and not "
          "revisited.")
check_time("after strata")
"""),
    code("""
# --- folds: patient-level CV, then leave-one-source-out -------------------------------------
# If the raw source values need remapping to the study's canonical groups, set SOURCE_MAPPING;
# every raw value must be covered or make_splits.py fails loudly rather than guessing, and
# whichever grouping is used is frozen into config/frozen_thresholds.yaml.
SOURCE_MAPPING = None   # e.g. {"RUMC": "Radboud", "UMCG": "UMCG", "MSD": "MSKCC", "NIH": "NIH"}
extra = ""
if SOURCE_MAPPING:
    (REPO / "splits" / "source_mapping.json").write_text(json.dumps(SOURCE_MAPPING, indent=1))
    extra = "--source-mapping splits/source_mapping.json"
sh(f"{sys.executable} scripts/data/make_splits.py --cohort splits/cohort.csv "
   f"--strata splits/strata.csv {extra}")

# Deterministic, no seed: each source held out in turn. --min-cases enforces the
# pre-registration's escalation rule rather than emitting a meaningless two-case LOSO fold.
sh(f"{sys.executable} scripts/data/make_loso_splits.py --exclude-source NIH --min-cases 25 "
   f"--emit-combined splits/splits_with_loso.json", check=False)
"""),
    code("""
splits = json.load(open(REPO / "splits" / "splits_final.json"))
fa = pd.read_csv(REPO / "splits" / "fold_assignment.csv")
print(f"{len(splits)} CV folds over {len(fa)} cases / {fa['patient_id'].nunique()} patients")
print(pd.crosstab(fa["fold"], fa["source"]).to_string() if "source" in fa.columns
      else fa["fold"].value_counts().sort_index().to_string())
loso_csv = REPO / "splits" / "loso_folds.csv"
if loso_csv.exists():
    print("\\nLeave-one-source-out folds:")
    print(pd.read_csv(loso_csv).to_string(index=False))
else:
    print("\\nNo LOSO folds: the cell above escalated. Decide (merge, drop, or lower the bar), "
          "record the decision, and re-run that cell with it applied.")
"""),
    code("""
# --- nnU-Net dataset and preprocessing ------------------------------------------------------
# Only manual-lesion cases enter Dataset501; the model-generated delineations are staged
# separately and never enter evaluation. Both live in SCRATCH: large and exactly regenerable.
sh(f"{sys.executable} scripts/data/convert_to_nnunet.py --data-root {DATA_ROOT} "
   f"--cohort splits/cohort.csv")

# Preset pairing. This is the study's only control, so it is measured, not chosen by taste:
# both arms must fit one VRAM ceiling. ResEnc VRAM per the upstream presets doc is M ~9-11 GB,
# L ~24 GB, XL ~40 GB. This notebook has no GPU, so the pairing is provisional here and the
# numbers are measured in the first worker session.
PROVISIONAL_VRAM_GB = 16      # the card the workers will use
if PROVISIONAL_VRAM_GB >= 40:
    CNN_PLANS, PRIMUS_TRAINER = "nnUNetResEncUNetXLPlans", "nnUNet_PrimusV2L_Trainer"
elif PROVISIONAL_VRAM_GB >= 22:
    CNN_PLANS, PRIMUS_TRAINER = "nnUNetResEncUNetLPlans", "nnUNet_PrimusV2M_Trainer"
else:
    CNN_PLANS, PRIMUS_TRAINER = "nnUNetResEncUNetMPlans", "nnUNet_PrimusV2S_Trainer"
PLANNER = "nnUNetPlannerResEnc" + CNN_PLANS.replace("nnUNetResEncUNet", "").replace("Plans", "")
print(f"CNN {CNN_PLANS} (planner {PLANNER}) | transformer {PRIMUS_TRAINER}")

sh(f"nnUNetv2_plan_and_preprocess -d 501 -pl {PLANNER} --verify_dataset_integrity")
check_time("after preprocessing")
"""),
    code("""
# --- install the frozen splits into the preprocessed dataset ---------------------------------
# Both arms must train on identical folds, so the frozen file is copied in rather than letting
# nnU-Net generate its own. With LOSO folds present the combined file goes in: it keeps the
# five CV folds byte-identical at indices 0-4 and appends LOSO at 5, 6, ...
ds_dir = Path(os.environ["nnUNet_preprocessed"]) / "Dataset501_PDAC"
src = REPO / "splits" / ("splits_with_loso.json"
                         if (REPO / "splits" / "splits_with_loso.json").exists()
                         else "splits_final.json")
shutil.copy(src, ds_dir / "splits_final.json")
print(f"installed {src.name}: {len(json.load(open(ds_dir / 'splits_final.json')))} folds")

plans = json.load(open(ds_dir / f"{CNN_PLANS}.json"))
patch_xyz = plans["configurations"]["3d_fullres"]["patch_size"]
grid = [p // 8 for p in patch_xyz]
print(f"patch {patch_xyz}, divides by the 8x8x8 Primus tokenizer stride: "
      f"{all(p % 8 == 0 for p in patch_xyz)}, token grid {grid} = {grid[0]*grid[1]*grid[2]}")
"""),
    code("""
# --- how the preprocessed data reaches the workers --------------------------------------------
# This is the decision the 20 GB cap actually forces. Preprocessed data is far too large to be
# notebook output for a real cohort, so there are exactly two honest options, and which one
# applies is a measurement, not a preference.
pre_gb = dir_gb(ds_dir)
print(f"preprocessed Dataset501_PDAC: {pre_gb:.1f} GB")
print(f"raw (SCRATCH): {dir_gb(os.environ['nnUNet_raw']):.1f} GB")

if pre_gb <= OUTPUT_LIMIT_GB - dir_gb(WORK) - 1:
    print("\\nSmall enough to travel as this notebook's output. Packing it into parts so the "
          "worker notebook can attach it and skip preprocessing entirely.")
    sh(f"{sys.executable} scripts/kaggle/pack_for_kaggle.py --src {ds_dir} "
       f"--out {WORK / 'preprocessed_pack'} --slug pdac-preprocessed "
       f"--title 'PDAC nnU-Net preprocessed (Dataset501)' --chunk-gb 4")
else:
    print(f"\\n{pre_gb:.0f} GB does NOT fit in a notebook output, and no amount of chunking "
          "changes that — the cap is on the whole output.")
    print("Two options, both fine, neither of them 'squeeze it in':")
    print("  A. Let each worker regenerate it. Every worker session then spends its first "
          "hour or two preprocessing before it trains. Correct, just wasteful.")
    print("  B. Stage it once as a dataset, which is NOT subject to the 20 GB cap. On a "
          "machine with the data and the disk:")
    print("       python scripts/kaggle/pack_for_kaggle.py --src <nnUNet_preprocessed> \\\\")
    print("           --out /tmp/pack --slug pdac-preprocessed --chunk-gb 15")
    print("       cd /tmp/pack && kaggle datasets create -d -r skip")
    print("     Then attach it to the worker notebook; restore_chunks() unpacks it and "
          "preprocessing is skipped.")
disk_report()
"""),
    code("""
# --- freeze and hand off ------------------------------------------------------------------------
save_state("splits", "config/frozen_thresholds.yaml", "preregistration/DEVIATIONS.md")
enforce_output_budget(where="end of preparation")

# The full task list the worker notebook is launched against, so "which runs are left" is a
# set difference rather than a memory exercise.
sh(f"{sys.executable} scripts/training/tasks.py")
"""),
    md("""
## Before any training: tag `prereg-v1`, on your own machine

Kaggle has no push credentials, and the tag is the whole point of the pre-registration. Copy
`splits/` and `config/frozen_thresholds.yaml` out of this notebook's output, then:

```bash
git add config/frozen_thresholds.yaml splits/
git commit -m "Freeze cohort strata, folds, and source grouping"
git tag -a prereg-v1 -m "Pre-registration frozen before any training run"
git push origin main --tags
bash scripts/check_prereg_tag.sh          # must print PASS
```

That check fails if the tag moves, if history is rewritten under it, or if any frozen file
differs between the tag and HEAD. Run it before every analysis session.

## Then: **Save Version → Save & Run All**

The output carries the frozen config, the splits, and (when it fits) the packed preprocessed
data. Attach this session to every worker.

Next: **01_run**, launched once per task — concurrently.
"""),
])

# ======================================================================================
NOTEBOOKS["01_run"] = (True, [
    md("""
# 01 · Run one task — training, then everything that needs this model

**Set `TASK` in the first cell and launch. This notebook is the unit of concurrency.** Run as
many copies at once as your Kaggle quota allows: each one writes only its own output, reads
nothing another worker writes, and needs no coordination. Nothing here has to be run in order.

`python scripts/training/tasks.py` lists all 25 task names. To launch several at once, fork
this notebook once per task and change the one line.

| Tier | Tasks |
| --- | --- |
| A | `cnn-fold0` … `cnn-fold4`, `tf-fold0` … `tf-fold4` |
| B | `cnn-loso_<Source>`, `tf-loso_<Source>`, `identity-fold0` … `identity-fold4`, `<arm>-fold<r>-seed1/2` |

## What one session does, and why it stays inside both limits

1. trains, stopping cleanly ~1.5 h before the session cap with a checkpoint on disk
2. if the run finished, writes **its own** out-of-fold predictions
3. occludes and re-infers **only the cases this fold held out** — the correct pairing (the
   model never saw them) and the only version with a bounded disk cost
4. predicts the NIH negatives when this task is the one that should
5. prunes every checkpoint the study does not evaluate, and asserts the output budget

**Time.** A pre-registered run is ~250,000 steps and spans many sessions. The trainer stops on
a wall-clock budget and raises rather than returning, because returning normally would make
nnU-Net write `checkpoint_final.pth` and mark an unfinished run as finished. Re-launch with
the same `TASK` and it resumes from `checkpoint_latest.pth`.

`RESERVE_HOURS` is held back from the training budget and has to cover three things, only one
of which this notebook controls: nnU-Net's own end-of-training validation pass over the
held-out fold (which happens inside training, after the last epoch), the occlusion inference,
and saving. Two hours is a starting estimate, not a measurement — **the first worker session
tells you the real numbers**, and the sensible response to running short is not a bigger
reserve but a second session: re-launching a finished task skips training entirely and does
inference only, resuming any prediction directory that was left partial.

**Storage.** One session holds one checkpoint plus a few hundred MB of predictions — about
2 GB against a 20 GB cap. The cap is per output, so twenty-five runs are twenty-five 2 GB
outputs, not one 50 GB pile. Preprocessed data never touches the output: it comes from an
attached dataset, or is regenerated into scratch.
"""),
    code("""
# ==========================================================================================
TASK = "cnn-fold0"     # one of `python scripts/training/tasks.py`
EPOCHS = None          # None = the frozen 1000-epoch schedule. A number is a DEVIATION.
RESERVE_HOURS = 2.0    # see the note below on what this has to cover
# ==========================================================================================
"""),
    code(BOOTSTRAP),
    code(ENV_INSTALL),
    code("""
restore_state("splits", "config/frozen_thresholds.yaml", "preregistration/DEVIATIONS.md")
import yaml, pandas as pd
frozen = yaml.safe_load(open(REPO / "config" / "frozen_thresholds.yaml"))
arch = frozen.get("architecture")

# The transformer trainer name is needed before nnU-Net is installed (the trainer shim imports
# it), so fall back to the config's candidate list until the first worker records the real one.
cfg = yaml.safe_load(open(REPO / "config" / "analysis_config.yaml"))
PRIMUS_TRAINER = (arch or {}).get("transformer", {}).get("trainer")
CNN_PLANS = (arch or {}).get("cnn", {}).get("plans")
if not PRIMUS_TRAINER:
    PRIMUS_TRAINER = "nnUNet_PrimusV2S_Trainer"
    CNN_PLANS = "nnUNetResEncUNetMPlans"
    print("No architecture record yet — using the provisional pairing and measuring it below. "
          "The first worker to get here freezes the measured numbers.")
os.environ["PRIMUS_BASE_TRAINER"] = PRIMUS_TRAINER

os.environ["PDAC_SAVE_EVERY"] = "5"
if EPOCHS:
    os.environ["PDAC_EPOCHS"] = str(int(EPOCHS))
BUDGET_SUFFIX = "_budget" + (f"{int(EPOCHS)}ep" if EPOCHS else "")
print(f"TASK={TASK}  CNN={CNN_PLANS}  transformer={PRIMUS_TRAINER}  suffix={BUDGET_SUFFIX}")
gpu = gpu_info()
"""),
    code(NNUNET_INSTALL),
    code(INSTALL_TRAINERS),
    code("""
# --- resolve the task ------------------------------------------------------------------------
# One place maps a task name to its trainer, plans, nnU-Net fold index, and results directory,
# so the run log, the checkpoint location, and the predictions can never disagree.
sys.path.insert(0, str(REPO / "scripts" / "training"))
import importlib, tasks as task_mod
importlib.reload(task_mod)
T = task_mod.parse(TASK, repo=REPO, budget_suffix=BUDGET_SUFFIX)
print(json.dumps(T, indent=1))

if EPOCHS:
    print("=" * 78)
    print(f"DEVIATION: {EPOCHS} epochs instead of the frozen {cfg['training']['epochs']}.")
    print("This run writes to its own results directory and is not a study result until the")
    print("deviation is recorded in preregistration/DEVIATIONS.md.")
    print("=" * 78)
    import datetime
    dev = REPO / "preregistration" / "DEVIATIONS.md"
    dev.write_text(dev.read_text().replace(
        "| — | — | none to date | — | — |",
        f"| {datetime.date.today().isoformat()} | training.epochs | "
        f"{cfg['training']['epochs']} -> {EPOCHS} | compute budget | (not yet tagged) |"
        + chr(10) + "| — | — | none to date | — | — |"))
"""),
    code("""
# --- preprocessed data: attached dataset, or regenerate into scratch ---------------------------
# Never into the output: it is large and exactly regenerable, and the 20 GB cap is for things
# that are neither.
ds_dir = Path(os.environ["nnUNet_preprocessed"]) / "Dataset501_PDAC"
if not (ds_dir / "dataset.json").exists():
    if restore_chunks(Path(os.environ["nnUNet_preprocessed"])):
        print("preprocessed data restored from an attached pack — preprocessing skipped")
    else:
        # Preprocessing is not interruptible and not time-checkable once started, so the
        # check is here. Attaching a preprocessed pack removes this risk entirely and is the
        # single biggest thing you can do for the session budget.
        if time_left_h() < 4:
            raise RuntimeError(
                f"Only {time_left_h():.1f} h left and preprocessing has not started. It is not "
                "interruptible, so starting now risks spending the whole session on it and "
                "training for nothing. Attach a preprocessed pack (see "
                "scripts/kaggle/pack_for_kaggle.py) or start a fresh session.")
        print("No preprocessed pack attached; regenerating (expected on a fresh session).")
        sh(f"{sys.executable} scripts/data/convert_to_nnunet.py --data-root {DATA_ROOT} "
           f"--cohort splits/cohort.csv")
        planner = ("nnUNetPlannerResEnc"
                   + CNN_PLANS.replace("nnUNetResEncUNet", "").replace("Plans", ""))
        sh(f"nnUNetv2_plan_and_preprocess -d 501 -pl {planner}")

src = REPO / "splits" / ("splits_with_loso.json"
                         if (REPO / "splits" / "splits_with_loso.json").exists()
                         else "splits_final.json")
shutil.copy(src, ds_dir / "splits_final.json")
print(f"{len(json.load(open(ds_dir / 'splits_final.json')))} folds installed from {src.name}")
check_time("before training")
"""),
    code("""
# --- record the matched-budget architecture, if this is the first worker to get here -----------
# Both arms measured on the same card at the same patch size. record_arch_stats.py writes it
# once and refuses to run again, so whichever worker arrives first freezes it and the rest
# read it back.
if not arch:
    import torch, time as _time
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    import nnunetv2

    plans = json.load(open(ds_dir / f"{CNN_PLANS}.json"))
    patch_xyz = plans["configurations"]["3d_fullres"]["patch_size"]
    dj = json.load(open(ds_dir / "dataset.json"))
    n_in, n_out = len(dj["channel_names"]), len(dj["labels"])

    def measure(build, label, batch=2, steps=3):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        net = build().cuda().train()
        n_params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.SGD(net.parameters(), lr=1e-2)
        x = torch.randn(batch, n_in, *patch_xyz[::-1], device="cuda")
        y = torch.randint(0, n_out, (batch, *patch_xyz[::-1]), device="cuda")
        t0 = None
        for i in range(steps + 1):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(x)
                out = out[0] if isinstance(out, (list, tuple)) else out
                loss = torch.nn.functional.cross_entropy(out.float(), y)
            loss.backward(); opt.step(); torch.cuda.synchronize()
            if i == 0:
                t0 = _time.time()     # first step includes allocator warm-up; not timed
        step_s = (_time.time() - t0) / steps
        peak = torch.cuda.max_memory_allocated() / 1e9
        del net, opt, x, y; torch.cuda.empty_cache()
        print(f"{label}: {n_params/1e6:.1f}M params, peak {peak:.1f} GB, {step_s:.2f} s/step")
        return n_params, peak, step_s

    pm = PlansManager(plans); cm = pm.get_configuration("3d_fullres")
    cnn_p, cnn_v, cnn_s = measure(
        lambda: get_network_from_plans(cm.network_arch_class_name, cm.network_arch_init_kwargs,
                                       cm.network_arch_init_kwargs_req_import, n_in, n_out,
                                       allow_init=True, deep_supervision=False), "CNN   ")
    search = str(Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer")
    PrimusCls = recursive_find_python_class(search, PRIMUS_TRAINER,
                                            current_module="nnunetv2.training.nnUNetTrainer")
    try:
        tf_p, tf_v, tf_s = measure(
            lambda: PrimusCls.build_network_architecture(
                cm.network_arch_class_name, cm.network_arch_init_kwargs,
                cm.network_arch_init_kwargs_req_import, n_in, n_out, False), "Primus")
    except Exception as e:
        raise SystemExit(
            f"Could not build {PRIMUS_TRAINER}'s network directly ({type(e).__name__}: {e}). "
            "Measure the transformer arm from a real 1-epoch run and pass the numbers to "
            "record_arch_stats.py by hand. Do not record estimates: the matched-budget "
            "ceiling is the study's only control, and an unmeasured ceiling is not one.")

    ratio = max(cnn_v, tf_v) / max(min(cnn_v, tf_v), 1e-9)
    print(f"VRAM ratio between arms: {ratio:.2f}x")
    if ratio > 1.25:
        print("*** The arms are NOT on a matched VRAM budget. That breaks the study's only "
              "control and is a same-day escalation. Fix the pairing before training. ***")
    sh(f"{sys.executable} scripts/training/record_arch_stats.py "
       f"--patch-size {patch_xyz[2]} {patch_xyz[1]} {patch_xyz[0]} "
       f'--nnunet-commit "$(pip freeze | grep -i nnunetv2)" --gpu "{(gpu or {}).get("name","unknown")}" '
       f"--cnn-plans {CNN_PLANS} --cnn-params {cnn_p} --cnn-vram-gb {cnn_v:.2f} "
       f"--cnn-step-time-s {cnn_s:.3f} --primus-trainer {PRIMUS_TRAINER} "
       f"--tf-params {tf_p} --tf-vram-gb {tf_v:.2f} --tf-step-time-s {tf_s:.3f}", check=False)
    save_state("config/frozen_thresholds.yaml")
else:
    print("architecture already frozen:")
    print(yaml.safe_dump(arch, sort_keys=False))
"""),
    code("""
# --- restore this task's checkpoint from an earlier session -------------------------------------
# Attach the previous session of THIS task (Add Input -> Your Work). Only this task's
# directory is copied: pulling in other workers' checkpoints would spend the output budget on
# models this session has no use for.
RES = Path(os.environ["nnUNet_results"])
out_dir = RES / "Dataset501_PDAC" / T["results_subdir"] / f"fold_{T['nnunet_fold']}"
restored = 0
if Path("/kaggle/input").exists():
    for cand in Path("/kaggle/input").rglob(f"Dataset501_PDAC/{T['results_subdir']}"):
        for f in cand.rglob("*"):
            if f.is_file() and f.suffix in (".pth", ".json", ".txt", ".pkl"):
                dst = RES / "Dataset501_PDAC" / T["results_subdir"] / f.relative_to(cand)
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(f, dst); restored += 1
print(f"restored {restored} file(s) for {T['results_subdir']}")

final = out_dir / "checkpoint_final.pth"
latest = out_dir / "checkpoint_latest.pth"
print("FINISHED" if final.exists() else ("resuming from checkpoint_latest.pth"
                                         if latest.exists() else "fresh start"))
disk_report()
"""),
    code("""
# --- train ---------------------------------------------------------------------------------------
# The wall-clock budget is whatever is left minus the reserve, so a session that spent two
# hours preprocessing trains for correspondingly less and still saves cleanly.
if final.exists():
    print("Already finished — skipping training.")
else:
    budget = max(0.25, time_left_h() - RESERVE_HOURS)
    os.environ["PDAC_MAX_HOURS"] = f"{budget:.3f}"
    print(f"training budget for this session: {budget:.2f} h")

    commit = subprocess.run("pip freeze | grep -i nnunetv2", shell=True, capture_output=True,
                            text=True).stdout.strip() or "unknown"
    run_id = subprocess.run(
        [sys.executable, "scripts/training/log_run.py", "start", "--arm", T["arm"],
         "--fold", T["fold_label"], "--seed", T["seed"], "--nnunet-commit", commit,
         "--trainer-or-plans", T["trainer"] + (f" / {T['plans']}" if T["plans"] else ""),
         "--gpu", (gpu or {}).get("name", "unknown"),
         "--notes", f"kaggle worker; task={TASK}; budget={budget:.2f}h"],
        cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    print("run_id:", run_id)

    cmd = f"nnUNetv2_train {T['train_cmd_args']}" + (" --c" if latest.exists() else "")
    rc = sh(cmd, check=False)

    if final.exists():
        status, note = "completed", "run finished"
    elif latest.exists():
        status, note = "crashed", "PAUSED on the wall-clock budget; re-launch to resume"
        print("\\n>>> Paused, not failed. Save this version, attach it to the next session, "
              "and run the same TASK again.")
    else:
        status, note = "crashed", f"no checkpoint written; exit {rc}"
    sh(f"{sys.executable} scripts/training/log_run.py finish --run-id {run_id} "
       f"--status {status} --checkpoint-path {final} --notes \\"{note}; exit {rc}\\"")

check_time("after training")
disk_report()
"""),
    code("""
# --- predictions, but only if the run actually finished -------------------------------------------
# nnU-Net writes this fold's held-out predictions during on_train_end, so out-of-fold
# predictions come out of training itself rather than a second inference pass.
FINISHED = final.exists()
val_dir = out_dir / "validation"
arm_tag = {"cnn": "cnn", "tf": "tf", "identity": "identity_control"}[T["arm_key"]]

if FINISHED and val_dir.exists():
    dst = PREDS / arm_tag / T["pred_key"]
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in val_dir.glob("*.nii.gz"):
        shutil.copy2(f, dst / f.name); n += 1
    print(f"{n} out-of-fold predictions -> {dst}")
else:
    print("Run not finished — no predictions this session. Everything below is skipped.")
"""),
    code("""
# --- occlusion, for this fold's held-out cases only (H2) --------------------------------------------
# Occluding the whole cohort centrally writes three copies of every image. Doing it per fold
# keeps the disk cost proportional to one fold AND gets the pairing right: every case here was
# held out by this model, so its occlusion Dice loss is measured on a case it never saw.
# Replicates and LOSO runs are excluded: the occlusion test is defined on the two arms'
# cross-validation folds, and running it again per replicate would re-do identical work into a
# path that already has an owner.
DO_OCCLUSION = (FINISHED and T["arm_key"] in ("cnn", "tf")
                and not T["is_replicate"] and not T["fold_label"].startswith("loso_"))
if DO_OCCLUSION and time_left_h() > 0.5:
    OCC = SCRATCH / "occlusion"
    sh(f"{sys.executable} scripts/analysis/occlusion_test.py --stage occlude "
       f"--cohort splits/cohort.csv --strata splits/strata.csv --workdir {OCC} "
       f"--fold {T['nnunet_fold']}")
    for lo, hi in cfg["hypotheses"]["h2"]["occlusion_shells_mm"]:
        if time_left_h() < 0.3:
            print("Out of session time — remaining shells next session."); break
        src_dir = OCC / "occluded" / f"shell_{lo}_{hi}" / "imagesTs"
        dst_dir = PREDS / "occlusion" / arm_tag / f"shell_{lo}_{hi}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        flag, value = ("-p", T["plans"]) if T["plans"] else ("-tr", T["trainer"])
        extra = f" -tr {T['trainer']}" if T["plans"] else ""
        # --continue_prediction: skip cases already written, so a session that ran out of
        # time mid-shell resumes instead of redoing an hour of inference.
        sh(f"nnUNetv2_predict -i {src_dir} -o {dst_dir} -d 501 -c 3d_fullres "
           f"-f {T['nnunet_fold']} {flag} {value}{extra} --continue_prediction", check=False)
        n_done = len(list(dst_dir.glob("*.nii.gz")))
        n_want = len(list(src_dir.glob("*.nii.gz")))
        print(f"shell {lo}-{hi}: {n_done}/{n_want} predictions"
              + ("" if n_done >= n_want else "  (partial — re-launch this task to finish)"))
        enforce_output_budget(where=f"after shell {lo}-{hi}")
else:
    print("Occlusion skipped for this task (it applies to the two arms' CV folds).")
"""),
    code("""
# --- NIH negative control ------------------------------------------------------------------------------
# The NIH cases have no lesions and are in no training fold, so they need a real inference pass.
# In domain = a CV-fold model; under source shift = a leave-one-source-out model, which never
# saw the acquisition source it is being asked about. Only the tasks that should do this, do it.
cohort = pd.read_csv(REPO / "splits" / "cohort.csv")
nih = cohort[cohort.get("source", pd.Series(dtype=str)) == "NIH"] if "source" in cohort else cohort.iloc[0:0]
CONDITION = (None if T["is_replicate"]
             else "in_domain" if T["fold_label"] == "fold0"
             else "source_shift" if T["fold_label"].startswith("loso_") else None)

if FINISHED and CONDITION and T["arm_key"] in ("cnn", "tf") and len(nih) and time_left_h() > 0.4:
    import SimpleITK as sitk
    nih_in = SCRATCH / "nih_images"; nih_in.mkdir(parents=True, exist_ok=True)
    for r in nih.itertuples():
        dst = nih_in / f"{r.case_id}_0000.nii.gz"
        if dst.exists():
            continue
        if str(r.path).endswith(".nii.gz"):
            shutil.copy2(r.path, dst)
        else:
            sitk.WriteImage(sitk.ReadImage(str(r.path)), str(dst), useCompression=True)
    out = PREDS / "nih" / arm_tag / CONDITION
    out.mkdir(parents=True, exist_ok=True)
    flag, value = ("-p", T["plans"]) if T["plans"] else ("-tr", T["trainer"])
    extra = f" -tr {T['trainer']}" if T["plans"] else ""
    sh(f"nnUNetv2_predict -i {nih_in} -o {out} -d 501 -c 3d_fullres "
       f"-f {T['nnunet_fold']} {flag} {value}{extra} --continue_prediction", check=False)
    print(f"NIH {arm_tag}/{CONDITION}: {len(list(out.glob('*.nii.gz')))} predictions")
else:
    print(f"NIH inference not this task's job (condition={CONDITION}).")
"""),
    code("""
# --- prune, budget-check, hand off ----------------------------------------------------------------------
# checkpoint_best is never evaluated by this study (the frozen schedule has no early stopping)
# and checkpoint_latest is dead weight once final exists. Left alone that is three checkpoints
# per run for no gain.
prune_checkpoints()

# The ERF measurement needs one real CNN checkpoint; every other analysis reads predictions.
# If the output is tight, dropping this task's checkpoint costs only the ability to re-infer.
DROP_CHECKPOINT = False
if DROP_CHECKPOINT and FINISHED:
    for f in out_dir.glob("checkpoint_*.pth"):
        print("dropping", f.name); f.unlink()

# The run log is per-session; the analysis notebook merges them. Naming it by task means two
# concurrent workers never collide.
log = REPO / "scripts" / "training" / "run_log.csv"
if log.exists():
    (STATE / "run_logs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(log, STATE / "run_logs" / f"{TASK}.csv")
save_state("config/frozen_thresholds.yaml", "preregistration/DEVIATIONS.md")

disk_report()
enforce_output_budget(where="end of worker")
print(f"\\nTASK {TASK}: {'FINISHED' if FINISHED else 'PAUSED — re-launch to resume'}")
"""),
    md("""
## Finishing

**Save Version → Save & Run All.**

- **Paused?** Attach this session to the next one and run the same `TASK` again. It resumes
  from `checkpoint_latest.pth`; nothing is lost and nothing is repeated.
- **Finished?** Its predictions are in the output. Move on to another task — or launch several
  at once, since no worker reads another worker's output.

When enough tasks are done, attach all of them to **02_analyze**. Tier A analysis needs the ten
`cnn-fold*` and `tf-fold*` outputs; the Tier B sections light up as their tasks land, and the
analysis notebook reports what is missing rather than silently analysing a partial study.
"""),
])

# ======================================================================================
NOTEBOOKS["02_analyze"] = (False, [
    md("""
# 02 · Analyze — every hypothesis, every figure, the verdict memo

**No GPU, runs in minutes, and safe to run early.** Attach every finished worker session
(*Add Input → Your Work*, one per task) plus the preparation session. Each section reports what
it is missing rather than quietly analysing a partial study, so this is worth running as soon
as Tier A lands and again after Tier B.

Its own output is tables and figures — a few tens of MB. The 20 GB cap is not in play here;
the guard runs anyway, because a guard you only run when you are worried is not a guard.

Order follows the hypotheses, not what came out best:

1. the paired per-case table — Dice, NSD at 2 mm, HD95, detection, false positives (deliv. 1)
2. **H1** volume, **H2** occlusion + the CNN effective receptive field, **H2b** identity
   control, **H3** source shift against the boundary-tolerance floor
3. the contrast sensitivity analysis and its pre-registered exploratory/confirmatory gate
4. the negative control, seed-versus-fold variance, the rule-selected failure gallery
5. six figures and the verdict memo
"""),
    code(BOOTSTRAP),
    code(ENV_INSTALL),
    code("""
restore_state("splits", "config/frozen_thresholds.yaml")
import pandas as pd, yaml, itertools

# Gather every attached worker's predictions into one tree. Workers write disjoint paths
# (preds/<arm>/<fold>/, preds/occlusion/<arm>/<shell>/, preds/nih/<arm>/<condition>/), so
# merging is a copy with no conflicts to resolve.
merged = 0
if Path("/kaggle/input").exists():
    for cand in Path("/kaggle/input").rglob("preds"):
        if not cand.is_dir():
            continue
        for f in cand.rglob("*.nii.gz"):
            dst = PREDS / f.relative_to(cand)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(f, dst); merged += 1
print(f"merged {merged} prediction files")

# Out-of-fold predictions are per fold on disk; the analysis wants one flat directory per arm.
OOF = WORK / "oof"
for arm in ("cnn", "tf", "identity_control"):
    src_root = PREDS / arm
    if not src_root.exists():
        continue
    dst = OOF / arm; dst.mkdir(parents=True, exist_ok=True)
    # "-seed" directories are replicate runs of a fold that already has an owner here; folding
    # them into the out-of-fold set would give some cases two predictions from the same arm.
    for fold_dir in sorted(d for d in src_root.glob("fold*") if "-seed" not in d.name):
        for f in fold_dir.glob("*.nii.gz"):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    print(f"{arm}: {len(list(dst.glob('*.nii.gz')))} out-of-fold predictions")

# Leave-one-source-out predictions, in the layout loso_analysis.py expects.
LOSO = WORK / "loso_preds"
for arm in ("cnn", "tf"):
    for fold_dir in sorted((PREDS / arm).glob("loso_*")) if (PREDS / arm).exists() else []:
        dst = LOSO / arm / fold_dir.name; dst.mkdir(parents=True, exist_ok=True)
        for f in fold_dir.glob("*.nii.gz"):
            if not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
        print(f"{arm}/{fold_dir.name}: {len(list(dst.glob('*.nii.gz')))}")
disk_report()
"""),
    code("""
# --- what is present, and what is still outstanding ----------------------------------------
# A study analysed on a partial set of runs is not the study, so this is stated up front
# rather than inferred later from a suspiciously small n.
sys.path.insert(0, str(REPO / "scripts" / "training"))
import tasks as task_mod
all_names = task_mod.all_tasks(REPO)

logs = sorted(Path("/kaggle/input").rglob("state/run_logs/*.csv")) if Path("/kaggle/input").exists() else []
logs += sorted((STATE / "run_logs").glob("*.csv"))
run_log = (pd.concat([pd.read_csv(p) for p in logs], ignore_index=True)
           if logs else pd.DataFrame(columns=["run_id", "arm", "fold", "seed", "status"]))
if len(run_log):
    run_log.to_csv(RESULTS / "run_log_merged.csv", index=False)
    print(run_log[["run_id", "arm", "fold", "seed", "status", "notes"]].to_string(index=False))

done = set()
for name in all_names:
    try:
        T = task_mod.parse(name, repo=REPO, budget_suffix="")
    except SystemExit:
        continue
    arm_tag = {"cnn": "cnn", "tf": "tf", "identity": "identity_control"}[T["arm_key"]]
    if (PREDS / arm_tag / T["pred_key"]).exists():
        done.add(name)
print(f"\\n{len(done)}/{len(all_names)} tasks have predictions.")
missing = [n for n in all_names if n not in done]
if missing:
    print("outstanding:", ", ".join(missing))
"""),
    code("""
# --- references ------------------------------------------------------------------------------
# The reference masks come from the nnU-Net raw dataset, which is regenerable, so it is built
# into scratch rather than carried between sessions.
REFS = Path(os.environ["nnUNet_raw"]) / "Dataset501_PDAC" / "labelsTr"
IMAGES = Path(os.environ["nnUNet_raw"]) / "Dataset501_PDAC" / "imagesTr"
if not REFS.exists():
    sh(f"{sys.executable} scripts/data/convert_to_nnunet.py --data-root {DATA_ROOT} "
       f"--cohort splits/cohort.csv")
print(f"{len(list(REFS.glob('*.nii.gz')))} reference masks")
"""),
    code("""
# --- 1. the paired per-case table (deliverable 1) ------------------------------------------------
# Surface distances are computed once here and every hypothesis reads from this table. The
# script refuses an incomplete pairing: a paired analysis with silently dropped cases is not a
# paired analysis.
MT = RESULTS / "per_case_metrics.csv"
sh(f"{sys.executable} scripts/analysis/build_per_case_table.py "
   f"--arm cnn:{OOF / 'cnn'} --arm tf:{OOF / 'tf'} --refs {REFS} "
   f"--strata splits/strata.csv --cohort splits/cohort.csv --out {MT}", check=False)
if MT.exists():
    mt = pd.read_csv(MT)
    print(mt.groupby("arm")[["dice", "nsd2mm", "hd95", "detected", "fp_count"]]
          .mean().round(4).to_string())
    print("\\n(aggregate means; nothing is claimable until the per-stratum intervals below)")
"""),
    code("""
# --- 2. H1 and the stratified maps ------------------------------------------------------------------
sh(f"{sys.executable} scripts/analysis/paired_analysis.py --metrics-table {MT} "
   f"--cohort splits/cohort.csv --out {RESULTS / 'h1'}", check=False)

h1p = RESULTS / "h1" / "summary.yaml"
if h1p.exists():
    h1 = yaml.safe_load(open(h1p))
    margin = yaml.safe_load(open(REPO / "config" / "analysis_config.yaml"))[
        "hypotheses"]["h1"]["equivalence_margin_dice_points"]
    print(f"H1 slope {h1['h1_slope']:+.5f}, 95% CI "
          f"[{h1['h1_ci'][0]:+.5f}, {h1['h1_ci'][1]:+.5f}] -> "
          f"{'SUPPORTED' if h1['h1_supported'] else 'NOT SUPPORTED'} (rule: CI below zero)")
    print(f"Large tertile TOST at ±{margin} points: "
          f"{'EQUIVALENT' if h1['large_tertile_equivalent'] else 'NOT SHOWN EQUIVALENT'}")
    print("\\n" + pd.read_csv(RESULTS / "h1" / "stratum_heatmap.csv").round(4).to_string(index=False))
"""),
    code("""
# --- 3. contrast sensitivity at 5 / 10 / 15 mm rings ---------------------------------------------------
# Pre-specified: if CNR ranks move across ring widths, the contrast axis is reported as
# exploratory. That call was made in August, so applying it now is not a retreat.
sh(f"{sys.executable} scripts/analysis/contrast_sensitivity.py --metrics-table {MT} "
   f"--strata splits/strata.csv --out {RESULTS / 'contrast_sensitivity'}", check=False)
"""),
    code("""
# --- 4. H2: score the occlusion predictions the workers produced -----------------------------------------
# Each worker occluded and re-inferred its own held-out fold, so the shell directories here are
# already the union over folds. The scoring stage only needs them plus the baseline predictions.
OCC = SCRATCH / "occlusion_score"
have_occ = (PREDS / "occlusion").exists()
if have_occ:
    for arm in ("cnn", "tf"):
        for shell_dir in sorted((PREDS / "occlusion" / arm).glob("shell_*")):
            dst = OCC / f"pred_{arm}" / shell_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in shell_dir.glob("*.nii.gz"):
                if not (dst / f.name).exists():
                    shutil.copy2(f, dst / f.name)
    sh(f"{sys.executable} scripts/analysis/occlusion_test.py --stage score --workdir {OCC} "
       f"--pred-base-cnn {OOF / 'cnn'} --pred-base-tf {OOF / 'tf'} --refs {REFS} "
       f"--cohort splits/cohort.csv --out {RESULTS / 'h2'}", check=False)
    h2p = RESULTS / "h2" / "summary.yaml"
    if h2p.exists():
        h2 = yaml.safe_load(open(h2p))
        print(f"H2 decisive shell: mean(tf loss − cnn loss) = {h2['h2_contrast_mean']:+.4f}, "
              f"CI [{h2['h2_ci'][0]:+.4f}, {h2['h2_ci'][1]:+.4f}] -> "
              f"{'SUPPORTED' if h2['h2_supported'] else 'NOT SUPPORTED'}")
else:
    print("No occlusion predictions yet — they come from the two arms' CV-fold workers.")
"""),
    code("""
# --- 5. the CNN effective receptive field (the H2 reference line) -------------------------------------------
# Measured from input gradients on the network that was actually trained: the theoretical
# receptive field only bounds what a voxel could see, not what it depends on.
sh(f"{sys.executable} scripts/analysis/effective_receptive_field.py --self-test", check=False)

ckpt = None
if Path("/kaggle/input").exists():
    hits = sorted(Path("/kaggle/input").rglob(
        "nnUNetTrainer*__nnUNetResEncUNet*Plans__3d_fullres/fold_0/checkpoint_final.pth"))
    ckpt = hits[0] if hits else None
if ckpt:
    # nnU-Net is a heavy install and this is the only cell in an otherwise CPU-only notebook
    # that needs it, so it goes in only when there is actually a network to rebuild.
    pip_install("'nnunetv2 @ git+https://github.com/MIC-DKFZ/nnUNet.git'")
    plans_name = ckpt.parent.parent.name.split("__")[1]
    sh(f"{sys.executable} scripts/analysis/effective_receptive_field.py "
       f"--nnunet-checkpoint {ckpt} --dataset 501 --configuration 3d_fullres "
       f"--plans {plans_name} --fold 0 --out {RESULTS / 'erf_cnn.yaml'}", check=False)
else:
    print("No CNN fold-0 checkpoint attached — attach the cnn-fold0 worker session for the "
          "receptive-field reference line on the H2 figure.")
"""),
    code("""
# --- 6. H3: leave-one-source-out and the boundary-tolerance floor ---------------------------------------------
# The floor needs no model output at all and is computed whether or not the LOSO runs exist;
# the share-of-loss statistic needs both.
if (LOSO / "cnn").exists() and (LOSO / "tf").exists():
    sh(f"{sys.executable} scripts/analysis/loso_analysis.py --metrics-table {MT} "
       f"--pred cnn:{LOSO / 'cnn'} --pred tf:{LOSO / 'tf'} --refs {REFS} "
       f"--cohort splits/cohort.csv --out {RESULTS / 'loso'}", check=False)
    if (RESULTS / "loso" / "by_source.csv").exists():
        print(pd.read_csv(RESULTS / "loso" / "by_source.csv").round(4).to_string(index=False))
else:
    print("No leave-one-source-out predictions yet (Tier B's cnn-loso_* / tf-loso_* tasks).")

loso_dice = RESULTS / "loso" / "loso_dice.csv"
sh(f"{sys.executable} scripts/analysis/boundary_tolerance.py --refs {REFS} "
   f"--cohort splits/cohort.csv {'--loso ' + str(loso_dice) if loso_dice.exists() else ''} "
   f"--out {RESULTS / 'h3'}", check=False)
"""),
    code("""
# --- 7. the inter-rater proxy ------------------------------------------------------------------------------------
# Two-sided on purpose: a list of paired cases, or a written statement that there are none. A
# synthetic boundary floor with no empirical check is a limitation to state, not to discover
# at review.
sh(f"{sys.executable} scripts/analysis/paired_delineation_check.py "
   f"--labels-root {DATA_ROOT / 'panorama' / 'panorama_labels'} --cohort splits/cohort.csv "
   f"--out {RESULTS / 'inter_rater'}", check=False)
p = RESULTS / "inter_rater" / "STATEMENT.md"
print(p.read_text() if p.exists() else "")
"""),
    code("""
# --- 8. H2b, the identity control -----------------------------------------------------------------------------------
if (OOF / "identity_control").exists() and any((OOF / "identity_control").glob("*.nii.gz")):
    sh(f"{sys.executable} scripts/analysis/build_per_case_table.py "
       f"--arm tf:{OOF / 'tf'} --arm identity_control:{OOF / 'identity_control'} "
       f"--refs {REFS} --strata splits/strata.csv --cohort splits/cohort.csv "
       f"--out {RESULTS / 'per_case_metrics_h2b.csv'}", check=False)
    sh(f"{sys.executable} scripts/analysis/identity_comparison.py "
       f"--metrics-table {RESULTS / 'per_case_metrics_h2b.csv'} --out {RESULTS / 'h2b'}",
       check=False)
else:
    print("No identity-control predictions yet (Tier B's identity-fold* tasks).")
"""),
    code("""
# --- 9. seed variance against fold variance --------------------------------------------------------------------------
# The seed replicates live under their own trainer class names, which is what stops a second
# seed on the same fold from overwriting the first.
rep = yaml.safe_load(open(REPO / "config" / "analysis_config.yaml"))["training"]["seed_replicates"]
for arm in ("cnn", "tf"):
    args = []
    for s in rep["extra_seeds"]:
        # No fallback to the default-seed directory: silently substituting the default run for
        # a missing replicate would compare a run against itself and report seed variance of
        # zero, which is worse than reporting nothing.
        d = PREDS / arm / f"fold{rep['fold']}-seed{s}"
        if d.exists():
            args += ["--seed-pred", f"{s}:{d}"]
    if not args or not MT.exists():
        print(f"{arm}: no seed replicates yet"); continue
    sh(f"{sys.executable} scripts/analysis/build_variance_tables.py --arm {arm} "
       f"--metrics-table {MT} --refs {REFS} {' '.join(args)} "
       f"--out {RESULTS / f'variance_inputs_{arm}'}", check=False)
    sh(f"{sys.executable} scripts/analysis/variance_decomposition.py --arm {arm} "
       f"--fold-dice {RESULTS / f'variance_inputs_{arm}' / 'fold_dice.csv'} "
       f"--seed-dice {RESULTS / f'variance_inputs_{arm}' / 'seed_dice.csv'} "
       f"--out {RESULTS / f'variance_{arm}.yaml'}", check=False)
"""),
    code("""
# --- 10. the NIH negative control (deliverable 8) --------------------------------------------------------------------
args = []
for arm, cond in itertools.product(("cnn", "tf"), ("in_domain", "source_shift")):
    d = PREDS / "nih" / arm / cond
    if d.exists() and any(d.glob("*.nii.gz")):
        args += ["--pred", f"{arm}:{cond}:{d}"]
if args:
    sh(f"{sys.executable} scripts/analysis/nih_false_positives.py {' '.join(args)} "
       f"--cohort splits/cohort.csv --out {RESULTS / 'nih_fp'}", check=False)
else:
    print("No NIH predictions yet — the fold0 and loso_* worker tasks produce them.")
"""),
    code("""
# --- 11. figures --------------------------------------------------------------------------------------------------------
# Each is skipped with a printed reason when its inputs are absent, so this cell is correct
# after Tier A and again after Tier B.
sh(f"{sys.executable} scripts/analysis/make_figures.py --results {RESULTS} "
   f"--out {RESULTS / 'figures'}", check=False)
from IPython.display import Image, Markdown, display
for p in sorted((RESULTS / "figures").glob("*.png")):
    print(p.name); display(Image(filename=str(p)))
"""),
    code("""
# --- 12. the failure gallery (deliverable 9) ------------------------------------------------------------------------------
# Selected by the frozen rule — largest |ΔDice| within each stratum cell — never by eye. The
# selection script is in the repo precisely so nobody has to be trusted about that.
if (RESULTS / "h1" / "per_case.csv").exists() and IMAGES.exists():
    sh(f"{sys.executable} scripts/analysis/failure_gallery.py "
       f"--per-case {RESULTS / 'h1' / 'per_case.csv'} --pred-cnn {OOF / 'cnn'} "
       f"--pred-tf {OOF / 'tf'} --refs {REFS} --images {IMAGES} "
       f"--out {RESULTS / 'failure_gallery'}", check=False)
    for p in sorted((RESULTS / "failure_gallery").glob("*.png")):
        print(p.name); display(Image(filename=str(p)))
"""),
    code("""
# --- 13. the verdict memo -----------------------------------------------------------------------------------------------------
# Rules quoted from the frozen config, numbers read from the analysis scripts' own summaries,
# and the rule applied to the number. It cannot invent a rule because it has nowhere else to
# read one from.
sh(f"{sys.executable} scripts/analysis/verdict_memo.py --results {RESULTS} "
   f"--out {RESULTS / 'VERDICT.md'}")
display(Markdown((RESULTS / "VERDICT.md").read_text()))
"""),
    code("""
save_state("scripts/training/run_log.csv")
disk_report()
enforce_output_budget(where="end of analysis")
print(f"\\nEverything for the paper is under {RESULTS}. Copy results/ out of this notebook's "
      "output, commit it, and run: bash scripts/check_prereg_tag.sh")
"""),
    md("""
## Reading this against the rules, not around them

The memo applies the frozen rules; it does not weigh or reinterpret them. A hypothesis that
reads NOT SUPPORTED is reported as a stratified negative result with its intervals, per
section 7 of the pre-registration — not re-tested under a rule invented after the fact. The
point of freezing in August was that this step is arithmetic.

Before writing any prose, run `bash scripts/check_prereg_tag.sh` and confirm PASS. If a frozen
file has changed, the numbers above are not what was pre-registered, and the difference belongs
in `preregistration/DEVIATIONS.md` and in the paper.
"""),
])


# ======================================================================================
def main():
    for old in sorted(OUT.glob("*.ipynb")):
        if old.stem not in NOTEBOOKS:
            old.unlink()
            print(f"  removed stale {old.name}")
    written = []
    for name, (gpu, cells) in NOTEBOOKS.items():
        path = OUT / f"{name}.ipynb"
        # ensure_ascii=False keeps the em dashes and middle dots readable in the JSON, and
        # matches what an editor writes when it saves the file.
        path.write_text(json.dumps(notebook(cells, gpu=gpu, name=name), indent=1,
                                   ensure_ascii=False) + "\n")
        written.append((name, gpu, sum(1 for c in cells if c["cell_type"] == "code")))
    width = max(len(n) for n, _, _ in written)
    for name, gpu, n_code in written:
        print(f"  {name.ljust(width)}  {'GPU' if gpu else '   '}  {n_code:2d} code cells")
    print(f"\n{len(written)} notebooks written to {OUT}")


if __name__ == "__main__":
    main()
