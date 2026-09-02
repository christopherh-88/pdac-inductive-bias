# Inductive Bias on a Hard Organ

**A pre-registered failure-mode comparison of a convolutional U-Net (nnU-Net ResEnc) and a
pure transformer (PrimusV2) for PDAC segmentation in CT.**

This repository holds the pre-registration, the frozen analysis configuration, and all code
for the study described in [docs/project_description_v2.md](docs/project_description_v2.md).
The output of the study is not a winner: it is a stratified map of where each architecture
fails (lesion volume, lesion-to-parenchyma contrast, acquisition source), two causal tests of
why (occlusion test, identity ablation), and an estimate of how much cross-source degradation
sits within annotation-boundary tolerance.

## Pre-registration

Hypotheses, stratum thresholds, margins, and decision rules are fixed **before any model is
trained** in [preregistration/PREREGISTRATION.md](preregistration/PREREGISTRATION.md) and
[config/analysis_config.yaml](config/analysis_config.yaml). The commit that freezes them is
tagged `prereg-v1`. Data-derived thresholds (volume tertiles, CNR tertiles) are computed once
by `scripts/analysis/compute_strata.py` and written to `config/frozen_thresholds.yaml`, which
is committed and tagged before training starts.

## The two arms

| Arm | Implementation | Command surface |
| --- | --- | --- |
| Convolutional | nnU-Net ResEnc preset M/L/XL (chosen by VRAM) | `nnUNetv2_train <ID> 3d_fullres <fold> -p nnUNetResEncUNet{M,L,XL}Plans` |
| Transformer | PrimusV2 (S/B/M/L), integrated in nnU-Net master | `nnUNetv2_train <ID> 3d_fullres <fold> -tr nnUNet_PrimusV2{S,B,M,L}_Trainer` |
| Identity control (H2b) | PrimusV2 with transformer blocks replaced by identity | `src/trainers/primus_identity_trainer.py` |

Both arms share nnU-Net preprocessing, augmentation, loss, schedule (1000 × 250), and
inference. See [environment/SETUP.md](environment/SETUP.md) for installation and VRAM-based
preset selection.

## Data

| Source | What | How to get it |
| --- | --- | --- |
| PANORAMA public training set (2,238 CECT nominal, 2,237 after deduplication; 675 PDAC, 481 manual delineations) | Primary cohort | `scripts/download/download_panorama.sh` (4 Zenodo batches + labels repo) |
| MSD Task07 (98 PDAC nominal per Suman et al. 2021 split, 97 after deduplication) | Third source for leave-one-source-out | Redistributed inside PANORAMA |
| NIH Pancreas-CT (80 cases, no tumors) | Negative control only | Redistributed inside PANORAMA |
| PanTS (36,390 CT, 145 centers) | Optional Tier C external test | https://github.com/MrGiovanni/PanTS |

Licensing: PANORAMA CC BY-NC 4.0, MSD Pancreas CC BY-SA 4.0, NIH Pancreas-CT CC BY.
No image data is stored in this repository. Code, frozen configuration, and per-fold case
lists are MIT (see [LICENSE](LICENSE)).

## Repository layout

```
preregistration/    Frozen hypotheses, decision rules, margins
config/             analysis_config.yaml (frozen) + frozen_thresholds.yaml (data-derived, frozen once)
environment/        SETUP.md, requirements.txt
scripts/download/   PANORAMA download (Zenodo) + labels
scripts/data/       deduplication, patient-level splits, nnU-Net dataset conversion
scripts/training/   week-one pipeline verification, Tier A launcher, RUN_LOG.md (GPU run log),
                    record_arch_stats.py (freezes patch/token size, params, VRAM)
scripts/analysis/   strata computation, paired analysis, occlusion test, boundary tolerance,
                    effective receptive field, NIH false-positive harness, failure gallery,
                    seed-vs-fold variance decomposition, metrics
src/trainers/       identity-ablation trainer (H2b)
splits/             frozen 5-fold splits + case lists (committed once, never edited)
docs/               project description v2
tests/              synthetic-data end-to-end smoke test for the data/analysis pipeline
```

Run `python tests/run_smoke_test.py` after changing any data or analysis script — see
[tests/README.md](tests/README.md).

## Workflow (Tier A)

```bash
# 1. Environment (see environment/SETUP.md)
pip install -r environment/requirements.txt

# 2. Data
bash scripts/download/download_panorama.sh /path/to/data

# 3. Deduplicate across PANORAMA / MSD / NIH, build cohort table
python scripts/data/deduplicate.py --data-root /path/to/data --out splits/cohort.csv

# 4. Strata (volume, CNR at 5/10/15 mm rings) -> freezes config/frozen_thresholds.yaml
python scripts/analysis/compute_strata.py --data-root /path/to/data --cohort splits/cohort.csv

# 5. Patient-level 5-fold splits, stratified by source x volume tertile (seed frozen)
python scripts/data/make_splits.py --cohort splits/cohort.csv --strata splits/strata.csv

# 6. nnU-Net dataset + week-one end-to-end verification of both arms
python scripts/data/convert_to_nnunet.py --data-root /path/to/data --cohort splits/cohort.csv
bash scripts/training/verify_pipeline.sh

# 7. Tier A training (5 folds x 2 arms)
bash scripts/training/run_tier_a.sh
```

### Disk-constrained alternative for steps 2-4

`download_panorama.sh` needs ~400 GB free (180 GB zipped, all 4 batches at once). Where that
much disk isn't available (e.g. a Kaggle notebook), `scripts/download/kaggle_stream_cohort.py`
processes one ~50 GB Zenodo batch at a time — extracting, hashing, and computing lesion
volume/diameter/CNR per case, then deleting the case and eventually the whole batch zip before
moving on — so raw CT never needs more than one batch's worth of disk at once. It ships back
only small CSVs (`case_records.csv`, `lesion_stats_raw.csv`, `clinical_information.xlsx`); the
actual images still have to be (re-)downloaded wherever training happens.
`scripts/data/finalize_cohort_from_stream.py` and `scripts/data/finalize_strata_from_stream.py`
turn those CSVs into the same `splits/cohort.csv` / `splits/strata.csv` (and
`config/frozen_thresholds.yaml` strata section) that `deduplicate.py` + `compute_strata.py`
would produce from live files — run them in place of steps 3-4 above once the CSVs are pulled
back down.

## Known open items (to resolve at freeze time, before training)

- PARTIALLY RESOLVED (2026-08-30): inspected the real `clinical_information.xlsx` (2,238 rows,
  columns `PANORAMA_patient_id`, `PANORAMA_study_id`, `anonymized_study_date`, `patient_age`,
  `patient_sex`, `scanner`, `label`, `level`). `label` is literally `PDAC` / `non-PDAC` — this
  **is** the Suman et al. (2021) split already applied by PANORAMA, no separate lookup needed.
  `level == MSD_dataset` (194 rows, 98 PDAC) and `level == NIH_dataset` (80 rows, 0 PDAC) match
  the project description's counts exactly and identify the MSKCC and NIH sources unambiguously.
  `scripts/data/deduplicate.py` now derives `is_pdac` from `label` and `source` from `level`
  (`MSD_dataset`->MSKCC, `NIH_dataset`->NIH, else `PANORAMA`) automatically. **Still open:** the
  remaining ~1,964 native-PANORAMA rows carry no institution field in this file — the project
  description's Radboud/UMCG split within them (vs. the five contributing institutions RUMC,
  UMCG, ZGT, Karolinska, Haukeland the PANORAMA site lists) is still undecided; fix it from
  `scanner`/`anonymized_study_date` patterns or another metadata file when splits are frozen,
  and record the decision in `config/frozen_thresholds.yaml`.
  Also fixed in the same pass: the clinical-merge join key previously picked by naive
  `"patient"`/`"case"`/`"id"` substring matching resolved to `panorama_patient_id` (bare patient
  id, e.g. `100000`) instead of `panorama_study_id` (the actual case-id grain used everywhere
  else in this repo, e.g. `100000_00001`) — every row would have silently failed to join,
  leaving `source`/`label` all-null with no error. Now prefers a `study_id` column explicitly
  and aborts if fewer than 99% of cohort case_ids match the chosen key.
- RESOLVED (2026-09-01): full 4-batch cohort built via the Kaggle streaming path
  (`scripts/download/kaggle_stream_cohort.py` v18-v21) and finalized locally
  (`finalize_cohort_from_stream.py`, `finalize_strata_from_stream.py`, `make_splits.py`).
  Cross-batch content-hash dedup found one genuine duplicate beyond the nominal counts above:
  `100278_00001` and `100205_00001` (two different `PANORAMA_patient_id`s, both `MSD_dataset`,
  both PDAC) share an identical `meta_fp` and `content_hash` — the same scan redistributed under
  two study IDs. One was dropped (`splits/duplicates.csv`), so the deduplicated cohort has 2,237
  scans, 675 PDAC (481 manual delineations), and MSD Task07 has 97 unique PDAC cases, not the
  194/98 nominal counts above. `splits/cohort.csv`, `splits/strata.csv`, and
  `splits/splits_final.json` are frozen from this deduplicated cohort.
- PrimusV2 preset (S/B/M/L) and ResEnc preset (M/L/XL) are chosen together once GPU VRAM is
  confirmed, to satisfy the matched-budget control; recorded in the frozen config.
- RESOLVED (2026-08-23): nnU-Net master now also ships `nnUNet_PrimusV3S_Trainer` and
  recommends it as the new default over V2. The transformer arm stays on **PrimusV2** — the
  project's rationale for a pure-transformer arm depends on the published Primus/TMLR
  parity-with-ResEnc-L benchmark, which V3 (documented upstream as "a preliminary version")
  doesn't yet have. See `preregistration/PREREGISTRATION.md` section 5.
- PARTIALLY VERIFIED (2026-08-23): installed nnU-Net master (commit `0e49508`) on CPU and
  confirmed `nnUNet_PrimusV2{S,B,M,L}_Trainer` all resolve via `recursive_find_python_class`
  exactly as `primus_identity_trainer.py` expects. Built `PrimusV2M` directly at our candidate
  patch size (96×160×160, divides evenly by the 8×8×8 tokenizer stride) and ran a real forward
  pass (150.4M params). Ran the identity trainer's `_find_transformer_blocks` logic against
  this real network: it correctly locates `eva.blocks` (16 blocks) and ablation drops 95.5% of
  parameters, with a valid post-ablation forward pass. **Not yet verified:** the full
  `nnUNetTrainer.initialize()` path (needs a real preprocessed dataset + plans.json) and
  anything GPU-dependent (VRAM, step time, multi-epoch training) — those remain week-one items.

## Key upstream references

- nnU-Net Revisited (Isensee et al., MICCAI 2024): arXiv:2404.09556
- Primus / PrimusV2 (Wald et al.): published in TMLR —
  [OpenReview](https://openreview.net/forum?id=x4vZE4PDEu); preprint arXiv:2503.01835 —
  [nnU-Net Primus documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/primus.md)
  (also documents PrimusV3, not used here — see "Known open items" above)
- [nnU-Net ResEnc presets](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md)
- [PANORAMA datasets page](https://panorama.grand-challenge.org/datasets-imaging-labels/) ·
  [panorama_labels](https://github.com/DIAGNijmegen/panorama_labels)
- [PanTS](https://github.com/MrGiovanni/PanTS)
