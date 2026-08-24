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
| PANORAMA public training set (2,238 CECT; 676 PDAC, 482 manual delineations) | Primary cohort | `scripts/download/download_panorama.sh` (4 Zenodo batches + labels repo) |
| MSD Task07 (98 PDAC per Suman et al. 2021 split) | Third source for leave-one-source-out | Redistributed inside PANORAMA |
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

## Known open items (to resolve at freeze time, before training)

- The PANORAMA public set lists five contributing institutions (RUMC, UMCG, ZGT, Karolinska,
  Haukeland) plus MSD and NIH; the project description groups sources as Radboud / UMCG /
  MSKCC. The leave-one-source-out grouping is fixed from `clinical_information.xlsx` when the
  splits are frozen, and recorded in `config/frozen_thresholds.yaml`.
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
