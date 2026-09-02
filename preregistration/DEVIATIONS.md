# Deviations from the frozen pre-registration

`PREREGISTRATION.md` is frozen at tag `prereg-v1` and is never edited after that tag. Any
clarification, correction, or deviation discovered after the freeze is recorded here instead,
with a date and a pointer to the section it concerns.

## H3 — no inter-rater proxy available (2026-09-02)

Section "H3 — Source shift vs boundary tolerance" names "paired independent delineations" as an
optional input to the boundary-tolerance floor, alongside 1- and 2-voxel morphological
perturbation of the reference. No such pairs exist in this dataset: the panorama_labels repo's
own README states manual PDAC lesion segmentations were each made by **one of two** trained
investigators (single annotator per case, supervised by one expert radiologist), not by both.

This does not change the test or decision rule — morphological perturbation alone was always
the primary method: it just means the "plus any paired independent delineations" clause has no
data to act on, so the boundary-tolerance floor rests on morphological perturbation of the
single reference alone. See `docs/project_description_v2.md` point 4 for the same note in the
supporting narrative (that document is not frozen and was updated directly).

## Architecture record is preliminary, from a Kaggle GPU substitute (2026-09-02)

`config/frozen_thresholds.yaml`'s `architecture` section was frozen from a Kaggle free-tier
P100 kernel (`scripts/verify/phase0_gpu_verify.py`), run as an interim substitute for confirmed
NYU/USC GPU access, solely to clear Phase 0's two-arm smoke-test verification (real end-to-end
train + predict + Dice for both arms and the identity control, on 8 real manual-lesion cases —
`Dataset999_PDACSmoke`, not the full 481-case cohort). It is not the matched-budget Tier A
configuration.

Two specific numbers in that record are diagnostic-only and will be superseded once real Tier A
planning runs on the full cohort:

- **Patch size mismatch between arms:** the CNN (`nnUNetResEncUNetMPlans`) auto-planned a 3D
  fullres patch of `(64, 160, 160)` on this tiny dataset, while the transformer's default-plans
  fingerprint (used for `nnUNet_PrimusV2S_Trainer`) auto-planned `(64, 192, 160)` — the frozen
  record uses the transformer's patch since that is the one PrimusV2's 8×8×8 tokenizer stride
  must divide evenly. Section 5's matched-budget control requires the two arms share one patch
  size; that alignment happens during real Tier A fingerprinting on the full cohort, not here.
- **CNN parameter count not captured:** the verification kernel measured the transformer's
  parameter count directly (25,477,406, incidental to the identity-control ablation's own
  parameter-drop assertion) but did not instrument the CNN arm's parameter count, so
  `architecture.cnn.n_params` is `null`. Filled in at real Tier A standup, not fabricated here.

Both `peak_vram_gb` figures (CNN 7.149 GB, transformer 6.341 GB) and the transformer's
`n_tokens` = 3,840 (token grid 8×24×20) are real measurements from that kernel run and are not
placeholders — they are simply from a 10-iteration smoke run on 8 cases on a P100, not the real
compute tier, and are recorded as such.
