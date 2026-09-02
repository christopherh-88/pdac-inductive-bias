# Pre-registration

**Study:** Inductive Bias on a Hard Organ: A Pre-Registered Failure-Mode Comparison of a
Convolutional U-Net and a Pure Transformer for PDAC Segmentation in CT.

**Status:** FROZEN at tag `prereg-v1`. Nothing in this file changes after that tag. Any
deviation during the study is reported in `preregistration/DEVIATIONS.md` (created only if a
deviation occurs), never by editing this file.

**Frozen before:** any training run of either arm on any fold.

---

## 1. Central question

Under one training pipeline (nnU-Net v2) and a matched compute budget, **where** (by lesion
volume, lesion-to-parenchyma contrast, and acquisition source) and **why** (context use,
attention dependence, annotation tolerance) do a convolutional segmenter (nnU-Net ResEnc) and
a pure-transformer segmenter (PrimusV2) fail differently on PDAC?

## 2. Hypotheses, tests, and decision rules

All margins below are fixed here and reported as pre-set whether or not they are met.

### H1 — Locality vs lesion size

- **Prediction:** The CNN advantage, ΔDice = Dice(CNN) − Dice(transformer) per case, is
  largest for small lesions and shrinks with volume.
- **Test:** Per-case paired differences regressed on log lesion volume, CNR, and source;
  patient-level bootstrap CIs (2,000 resamples); Wilcoxon signed-rank within volume tertiles.
- **Decision rule:** Supported if the 95% CI of the slope of ΔDice on log volume lies below
  zero. The large-lesion tertile is additionally reported with a two one-sided equivalence
  test (TOST) at **±3 Dice points**.

### H2 — Context use

- **Prediction:** The transformer's predictions depend on anatomy farther from the lesion
  than the CNN's effective receptive field.
- **Test:** Occlusion test applied identically to both arms: replace spherical shells at
  **10–20, 20–40, and 40–80 mm** from the manual lesion surface with the shell's own mean HU,
  re-infer, record Dice loss per shell. The CNN's effective receptive field is measured
  empirically from input gradients as a reference.
- **Decision rule:** Supported if the transformer's Dice loss in the **40–80 mm** shell
  exceeds the CNN's by at least **2 Dice points** with a patient-level bootstrap CI excluding
  zero.

### H2b — Attention dependence

- **Prediction:** The transformer's stratified failure map is produced by its attention
  blocks, not by its patch embedding and decoder.
- **Test:** Identity control — the same architecture trained on the same folds with the
  transformer blocks replaced by identity (the Primus stress test).
- **Decision rule:** If the identity control reproduces the full model's stratified map
  within CI, attention is not the cause, and the paper says so.

### H3 — Source shift vs boundary tolerance

- **Prediction:** A measurable share of the Dice lost under cross-source evaluation lies
  within the tolerance implied by boundary ambiguity in the reference.
- **Test:** Leave-one-source-out training per arm; boundary-tolerance floor from 1- and
  2-voxel morphological perturbation of the reference; NSD at 2 mm reported alongside Dice.
  RESOLVED (2026-09-02): no inter-rater proxy is available from paired independent
  delineations — the panorama_labels repo's own README states manual PDAC lesion
  segmentations were each made by **one of two** trained investigators (single annotator per
  case, supervised by one expert radiologist), not by both. The boundary-tolerance floor
  therefore rests on morphological perturbation of the single reference alone.
- **Decision rule:** Report the share of each arm's cross-source Dice loss that falls inside
  the floor. Supported if that share exceeds **50%** for at least one arm.

## 3. Stratification axes

- **Lesion volume:** tertiles of manual lesion volume over the cohort, computed once by
  `scripts/analysis/compute_strata.py` and written to `config/frozen_thresholds.yaml`.
  Secondary cut at 20 mm maximum in-plane diameter (the 2 cm clinical threshold).
- **Contrast:** CNR = |mean HU(lesion) − mean HU(ring)| / SD(ring), ring = automatic
  parenchyma label (label 4) within 10 mm of the lesion, excluding lesion, duct, and vessels.
  Pre-specified sensitivity analysis repeats every contrast-stratified result with 5 mm and
  15 mm rings. If CNR ranks are unstable across ring widths (Spearman ρ < 0.8 between any
  pair of ring widths), the contrast axis is reported as exploratory.
- **Source:** acquisition source per `clinical_information.xlsx`, grouped as recorded in
  `config/frozen_thresholds.yaml` at split freeze. Scanner vendor/model within source where
  metadata allow.

## 4. Cohort and evaluation rules

- Only manually delineated PDAC cases (n = 481 in PANORAMA after deduplication) enter test
  folds. The 194 model-generated delineations never enter evaluation; their use as extra
  training data is an optional, separately reported ablation.
- MSD Task07 restricted to PDAC (98 nominal cases per Suman et al. 2021; stented patients
  removed; 97 after deduplication — `100278_00001` and `100205_00001` are the same underlying
  scan under two study IDs, see `splits/duplicates.csv`); non-PDAC masses excluded from all
  PDAC analyses.
- NIH Pancreas-CT (80 cases, no tumors) is a negative control only: false-positive lesions
  per case, never a tumor test set.
- Duplicate scans across PANORAMA, MSD Task07, and NIH Pancreas-CT are removed before
  splitting (`scripts/data/deduplicate.py`).
- Patient-level 5-fold splits, stratified by source and lesion-volume tertile, fixed once
  (`splits/`), so every case gets one out-of-fold prediction per arm and the analysis is
  paired over the whole cohort. Three seed replicates on fold 0 separate initialization
  variance from fold variance.

## 5. Controls

- One framework: nnU-Net v2 preprocessing, target spacing, augmentation, loss, optimizer
  schedule (1000 epochs × 250 iterations), and inference shared by both arms. Architecture is
  the only variable.
- Matched budget, not matched parameter count: same VRAM ceiling, same steps, same wall-clock
  allowance. Parameter counts reported but not controlled.
- Token resolution stated and controlled (recorded in frozen config) so small-lesion failure
  in the transformer cannot be attributed to tokenization alone.
- Both arms trained from scratch in Tiers A and B. Pretraining, if any, applied to both arms
  or neither, and only in Tier C.
- Fallback: if PrimusV2 cannot be integrated and verified in week one, Primus v1 is used and
  the gap to published numbers reported. No fallback to hybrids (SwinUNETR, CoTr, nnFormer,
  TransUNet).
### Primus V2 vs V3S

**Decision:** the transformer arm uses **PrimusV2** (`nnUNet_PrimusV2{S,B,M,L}_Trainer`), not
the upstream-recommended `nnUNet_PrimusV3S_Trainer`. Settled here, before freeze (2026-08-23),
per the project description's requirement; not revisited inside Tiers A/B.

**Reasoning:**
- nnU-Net master now ships `nnUNet_PrimusV3S_Trainer` and flags it `# (recommended)`, but its
  own documentation calls PrimusV3 "a preliminary version ... with a more convolution heavy
  patch embedding" — i.e. upstream's own stated status, not just our caution.
- The project's rationale for a pure-transformer arm rests on the published Primus/TMLR
  benchmark showing PrimusV2 reaches parity with ResEnc-L. V3S has no equivalent independent
  validation yet; switching would sever the study's stated grounding evidence.
- V3S uses a materially different tokenizer than V2 — aggressive channel scaling (32 → 64 →
  256 → 1024) with multi-resolution skip connections, versus V2's iterative residual tokenizer
  (32 → 32 → 64 → 128). Switching now would mean re-doing VRAM/step-budget matching against a
  different architecture family, not swapping a trainer flag.
- V2 is the version this study's design, budget matching, and identity-control trainer
  (`src/trainers/primus_identity_trainer.py`) were built against.
- V3S's own published ablation (identity block: 84.48 vs. full model 87.98 Dice, five-fold
  average, generic segmentation tasks per the Primus documentation page) is a useful external
  sanity check for the direction of our own H2b, but is not PDAC evidence and doesn't on its
  own justify switching architecture families three weeks before freeze.

**Revisit policy:** V3S is a candidate for Tier C (Nov 2026+) only, added as a new tag with a
written reason, per the deviations policy in the header of this document — never a silent
substitution inside Tier A/B.

### Token resolution

Primus uses fixed **8 × 8 × 8 voxel tokens**, constant across the S/B/M/L variants for both V2
and V3S (per the nnU-Net Primus documentation page). Effective token count along each axis is
`patch_size_axis / 8`; total effective tokens is the product across the three axes.

The numeric patch size — and therefore the numeric effective token count — is a property of
the frozen cohort's nnU-Net fingerprint, which does not exist until PANORAMA is downloaded and
preprocessed (R2, Phase 0). That number is written to `config/frozen_thresholds.yaml` at
fingerprinting time, not fabricated here. What is fixed now, before any data-dependent number
exists, is the 8 × 8 × 8 stride itself and the rule that patch size must be chosen to divide
evenly by it in all three axes — so a downstream small-lesion failure can never be blamed on
an unrecorded or accidental tokenization mismatch.

## 6. Metrics and reporting

Dice, normalized surface Dice at 2 mm, HD95, case-level lesion detection sensitivity, and
false-positive lesions per case (including on NIH negatives). Frozen operational definitions:

- **Detection:** a reference lesion counts as detected if the predicted lesion mask covers at
  least 10% of its volume.
- **False positive:** a connected component of the prediction with zero overlap with the
  reference lesion and volume ≥ 100 mm³.

Everything is reported per stratum with per-cell n and patient-level bootstrap CIs. No
aggregate leaderboard table without intervals. Failure gallery cases are selected by rule
(largest |ΔDice| within each stratum), never by hand.

## 7. What the paper says if a hypothesis fails

- CNN wins every stratum → the stratified negative result is reported with intervals,
  occlusion curves, and the identity control; the claim remains "where and why".
- PrimusV2 not reproducible at matched budget → Primus v1, gap reported first.
- CNR unstable across ring widths → contrast axis reported as exploratory.
- Underpowered small-lesion cells → intervals and equivalence bounds reported, no null
  claimed.
- No 3D compute by Aug 28, 2026 → ML4H and CJSJ dropped; plan runs to ISBI and MIDL.

## 8. Machine-readable configuration

Every number in this document is duplicated in `config/analysis_config.yaml`, which is the
version the analysis code actually reads. If the two ever disagree, the YAML at tag
`prereg-v1` governs and the disagreement is reported.
