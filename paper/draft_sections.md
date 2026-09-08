# ISBI 2027 — draft prose for sections that don't depend on Tier A results

Companion to `ISBI2027_outline.md`. That file is the full structure with every claim's
source artifact named; this file writes real prose for the subsections whose inputs are
already frozen (cohort, splits, pre-registration, metrics definitions) and does not touch
anything gated on Tier A training. **Nothing below states a result.** Sections not included
here (Results, Discussion, the "Arms and matched-budget control" paragraph, Limitations)
still have no numbers to draw on and stay in the outline only.

Not yet draftable and why:
- **Arms and matched-budget control** — `config/frozen_thresholds.yaml`'s `architecture`
  block is explicitly preliminary (Kaggle P100 substitute; CNN `n_params` and both arms'
  `step_time_s` are still `null`; see `preregistration/DEVIATIONS.md`). This paragraph needs
  the real Tier A fingerprint.
- **Results, Discussion, Limitations** — depend on `results/*` artifacts that don't exist
  until Tier A trains.

---

## 1. Introduction (~450 words)

Pancreatic ductal adenocarcinoma (PDAC) is among the hardest lesion types in CT segmentation
for a specific, structural reason: at the resectable stage most lesions are small, present
low intrinsic contrast against normal parenchyma, and sit in an organ whose shape, position,
and boundary with adjacent structures varies substantially across patients. Small size limits
the number of voxels an evaluation metric or a model can use to resolve the lesion boundary;
low contrast means the boundary itself is frequently ambiguous even to an expert annotator;
and anatomical variability means a model cannot rely on a stable spatial prior for where the
organ — let alone the lesion — will be. Any architecture comparison on this task has to
contend with all three properties at once, and they do not act uniformly across the cohort.

Architecture comparisons in medical imaging typically report a single aggregate metric across
the test cohort: model A beats model B by some margin in mean Dice. An aggregate number over
a cohort with this much within-cohort heterogeneity — in lesion size, in contrast, in
acquisition source — collapses exactly the information a clinician or a downstream system
designer needs: not which model wins on average, but which cases each model is likely to fail
on, and why. Two models with identical aggregate Dice can have entirely different failure
maps, and a single mean obscures that difference completely.

This paper reports a stratified failure map — Dice, boundary-aware surface distance, and
detection/false-positive behavior broken out by lesion volume, contrast, and acquisition
source — for a convolutional U-Net and a pure-transformer architecture trained under one
shared nnU-Net pipeline at a matched compute budget. Beyond the stratified comparison, we
test two specific mechanistic claims about *why* the architectures might diverge where they
do: whether performance differences track each architecture's use of image context beyond the
lesion itself (via an occlusion test referenced against the CNN's effective receptive field),
and whether the transformer's attention mechanism specifically, as opposed to its parameter
budget or patch embedding, is responsible for any context-use difference (via an identity-block
ablation that removes attention while holding capacity fixed). Every hypothesis, margin, and
decision rule governing these tests was fixed before any model saw training data, and is
recorded in a tagged, machine-readable pre-registration (`prereg-v1`) alongside this paper's
code and configuration.

*Cite: nnU-Net Revisited (arXiv:2404.09556); Primus (TMLR, OpenReview `x4vZE4PDEu`); PANORAMA
datasets page and study-protocol DOI.*

State the pre-registration tag (`prereg-v1`) and repository URL directly in this section, not
only in a data-availability footnote.

---

## 2. Data and methods (partial — frozen subsections only)

### Cohort

Manually delineated PDAC cases from PANORAMA form the core cohort: 481 cases after
deduplication carry a manual lesion delineation, of which 478 have a positive PDAC lesion
label and enter analysis (3 cases' manual labels delineate a non-PDAC finding and are excluded
per the pre-registration's non-PDAC exclusion rule; see `preregistration/DEVIATIONS.md`). The
478-case cohort spans two acquisition sources: PANORAMA (382 cases) and MSKCC (96 cases). The
194 PANORAMA cases with only model-generated (automatic) delineations never enter evaluation;
their use as additional training data is a separate, optional ablation, reported apart from
the main comparison.

MSD Task07, restricted to PDAC cases (98 nominal cases per Suman et al. 2021, stented patients
removed, 97 after deduplication — two study IDs, `100278_00001` and `100205_00001`, resolve to
the same underlying scan, see `splits/duplicates.csv`), and NIH Pancreas-CT (80 cases, no
tumors) extend the evaluation cohort for two different purposes: MSD Task07 as an additional
source-shift test, NIH Pancreas-CT strictly as a negative control for false-positive lesion
counts, never as a tumor test set. Duplicate scans across all three source datasets are
removed before splitting (`scripts/data/deduplicate.py`).

Patient-level 5-fold splits, stratified by acquisition source and lesion-volume tertile, are
fixed once (`splits/`, split seed `20260823`) so that every case receives exactly one
out-of-fold prediction per arm and the whole-cohort analysis is paired. Three seed replicates
are additionally trained on fold 0 to separate initialization variance from fold-to-fold
variance.

### Stratification

Three axes stratify every reported result: lesion-volume tertile (frozen cut points
3827.13 mm³ and 10047.77 mm³), acquisition source, and contrast-to-noise ratio (CNR) at a
10 mm parenchyma ring, with 5 mm and 15 mm rings as a pre-specified sensitivity analysis. The
pre-registration downgrades the contrast axis to exploratory if CNR-based rank orderings are
unstable (Spearman ρ < 0.8) between any pair of ring widths; the frozen record shows this rule
was **not** triggered (ρ = 0.93 for 10 mm vs. 5 mm, 0.97 for 10 mm vs. 15 mm, 0.86 for 5 mm vs.
15 mm — all above the 0.8 floor), so the contrast axis is reported as a primary stratification
axis, not an exploratory one (`config/frozen_thresholds.yaml` → `strata.contrast_axis_exploratory:
false`).

### Metrics

Five metrics are reported for every stratified cell: Dice, normalized surface Dice at 2 mm,
HD95, case-level lesion detection sensitivity, and false-positive lesions per case (including
on the NIH negative-control cohort). Two operational definitions are frozen ahead of any
result: a reference lesion counts as **detected** if the predicted mask covers at least 10% of
its volume, and a predicted connected component counts as a **false positive** if it has zero
overlap with the reference lesion and a volume of at least 100 mm³. Every number is reported
per stratum with per-cell n and patient-level bootstrap confidence intervals; no aggregate
leaderboard table is reported without intervals, and failure-gallery cases are selected by a
fixed rule (largest |ΔDice| within each stratum), never chosen by eye.

### Pre-registration

Four hypotheses are pre-registered, each with a fixed margin and decision rule, before any
training run of either arm on any fold (tag `prereg-v1`): H1 tests whether the performance gap
between the two architectures tracks lesion volume (locality vs. lesion size); H2 tests
whether it tracks each architecture's use of image context beyond the lesion, referenced
against the CNN's measured effective receptive field; H2b isolates whether attention
specifically (as opposed to capacity) drives any H2 effect, via an identity-block ablation
that swaps attention for a fixed block while holding parameter count constant; H3 tests
whether cross-source performance degradation exceeds a boundary-tolerance floor derived from
morphological perturbation of the reference annotation (no paired independent delineations
exist in this dataset to supplement that floor — see `preregistration/DEVIATIONS.md`). Any
deviation from the frozen configuration, including a reduced training schedule, is disclosed
in `preregistration/DEVIATIONS.md` and cited here rather than silently absorbed into the
results.

---

## Still needed before this file can grow further

- Real Tier A fingerprint (patch size, parameter counts, step time, GPU) to write "Arms and
  the matched-budget control" — blocked on confirmed NYU/USC GPU access.
- Any `results/*` artifact to write Results, Discussion, or Limitations.
