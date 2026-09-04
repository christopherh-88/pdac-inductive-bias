# ISBI 2027 — four-page outline

**Status: structure only. There are no numbers in this file and none should be added by hand.**
Every claim below names the artifact that fills it. If an artifact does not exist yet, the
claim is not made — that is the whole point of having written this before the results existed.

Deadline: Oct 26, 2026. Internal freeze Oct 19 — no new results enter after that date.
Division of writing: methods and statistics, training and compute, figures and the mechanistic
section.

Word budgets are for a four-page ISBI paper at roughly 2,400 words of body text plus figures.

---

## Title and abstract (~180 words)

Working title: *Inductive Bias on a Hard Organ: A Pre-Registered Failure-Mode Comparison of a
Convolutional U-Net and a Pure Transformer for PDAC Segmentation in CT.*

The abstract states **where** and **why**, never a winner. One sentence per element: the two
arms at a matched budget inside one pipeline; the three stratification axes; the two mechanism
tests; the boundary-tolerance floor; the pre-registration tag and public repository.

Fills from: `results/VERDICT.md` (the four verdicts), `config/frozen_thresholds.yaml`
(cohort size, folds).

## 1. Introduction (~450 words)

Three moves, no more:

1. PDAC segmentation is hard for a stated, specific reason — small, low-contrast lesions with
   ambiguous boundaries in an organ with high anatomical variability.
2. Architecture comparisons in medical imaging usually report an aggregate winner. An
   aggregate number over a cohort with this much within-cohort heterogeneity hides exactly the
   thing a clinician needs to know: which cases each model fails on.
3. This paper reports a stratified failure map plus three tests of the mechanism behind it,
   with every rule fixed before training.

Cite: nnU-Net Revisited (arXiv:2404.09556), Primus (TMLR, OpenReview `x4vZE4PDEu`), PANORAMA
(the datasets page and the study-protocol DOI).

State the pre-registration tag and repository URL here, not only in a data-availability
footnote.

## 2. Data and methods (~700 words)

**Cohort.** PANORAMA PDAC cases, manual delineations only in test folds; MSD Task07 restricted
to PDAC; NIH Pancreas-CT as a negative control and never a tumour test set; duplicates removed
by image content, not by case ID.
Fills from: `splits/cohort_provenance.yaml`, `splits/duplicates.csv`,
`config/frozen_thresholds.yaml` → `splits`.

**Stratification.** Volume tertiles; CNR at a 10 mm parenchyma ring with 5 mm and 15 mm as a
pre-specified sensitivity analysis; acquisition source.
Fills from: `config/frozen_thresholds.yaml` → `strata`,
`results/contrast_sensitivity/summary.yaml`.
**If the rank-stability rule fired, the contrast axis is labelled exploratory here, in the
methods, not softened in the discussion.**

**Arms and the matched-budget control.** One nnU-Net pipeline: same preprocessing, target
spacing, augmentation, loss, 1000 × 250 schedule, and inference. Same VRAM ceiling, same
steps, comparable wall-clock. Parameter counts reported, not controlled. Token resolution
stated explicitly so a small-lesion result can never be attributed to tokenization alone.
Fills from: `config/frozen_thresholds.yaml` → `architecture` (patch size, token grid, params,
peak VRAM, step time, GPU, nnU-Net commit).

**Metrics.** Dice, NSD at 2 mm, HD95, detection at 10% reference coverage, false positives as
zero-overlap components ≥ 100 mm³. Every number per stratum with per-cell n and patient-level
bootstrap intervals; no aggregate leaderboard table without intervals.

**Pre-registration.** One paragraph: H1, H2, H2b, H3, each with its margin, stated as fixed
before training, with the tag name. Deviations, if any, cited to
`preregistration/DEVIATIONS.md` — including any reduced schedule.

## 3. Results (~750 words, and the figures carry most of the load)

Order is fixed by the hypotheses, not by what came out best.

**3.1 H1 — volume.** Slope of ΔDice on log volume with its bootstrap CI; Wilcoxon within
tertiles; the large-tertile equivalence result against the ±3-point margin.
Fills from: `results/h1/summary.yaml`, `results/h1/wilcoxon_by_tertile.csv`.
Figures 1 and 2.

**3.2 The stratified map.** Volume × contrast, one panel per arm, per-cell n and CI.
Fills from: `results/h1/stratum_heatmap.csv`. Figures 2 and 3.

**3.3 H2 — context use.** Dice loss per occlusion shell per arm, with the CNN effective
receptive field marked on the same axis.
Fills from: `results/h2/summary.yaml`, `results/erf_cnn.yaml`. Figure 4.

**3.4 H2b — attention dependence.** The identity control's map in the identical layout, and
how many cells agree within CI.
Fills from: `results/h2b/summary.yaml`. Figure 5.

**3.5 H3 — source shift against boundary tolerance.** Cross-source Dice loss per arm per
source, and the share of it that sits inside the floor.
Fills from: `results/loso/by_source.csv`, `results/h3/summary.yaml`. Figure 6.

**3.6 Negative control and variance.** False positives per case on the NIH negatives, in
domain and under source shift; seed variance separated from fold variance.
Fills from: `results/nih_fp/summary.yaml`, `results/variance_*.yaml`.

## 4. Discussion (~350 words)

What the failure map says a practitioner should expect from each architecture on which cases.
What the occlusion curves and the identity control do and do not license as a mechanistic
claim. How much of the cross-source gap is annotation tolerance rather than model degradation.

If a hypothesis reads NOT SUPPORTED, it is reported here as a stratified negative result with
its intervals — per section 7 of the pre-registration — and not re-tested under a rule invented
after the fact.

## 5. Limitations (~180 words)

Named, not hedged:

- the boundary-tolerance floor is synthetic unless
  `results/inter_rater/STATEMENT.md` says otherwise; quote that statement's conclusion
- cell counts in the small-lesion / low-contrast corners, and what the intervals there permit
- seed versus fold variance, from `results/variance_*.yaml`
- one framework, two architectures: nothing here generalises past nnU-Net's pipeline
- any deviation in `preregistration/DEVIATIONS.md`, including a reduced training schedule

## Figures (six, of which four fit a four-page paper — choose by what the verdicts need)

All produced by `scripts/analysis/make_figures.py`; the gallery by
`scripts/analysis/failure_gallery.py`, selected by the frozen rule and never by eye.

| Fig | Content | File |
| --- | --- | --- |
| 1 | ΔDice against log volume, with the fitted slope and its bootstrap band | `fig1_delta_dice_vs_volume.png` |
| 2 | Stratified mean-Dice maps, one panel per arm | `fig2_stratified_maps.png` |
| 3 | Paired ΔDice map, diverging scale | `fig3_delta_map.png` |
| 4 | Occlusion curves with the CNN effective receptive field | `fig4_occlusion_curves.png` |
| 5 | Identity control in the identical layout | `fig5_identity_control_map.png` |
| 6 | Cross-source degradation against the boundary floor | `fig6_loso.png` |

## Submission checklist

- [ ] `bash scripts/check_prereg_tag.sh` prints PASS
- [ ] every number in the text traced to a file under `results/`
- [ ] `results/VERDICT.md` regenerated after the last analysis run, and the paper's verdicts
      match it word for word
- [ ] `preregistration/DEVIATIONS.md` reflects every deviation, including training schedule
- [ ] repository public, `prereg-v1` pushed, URL in the paper
- [ ] licences stated: PANORAMA CC BY-NC 4.0, MSD CC BY-SA 4.0, NIH CC BY, our code MIT
- [ ] formatting and anonymisation checked against the ISBI 2027 author instructions
- [ ] the unpublished / not-under-review rule confirmed in writing against anything else out
- [ ] internal freeze honoured: no result in the paper postdates it
