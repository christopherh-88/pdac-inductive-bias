# Deviations from prereg-v1

**Empty is the expected state.** This file exists so that a deviation has somewhere to go
other than an edit to `PREREGISTRATION.md`, which is frozen. Nothing in the pre-registration
or in `config/analysis_config.yaml` is ever changed in place after the `prereg-v1` tag; if a
frozen value genuinely has to change, it becomes a new tag (`prereg-v2`, …) with the reason
recorded here, and the paper reports both the original and the revised value.

`scripts/check_prereg_tag.sh` fails if any frozen file differs between the tag and HEAD, so a
deviation cannot pass unnoticed — it either appears here or it breaks the check.

## What counts as a deviation

- Any change to `config/analysis_config.yaml`, `config/frozen_thresholds.yaml`,
  `preregistration/PREREGISTRATION.md`, `splits/splits_final.json`, or
  `splits/fold_assignment.csv` after the tag.
- Training a schedule other than the frozen 1000 × 250 (including any reduced schedule run
  for compute reasons — it is a deviation even when the reason is good, and it is reported
  as one).
- Switching the transformer arm away from PrimusV2, the CNN preset away from the recorded
  one, or the identity control away from the block-replacement defined in
  `src/trainers/primus_identity_trainer.py`.
- Any analysis run under a rule, margin, or threshold not already in the frozen config.
- Any case entering or leaving the cohort after the splits were frozen.

## What is *not* a deviation

- Bug fixes to analysis code that leave every frozen number unchanged (record the commit in
  the run log; the frozen config is what is pre-registered, not the implementation).
- Additional exploratory analyses, as long as they are labelled exploratory in the paper and
  do not replace a pre-registered test.
- Anything the pre-registration itself already provides for: the Primus v1 fallback, the
  contrast axis being downgraded to exploratory by its own rank-stability rule, or the
  optional model-generated-delineation training ablation.

## Log

| Date | Frozen item | Change | Reason | New tag |
| --- | --- | --- | --- | --- |
| — | — | none to date | — | — |
