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
