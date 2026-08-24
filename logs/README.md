# Run log

`run_log.csv` is the only source of truth for anything that touched a GPU. A result with no
run ID does not exist (see the project description's "How we run this").

One row per training run: `run_id, arm, fold, seed, nnunet_commit, gpu, start, end,
checkpoint_path, status, notes`.

- `run_id`: short unique string, e.g. `cnn-f0-s0-20260901`.
- `arm`: `cnn`, `transformer`, or `identity_control`.
- `nnunet_commit`: the pinned nnU-Net commit hash the run used (see `environment/SETUP.md`).
- `status`: `running`, `done`, `crashed`, `resumed`.
- Append-only. Crashes and resumes get their own rows rather than edited-in-place entries, so
  the log has no unexplained gaps.

The two week-one smoke tests (Phase 0, "All: smoke test both arms") are the first entries once
a GPU is available.
