# Pipeline smoke test

`run_smoke_test.py` exercises every script in `scripts/data/`, `scripts/analysis/`, and
`scripts/training/` against a small synthetic PANORAMA-like dataset (`synthetic_data.py`) — no
real data, no GPU, no nnU-Net training. Predictions are built with known, designed patterns
(CNN better on small lesions, worse on large; transformer degrades more in the far occlusion
shell and more under source shift), so the checks assert on the actual statistical outcome —
H1's slope sign, H2's decisive-shell contrast, H3's share-inside-floor, exact case counts —
not just "the script didn't crash".

Where a script produces a verdict, the fixtures exercise **both** directions: H2b is run
against an identity control that reproduces the transformer's map and against one that is
degraded everywhere, so a hard-wired verdict fails the test rather than passing it.

The concurrency sections check what a fan-out would corrupt silently: that no two of the 25
tasks share an nnU-Net results directory, that no two share a prediction directory (seed
replicates carry their seed, or a replicate would overwrite the default run of the same fold),
and that per-fold occlusion covers exactly the cases that fold held out.

`test_trainers.py` runs the custom trainers against a stubbed nnU-Net — no GPU, no install —
and checks the things that would otherwise only surface after wasting GPU hours: that seed
replicates really do change the initialization, that an epoch override lands *before* the LR
schedule is built, and that the wall-clock stop raises rather than returning, so nnU-Net never
marks an unfinished run as finished.

The last section checks the generated Kaggle notebooks: that all three are committed, that
every code cell is valid Python, and that they still match what `notebooks/build_notebooks.py`
produces — otherwise editing the generator silently does nothing.

Runs entirely inside a temp directory holding a throwaway copy of `scripts/`, `src/`, and
`config/` — it never touches or pollutes the real repo's `config/frozen_thresholds.yaml` or
`splits/`.

Run it whenever a data or analysis script changes:

```bash
pip install -r environment/requirements.txt   # nnunetv2/torch not needed for this test
python tests/run_smoke_test.py
```

123 checks. Exits 0 with `ALL CHECKS PASSED`, or exits 1 and lists every failure.

`--keep` leaves the working directory in place (tables, figures, the verdict memo) so the
outputs can be looked at, which is the one check an assertion cannot make.
