# Pipeline smoke test

`run_smoke_test.py` exercises every script in `scripts/data/` and `scripts/analysis/` against a
small synthetic PANORAMA-like dataset (`synthetic_data.py`) — no real data, no GPU, no nnU-Net
training. Predictions are built with a known, designed pattern (CNN better on small lesions,
worse on large; transformer degrades more in the far occlusion shell), so the checks assert on
the actual statistical outcome (H1 slope sign, H2 decisive-shell contrast, exact case counts),
not just "the script didn't crash."

Runs entirely inside a temp directory holding a throwaway copy of `scripts/`, `src/`, and
`config/` — it never touches or pollutes the real repo's `config/frozen_thresholds.yaml` or
`splits/`.

Run it whenever a data or analysis script changes:

```bash
pip install -r environment/requirements.txt   # nnunetv2/torch not needed for this test
python tests/run_smoke_test.py
```

Exits 0 with `ALL CHECKS PASSED` on success, exits 1 and lists every failed check otherwise.
