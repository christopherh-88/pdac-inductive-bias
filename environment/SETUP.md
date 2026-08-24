# Environment setup

## 1. Python environment

```bash
python -m venv .venv && source .venv/bin/activate   # or conda create -n pdac python=3.11
pip install -r environment/requirements.txt
```

`nnunetv2` is installed from GitHub master because the PrimusV2 trainers
(`nnUNet_PrimusV2S/B/M/L_Trainer`, see
[documentation/primus.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/primus.md))
are not guaranteed to be in the PyPI release. **At week-one verification, pin the commit:**

```bash
pip freeze | grep nnunetv2      # record the commit hash in config/frozen_thresholds.yaml
```

Verify the trainers exist before anything else:

```bash
python -c "from nnunetv2.training.nnUNetTrainer.variants.primus.nnUNet_PrimusTrainers import *" \
  2>/dev/null || python - <<'EOF'
import importlib, pkgutil, nnunetv2.training.nnUNetTrainer as t
names = [m.name for m in pkgutil.walk_packages(t.__path__, t.__name__ + '.')]
print([n for n in names if 'rimus' in n] or 'NO PRIMUS TRAINER FOUND — check nnU-Net commit')
EOF
```

**Verified 2026-08-24 (CPU-only, no GPU needed for this check):** nnU-Net master @
`0e495086eb108ff79afe106291e8c15bd2f2bc3a` exports all four required trainer classes —
`nnUNet_PrimusV2S_Trainer`, `nnUNet_PrimusV2B_Trainer`, `nnUNet_PrimusV2M_Trainer`,
`nnUNet_PrimusV2L_Trainer` — plus the V3 family (`nnUNet_PrimusV3{S,B,M,L}_Trainer`, not used;
see `preregistration/PREREGISTRATION.md` § "Primus V2 vs V3S"). Full `pip freeze` from that
check is in `environment/pip_freeze_verification.txt`. This confirms the trainers *import*;
it does not confirm they *instantiate on our patch size* or fit VRAM — those still need real
data (fingerprinting) and a real GPU, and are still open Phase 0 items for R2/R3.

## 2. nnU-Net environment variables

```bash
export nnUNet_raw=/path/to/data/nnUNet_raw
export nnUNet_preprocessed=/path/to/data/nnUNet_preprocessed
export nnUNet_results=/path/to/data/nnUNet_results
```

Add these to your shell profile on the training machine. None of these directories are inside
the repo.

## 3. Preset selection by confirmed VRAM (matched-budget control)

The two arms must run under the same VRAM ceiling, step count (1000 × 250), and wall-clock
allowance. Choose the pair once GPU access is confirmed and record it in
`config/frozen_thresholds.yaml`:

| VRAM | CNN arm (plans) | Transformer arm (trainer) |
| --- | --- | --- |
| ~24 GB | `nnUNetResEncUNetLPlans` | `nnUNet_PrimusV2M_Trainer` (verify fit; drop to B if OOM) |
| ~40–48 GB | `nnUNetResEncUNetXLPlans` | `nnUNet_PrimusV2L_Trainer` (verify fit) |

ResEnc VRAM requirements per the
[ResEnc presets doc](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md):
M ≈ 9–11 GB, L ≈ 24 GB, XL ≈ 40 GB. PrimusV2 memory use must be measured on this data during
week-one verification — the pairing above is a starting point, and the final matched pair is
whatever fits the same ceiling. Record the actual patch size, token/patch-embedding
resolution, parameter counts, and peak VRAM of both arms with
`scripts/training/record_arch_stats.py` (called at the end of `verify_pipeline.sh`), which
writes them into `config/frozen_thresholds.yaml` and refuses to run twice. Token resolution is
a pre-registered control: PrimusV2 tokenizes with an 8×8×8 voxel stride, so a lesion smaller
than one token cannot be localized any finer than the token grid regardless of attention
pattern — the script computes and freezes the token grid so this confound is checkable, not
just asserted.

## 4. Disk

- PANORAMA batches: ~180 GB zipped; keep ~400 GB free for extraction + nnU-Net preprocessed.
- PanTS (Tier C only): ~300 GB additional. Not needed for Tiers A/B.

## 5. Reproducibility

- Default nnU-Net seed for the main runs; seed replicates on fold 0 use seeds 1 and 2
  (`training.seed_replicates` in `config/analysis_config.yaml`).
- Record GPU model, driver, CUDA, and PyTorch versions in `config/frozen_thresholds.yaml`
  when training starts.
