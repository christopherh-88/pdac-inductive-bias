#!/usr/bin/env python
"""Freeze the matched-budget architecture record: patch size, PrimusV2 token resolution,
parameter counts, and peak VRAM for both arms, in one place, once.

Closes a gap left by verify_pipeline.sh, which ends with an instruction to "record patch/token
sizes, parameter counts, peak VRAM" in config/frozen_thresholds.yaml by hand -- an unwritten
manual step is exactly how the source-grouping decision went undocumented before make_splits.py
was fixed to freeze it automatically. This script is the automated version for architecture.

Token resolution matters here specifically because it is a pre-registered control: if PrimusV2
under-performs on small lesions, that has to be distinguishable from "the lesion is smaller than
a token" rather than attributed to the architecture's inductive bias. PrimusV2 tokenizes with an
8x8x8 voxel stride (github.com/MIC-DKFZ/nnUNet/blob/master/documentation/primus.md); this is
recorded as a default, not hardcoded, in case the installed commit differs.

Run once, at week-one verification, after the CNN/Primus VRAM pairing is confirmed and a
smoke-test forward/backward pass has produced parameter counts and peak VRAM for both arms:

  python scripts/training/record_arch_stats.py \
      --patch-size 96 160 160 --nnunet-commit <hash> --gpu A100-40GB \
      --cnn-plans nnUNetResEncUNetLPlans --cnn-params 102000000 --cnn-vram-gb 22.4 \
      --primus-trainer nnUNet_PrimusV2M_Trainer --tf-params 98000000 --tf-vram-gb 23.1

Patch size must divide evenly by the tokenizer stride in every axis -- if it doesn't, PrimusV2
cannot tokenize the patch, which is itself information: it means the confirmed patch size and
the Primus variant are incompatible and the pairing in environment/SETUP.md needs to change
before training starts, not after.
"""
import argparse
from pathlib import Path

import yaml

FROZEN_PATH = Path(__file__).parents[2] / "config" / "frozen_thresholds.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-size", nargs=3, type=int, required=True, metavar=("Z", "Y", "X"),
                    help="confirmed 3d_fullres patch size from nnU-Net fingerprinting")
    ap.add_argument("--tokenizer-stride", nargs=3, type=int, default=(8, 8, 8),
                    metavar=("Z", "Y", "X"), help="PrimusV2 patch-embedding stride in voxels")
    ap.add_argument("--nnunet-commit", required=True)
    ap.add_argument("--gpu", required=True, help="e.g. A100-40GB")
    ap.add_argument("--cnn-plans", required=True, help="e.g. nnUNetResEncUNetLPlans")
    ap.add_argument("--cnn-params", type=int, default=None)
    ap.add_argument("--cnn-vram-gb", type=float, default=None)
    ap.add_argument("--cnn-step-time-s", type=float, default=None)
    ap.add_argument("--primus-trainer", required=True, help="e.g. nnUNet_PrimusV2M_Trainer")
    ap.add_argument("--tf-params", type=int, default=None)
    ap.add_argument("--tf-vram-gb", type=float, default=None)
    ap.add_argument("--tf-step-time-s", type=float, default=None)
    ap.add_argument("--frozen-out", default=None, type=Path,
                     help="where to write the frozen architecture record; defaults to "
                          "config/frozen_thresholds.yaml. Override for a dry run/smoke test so "
                          "it doesn't write (or refuse to overwrite) the real frozen config.")
    args = ap.parse_args()

    frozen_path = args.frozen_out or FROZEN_PATH
    frozen = yaml.safe_load(open(frozen_path)) if frozen_path.exists() else {}
    frozen = frozen or {}
    if "architecture" in frozen:
        raise SystemExit(f"{frozen_path} already has an 'architecture' section — frozen once, "
                         "same as splits. Delete it by hand first if this is a genuine redo.")

    pz, py, px = args.patch_size
    sz, sy, sx = args.tokenizer_stride
    if pz % sz or py % sy or px % sx:
        raise SystemExit(
            f"Patch size {args.patch_size} does not divide evenly by tokenizer stride "
            f"{args.tokenizer_stride} — PrimusV2 cannot tokenize this patch. Fix the "
            "patch/preset pairing before training, not after.")
    token_grid = (pz // sz, py // sy, px // sx)
    n_tokens = token_grid[0] * token_grid[1] * token_grid[2]

    frozen["architecture"] = {
        "nnunet_commit": args.nnunet_commit,
        "gpu": args.gpu,
        "patch_size_zyx": list(args.patch_size),
        "cnn": {
            "plans": args.cnn_plans,
            "n_params": args.cnn_params,
            "peak_vram_gb": args.cnn_vram_gb,
            "step_time_s": args.cnn_step_time_s,
        },
        "transformer": {
            "trainer": args.primus_trainer,
            "n_params": args.tf_params,
            "peak_vram_gb": args.tf_vram_gb,
            "step_time_s": args.tf_step_time_s,
            "tokenizer_stride_zyx": list(args.tokenizer_stride),
            "token_grid_zyx": list(token_grid),
            "n_tokens": n_tokens,
            "note": "token resolution is a pre-registered control (H1/H2 confound check): a "
                    "lesion smaller than one token cannot be localized any finer than the "
                    "token grid regardless of attention pattern",
        },
    }
    yaml.safe_dump(frozen, open(frozen_path, "w"), sort_keys=False)

    print(f"Patch {tuple(args.patch_size)} -> token grid {token_grid} = {n_tokens} tokens "
          f"(stride {tuple(args.tokenizer_stride)})")
    print(f"CNN: {args.cnn_plans}"
          + (f", {args.cnn_params:,} params" if args.cnn_params else "")
          + (f", {args.cnn_vram_gb} GB peak" if args.cnn_vram_gb else ""))
    print(f"Transformer: {args.primus_trainer}"
          + (f", {args.tf_params:,} params" if args.tf_params else "")
          + (f", {args.tf_vram_gb} GB peak" if args.tf_vram_gb else ""))
    print(f"Wrote architecture record to {frozen_path}.")


if __name__ == "__main__":
    main()
