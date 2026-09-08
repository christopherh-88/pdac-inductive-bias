#!/usr/bin/env python
"""H2 reference measurement: the CNN's effective receptive field (ERF), from input gradients.

Method (Luo et al., NeurIPS 2016 -- "Understanding the Effective Receptive Field"): feed a
small-noise input through the network, backpropagate a unit gradient from the output voxel at
the patch center, and look at the resulting |gradient| map over the input. Unlike the
*theoretical* receptive field (which only bounds what a voxel could see), this measures what it
actually, empirically depends on -- the concentration of that gradient mass falls off with
distance from the center rather than being uniform inside a hard boundary.

We report, in mm (accounting for real voxel spacing): the radius containing 50/90/95% of the
total gradient mass, and the fraction of gradient mass falling inside each of the study's own
occlusion shells (10-20, 20-40, 40-80 mm) so the two measurements sit on the same axis in the
H2 figure (a vertical line at the 95%-mass radius, per the pre-registration).

Two usage modes:
  --self-test       Runs against a small synthetic 3D CNN (no nnU-Net, no checkpoint needed).
                     Verifies the measurement code is sound: ERF must grow monotonically with
                     network depth. Safe to run any time, including with no GPU.
  --nnunet-checkpoint / --plans / --dataset / --configuration / --fold
                     Loads the actual trained ResEnc network via nnU-Net's own architecture
                     builder and checkpoint loader, and measures its real ERF. Requires
                     nnunetv2 installed and a completed training run -- not runnable in this
                     repo's dry-run environment, wired up for week-one verification / Tier A.

Usage:
  python scripts/analysis/effective_receptive_field.py --self-test
  python scripts/analysis/effective_receptive_field.py \
      --nnunet-checkpoint /path/checkpoint_final.pth --dataset 501 --configuration 3d_fullres \
      --plans nnUNetResEncUNetLPlans --fold 0 --out results/erf_cnn.yaml
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

CFG = yaml.safe_load(open(Path(__file__).parents[2] / "config" / "analysis_config.yaml"))


def measure_erf(model: nn.Module, input_shape_zyx, spacing_zyx_mm, device="cpu",
                shell_bounds_mm=None, seed=20260823):
    """Returns a dict of ERF summary statistics. input_shape_zyx: (Z, Y, X) voxel patch size.
    spacing_zyx_mm: (sz, sy, sx) mm per voxel, matching numpy's (z, y, x) array axis order.
    """
    torch.manual_seed(seed)
    model = model.to(device).eval()
    z, y, x = input_shape_zyx
    inp = (0.1 * torch.randn(1, 1, z, y, x, device=device)).requires_grad_(True)

    out = model(inp)
    if out.ndim != 5:
        raise ValueError(f"expected a 5D (N,C,Z,Y,X) output, got shape {tuple(out.shape)}")
    oz, oy, ox = out.shape[2], out.shape[3], out.shape[4]
    cz, cy, cx = oz // 2, oy // 2, ox // 2

    # Sum over output channels at the single center voxel -- a unit gradient signal there,
    # backpropagated to see which input voxels it actually depended on.
    target = out[0, :, cz, cy, cx].sum()
    target.backward()

    grad = inp.grad[0, 0].detach().abs().cpu().numpy()  # (Z, Y, X)
    total = grad.sum()
    if total <= 0:
        raise RuntimeError("all-zero gradient -- network is not differentiable end to end "
                           "w.r.t. the input at this output location; check the model")

    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    icz, icy, icx = z // 2, y // 2, x // 2
    sz, sy, sx = spacing_zyx_mm
    dist_mm = np.sqrt(((zz - icz) * sz) ** 2 + ((yy - icy) * sy) ** 2 + ((xx - icx) * sx) ** 2)

    order = np.argsort(dist_mm, axis=None)
    sorted_dist = dist_mm.ravel()[order]
    sorted_mass = grad.ravel()[order] / total
    cum_mass = np.cumsum(sorted_mass)

    def radius_at(fraction):
        idx = np.searchsorted(cum_mass, fraction)
        idx = min(idx, len(sorted_dist) - 1)
        return float(sorted_dist[idx])

    result = {
        "erf_radius_mm_50pct": radius_at(0.50),
        "erf_radius_mm_90pct": radius_at(0.90),
        "erf_radius_mm_95pct": radius_at(0.95),
        "input_shape_zyx": list(input_shape_zyx),
        "spacing_zyx_mm": list(spacing_zyx_mm),
    }
    if shell_bounds_mm:
        shell_frac = {}
        for lo, hi in shell_bounds_mm:
            in_shell = (dist_mm > lo) & (dist_mm <= hi)
            shell_frac[f"{lo}-{hi}mm"] = float(grad[in_shell].sum() / total)
        result["gradient_mass_fraction_by_occlusion_shell"] = shell_frac
        inside_10mm = float(grad[dist_mm <= 10].sum() / total)
        result["gradient_mass_fraction_within_10mm"] = inside_10mm
    return result


class _ToyConvNet(nn.Module):
    """Minimal stand-in for self-testing the measurement code: a stack of stride-1, 3x3x3
    'same'-padded conv+ReLU blocks. Not the real ResEnc architecture -- only used to verify
    that measure_erf() produces a sensible, monotonically-growing ERF as depth increases."""

    def __init__(self, n_layers, channels=8):
        super().__init__()
        layers = []
        c_in = 1
        for _ in range(n_layers):
            layers += [nn.Conv3d(c_in, channels, 3, padding=1), nn.ReLU(inplace=True)]
            c_in = channels
        layers += [nn.Conv3d(c_in, 1, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def self_test():
    print("Self-test: ERF must grow monotonically with synthetic network depth "
          "(no nnU-Net / GPU required)\n")
    shape = (32, 32, 32)
    spacing = (1.0, 1.0, 1.0)
    radii = []
    for n_layers in (2, 4, 8, 16):
        model = _ToyConvNet(n_layers)
        r = measure_erf(model, shape, spacing)
        radii.append(r["erf_radius_mm_95pct"])
        print(f"  {n_layers:2d} conv layers -> 95%-mass ERF radius = {r['erf_radius_mm_95pct']:.2f} mm "
              f"(50%={r['erf_radius_mm_50pct']:.2f}, 90%={r['erf_radius_mm_90pct']:.2f})")
    monotonic = all(radii[i] <= radii[i + 1] for i in range(len(radii) - 1))
    print(f"\n{'PASS' if monotonic else 'FAIL'}: ERF radius {'is' if monotonic else 'is NOT'} "
          f"monotonically non-decreasing with depth: {[round(r, 2) for r in radii]}")
    if not monotonic:
        raise SystemExit(1)


def load_nnunet_network(checkpoint, dataset, configuration, plans, fold, trainer="nnUNetTrainer"):
    """Builds the trained network from an nnU-Net checkpoint using the exact trainer class that
    produced it, so the ERF is measured on the exact network that was trained -- not a
    re-implementation. Requires nnunetv2 installed and nnUNet_preprocessed/results set.

    The trainer class matters, not just the plans: nnUNetTrainer's own build_network_architecture
    goes through the generic get_network_from_plans path (fine for ResEnc/CNN arms), but PrimusV2
    trainers override build_network_architecture entirely to construct the transformer (patch
    embedding, attention blocks) -- get_network_from_plans cannot build that network at all, it
    silently builds a PlainConvUNet instead and load_state_dict then fails on every key. Reading
    the trainer name straight out of the checkpoint (nnU-Net writes it to every checkpoint as
    'trainer_name') and locating that exact class is what makes this work for both arms.
    """
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    import nnunetv2
    import json as _json

    # nnU-Net's directories are named Dataset<ID>_<Name> ("Dataset501_PDAC"), not Dataset<ID>,
    # so resolve by ID prefix. --dataset also accepts the full folder name.
    root = Path(nnUNet_preprocessed)
    if (root / dataset).exists():
        ds_dir = root / dataset
    else:
        matches = sorted(root.glob(f"Dataset{int(dataset):03d}_*"))
        if not matches:
            raise SystemExit(f"No preprocessed dataset for ID {dataset} under {root} "
                             f"(found: {[p.name for p in root.iterdir() if p.is_dir()]})")
        ds_dir = matches[0]
    plans_path = ds_dir / f"{plans}.json"
    dataset_json_path = ds_dir / "dataset.json"
    plans_manager = PlansManager(_json.load(open(plans_path)))
    dataset_json = _json.load(open(dataset_json_path))
    config_manager = plans_manager.get_configuration(configuration)

    # weights_only=False: nnU-Net checkpoints store numpy scalars (e.g. best-EMA dice)
    # alongside the state dict, which torch>=2.6's default weights_only=True rejects.
    # Safe here because these are our own training runs' checkpoints, not third-party files.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    trainer_name = ckpt.get("trainer_name", trainer)
    trainer_class = recursive_find_python_class(
        os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), trainer_name,
        current_module="nnunetv2.training.nnUNetTrainer")
    if trainer_class is None:
        raise SystemExit(f"Could not locate trainer class '{trainer_name}' (from the "
                          f"checkpoint) under nnunetv2.training.nnUNetTrainer -- if it was a "
                          f"dynamically-installed variant (e.g. the TierALite trainers), it "
                          f"must be installed into this environment's nnunetv2 package first.")

    num_input_channels = len(dataset_json["channel_names"])
    num_output_channels = len(dataset_json["labels"])
    network = trainer_class.build_network_architecture(
        plans_manager, config_manager, num_input_channels, num_output_channels,
        enable_deep_supervision=False)

    state_dict = ckpt.get("network_weights", ckpt)
    network.load_state_dict(state_dict)
    patch_size = config_manager.patch_size  # (X, Y, Z) in nnU-Net's convention
    spacing = config_manager.spacing        # (X, Y, Z) mm
    # measure_erf expects (Z, Y, X) to match numpy array axis order used throughout this repo
    return network, tuple(reversed(patch_size)), tuple(reversed(spacing)), trainer_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="run the no-GPU synthetic-network sanity check and exit")
    ap.add_argument("--nnunet-checkpoint", type=Path)
    ap.add_argument("--dataset", default="501")
    ap.add_argument("--configuration", default="3d_fullres")
    ap.add_argument("--plans", default="nnUNetResEncUNetLPlans")
    ap.add_argument("--fold", default="0")
    ap.add_argument("--out", type=Path, default=Path("results/erf_cnn.yaml"))
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.nnunet_checkpoint:
        raise SystemExit("Provide --nnunet-checkpoint (or run --self-test for the no-GPU check)")

    network, patch_shape_zyx, spacing_zyx, trainer_name = load_nnunet_network(
        args.nnunet_checkpoint, args.dataset, args.configuration, args.plans, args.fold)
    shells = [tuple(s) for s in CFG["hypotheses"]["h2"]["occlusion_shells_mm"]]
    result = measure_erf(network, patch_shape_zyx, spacing_zyx, shell_bounds_mm=shells)
    result["trainer_name"] = trainer_name

    args.out.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(result, open(args.out, "w"), sort_keys=False)
    print(f"{trainer_name} effective receptive field: 95%-mass radius = "
          f"{result['erf_radius_mm_95pct']:.1f} mm (50%={result['erf_radius_mm_50pct']:.1f}, "
          f"90%={result['erf_radius_mm_90pct']:.1f})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
