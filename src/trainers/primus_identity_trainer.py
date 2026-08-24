"""H2b identity control: PrimusV2 with its transformer blocks replaced by identity.

This is the Primus 'stress test' (Wald et al., 2025): if the identity-ablated model
reproduces the full model's stratified failure map, the attention blocks are not the cause.

The exact module layout of PrimusV2 depends on the installed nnU-Net commit, so this trainer
locates the transformer block stack structurally instead of by a hard-coded attribute name:
it finds the largest nn.ModuleList whose depth matches the preset's expected layer count and
whose members contain attention submodules, then replaces every member with a pass-through.
It asserts loudly (parameter-count drop, block count) rather than training silently on the
wrong ablation. VERIFY during week-one pipeline check.

Usage:
  export PRIMUS_BASE_TRAINER=nnUNet_PrimusV2M_Trainer   # match the frozen transformer arm
  nnUNetv2_train 501 3d_fullres FOLD -tr nnUNet_PrimusV2_Identity_Trainer
(requires this file on the nnU-Net trainer search path — e.g. symlink into
 nnunetv2/training/nnUNetTrainer/variants/ or install this repo with `pip install -e .`)
"""
import os

import torch.nn as nn
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2

_BASE_NAME = os.environ.get("PRIMUS_BASE_TRAINER", "nnUNet_PrimusV2M_Trainer")
_Base = recursive_find_python_class(
    os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), _BASE_NAME,
    current_module="nnunetv2.training.nnUNetTrainer")
if _Base is None:
    raise ImportError(
        f"Base trainer {_BASE_NAME} not found in the installed nnunetv2 — PrimusV2 requires "
        "nnU-Net master (see environment/SETUP.md).")


class _PassThrough(nn.Module):
    """Identity that tolerates the extra args Eva-style blocks receive (e.g. RoPE)."""

    def forward(self, x, *args, **kwargs):
        return x


def _find_transformer_blocks(network: nn.Module) -> nn.ModuleList:
    candidates = []
    for name, module in network.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 4:
            has_attn = any(
                any("attn" in sub_name.lower() for sub_name, _ in block.named_modules())
                for block in module)
            if has_attn:
                candidates.append((name, module))
    if not candidates:
        raise RuntimeError(
            "Could not locate the transformer block stack in the PrimusV2 network. "
            "Inspect the architecture and adapt _find_transformer_blocks — do NOT train "
            "this ablation until it is verified.")
    # the transformer trunk is the deepest such stack
    candidates.sort(key=lambda c: len(c[1]), reverse=True)
    name, blocks = candidates[0]
    print(f"[identity control] replacing {len(blocks)} blocks at '{name}' with identity")
    return blocks


class nnUNet_PrimusV2_Identity_Trainer(_Base):
    def initialize(self):
        super().initialize()
        net = self.network.module if hasattr(self.network, "module") else self.network
        params_before = sum(p.numel() for p in net.parameters())
        blocks = _find_transformer_blocks(net)
        n_blocks = len(blocks)
        for i in range(n_blocks):
            blocks[i] = _PassThrough()
        params_after = sum(p.numel() for p in net.parameters())
        assert params_after < params_before * 0.8, (
            f"Identity ablation removed too few parameters "
            f"({params_before} -> {params_after}); the wrong module list was replaced.")

        # super().initialize() already built self.optimizer / self.lr_scheduler from the
        # pre-ablation network. Left alone, the optimizer keeps stale param_groups pointing at
        # the now-orphaned transformer-block weights (dead momentum/state buffers, no gradient
        # ever reaches them) — wasted VRAM against the pre-registered matched-budget control.
        # Rebuild both from the post-ablation parameters.
        self.optimizer, self.lr_scheduler = self.configure_optimizers()

        self.print_to_log_file(
            f"IDENTITY CONTROL ACTIVE (base={_BASE_NAME}): {n_blocks} transformer blocks "
            f"-> identity; params {params_before:,} -> {params_after:,}; "
            f"optimizer rebuilt post-ablation")
