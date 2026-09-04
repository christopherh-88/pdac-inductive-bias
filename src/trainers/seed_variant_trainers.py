"""Seed-replicate trainers: the same arm, the same fold, a different initialization.

The pre-registration asks for seed replicates on the replicate fold so initialization
variance can be separated from fold variance. nnU-Net has no `--seed` flag: `nnUNetv2_train`
run twice on the same fold reproduces the same initialization *and* writes to the same
`nnUNet_results/<Dataset>/<Trainer>__<Plans>__<config>/fold_<k>/` directory, so a naive
"just run it again" both fails to vary the seed and silently overwrites the first checkpoint.

Each generated class fixes both problems at once, because nnU-Net keys the output directory
off the trainer class name:
  - a distinct class name  -> a distinct results directory, no clobbering
  - a seed set before the network is built -> a genuinely different initialization

Generated from config/analysis_config.yaml's `training.seed_replicates.extra_seeds`, so the
set of replicate seeds lives in the frozen config rather than in this file:

  nnUNetTrainer_seed1, nnUNetTrainer_seed2, ...             CNN arm (ResEnc plans via -p)
  nnUNet_Primus_seed1_Trainer, nnUNet_Primus_seed2_Trainer  transformer arm

The transformer variants subclass whatever `PRIMUS_BASE_TRAINER` names, exactly as
primus_identity_trainer.py does, so the replicate always matches the frozen transformer arm.

Usage:
  export PRIMUS_BASE_TRAINER=nnUNet_PrimusV2M_Trainer
  nnUNetv2_train 501 3d_fullres 0 -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_seed1
  nnUNetv2_train 501 3d_fullres 0 -tr nnUNet_Primus_seed1_Trainer
(requires this file on the nnU-Net trainer search path — same as the identity trainer)
"""
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2

_CFG = yaml.safe_load(open(Path(__file__).parents[2] / "config" / "analysis_config.yaml"))
_EXTRA_SEEDS = _CFG["training"]["seed_replicates"]["extra_seeds"]

_PRIMUS_BASE_NAME = os.environ.get("PRIMUS_BASE_TRAINER", "nnUNet_PrimusV2M_Trainer")
_PrimusBase = recursive_find_python_class(
    os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"), _PRIMUS_BASE_NAME,
    current_module="nnunetv2.training.nnUNetTrainer")


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _make(name: str, base, seed: int):
    def initialize(self):
        # Before super().initialize(), which is where the network is constructed and its
        # weights drawn — seeding after that point would change nothing about initialization.
        _seed_everything(seed)
        base.initialize(self)   # bound to the captured base, not type(self): a further
                                # subclass of a replicate must not re-enter this method
        self.print_to_log_file(f"SEED REPLICATE: {name} (base={base.__name__}, seed={seed})")

    return type(name, (base,), {"initialize": initialize, "replicate_seed": seed})


for _seed in _EXTRA_SEEDS:
    globals()[f"nnUNetTrainer_seed{_seed}"] = _make(f"nnUNetTrainer_seed{_seed}",
                                                    nnUNetTrainer, int(_seed))
    if _PrimusBase is not None:
        globals()[f"nnUNet_Primus_seed{_seed}_Trainer"] = _make(
            f"nnUNet_Primus_seed{_seed}_Trainer", _PrimusBase, int(_seed))

__all__ = ([f"nnUNetTrainer_seed{s}" for s in _EXTRA_SEEDS]
           + ([f"nnUNet_Primus_seed{s}_Trainer" for s in _EXTRA_SEEDS]
              if _PrimusBase is not None else []))
