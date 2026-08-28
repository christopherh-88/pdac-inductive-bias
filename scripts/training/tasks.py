#!/usr/bin/env python
"""The 25 training runs, as one named list that both the launcher and the notebooks read.

Every run in the study is `<arm>-<fold>[-seed<N>]`. Naming them in one place means the worker
notebook takes a single string, the run log and the results directories agree with each other,
and "which runs are left" is a set difference rather than a memory exercise.

  Tier A   cnn-fold0 .. cnn-fold4, tf-fold0 .. tf-fold4                        10 runs
  Tier B   cnn-loso_<Source>, tf-loso_<Source>                                  6 runs
           identity-fold0 .. identity-fold4                                     5 runs
           cnn-fold<r>-seed1/2, tf-fold<r>-seed1/2                              4 runs

Each task resolves to the trainer class and plans that run actually uses, including the
`_budget` suffix from src/trainers/budget_trainers.py when a wall-clock budget or an epoch
override is in play — so a task name maps to exactly one nnU-Net results directory and two
different runs can never land in the same one.

  python scripts/training/tasks.py                 # list every task
  python scripts/training/tasks.py --tier A        # just Tier A
  python scripts/training/tasks.py --describe cnn-fold0
"""
import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).parents[2]
ARMS = {"cnn": "cnn", "tf": "transformer", "identity": "identity_control"}


def _frozen(repo: Path = REPO):
    p = repo / "config" / "frozen_thresholds.yaml"
    frozen = (yaml.safe_load(open(p)) or {}) if p.exists() else {}
    arch = frozen.get("architecture") or {}
    return (arch.get("cnn", {}).get("plans"),
            arch.get("transformer", {}).get("trainer"))


def all_tasks(repo: Path = REPO):
    """Every task name, in the order the study needs them (Tier A first)."""
    cfg = yaml.safe_load(open(repo / "config" / "analysis_config.yaml"))
    n_folds = cfg["splits"]["n_folds"]
    rep = cfg["training"]["seed_replicates"]

    names = [f"{arm}-fold{f}" for f in range(n_folds) for arm in ("cnn", "tf")]

    loso_csv = repo / "splits" / "loso_folds.csv"
    if loso_csv.exists():
        import pandas as pd
        for src in pd.read_csv(loso_csv)["held_out_source"]:
            names += [f"cnn-loso_{src}", f"tf-loso_{src}"]

    names += [f"identity-fold{f}" for f in range(n_folds)]
    names += [f"{arm}-fold{rep['fold']}-seed{s}"
              for s in rep["extra_seeds"] for arm in ("cnn", "tf")]
    return names


def tier_of(name: str) -> str:
    return "A" if ("-fold" in name and "loso" not in name and "seed" not in name
                   and not name.startswith("identity")) else "B"


def parse(name: str, repo: Path = REPO, budget_suffix: str = ""):
    """Resolve a task name to everything needed to launch and to find its output.

    budget_suffix comes from src/trainers/budget_trainers.SUFFIX ("_budget", or
    "_budget50ep" when PDAC_EPOCHS is set). Passing "" gives the plain trainer names, which
    is what a cluster run without a session cap uses.
    """
    parts = name.split("-")
    if len(parts) < 2 or parts[0] not in ARMS:
        raise ValueError(f"bad task name {name!r}; expected <arm>-<fold>[-seed<N>] with arm in "
                         f"{sorted(ARMS)}")
    arm_key, fold_label = parts[0], parts[1]
    seed = "default"
    if len(parts) == 3:
        if not parts[2].startswith("seed"):
            raise ValueError(f"bad task name {name!r}; third field must be seed<N>")
        seed = parts[2][len("seed"):]
    elif len(parts) > 3:
        raise ValueError(f"bad task name {name!r}; too many fields")

    cnn_plans, primus_trainer = _frozen(repo)

    if fold_label.startswith("loso_"):
        loso_csv = repo / "splits" / "loso_folds.csv"
        if not loso_csv.exists():
            raise SystemExit(f"{loso_csv} is missing — run scripts/data/make_loso_splits.py "
                             "before any leave-one-source-out task.")
        import pandas as pd
        lf = pd.read_csv(loso_csv)
        source = fold_label[len("loso_"):]
        row = lf[lf["held_out_source"] == source]
        if row.empty:
            raise SystemExit(f"{source!r} is not a LOSO source; have "
                             f"{lf['held_out_source'].tolist()}")
        nnunet_fold = int(row.iloc[0]["nnunet_fold"])
    elif fold_label.startswith("fold"):
        nnunet_fold = int(fold_label[len("fold"):])
    else:
        raise ValueError(f"bad fold {fold_label!r} in {name!r}; expected fold<N> or loso_<Source>")

    if seed != "default" and arm_key == "identity":
        raise SystemExit("Seed replicates are pre-registered for the two arms only, not for "
                         "the identity control.")

    if arm_key == "cnn":
        trainer, plans = "nnUNetTrainer", cnn_plans
        if seed != "default":
            trainer = f"nnUNetTrainer_seed{seed}"
    elif arm_key == "tf":
        trainer, plans = primus_trainer, None
        if seed != "default":
            trainer = f"nnUNet_Primus_seed{seed}_Trainer"
    else:
        trainer, plans = "nnUNet_PrimusV2_Identity_Trainer", None

    if trainer is None:
        raise SystemExit("The frozen config has no 'architecture' section yet — run the "
                         "preparation notebook (or record_arch_stats.py) before training. "
                         "Training before the matched-budget record exists means the study's "
                         "only control was never established.")

    # Where this task's predictions live. Two runs that differ only by seed share a fold
    # label, so the seed has to be in the path or a replicate would overwrite the default
    # run's out-of-fold predictions and the variance decomposition would compare a run
    # against itself.
    pred_key = fold_label if seed == "default" else f"{fold_label}-seed{seed}"

    return {
        "task": name,
        "arm": ARMS[arm_key],
        "arm_key": arm_key,
        "fold_label": fold_label,
        "pred_key": pred_key,
        "is_replicate": seed != "default",
        "nnunet_fold": nnunet_fold,
        "seed": seed,
        "trainer": trainer + budget_suffix,
        "base_trainer": trainer,
        "plans": plans,
        "primus_base": primus_trainer,
        "tier": tier_of(name),
        "results_subdir": f"{trainer}{budget_suffix}__{plans or 'nnUNetPlans'}__3d_fullres",
        # No --npz. It makes nnU-Net save the full predicted softmax map as .npz for every
        # validation case, which is needed only for cross-configuration ensembling and
        # find_best_configuration — neither of which this study does. On a capped output it is
        # the largest uncontrolled writer there is: gigabytes per run, for files nothing reads.
        "train_cmd_args": (f"501 3d_fullres {nnunet_fold} -tr {trainer}{budget_suffix}"
                           + (f" -p {plans}" if plans else "")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["A", "B"], default=None)
    ap.add_argument("--describe", default=None, metavar="TASK")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--budget-suffix", default="")
    args = ap.parse_args()

    if args.describe:
        print(json.dumps(parse(args.describe, budget_suffix=args.budget_suffix), indent=1))
        return

    names = [n for n in all_tasks() if args.tier is None or tier_of(n) == args.tier]
    if args.json:
        print(json.dumps(names, indent=1))
        return
    for tier in ("A", "B"):
        group = [n for n in names if tier_of(n) == tier]
        if group:
            print(f"Tier {tier} ({len(group)} runs):")
            for n in group:
                print(f"  {n}")
    print(f"\n{len(names)} runs total")
    if not (REPO / "splits" / "loso_folds.csv").exists():
        print("(leave-one-source-out tasks are absent until make_loso_splits.py has run)")


if __name__ == "__main__":
    main()
