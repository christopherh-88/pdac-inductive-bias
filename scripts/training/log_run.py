#!/usr/bin/env python
"""Append/update rows in scripts/training/run_log.csv (see RUN_LOG.md).

RUN_LOG.md requires every GPU command in verify_pipeline.sh / run_tier_a.sh to get a row
before it starts and have that row updated when it ends — otherwise the log is a policy with
nothing enforcing it. This is the enforcement: two subcommands, called from the shell scripts.

  start:  appends a row with status=running, prints the run_id to stdout (capture it in bash)
  finish: rewrites that run_id's row with end_time/status/checkpoint_path

Usage:
  RUN_ID=$(python scripts/training/log_run.py start --arm cnn --fold 0 --seed 0 \
      --trainer-or-plans nnUNetResEncUNetLPlans --gpu A100-40GB --nnunet-commit "$(pip freeze | grep nnunetv2)")
  # ... run the actual training command ...
  python scripts/training/log_run.py finish --run-id "$RUN_ID" --status completed \
      --checkpoint-path /path/to/checkpoint_final.pth
"""
import argparse
import csv
import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent / "run_log.csv"
COLUMNS = ["run_id", "arm", "fold", "seed", "nnunet_commit", "trainer_or_plans", "gpu",
           "start_time", "end_time", "status", "checkpoint_path", "notes"]


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_start(args):
    run_id = f"{args.arm}-fold{args.fold}-seed{args.seed}-{_now().replace(':', '').replace('-', '')}"
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            w.writeheader()
        w.writerow({
            "run_id": run_id, "arm": args.arm, "fold": args.fold, "seed": args.seed,
            "nnunet_commit": args.nnunet_commit, "trainer_or_plans": args.trainer_or_plans,
            "gpu": args.gpu, "start_time": _now(), "end_time": "", "status": "running",
            "checkpoint_path": "", "notes": args.notes,
        })
    print(run_id)


def cmd_finish(args):
    if not LOG_PATH.exists():
        raise SystemExit(f"{LOG_PATH} does not exist — no run to finish")
    rows = list(csv.DictReader(open(LOG_PATH)))
    matched = False
    for row in rows:
        if row["run_id"] == args.run_id:
            row["end_time"] = _now()
            row["status"] = args.status
            row["checkpoint_path"] = args.checkpoint_path or row["checkpoint_path"]
            if args.notes:
                row["notes"] = (row["notes"] + "; " + args.notes) if row["notes"] else args.notes
            matched = True
            break
    if not matched:
        raise SystemExit(f"run_id {args.run_id!r} not found in {LOG_PATH}")
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("start")
    s.add_argument("--arm", required=True, choices=["cnn", "transformer", "identity_control"])
    s.add_argument("--fold", required=True, help="0-4, or loso_<Source>")
    s.add_argument("--seed", default="default")
    s.add_argument("--nnunet-commit", required=True)
    s.add_argument("--trainer-or-plans", required=True)
    s.add_argument("--gpu", default="unknown")
    s.add_argument("--notes", default="")
    s.set_defaults(func=cmd_start)

    f = sub.add_parser("finish")
    f.add_argument("--run-id", required=True)
    f.add_argument("--status", required=True, choices=["completed", "crashed"])
    f.add_argument("--checkpoint-path", default="")
    f.add_argument("--notes", default="")
    f.set_defaults(func=cmd_finish)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
