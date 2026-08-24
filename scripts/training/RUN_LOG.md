# GPU run log

The only source of truth for anything that touched a GPU. A result with no run ID here does
not exist — every training/inference command in `run_tier_a.sh`, `verify_pipeline.sh`, and any
manual leave-one-source-out / seed-replicate / identity-control run gets a row before it starts
and is updated when it ends. No unexplained gaps: a crash or restart is its own row, not an
edit to the row it interrupted.

`verify_pipeline.sh` and `run_tier_a.sh` write rows automatically via `log_run.py` (start
before each `nnUNetv2_train` call, finish after — including on failure, logged as `crashed`).
`run_log.csv` is created on the first real run, not committed as an empty file so it's obvious
no run has happened yet. Do not edit past rows by hand except through `log_run.py finish`.

For a manual run not covered by those two scripts (leave-one-source-out, a seed replicate, an
identity-control run), call it directly:

```bash
RUN_ID=$(python scripts/training/log_run.py start --arm cnn --fold loso_Radboud --seed 1 \
    --nnunet-commit "$(pip freeze | grep nnunetv2)" --trainer-or-plans nnUNetResEncUNetLPlans \
    --gpu A100-40GB)
# ... run the training command ...
python scripts/training/log_run.py finish --run-id "$RUN_ID" --status completed \
    --checkpoint-path /path/to/checkpoint_final.pth
```

## Columns

| Column | Meaning |
| --- | --- |
| `run_id` | Unique, e.g. `cnn-fold0-seed0-20260829T1400` |
| `arm` | `cnn` \| `transformer` \| `identity_control` |
| `fold` | 0-4, or `loso_Radboud` / `loso_UMCG` / `loso_MSKCC` for Tier B |
| `seed` | Training seed (default seed, or 1/2 for the fold-0 replicates) |
| `nnunet_commit` | Output of `pip freeze \| grep nnunetv2`, pinned at week-one verification |
| `trainer_or_plans` | e.g. `nnUNetResEncUNetLPlans` or `nnUNet_PrimusV2M_Trainer` |
| `gpu` | Card model, e.g. `A100-40GB` |
| `start_time` | ISO 8601, UTC |
| `end_time` | ISO 8601, UTC; blank while running |
| `status` | `running` \| `completed` \| `crashed` \| `resumed_from:<run_id>` |
| `checkpoint_path` | Absolute path under `$nnUNet_results`; blank until `completed` |
| `notes` | Anything that would otherwise only live in someone's memory |

## Template row (copy into `run_log.csv`)

```csv
run_id,arm,fold,seed,nnunet_commit,trainer_or_plans,gpu,start_time,end_time,status,checkpoint_path,notes
```
