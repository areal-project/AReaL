# ThreadWeaver SFT on Ray Server

This adds a new launcher: `train_ray.sh`.

It keeps the same SFT training arguments as `train.sh`, but submits the run to a Ray Jobs server so training can execute remotely on your Ray cluster.

## What Was Added

- `train_ray.sh`: Ray Job submission wrapper for `src/sft_threadweaver.py`
- Existing training code is unchanged.

## Prerequisites

1. A reachable Ray Jobs server (`ray dashboard` / Jobs API), for example:
```bash
ray start --head --dashboard-host 0.0.0.0 --dashboard-port 8265
```
2. Ray CLI installed on the machine that submits jobs:
```bash
pip install "ray[default]"
```
3. The model path and dataset path used in the command must be readable by the Ray worker node that runs the job.

## Basic Usage

From `threadweaver/threadweaver_sft`:

```bash
bash train_ray.sh \
  --ray_address http://<ray-head-ip>:8265 \
  --ray_num_gpus 8 \
  --dataset_dir /shared/data/train.parquet \
  --output_dir /shared/ckpts/Q3-8B-131072-SFT-ray
```

This submits a Ray job that runs:

- `torchrun --nproc-per-node gpu src/sft_threadweaver.py ...`
- with the same core hyperparameters used by `train.sh`

## Common Options

Same as `train.sh`:
- `--original_model_path` / `--base_model`
- `--dataset_dir` / `--dataset_path` / `--train_data`
- `--output_dir`
- extra args are forwarded to `src/sft_threadweaver.py`

Ray-specific:
- `--ray_address`: Ray Jobs API URL (default: `http://127.0.0.1:8265`)
- `--ray_working_dir`: directory uploaded to Ray (default: this script directory)
- `--ray_num_gpus`: GPUs requested by the Ray entrypoint
- `--ray_num_cpus`: CPUs requested by the Ray entrypoint
- `--ray_runtime_env_json`: runtime env JSON for `ray job submit`
- `--ray_submission_id`: custom submission id
- `--ray_no_wait`: return immediately after submission
- `--dry_run`: print the resolved `ray job submit` command without running it

## Example: Pass Extra Trainer Args

```bash
bash train_ray.sh \
  --ray_address http://<ray-head-ip>:8265 \
  --ray_num_gpus 4 \
  --dataset_dir /shared/data/train.parquet \
  --output_dir /shared/ckpts/run-ray \
  --logging_steps 5 \
  --save_strategy steps \
  --save_steps 100
```

## Notes

- Prefer shared storage paths (NFS/S3-mounted paths) for `--dataset_dir`, `--base_model`, and `--output_dir`.
- If you only want to validate command composition, run with `--dry_run`.
