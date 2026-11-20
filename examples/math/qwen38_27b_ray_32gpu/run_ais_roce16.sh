#!/usr/bin/env bash
# 2 × 8 B300 GSM8K recipe. The AIS submitter has already started Ray.
set -euo pipefail

export WANDB_BASE_URL="${WANDB_BASE_URL:-https://slurm.alipay.com/}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"  # Fill in your W&B token here.

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QWEN_REPO="$(cd -- "$script_dir/../../.." && pwd)"
: "${QWEN_MODEL:?Set QWEN_MODEL}" "${QWEN_DATASET:?Set QWEN_DATASET}"
: "${QWEN_ARENA_OUTPUT:?Set QWEN_ARENA_OUTPUT}"
export QWEN_ARENA_EXPERIMENT="${QWEN_ARENA_EXPERIMENT:-qwen38-27b-gsm8k-ais-oss}"
export QWEN_ARENA_TRIAL="${QWEN_ARENA_TRIAL:-${AREAL_AIS_JOB_NAME:-roce16-$(date -u +%Y%m%d-%H%M%S)-$}}"
dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi
overrides=(
    cluster.n_nodes=2
    actor.backend=megatron:d2p2t4c1e1
    rollout.backend=sglang:d16t1p1
    "experiment_name=$QWEN_ARENA_EXPERIMENT"
    stats_logger.wandb.mode=online
    total_train_steps=10
    saver.freq_steps=2
    "$@"
)
if (( ! dry_run )); then
    : "${RAY_ADDRESS:?Run this command after AIS Ray bootstrap}"
    : "${WANDB_API_KEY:?Fill in WANDB_API_KEY at the top of this script}"
    export WANDB_MODE=online

    ca_bundle=$(mktemp "${TMPDIR:-/tmp}/areal-wandb-ca.XXXXXX")
    cat /etc/ssl/certs/ca-certificates.crt "$script_dir/nautilus-swg-ca.crt" > "$ca_bundle"
    export SSL_CERT_FILE="$ca_bundle"
    export REQUESTS_CA_BUNDLE="$ca_bundle"
    export CURL_CA_BUNDLE="$ca_bundle"
fi

export NVTE_FLASH_ATTN_V4=0
export PYTHONPATH="$QWEN_REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false TMS_INIT_ENABLE=0 TMS_INIT_ENABLE_CPU_BACKUP=0
export QWEN_PLE_CHUNK_TOKENS=0 QWEN_GDN_CP_COMPAT=0
# Workers communicate with this host directly, independently of HTTP proxy settings.
export NO_PROXY='*' no_proxy='*'
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1
unset AREAL_SPMD_MODE RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

# Reuse the JIT cache across runs; cold FlashInfer compilation can take minutes.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"

cd -- "$QWEN_REPO"
# Validate without load_expr_config(), which would write a run directory.
python3 - "$script_dir/train.yaml" "$dry_run" ${overrides[@]+"${overrides[@]}"} <<'PY'
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from areal.api.cli_args import GRPOConfig, parse_cli_args, to_structured_cfg

raw, _ = parse_cli_args(['--config', sys.argv[1], *sys.argv[3:]])
config = OmegaConf.to_object(to_structured_cfg(raw, GRPOConfig))
if config.scheduler.type != 'ray' or (config.cluster.n_nodes, config.cluster.n_gpus_per_node) != (2, 8):
    raise SystemExit('This recipe requires RayScheduler with two 8-GPU nodes.')
if config.actor._version != 'v1' or config.rollout._version != 'v1' or config.actor.weight_update_mode != 'awex':
    raise SystemExit('This shared-GPU recipe requires both v1 controllers and AWEX.')
print(f'Config OK: Ray 2 × 8 GPUs; actor={config.actor.backend}; rollout={config.rollout.backend}')
print(f'Model: {config.actor.path}\nDataset: {config.train_dataset.path}')
print(f'Steps: {config.total_train_steps}; save every {config.saver.freq_steps}; WandB: {config.stats_logger.wandb.mode}')
if sys.argv[2] == '1':
    raise SystemExit(0)
model = Path(config.actor.path)
if not (model / 'config.json').is_file():
    raise SystemExit(f'QWEN_MODEL must point to a local model directory containing config.json: {model}')
dataset = config.train_dataset.path
if dataset.startswith(('/', '.')) and not Path(dataset).is_dir():
    raise SystemExit(f'Dataset directory does not exist: {dataset}')
import ray

ray.init(address=os.environ.get('RAY_ADDRESS', 'auto'), logging_level='ERROR')
try:
    nodes = [
        node for node in ray.nodes()
        if node['Alive'] and node['Resources'].get('GPU', 0) > 0
    ]
    if len(nodes) != 2 or any(node['Resources']['GPU'] != 8 for node in nodes):
        raise SystemExit('The Ray cluster must have two live nodes with 8 GPUs each.')
    if ray.available_resources().get('GPU', 0) < 16:
        raise SystemExit('At least 16 GPUs must be free in Ray.')
finally:
    ray.shutdown()
PY

# Ray workers inherit the cluster image's working directory. Keep that
# directory out of Python's import path so every worker uses this checkout.
ray_driver="$(cat <<'PYTHON'
import os
import runpy
import sys

import ray

ray.init(
    address=os.environ.get("RAY_ADDRESS", "auto"),
    runtime_env={
        "env_vars": {
            "PYTHONSAFEPATH": "1",
            "NCCL_SOCKET_IFNAME": "=eth0",
            "NCCL_SOCKET_FAMILY": "AF_INET",
            "NCCL_NET": "IB",
            "NCCL_IB_DISABLE": "0",
            "NCCL_IB_HCA": "mlx5_bond_",
            "NCCL_IB_GID_INDEX": "-1",
            "NCCL_IB_ADDR_FAMILY": "AF_INET",
            "NCCL_IB_ROCE_VERSION_NUM": "2",
            "NCCL_DEBUG_SUBSYS": "INIT,NET",
            "GLOO_SOCKET_IFNAME": "eth0",
        }
    },
)
try:
    sys.argv.pop(0)
    runpy.run_module("examples.math.gsm8k_rl", run_name="__main__", alter_sys=True)
finally:
    ray.shutdown()
PYTHON
)"
command=(python3 -c "$ray_driver" examples.math.gsm8k_rl --config "$script_dir/train.yaml")
command+=(${overrides[@]+"${overrides[@]}"})
printf 'Launch:'
printf ' %q' "${command[@]}"
printf '\n'
if (( dry_run )); then exit 0; fi

run_dir="$QWEN_ARENA_OUTPUT/submissions/$QWEN_ARENA_EXPERIMENT/$QWEN_ARENA_TRIAL"
mkdir -p -- "$run_dir"
cp -- "$script_dir/train.yaml" "$run_dir/"
printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
exec "${command[@]}"
