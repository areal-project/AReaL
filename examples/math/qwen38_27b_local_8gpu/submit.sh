#!/usr/bin/env bash
# Run inside the prepared SGLang 0.5.20 / Megatron environment, without Slurm.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: QWEN_MODEL=/models/Qwen3.8-27B QWEN_DATASET=/data/gsm8k bash submit.sh [--dry-run] [key=value ...]

Single node, 8 GPUs, LocalScheduler, BF16, v1 AWEX shared GPUs.
Activate the prepared SGLang 0.5.20 Python environment before running.
QWEN_REPO          AReaL checkout (auto-detected when this file is in the repo)
QWEN_MODEL         Local Qwen3.8-27B model directory (required to train)
QWEN_DATASET       HF dataset snapshot directory or ID (default: openai/gsm8k)
QWEN_ARENA_OUTPUT  Output root (default: <repo>/output/local-gsm8k)
QWEN_ARENA_TRIAL   Run name (default: local8gpu<timestamp>-<pid>)
CUDA_VISIBLE_DEVICES  Exactly 8 numeric GPU IDs (default: 0,1,2,3,4,5,6,7)
WANDB_MODE         online (default), offline, or disabled; credentials from env/login
WANDB_ENTITY / WANDB_PROJECT / WANDB_BASE_URL / WANDB_API_KEY are honored.
--dry-run parses the config and prints the launch command without starting workers.
Additional arguments use Hydra overrides, e.g. actor.mb_spec.max_tokens_per_mb=20480.
EOF
}

dry_run=0
overrides=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) dry_run=1 ;;
        -h|--help) usage; exit 0 ;;
        --*) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
        *) overrides+=("$arg") ;;
    esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${QWEN_REPO:-}" ]]; then
    # Also support <experiment-root>/gsm8k/local-8gpu beside <root>/AReaL.
    if [[ -f "$script_dir/../../AReaL/areal/api/cli_args.py" ]]; then
        QWEN_REPO="$script_dir/../../AReaL"
    else
        QWEN_REPO="$(cd -- "$script_dir/../../.." && pwd)"
    fi
fi
export QWEN_REPO
test -f "$QWEN_REPO/areal/api/cli_args.py" || {
    printf 'Set QWEN_REPO to the AReaL checkout.\n' >&2; exit 2;
}
export QWEN_REPO="$(cd -- "$QWEN_REPO" && pwd)"
export QWEN_MODEL="${QWEN_MODEL:-/path/to/Qwen3.8-27B}"
export QWEN_DATASET="${QWEN_DATASET:-openai/gsm8k}"
export QWEN_ARENA_OUTPUT="${QWEN_ARENA_OUTPUT:-$QWEN_REPO/output/local-gsm8k}"
export QWEN_ARENA_TRIAL="${QWEN_ARENA_TRIAL:-local8gpu$(date +%Y%m%d%H%M%S)-$$}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONPATH="$QWEN_REPO"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false TMS_INIT_ENABLE=0 TMS_INIT_ENABLE_CPU_BACKUP=0
export QWEN_PLE_CHUNK_TOKENS=0 QWEN_GDN_CP_COMPAT=0
# Workers communicate with this host directly, independently of HTTP proxy settings.
export NO_PROXY='*' no_proxy='*'
unset AREAL_SPMD_MODE RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

# TE and PyTorch must use the same cuDNN installation. Never fall back silently
# to a second system cuDNN when the runtime is missing the Python cuDNN package.
cudnn_home="$(python3 - <<'PY'
import importlib.util
try:
    spec = importlib.util.find_spec('nvidia.cudnn')
except ModuleNotFoundError:
    spec = None
if spec is not None and spec.submodule_search_locations:
    print(next(iter(spec.submodule_search_locations)))
PY
)" || {
    if (( ! dry_run )); then exit 2; fi
    cudnn_home=''
}
if [[ -n "$cudnn_home" && -d "$cudnn_home/lib" ]]; then
    export CUDNN_HOME="$cudnn_home"
    export LD_LIBRARY_PATH="$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif (( ! dry_run )); then
    printf 'No Python cuDNN package found; activate the prepared CUDA runtime.\n' >&2
    exit 2
fi
if [[ -d /usr/local/cuda/compat ]]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="/usr/local/cuda/compat${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
# Reuse the JIT cache across runs; cold FlashInfer compilation can take minutes.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"

cd -- "$QWEN_REPO"
# Validate without load_expr_config(), which would write a run directory.
python3 - "$script_dir/train.yaml" "$dry_run" ${overrides[@]+"${overrides[@]}"} <<'PY'
import os
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

from areal.api.cli_args import GRPOConfig, parse_cli_args, to_structured_cfg

raw, _ = parse_cli_args(['--config', sys.argv[1], *sys.argv[3:]])
config = OmegaConf.to_object(to_structured_cfg(raw, GRPOConfig))
if config.scheduler.type != 'local' or (config.cluster.n_nodes, config.cluster.n_gpus_per_node) != (1, 8):
    raise SystemExit('This recipe requires LocalScheduler and exactly one 8-GPU node.')
if config.actor.backend != 'megatron:d1p2t4c1e1' or config.rollout.backend != 'sglang:d2t4p1':
    raise SystemExit('This v1 AWEX recipe requires actor PP2/TP4 and rollout DP2/TP4.')
if config.actor._version != 'v1' or config.rollout._version != 'v1' or config.actor.weight_update_mode != 'awex':
    raise SystemExit('This shared-GPU recipe requires both v1 controllers and AWEX.')
gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8 or any(not re.fullmatch(r'\d+', gpu) for gpu in gpu_ids):
    raise SystemExit('CUDA_VISIBLE_DEVICES must contain 8 distinct numeric GPU IDs.')
for role in (config.actor, config.rollout):
    if role.scheduling_strategy.type != 'separation' or len(role.scheduling_spec) != 1:
        raise SystemExit('Use one worker spec and separation strategy per v1 AWEX role.')
    if 'CUDA_VISIBLE_DEVICES' in role.scheduling_spec[0].env_vars:
        raise SystemExit('Do not override worker CUDA_VISIBLE_DEVICES; LocalScheduler assigns it.')
print(f'Config OK: local 8 GPUs; actor={config.actor.backend}; rollout={config.rollout.backend}')
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
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit('Exactly 8 visible CUDA GPUs are required.')
PY

command=(python3 -m examples.math.gsm8k_rl --config "$script_dir/train.yaml")
command+=(${overrides[@]+"${overrides[@]}"})
printf 'Launch:'
printf ' %q' "${command[@]}"
printf '\n'
if (( dry_run )); then exit 0; fi

run_dir="$QWEN_ARENA_OUTPUT/submissions/$QWEN_ARENA_TRIAL"
mkdir -p -- "$run_dir"
cp -- "$script_dir/submit.sh" "$script_dir/train.yaml" "$run_dir/"
printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
exec "${command[@]}"
