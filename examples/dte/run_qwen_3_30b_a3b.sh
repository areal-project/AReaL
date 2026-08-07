#!/usr/bin/env bash
set -euo pipefail

# Qwen3-30B-A3B AdamW DTE separation validation launcher.
# This script is intentionally scoped to the AWEX v0.8.0-compatible
# Qwen3-MoE path and requires site-specific paths to be provided via env vars.

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

usage() {
  cat <<'USAGE'
Usage:
  IMAGE=/path/to/areal.sif \
  MODEL_PATH=/path/to/Qwen3-30B-A3B \
  DATASET_PATH=/path/to/gsm8k \
  examples/dte/run_qwen_3_30b_a3b.sh

Required environment:
  IMAGE                         Singularity image used for the controller job.
  MODEL_PATH                    Local/shared path to Qwen3-30B-A3B.
  DATASET_PATH                  Local/shared path to the GSM8K dataset.

Common optional environment:
  DTE_SRC=/path/to/dte/src       Delta Transfer Engine source path. If unset,
                                the script expects dte to be installed in IMAGE.
  DTE_AWEX_SRC=/path/to/awex     AWEX v0.8.0 source checkout. If unset, the
                                script expects awex to be installed in IMAGE.
  FILEROOT=/shared/output/root   Shared output root for logs, name resolution,
                                and launch artifacts.
  SUBMIT=0                       Generate sbatch without submitting.
  RESERVATION=<reservation>      Optional Slurm reservation for controller and
                                child worker sbatch jobs.
  CONTROLLER_NODELIST=<nodes>    Optional controller sbatch nodelist.
  DTE_NODELIST=<nodes>           Optional worker nodelist for separation jobs.
  TRIAL_NAME=<name>              Set the trial name.

Fixed defaults:
  topology: separation
  actor.backend: megatron:(attn:d2p1t4|ffn:d1p1e8)
  rollout.backend: sglang:d2t4p1
  delta method: adamw
  train steps: 20
  batch size: 4
  n samples: 4
USAGE
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
fi

require_env() {
  local name=$1
  if [[ -z "${!name:-}" ]]; then
    echo "${name} is required. See --help for an example." >&2
    exit 2
  fi
}

AREAL_SRC=${AREAL_SRC:-${REPO_ROOT}}
IMAGE=${IMAGE:-}
DTE_SRC=${DTE_SRC:-}
DTE_AWEX_SRC=${DTE_AWEX_SRC:-}
DTE_AWEX_PYTHONPATH=${DTE_AWEX_PYTHONPATH:-}
if [[ -z "${DTE_AWEX_PYTHONPATH}" && -n "${DTE_AWEX_SRC}" ]]; then
  DTE_AWEX_PYTHONPATH="${DTE_AWEX_SRC}:"
fi
BASE_PYTHONPATH="${DTE_SRC:+${DTE_SRC}:}${DTE_AWEX_PYTHONPATH}${AREAL_SRC}"

CONFIG_PATH=${CONFIG_PATH:-examples/dte/qwen3_30b_dte_adamw.yaml}
DATASET_PATH=${DATASET_PATH:-}
MODEL_PATH=${MODEL_PATH:-}

require_env IMAGE
require_env MODEL_PATH
require_env DATASET_PATH

TOPOLOGY=${TOPOLOGY:-separation}
if [[ "${TOPOLOGY}" != "separation" ]]; then
  echo "This launcher only supports TOPOLOGY=separation, got ${TOPOLOGY}" >&2
  exit 2
fi

ACTOR_BACKEND=${ACTOR_BACKEND:-"megatron:(attn:d2p1t4|ffn:d1p1e8)"}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-"sglang:d2t4p1"}

EXP_NAME=${EXP_NAME:-areal-dte-qwen3-30b-a3b}
TRIAL_SUFFIX=${TRIAL_SUFFIX:-$(date +%m%d_%H%M%S)}
TRIAL_NAME=${TRIAL_NAME:-qwen3_30b_a3b_separation_adamw_20step_bs4_ns4_${TRIAL_SUFFIX}}
JOB_NAME=${JOB_NAME:-dte-qwen3-30b-a3b-separation}
SUBMIT=${SUBMIT:-1}
SBATCH_EXCLUSIVE=${SBATCH_EXCLUSIVE:-0}
RESERVATION=${RESERVATION:-}
CONTROLLER_NODELIST=${CONTROLLER_NODELIST:-}
DTE_NODELIST=${DTE_NODELIST:-${NODELIST:-}}

FILEROOT=${FILEROOT:-${REPO_ROOT}/outputs/dte/qwen3-30b-a3b}
NFS_ROOT=${NFS_ROOT:-${FILEROOT}/name_resolve/${EXP_NAME}/${TRIAL_NAME}}
RUN_ROOT=${RUN_ROOT:-${FILEROOT}/runs/${TRIAL_NAME}}
LAUNCH_DIR=${LAUNCH_DIR:-${FILEROOT}/launch/${TRIAL_NAME}}
CACHE_ROOT=${CACHE_ROOT:-${FILEROOT}/cache/${TRIAL_NAME}}
DTE_WORKER_TMPDIR=${DTE_WORKER_TMPDIR:-}
DTE_WORKER_CACHE_ROOT=${DTE_WORKER_CACHE_ROOT:-}
DTE_IMAGE_CACHE_TAG=${DTE_IMAGE_CACHE_TAG:-$(basename "${IMAGE}" .sif)}
DTE_WORKER_CACHE_SUFFIX=${DTE_WORKER_CACHE_SUFFIX:-}
LOCAL_TMP_ROOT=${LOCAL_TMP_ROOT:-}
SBATCH_PATH=${SBATCH_PATH:-${LAUNCH_DIR}/job.sbatch}
JOB_LOG=${JOB_LOG:-${LAUNCH_DIR}/job.log}

TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-20}
SAVE_FREQ_STEPS=${SAVE_FREQ_STEPS:-null}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
N_SAMPLES=${N_SAMPLES:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
MAX_TOKENS=${MAX_TOKENS:-3072}
MAX_CONCURRENT_ROLLOUTS=${MAX_CONCURRENT_ROLLOUTS:-64}
MAX_HEAD_OFFPOLICYNESS=${MAX_HEAD_OFFPOLICYNESS:-8}
ACTOR_MAX_TOKENS_PER_MB=${ACTOR_MAX_TOKENS_PER_MB:-6144}
WORKER_SETUP_TIMEOUT=${WORKER_SETUP_TIMEOUT:-43200}
WORKERS_READY_TIMEOUT=${WORKERS_READY_TIMEOUT:-43200}
SGLANG_WAIT_WEIGHTS_READY_TIMEOUT=${SGLANG_WAIT_WEIGHTS_READY_TIMEOUT:-1800}
NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
AWEX_COLOCATE_TIMEOUT_S=${AWEX_COLOCATE_TIMEOUT_S:-3600}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-32768}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.65}
DTE_DELTA_METHOD=${DTE_DELTA_METHOD:-adamw}
DTE_VERIFY_SNAPSHOT=${DTE_VERIFY_SNAPSHOT:-false}
DTE_DELTA_P2P_COALESCE=${DTE_DELTA_P2P_COALESCE:-1}
AWEX_WU_USE_GROUP=${AWEX_WU_USE_GROUP:-0}
AWEX_EXPECTED_MODEL_ARCH=${AWEX_EXPECTED_MODEL_ARCH:-Qwen3MoeForCausalLM}
AWEX_MIN_VERSION=${AWEX_MIN_VERSION:-0.8.0}
CHECK_MODEL_ARCH=${CHECK_MODEL_ARCH:-1}

SCHEDULER_TYPE=slurm
CLUSTER_NODES=${CLUSTER_NODES:-2}
CLUSTER_GPUS_PER_NODE=${CLUSTER_GPUS_PER_NODE:-8}
SBATCH_NODES=${SBATCH_NODES:-1}
SBATCH_GPUS=${SBATCH_GPUS:-0}

check_smoke_logs() {
  local run_log_dir=$1
  local job_log=$2
  python3 - "${run_log_dir}" "${job_log}" "${TOTAL_TRAIN_STEPS}" <<'PY'
import os
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
job_log = Path(sys.argv[2])
required_steps = int(sys.argv[3])
paths = []
if job_log.exists():
    paths.append(job_log)
if run_dir.exists():
    for root, _, files in os.walk(run_dir):
        for name in files:
            if name.endswith((".log", ".out", ".err")):
                paths.append(Path(root) / name)
text = []
for path in paths:
    try:
        text.append(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        pass
clean = re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(text))
train_steps = [int(x) for x in re.findall(r"Train step\s+(\d+)/", clean)]
updates = sorted({int(x) for x in re.findall(r"Weight update v(\d+) completed", clean)})
failures = []
if not train_steps:
    failures.append("train step log missing")
elif max(train_steps) < required_steps:
    failures.append(f"train_step {max(train_steps)} < {required_steps}")
if len(updates) < required_steps:
    failures.append(f"weight updates {updates} < {required_steps} versions")
required = [
    r"WeightUpdateController connected .*colocate=False",
    r"separation delta v1: FULL sync",
    r"separation delta v\d+: sparse path",
    r"separation delta v\d+ sent \d+ payload ops",
    r"separation delta v\d+ received \d+ ops",
]
for pat in required:
    if re.search(pat, clean) is None:
        failures.append(f"missing log pattern: {pat}")
for pat in [
    "Model Qwen3ForCausalLM not found",
    "Qwen3MoeForCausalLM not found",
    "Unsupported layer norm parameter name",
    "Train/infer key mismatch",
]:
    if pat in clean:
        failures.append(f"unexpected log pattern: {pat}")
print("DTE_SMOKE_TOPOLOGY=separation")
print(f"DTE_SMOKE_TRAIN_STEP_MAX={max(train_steps) if train_steps else 0}")
print(f"DTE_SMOKE_WEIGHT_UPDATE_VERSIONS={updates}")
if failures:
    print("DTE_SMOKE_CHECK_FAILED=" + "; ".join(failures))
    raise SystemExit(1)
print("DTE_SMOKE_CHECK_OK=1")
PY
}

hydra_quote() {
  local value=$1
  value=${value//\\/\\\\}
  value=${value//\"/\\\"}
  printf '"%s"' "${value}"
}

check_qwen3_a3b_awex_stack() {
  python3 - <<'PY'
import importlib.metadata as importlib_metadata
import os
import re

import areal
import awex
import dte
import sglang
import torch

expected_arch = os.environ["AWEX_EXPECTED_MODEL_ARCH"]
required_version = os.environ.get("AWEX_MIN_VERSION", "0.8.0")
model_path = os.environ["MODEL_PATH"]

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("areal", areal.__file__)
print("dte", getattr(dte, "__file__", ""))
print("awex", getattr(awex, "__file__", ""))
print("sglang", getattr(sglang, "__version__", "unknown"), getattr(sglang, "__file__", ""))

version = getattr(awex, "__version__", "")
if not version:
    try:
        version = importlib_metadata.version("awex")
    except importlib_metadata.PackageNotFoundError:
        pass

def version_tuple(value):
    nums = [int(x) for x in re.findall(r"\d+", value)[:3]]
    return tuple(nums + [0] * (3 - len(nums)))

if version:
    print("awex_version", version)
    if version_tuple(version) < version_tuple(required_version):
        raise RuntimeError(f"AWEX version {version} is older than required {required_version}")
else:
    print("awex_version unknown; continuing because source checkouts may not expose package metadata")

from awex.models import registry

models = getattr(registry.ModelRegistry, "models", {}) or {}
if expected_arch not in models:
    raise RuntimeError(
        f"AWEX registry does not contain {expected_arch}; use AWEX v0.8.0 or a compatible checkout"
    )

if os.environ.get("CHECK_MODEL_ARCH", "1") == "1":
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    archs = list(getattr(cfg, "architectures", None) or [])
    print("model_architectures", archs)
    if expected_arch not in archs:
        raise RuntimeError(
            f"MODEL_PATH architectures {archs} do not include expected {expected_arch}"
        )

print("AWEX_QWEN3_A3B_ADAPTATION_OK=1")
PY
}

run_inside_container() {
  cd "${AREAL_SRC}"

  DTE_WORKER_TMPDIR=${DTE_WORKER_TMPDIR:-/tmp/areal-dte-${SLURM_JOB_ID:-local}}
  DTE_WORKER_CACHE_ROOT=${DTE_WORKER_CACHE_ROOT:-${DTE_WORKER_TMPDIR}/worker_cache/${TRIAL_NAME}}
  DTE_WORKER_CACHE_SUFFIX=${DTE_WORKER_CACHE_SUFFIX:-$(hostname 2>/dev/null || echo node)}
  DTE_WORKER_CACHE_SUFFIX=${DTE_WORKER_CACHE_SUFFIX//[^A-Za-z0-9_.-]/_}
  # SGLang/AWEX result channels use ZMQ ipc:// sockets under TMPDIR.
  # Keep this path short enough for Unix-domain socket limits.
  LOCAL_TMP_ROOT=${LOCAL_TMP_ROOT:-${DTE_WORKER_TMPDIR}/rt}

  export AREAL_SRC IMAGE DTE_SRC DTE_AWEX_SRC DTE_AWEX_PYTHONPATH BASE_PYTHONPATH
  export DTE_WORKER_TMPDIR DTE_WORKER_CACHE_ROOT
  export DTE_IMAGE_CACHE_TAG DTE_WORKER_CACHE_SUFFIX DTE_DELTA_P2P_COALESCE
  export AWEX_WU_USE_GROUP AWEX_EXPECTED_MODEL_ARCH AWEX_MIN_VERSION CHECK_MODEL_ARCH
  export MODEL_PATH DATASET_PATH
  export PYTHONUNBUFFERED=1
  export PYTHONNOUSERSITE=1
  export PYTHONFAULTHANDLER=1
  export HYDRA_FULL_ERROR=1
  export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1
  export PYTHONPATH="${BASE_PYTHONPATH}:${PYTHONPATH:-}"
  export AREAL_CACHE_DIR="${CACHE_ROOT}"
  export HOME="${CACHE_ROOT}/home"
  export HF_HOME="${CACHE_ROOT}/hf"
  export HF_DATASETS_CACHE="${CACHE_ROOT}/hf_datasets"
  export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
  export UV_CACHE_DIR="${CACHE_ROOT}/uv_cache"
  export PIP_CACHE_DIR="${CACHE_ROOT}/pip_cache"
  export TMPDIR="${LOCAL_TMP_ROOT}"
  export TRITON_CACHE_DIR="${DTE_WORKER_CACHE_ROOT}/jit/${DTE_IMAGE_CACHE_TAG}/triton/controller"
  export TORCHINDUCTOR_CACHE_DIR="${DTE_WORKER_CACHE_ROOT}/jit/${DTE_IMAGE_CACHE_TAG}/torchinductor/controller"
  export VLLM_CACHE_ROOT="${DTE_WORKER_CACHE_ROOT}/jit/${DTE_IMAGE_CACHE_TAG}/vllm/controller"
  export FLASHINFER_WORKSPACE_DIR="${DTE_WORKER_CACHE_ROOT}/jit/${DTE_IMAGE_CACHE_TAG}/flashinfer/controller"
  export PYTORCH_KERNEL_CACHE_PATH="${DTE_WORKER_CACHE_ROOT}/jit/${DTE_IMAGE_CACHE_TAG}/torch_kernel/controller"
  export TORCH_EXTENSIONS_DIR="${DTE_WORKER_CACHE_ROOT}/jit/${DTE_IMAGE_CACHE_TAG}/torch_extensions/controller"
  export SGLANG_DG_CACHE_DIR="${DTE_WORKER_CACHE_ROOT}/jit/${DTE_IMAGE_CACHE_TAG}/deep_gemm/controller"
  export UV_LINK_MODE=copy
  export NCCL_CUMEM_ENABLE=0
  export NCCL_NVLS_ENABLE=0
  export NCCL_TIMEOUT
  export NCCL_DEBUG=WARN
  export TOKENIZERS_PARALLELISM=false
  export OMP_NUM_THREADS=4
  export MKL_NUM_THREADS=4
  export OPENBLAS_NUM_THREADS=4
  export NUMEXPR_NUM_THREADS=4
  export SGLANG_WAIT_WEIGHTS_READY_TIMEOUT
  export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0
  export AWEX_COLOCATE_TIMEOUT_S
  if [[ -n "${UV_INDEX_URL:-}" ]]; then
    export UV_INDEX_URL
  fi

  mkdir -p "${HOME}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" \
    "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${TMPDIR}" \
    "${DTE_WORKER_TMPDIR}/guard_ports" "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
    "${FLASHINFER_WORKSPACE_DIR}" "${PYTORCH_KERNEL_CACHE_PATH}" \
    "${TORCH_EXTENSIONS_DIR}" "${SGLANG_DG_CACHE_DIR}"
  trap 'rm -rf "${TMPDIR}" 2>/dev/null || true' EXIT

  echo "==> host: $(hostname)"
  echo "==> date: $(date -Is)"
  echo "==> repo: ${AREAL_SRC}"
  git config --global --add safe.directory "${AREAL_SRC}" || true
  git -C "${AREAL_SRC}" rev-parse --short HEAD || true
  python3 --version
  check_qwen3_a3b_awex_stack

  RUN_LOG_DIR="${FILEROOT}/logs/$(whoami)/${EXP_NAME}/${TRIAL_NAME}"
  rm -rf "${NFS_ROOT}" 2>/dev/null || true
  mkdir -p "${RUN_LOG_DIR}" "${NFS_ROOT}"

  python3 -u examples/math/gsm8k_rl.py \
    --config "${CONFIG_PATH}" \
    "++scheduler.type=$(hydra_quote "${SCHEDULER_TYPE}")" \
    "++experiment_name=$(hydra_quote "${EXP_NAME}")" \
    "++trial_name=$(hydra_quote "${TRIAL_NAME}")" \
    ++cluster.n_nodes="${CLUSTER_NODES}" \
    ++cluster.n_gpus_per_node="${CLUSTER_GPUS_PER_NODE}" \
    "++cluster.fileroot=$(hydra_quote "${FILEROOT}")" \
    "++cluster.name_resolve.nfs_record_root=$(hydra_quote "${NFS_ROOT}")" \
    "++actor.backend=$(hydra_quote "${ACTOR_BACKEND}")" \
    "++rollout.backend=$(hydra_quote "${ROLLOUT_BACKEND}")" \
    "++rollout.scheduling_strategy.type=$(hydra_quote "${TOPOLOGY}")" \
    "++actor.path=$(hydra_quote "${MODEL_PATH}")" \
    "++ref.path=$(hydra_quote "${MODEL_PATH}")" \
    "++sglang.model_path=$(hydra_quote "${MODEL_PATH}")" \
    "++tokenizer_path=$(hydra_quote "${MODEL_PATH}")" \
    ++actor.mb_spec.max_tokens_per_mb="${ACTOR_MAX_TOKENS_PER_MB}" \
    ++actor.setup_timeout="${WORKER_SETUP_TIMEOUT}" \
    ++actor.workers_ready_timeout="${WORKERS_READY_TIMEOUT}" \
    ++rollout.setup_timeout="${WORKER_SETUP_TIMEOUT}" \
    ++rollout.workers_ready_timeout="${WORKERS_READY_TIMEOUT}" \
    "++actor.dte.delta_method=$(hydra_quote "${DTE_DELTA_METHOD}")" \
    ++actor.dte.verify_snapshot="${DTE_VERIFY_SNAPSHOT}" \
    ++total_train_steps="${TOTAL_TRAIN_STEPS}" \
    ++saver.freq_steps="${SAVE_FREQ_STEPS}" \
    "++train_dataset.path=$(hydra_quote "${DATASET_PATH}")" \
    "++valid_dataset.path=$(hydra_quote "${DATASET_PATH}")" \
    "++train_dataset.scheduling_spec.image=$(hydra_quote "${IMAGE}")" \
    "++valid_dataset.scheduling_spec.image=$(hydra_quote "${IMAGE}")" \
    ++train_dataset.batch_size="${TRAIN_BATCH_SIZE}" \
    ++valid_dataset.batch_size="${TRAIN_BATCH_SIZE}" \
    ++rollout.consumer_batch_size="${TRAIN_BATCH_SIZE}" \
    ++rollout.max_concurrent_rollouts="${MAX_CONCURRENT_ROLLOUTS}" \
    ++rollout.max_head_offpolicyness="${MAX_HEAD_OFFPOLICYNESS}" \
    ++gconfig.n_samples="${N_SAMPLES}" \
    ++gconfig.max_new_tokens="${MAX_NEW_TOKENS}" \
    ++gconfig.max_tokens="${MAX_TOKENS}" \
    ++sglang.context_length="${SGLANG_CONTEXT_LENGTH}" \
    ++sglang.mem_fraction_static="${SGLANG_MEM_FRACTION}" \
    "++actor.scheduling_spec.0.image=$(hydra_quote "${IMAGE}")" \
    "++actor.scheduling_spec.0.nodelist=$(hydra_quote "${DTE_NODELIST}")"

  echo "==> training finished: $(date -Is)"
  echo "==> rendered config: ${RUN_LOG_DIR}/config.yaml"
  check_smoke_logs "${RUN_LOG_DIR}" "${JOB_LOG}"
}

if [[ "${INSIDE_QWEN3_30B_A3B_DTE_CONTAINER:-0}" == "1" ]]; then
  run_inside_container
  exit 0
fi

mkdir -p "${LAUNCH_DIR}" "${RUN_ROOT}" "${CACHE_ROOT}" "${NFS_ROOT}"

SBATCH_NODELIST_LINE=
if [[ -n "${CONTROLLER_NODELIST}" ]]; then
  SBATCH_NODELIST_LINE="#SBATCH --nodelist=${CONTROLLER_NODELIST}"
fi
SBATCH_RESERVATION_LINE=
if [[ -n "${RESERVATION}" ]]; then
  SBATCH_RESERVATION_LINE="#SBATCH --reservation=${RESERVATION}"
fi
if [[ "${SBATCH_EXCLUSIVE}" == "1" ]]; then
  SBATCH_SHARE_LINE="#SBATCH --exclusive"
else
  SBATCH_SHARE_LINE="#SBATCH --oversubscribe"
fi

cat > "${SBATCH_PATH}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=${SBATCH_NODES}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:${SBATCH_GPUS}
#SBATCH --mem=0
#SBATCH --time=06:00:00
${SBATCH_NODELIST_LINE}
${SBATCH_RESERVATION_LINE}
${SBATCH_SHARE_LINE}
#SBATCH --no-requeue
#SBATCH --chdir=${AREAL_SRC}
#SBATCH --output=${JOB_LOG}
#SBATCH --open-mode=append

set -euo pipefail
if [[ -n "${RESERVATION}" ]]; then
  # Slurm consumes SBATCH_* environment variables for nested sbatch calls. The
  # controller's #SBATCH line only applies to the controller job, so keep the
  # reservation exported for SlurmScheduler-created data/actor/rollout jobs.
  export SBATCH_RESERVATION="${RESERVATION}"
fi
SINGULARITY_RESERVATION_ENV=()
if [[ -n "\${SBATCH_RESERVATION:-}" ]]; then
  SINGULARITY_RESERVATION_ENV=(--env "SBATCH_RESERVATION=\${SBATCH_RESERVATION}")
fi
echo "==> sbatch job: \${SLURM_JOB_ID:-unknown}"
echo "==> nodelist: \${SLURM_JOB_NODELIST:-unknown}"
echo "==> topology: ${TOPOLOGY}"
echo "==> child sbatch reservation: \${SBATCH_RESERVATION:-unset}"
echo "==> batch CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
echo "==> batch SLURM_JOB_GPUS=\${SLURM_JOB_GPUS:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv || true
df -h /dev/shm || true

exec srun --mpi=pmi2 --ntasks=1 --cpus-per-task="\${SLURM_CPUS_PER_TASK:-64}" \\
  singularity exec --no-home --writable-tmpfs --nv \\
  --bind /storage:/storage \\
  --bind /home:/home \\
  --bind /etc/slurm/:/etc/slurm/ \\
  --bind /etc/passwd:/etc/passwd:ro \\
  --bind /etc/group:/etc/group:ro \\
  --bind /etc/munge:/etc/munge:ro \\
  --bind /var/run/munge:/var/run/munge \\
  --bind /usr/bin/sbatch:/usr/bin/sbatch \\
  --bind /usr/bin/srun:/usr/bin/srun \\
  --bind /usr/bin/squeue:/usr/bin/squeue \\
  --bind /usr/bin/scancel:/usr/bin/scancel \\
  --bind /usr/bin/scontrol:/usr/bin/scontrol \\
  --bind /usr/lib64/slurm:/usr/lib64/slurm \\
  \${SINGULARITY_RESERVATION_ENV[@]+"\${SINGULARITY_RESERVATION_ENV[@]}"} \\
  --env "INSIDE_QWEN3_30B_A3B_DTE_CONTAINER=1" \\
  --env "AREAL_SRC=${AREAL_SRC}" \\
  --env "IMAGE=${IMAGE}" \\
  --env "DTE_SRC=${DTE_SRC}" \\
  --env "DTE_AWEX_SRC=${DTE_AWEX_SRC}" \\
  --env "DTE_AWEX_PYTHONPATH=${DTE_AWEX_PYTHONPATH}" \\
  --env "BASE_PYTHONPATH=${BASE_PYTHONPATH}" \\
  --env "CONFIG_PATH=${CONFIG_PATH}" \\
  --env "DATASET_PATH=${DATASET_PATH}" \\
  --env "TOPOLOGY=${TOPOLOGY}" \\
  --env "SCHEDULER_TYPE=${SCHEDULER_TYPE}" \\
  --env "EXP_NAME=${EXP_NAME}" \\
  --env "TRIAL_NAME=${TRIAL_NAME}" \\
  --env "FILEROOT=${FILEROOT}" \\
  --env "NFS_ROOT=${NFS_ROOT}" \\
  --env "CACHE_ROOT=${CACHE_ROOT}" \\
  --env "DTE_WORKER_TMPDIR=${DTE_WORKER_TMPDIR}" \\
  --env "DTE_WORKER_CACHE_ROOT=${DTE_WORKER_CACHE_ROOT}" \\
  --env "DTE_IMAGE_CACHE_TAG=${DTE_IMAGE_CACHE_TAG}" \\
  --env "DTE_WORKER_CACHE_SUFFIX=${DTE_WORKER_CACHE_SUFFIX}" \\
  --env "LOCAL_TMP_ROOT=${LOCAL_TMP_ROOT}" \\
  --env "MODEL_PATH=${MODEL_PATH}" \\
  --env "ACTOR_BACKEND=${ACTOR_BACKEND}" \\
  --env "ROLLOUT_BACKEND=${ROLLOUT_BACKEND}" \\
  --env "CLUSTER_NODES=${CLUSTER_NODES}" \\
  --env "CLUSTER_GPUS_PER_NODE=${CLUSTER_GPUS_PER_NODE}" \\
  --env "DTE_NODELIST=${DTE_NODELIST}" \\
  --env "TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS}" \\
  --env "SAVE_FREQ_STEPS=${SAVE_FREQ_STEPS}" \\
  --env "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}" \\
  --env "N_SAMPLES=${N_SAMPLES}" \\
  --env "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}" \\
  --env "MAX_TOKENS=${MAX_TOKENS}" \\
  --env "MAX_CONCURRENT_ROLLOUTS=${MAX_CONCURRENT_ROLLOUTS}" \\
  --env "MAX_HEAD_OFFPOLICYNESS=${MAX_HEAD_OFFPOLICYNESS}" \\
  --env "ACTOR_MAX_TOKENS_PER_MB=${ACTOR_MAX_TOKENS_PER_MB}" \\
  --env "WORKER_SETUP_TIMEOUT=${WORKER_SETUP_TIMEOUT}" \\
  --env "WORKERS_READY_TIMEOUT=${WORKERS_READY_TIMEOUT}" \\
  --env "SGLANG_WAIT_WEIGHTS_READY_TIMEOUT=${SGLANG_WAIT_WEIGHTS_READY_TIMEOUT}" \\
  --env "NCCL_TIMEOUT=${NCCL_TIMEOUT}" \\
  --env "AWEX_COLOCATE_TIMEOUT_S=${AWEX_COLOCATE_TIMEOUT_S}" \\
  --env "SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION}" \\
  --env "SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH}" \\
  --env "DTE_DELTA_METHOD=${DTE_DELTA_METHOD}" \\
  --env "DTE_VERIFY_SNAPSHOT=${DTE_VERIFY_SNAPSHOT}" \\
  --env "DTE_DELTA_P2P_COALESCE=${DTE_DELTA_P2P_COALESCE}" \\
  --env "AWEX_WU_USE_GROUP=${AWEX_WU_USE_GROUP}" \\
  --env "AWEX_EXPECTED_MODEL_ARCH=${AWEX_EXPECTED_MODEL_ARCH}" \\
  --env "AWEX_MIN_VERSION=${AWEX_MIN_VERSION}" \\
  --env "CHECK_MODEL_ARCH=${CHECK_MODEL_ARCH}" \\
  --env "PYTHONPATH=${BASE_PYTHONPATH}" \\
  --env "PYTHONUNBUFFERED=1" \\
  --env "PYTHONFAULTHANDLER=1" \\
  --env "HYDRA_FULL_ERROR=1" \\
  --env "PYTHONNOUSERSITE=1" \\
  --env "AREAL_ALLOW_DEFAULT_ADMIN_KEY=1" \\
  "${IMAGE}" \\
  bash "${SCRIPT_PATH}"
EOF

echo "==> topology: ${TOPOLOGY}"
echo "==> scheduler: ${SCHEDULER_TYPE}"
echo "==> model: ${MODEL_PATH}"
echo "==> actor backend: ${ACTOR_BACKEND}"
echo "==> rollout backend: ${ROLLOUT_BACKEND}"
echo "==> cluster nodes: ${CLUSTER_NODES}"
echo "==> worker nodelist: ${DTE_NODELIST:-unset}"
echo "==> controller nodelist: ${CONTROLLER_NODELIST:-unset}"
echo "==> sbatch: ${SBATCH_PATH}"
echo "==> log: ${JOB_LOG}"

if [[ "${SUBMIT}" != "1" ]]; then
  echo "==> SUBMIT=${SUBMIT}; generated files only"
  exit 0
fi

job_id=$(sbatch --parsable "${SBATCH_PATH}")
echo "==> submitted job: ${job_id}"
echo "squeue -j ${job_id}"
echo "tail -f ${JOB_LOG}"
