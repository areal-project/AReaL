#!/usr/bin/env bash
set -euo pipefail

# Qwen3-30B-A3B AdamW DTE validation launcher.
# Static model/training settings live in qwen3_30b_dte_adamw.yaml.

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

usage() {
  cat <<'USAGE'
Usage:
  examples/dte/run_qwen3_30b.sh

Environment:
  TOPOLOGY=colocation|separation  Select colocated or separated DTE update mode.
  SUBMIT=0                         Generate sbatch without submitting.
  RESERVATION=shanghai             Optional Slurm reservation for controller sbatch.
  CONTROLLER_NODELIST=slurmd-13     Optional node for the controller sbatch.
  DTE_NODELIST=slurmd-[63-64]       Optional worker nodelist for separation child jobs.
  TRIAL_NAME=qwen3_30b_manual       Set the trial name.
  FILEROOT=/storage/.../fileroot    Set output root.

Defaults:
  model: /storage/openpsi/models/Qwen__Qwen3-30B-A3B
  actor.backend: megatron:(attn:d2p1t4|ffn:d1p1e8)
  rollout.backend: sglang:d2t4p1
  delta method: adamw
  train steps: 3
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

AREAL_SRC=${AREAL_SRC:-${REPO_ROOT}}
IMAGE=${IMAGE:-/storage/openpsi/images/areal-dev-20260508.sif}
DTE_SRC=${DTE_SRC:-/storage/openpsi/users/pengzai.pyq/delta-transfer-engine-pyq-perf-sparse-p2p/src}
DTE_AWEX_SRC=${DTE_AWEX_SRC:-/storage/openpsi/users/pengzai.pyq/asystem-awex}
CONFIG_PATH=${CONFIG_PATH:-examples/dte/qwen3_30b_dte_adamw.yaml}
DATASET_PATH=${DATASET_PATH:-/storage/openpsi/data/gsm8k}

TOPOLOGY=${TOPOLOGY:-colocation}
if [[ "${TOPOLOGY}" != "colocation" && "${TOPOLOGY}" != "separation" ]]; then
  echo "TOPOLOGY must be colocation or separation, got ${TOPOLOGY}" >&2
  exit 2
fi

MODEL_PATH=${MODEL_PATH:-/storage/openpsi/models/Qwen__Qwen3-30B-A3B}
ACTOR_BACKEND=${ACTOR_BACKEND:-"megatron:(attn:d2p1t4|ffn:d1p1e8)"}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-"sglang:d2t4p1"}

EXP_NAME=${EXP_NAME:-pyq-areal-port-dte-qwen3-30b}
TRIAL_SUFFIX=${TRIAL_SUFFIX:-$(date +%m%d_%H%M%S)}
TRIAL_NAME=${TRIAL_NAME:-qwen3_30b_${TOPOLOGY}_adamw_3step_bs8_ns2_${TRIAL_SUFFIX}}
JOB_NAME=${JOB_NAME:-pyq-dte-q3-30b-${TOPOLOGY}}
SUBMIT=${SUBMIT:-1}
SBATCH_EXCLUSIVE=${SBATCH_EXCLUSIVE:-0}
RESERVATION=${RESERVATION:-}
CONTROLLER_NODELIST=${CONTROLLER_NODELIST:-}
DTE_NODELIST=${DTE_NODELIST:-${NODELIST:-}}

FILEROOT=${FILEROOT:-/storage/openpsi/users/pengzai.pyq/areal_port_dte_qwen3_30b/fileroot}
NFS_ROOT=${NFS_ROOT:-${FILEROOT}/name_resolve/${EXP_NAME}/${TRIAL_NAME}}
RUN_ROOT=${RUN_ROOT:-${FILEROOT}/runs/${TRIAL_NAME}}
LAUNCH_DIR=${LAUNCH_DIR:-${FILEROOT}/launch/${TRIAL_NAME}}
CACHE_ROOT=${CACHE_ROOT:-${FILEROOT}/cache/${TRIAL_NAME}}
DTE_WORKER_TMPDIR=${DTE_WORKER_TMPDIR:-}
DTE_WORKER_CACHE_ROOT=${DTE_WORKER_CACHE_ROOT:-}
DTE_IMAGE_CACHE_TAG=${DTE_IMAGE_CACHE_TAG:-$(basename "${IMAGE}" .sif)}
LOCAL_TMP_ROOT=${LOCAL_TMP_ROOT:-}
SBATCH_PATH=${SBATCH_PATH:-${LAUNCH_DIR}/job.sbatch}
JOB_LOG=${JOB_LOG:-${LAUNCH_DIR}/job.log}

TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-3}
SAVE_FREQ_STEPS=${SAVE_FREQ_STEPS:-null}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
N_SAMPLES=${N_SAMPLES:-2}
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
DTE_DELTA_METHOD=${DTE_DELTA_METHOD:-adamw}
DTE_VERIFY_SNAPSHOT=${DTE_VERIFY_SNAPSHOT:-false}
DTE_DELTA_P2P_COALESCE=${DTE_DELTA_P2P_COALESCE:-1}
AWEX_WU_USE_GROUP=${AWEX_WU_USE_GROUP:-0}

if [[ "${TOPOLOGY}" == "colocation" ]]; then
  SCHEDULER_TYPE=local
  CLUSTER_NODES=${CLUSTER_NODES:-1}
  SBATCH_GPUS=${SBATCH_GPUS:-8}
  SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.35}
else
  SCHEDULER_TYPE=slurm
  CLUSTER_NODES=${CLUSTER_NODES:-2}
  SBATCH_GPUS=${SBATCH_GPUS:-0}
  SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.65}
fi
CLUSTER_GPUS_PER_NODE=${CLUSTER_GPUS_PER_NODE:-8}
SBATCH_NODES=${SBATCH_NODES:-1}

check_smoke_logs() {
  local run_log_dir=$1
  local job_log=$2
  python3 - "${run_log_dir}" "${job_log}" "${TOTAL_TRAIN_STEPS}" "${TOPOLOGY}" <<'PY'
import os
import re
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
job_log = Path(sys.argv[2])
required_steps = int(sys.argv[3])
topology = sys.argv[4]
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
if topology == "colocation":
    required = [
        r"WeightUpdateController connected .*colocate=True",
        r"colocate delta enabled \(sender\); DTE components ready",
        r"colocate delta enabled \(receiver\); DTE DeltaEngine ready",
        r"colocate delta v1: FULL sync",
        r"colocate delta v\d+ \[.*\]: changed",
        r"Colocate DTE weight update completed",
    ]
else:
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
    "Unsupported layer norm parameter name",
    "Train/infer key mismatch",
]:
    if pat in clean:
        failures.append(f"unexpected log pattern: {pat}")
print(f"DTE_SMOKE_TOPOLOGY={topology}")
print(f"DTE_SMOKE_TRAIN_STEP_MAX={max(train_steps) if train_steps else 0}")
print(f"DTE_SMOKE_WEIGHT_UPDATE_VERSIONS={updates}")
if failures:
    print("DTE_SMOKE_CHECK_FAILED=" + "; ".join(failures))
    raise SystemExit(1)
print("DTE_SMOKE_CHECK_OK=1")
PY
}

run_inside_container() {
  cd "${AREAL_SRC}"

  DTE_WORKER_TMPDIR=${DTE_WORKER_TMPDIR:-/tmp/dte-pyq-${SLURM_JOB_ID:-local}}
  DTE_WORKER_CACHE_ROOT=${DTE_WORKER_CACHE_ROOT:-${DTE_WORKER_TMPDIR}/areal_worker_cache/${TRIAL_NAME}}
  LOCAL_TMP_ROOT=${LOCAL_TMP_ROOT:-${DTE_WORKER_TMPDIR}/driver_tmp/${TRIAL_NAME}}

  export AREAL_SRC IMAGE DTE_SRC DTE_AWEX_SRC DTE_WORKER_TMPDIR DTE_WORKER_CACHE_ROOT
  export DTE_IMAGE_CACHE_TAG DTE_DELTA_P2P_COALESCE AWEX_WU_USE_GROUP
  export PYTHONUNBUFFERED=1
  export PYTHONNOUSERSITE=1
  export PYTHONFAULTHANDLER=1
  export HYDRA_FULL_ERROR=1
  export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1
  export PYTHONPATH="${DTE_SRC}:${DTE_AWEX_SRC}:${AREAL_SRC}:${PYTHONPATH:-}"
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
  export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
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
  python3 - <<'PY'
import awex
import dte
import sglang
import torch
import areal

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("areal", areal.__file__)
print("dte", getattr(dte, "__file__", ""))
print("awex", getattr(awex, "__file__", ""))
print("sglang", getattr(sglang, "__version__", "unknown"), getattr(sglang, "__file__", ""))
PY

  RUN_LOG_DIR="${FILEROOT}/logs/$(whoami)/${EXP_NAME}/${TRIAL_NAME}"
  rm -rf "${NFS_ROOT}" 2>/dev/null || true
  mkdir -p "${RUN_LOG_DIR}" "${NFS_ROOT}"

  python3 -u examples/math/gsm8k_rl.py \
    --config "${CONFIG_PATH}" \
    ++scheduler.type="${SCHEDULER_TYPE}" \
    ++experiment_name="${EXP_NAME}" \
    ++trial_name="${TRIAL_NAME}" \
    ++cluster.n_nodes="${CLUSTER_NODES}" \
    ++cluster.n_gpus_per_node="${CLUSTER_GPUS_PER_NODE}" \
    ++cluster.fileroot="${FILEROOT}" \
    ++cluster.name_resolve.nfs_record_root="${NFS_ROOT}" \
    ++actor.backend="${ACTOR_BACKEND}" \
    ++rollout.backend="${ROLLOUT_BACKEND}" \
    ++rollout.scheduling_strategy.type="${TOPOLOGY}" \
    ++actor.path="${MODEL_PATH}" \
    ++ref.path="${MODEL_PATH}" \
    ++sglang.model_path="${MODEL_PATH}" \
    ++tokenizer_path="${MODEL_PATH}" \
    ++actor.mb_spec.max_tokens_per_mb="${ACTOR_MAX_TOKENS_PER_MB}" \
    ++actor.setup_timeout="${WORKER_SETUP_TIMEOUT}" \
    ++actor.workers_ready_timeout="${WORKERS_READY_TIMEOUT}" \
    ++rollout.setup_timeout="${WORKER_SETUP_TIMEOUT}" \
    ++rollout.workers_ready_timeout="${WORKERS_READY_TIMEOUT}" \
    ++actor.dte.delta_method="${DTE_DELTA_METHOD}" \
    ++actor.dte.verify_snapshot="${DTE_VERIFY_SNAPSHOT}" \
    ++total_train_steps="${TOTAL_TRAIN_STEPS}" \
    ++saver.freq_steps="${SAVE_FREQ_STEPS}" \
    ++train_dataset.path="${DATASET_PATH}" \
    ++valid_dataset.path="${DATASET_PATH}" \
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
    ++actor.scheduling_spec.0.image="${IMAGE}" \
    ++actor.scheduling_spec.0.nodelist="${DTE_NODELIST}"

  echo "==> training finished: $(date -Is)"
  echo "==> rendered config: ${RUN_LOG_DIR}/config.yaml"
  check_smoke_logs "${RUN_LOG_DIR}" "${JOB_LOG}"
}

if [[ "${INSIDE_QWEN3_30B_DTE_CONTAINER:-0}" == "1" ]]; then
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
echo "==> sbatch job: \${SLURM_JOB_ID:-unknown}"
echo "==> nodelist: \${SLURM_JOB_NODELIST:-unknown}"
echo "==> topology: ${TOPOLOGY}"
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
  --env "INSIDE_QWEN3_30B_DTE_CONTAINER=1" \\
  --env "AREAL_SRC=${AREAL_SRC}" \\
  --env "IMAGE=${IMAGE}" \\
  --env "DTE_SRC=${DTE_SRC}" \\
  --env "DTE_AWEX_SRC=${DTE_AWEX_SRC}" \\
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
  --env "PYTHONPATH=${DTE_SRC}:${DTE_AWEX_SRC}:${AREAL_SRC}" \\
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
