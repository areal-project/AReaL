#!/usr/bin/env bash
set -euo pipefail

# Qwen3-0.6B colocated DTE smoke test.
#
# Defaults match the validated run from 2026-08-06:
#   actor.backend=megatron:d2p1t1
#   rollout.backend=sglang:d2p1t1
#   rollout.scheduling_strategy.type=colocation
#   actor.dte.enabled=true, actor.dte.transfer=delta
#
# Useful overrides:
#   NODELIST=slurmd-21 SUBMIT=1 ./examples/dte/run_qwen3_0_6b_colocation_dte_smoke.sh
#   SUBMIT=0 ./examples/dte/run_qwen3_0_6b_colocation_dte_smoke.sh

SCRIPT_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

AREAL_SRC=${AREAL_SRC:-${REPO_ROOT}}
IMAGE=${IMAGE:-/storage/openpsi/images/areal-dev-20260508.sif}
DTE_SRC=${DTE_SRC:-/storage/openpsi/users/pengzai.pyq/delta-transfer-engine-pyq-perf-sparse-p2p/src}
DTE_AWEX_SRC=${DTE_AWEX_SRC:-/storage/openpsi/users/pengzai.pyq/asystem-awex}
DATASET_PATH=${DATASET_PATH:-/storage/openpsi/data/gsm8k}
CONFIG_PATH=${CONFIG_PATH:-examples/math/gsm8k_grpo_megatron.yaml}

EXP_NAME=${EXP_NAME:-pyq-areal-port-dte-smoke}
TRIAL_SUFFIX=${TRIAL_SUFFIX:-$(date +%m%d_%H%M%S)}
TRIAL_NAME=${TRIAL_NAME:-colocation_dte_qwen3_0_6b_${TRIAL_SUFFIX}}
JOB_NAME=${JOB_NAME:-pyq-dte-q3-0p6b-col}
NODELIST=${NODELIST:-}
RESERVATION=${RESERVATION:-}
SUBMIT=${SUBMIT:-1}

FILEROOT=${FILEROOT:-/storage/openpsi/users/pengzai.pyq/areal_port_dte_smoke/fileroot}
NFS_ROOT=${NFS_ROOT:-${FILEROOT}/name_resolve/${EXP_NAME}/${TRIAL_NAME}}
RUN_ROOT=${RUN_ROOT:-${FILEROOT}/runs/${TRIAL_NAME}}
LAUNCH_DIR=${LAUNCH_DIR:-${FILEROOT}/launch/${TRIAL_NAME}}
CACHE_ROOT=${CACHE_ROOT:-${FILEROOT}/cache/${TRIAL_NAME}}
LOCAL_TMP_ROOT=${LOCAL_TMP_ROOT:-/tmp/areal_port_dte_${TRIAL_NAME}}
SBATCH_PATH=${SBATCH_PATH:-${LAUNCH_DIR}/job.sbatch}
JOB_LOG=${JOB_LOG:-${LAUNCH_DIR}/job.log}

MODEL_PATH=${MODEL_PATH:-/storage/openpsi/models/Qwen__Qwen3-0.6B}
ACTOR_BACKEND=${ACTOR_BACKEND:-megatron:d2p1t1}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-sglang:d2p1t1}
CLUSTER_GPUS=${CLUSTER_GPUS:-2}
SBATCH_GPUS=${SBATCH_GPUS:-2}
SCHED_TARGET=${SCHED_TARGET:-actor}

TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-3}
SAVE_FREQ_STEPS=${SAVE_FREQ_STEPS:-null}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
N_SAMPLES=${N_SAMPLES:-2}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
MAX_TOKENS=${MAX_TOKENS:-1024}
MAX_CONCURRENT_ROLLOUTS=${MAX_CONCURRENT_ROLLOUTS:-16}
LR=${LR:-3e-6}
EPS_CLIP=${EPS_CLIP:-1}
WORKER_SETUP_TIMEOUT=${WORKER_SETUP_TIMEOUT:-7200}
WORKERS_READY_TIMEOUT=${WORKERS_READY_TIMEOUT:-7200}
SGLANG_WAIT_WEIGHTS_READY_TIMEOUT=${SGLANG_WAIT_WEIGHTS_READY_TIMEOUT:-900}
NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
AWEX_COLOCATE_TIMEOUT_S=${AWEX_COLOCATE_TIMEOUT_S:-1800}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.72}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-2048}
DTE_DELTA_METHOD=${DTE_DELTA_METHOD:-adamw}
DTE_VERIFY_SNAPSHOT=${DTE_VERIFY_SNAPSHOT:-false}

run_inside_container() {
  cd "${AREAL_SRC}"

  export PYTHONUNBUFFERED=1
  export PYTHONNOUSERSITE=1
  export PYTHONFAULTHANDLER=1
  export HYDRA_FULL_ERROR=1
  export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1
  export PYTHONPATH="${DTE_SRC}:${DTE_AWEX_SRC}:${AREAL_SRC}:${PYTHONPATH:-}"
  export DTE_SRC
  export DTE_AWEX_SRC
  export AREAL_CACHE_DIR="${CACHE_ROOT}"
  export HOME="${CACHE_ROOT}/home"
  export HF_HOME="${CACHE_ROOT}/hf"
  export HF_DATASETS_CACHE="${CACHE_ROOT}/hf_datasets"
  export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
  export UV_CACHE_DIR="${CACHE_ROOT}/uv_cache"
  export PIP_CACHE_DIR="${CACHE_ROOT}/pip_cache"
  export TMPDIR="${LOCAL_TMP_ROOT}"
  export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
  export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/inductor"
  export VLLM_CACHE_ROOT="${CACHE_ROOT}/vllm"
  export FLASHINFER_WORKSPACE_DIR="${CACHE_ROOT}/flashinfer"
  export PYTORCH_KERNEL_CACHE_PATH="${CACHE_ROOT}/torch_kernel_cache"
  export TORCH_EXTENSIONS_DIR="${CACHE_ROOT}/torch_extensions"
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
  if [[ -x /opt/.venv/bin/python3 ]]; then
    export PATH="/opt/.venv/bin:${PATH}"
  fi

  mkdir -p "${HOME}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" \
    "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
    "${FLASHINFER_WORKSPACE_DIR}" "${PYTORCH_KERNEL_CACHE_PATH}" \
    "${TORCH_EXTENSIONS_DIR}"
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
    ++scheduler.type=local \
    ++experiment_name="${EXP_NAME}" \
    ++trial_name="${TRIAL_NAME}" \
    ++cluster.n_nodes=1 \
    ++cluster.n_gpus_per_node="${CLUSTER_GPUS}" \
    ++cluster.fileroot="${FILEROOT}" \
    ++cluster.name_resolve.nfs_record_root="${NFS_ROOT}" \
    +actor._version=v2 \
    +rollout._version=v2 \
    ++actor.backend="${ACTOR_BACKEND}" \
    ++rollout.backend="${ROLLOUT_BACKEND}" \
    ++rollout.scheduling_strategy.type=colocation \
    ++rollout.scheduling_strategy.target="${SCHED_TARGET}" \
    ++rollout.scheduling_strategy.fork=true \
    ++actor.path="${MODEL_PATH}" \
    ++ref.path="${MODEL_PATH}" \
    ++sglang.model_path="${MODEL_PATH}" \
    ++tokenizer_path="${MODEL_PATH}" \
    ++actor.gradient_checkpointing=true \
    ++actor.optimizer.lr="${LR}" \
    ++actor.eps_clip="${EPS_CLIP}" \
    ++actor.eps_clip_higher=null \
    ++actor.setup_timeout="${WORKER_SETUP_TIMEOUT}" \
    ++actor.workers_ready_timeout="${WORKERS_READY_TIMEOUT}" \
    ++rollout.setup_timeout="${WORKER_SETUP_TIMEOUT}" \
    ++rollout.workers_ready_timeout="${WORKERS_READY_TIMEOUT}" \
    ++actor.dte.enabled=true \
    ++actor.dte.transfer=delta \
    ++actor.dte.delta_method="${DTE_DELTA_METHOD}" \
    ++actor.dte.anchor_interval=0 \
    ++actor.dte.verify_snapshot="${DTE_VERIFY_SNAPSHOT}" \
    ++total_train_steps="${TOTAL_TRAIN_STEPS}" \
    ++saver.mode=auto \
    ++saver.freq_epochs=null \
    ++saver.freq_steps="${SAVE_FREQ_STEPS}" \
    ++evaluator.freq_epochs=null \
    ++evaluator.freq_steps=null \
    ++evaluator.freq_secs=null \
    ++recover.mode=disabled \
    ++recover.freq_epochs=null \
    ++recover.freq_steps=null \
    ++recover.freq_secs=null \
    ++train_dataset.path="${DATASET_PATH}" \
    ++valid_dataset.path="${DATASET_PATH}" \
    ++train_dataset.batch_size="${TRAIN_BATCH_SIZE}" \
    ++valid_dataset.batch_size="${TRAIN_BATCH_SIZE}" \
    ++rollout.consumer_batch_size="${TRAIN_BATCH_SIZE}" \
    ++rollout.max_concurrent_rollouts="${MAX_CONCURRENT_ROLLOUTS}" \
    ++rollout.max_head_offpolicyness=8 \
    ++rollout.dump_to_file=true \
    ++gconfig.n_samples="${N_SAMPLES}" \
    ++gconfig.max_new_tokens="${MAX_NEW_TOKENS}" \
    ++gconfig.max_tokens="${MAX_TOKENS}" \
    ++sglang.context_length="${SGLANG_CONTEXT_LENGTH}" \
    ++sglang.mem_fraction_static="${SGLANG_MEM_FRACTION}" \
    ++stats_logger.wandb.mode=disabled

  echo "==> training finished: $(date -Is)"
  echo "==> rendered config: ${RUN_LOG_DIR}/config.yaml"
  check_smoke_logs "${RUN_LOG_DIR}" "${JOB_LOG}"
}

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
    r"WeightUpdateController connected .*colocate=True",
    r"colocate delta enabled \(sender\); DTE components ready",
    r"colocate delta enabled \(receiver\); DTE DeltaEngine ready",
    r"colocate delta v1: FULL sync",
    r"colocate delta v\d+ \[.*\]: changed",
    r"Colocate DTE weight update completed",
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
print("DTE_SMOKE_TOPOLOGY=colocation")
print(f"DTE_SMOKE_TRAIN_STEP_MAX={max(train_steps) if train_steps else 0}")
print(f"DTE_SMOKE_WEIGHT_UPDATE_VERSIONS={updates}")
if failures:
    print("DTE_SMOKE_CHECK_FAILED=" + "; ".join(failures))
    raise SystemExit(1)
print("DTE_SMOKE_CHECK_OK=1")
PY
}

if [[ "${INSIDE_DTE_SMOKE_CONTAINER:-0}" == "1" ]]; then
  run_inside_container
  exit 0
fi

mkdir -p "${LAUNCH_DIR}" "${RUN_ROOT}" "${CACHE_ROOT}" "${NFS_ROOT}"

SBATCH_NODELIST_LINE=
if [[ -n "${NODELIST}" ]]; then
  SBATCH_NODELIST_LINE="#SBATCH --nodelist=${NODELIST}"
fi
SBATCH_RESERVATION_LINE=
if [[ -n "${RESERVATION}" ]]; then
  SBATCH_RESERVATION_LINE="#SBATCH --reservation=${RESERVATION}"
fi

cat > "${SBATCH_PATH}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:${SBATCH_GPUS}
#SBATCH --mem=0
#SBATCH --time=03:00:00
${SBATCH_NODELIST_LINE}
${SBATCH_RESERVATION_LINE}
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --chdir=${AREAL_SRC}
#SBATCH --output=${JOB_LOG}
#SBATCH --open-mode=append

set -euo pipefail
echo "==> sbatch job: \${SLURM_JOB_ID:-unknown}"
echo "==> nodelist: \${SLURM_JOB_NODELIST:-unknown}"
echo "==> CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}"
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
  --env INSIDE_DTE_SMOKE_CONTAINER=1 \\
  --env AREAL_SRC=${AREAL_SRC} \\
  --env IMAGE=${IMAGE} \\
  --env DTE_SRC=${DTE_SRC} \\
  --env DTE_AWEX_SRC=${DTE_AWEX_SRC} \\
  --env DATASET_PATH=${DATASET_PATH} \\
  --env CONFIG_PATH=${CONFIG_PATH} \\
  --env EXP_NAME=${EXP_NAME} \\
  --env TRIAL_NAME=${TRIAL_NAME} \\
  --env FILEROOT=${FILEROOT} \\
  --env NFS_ROOT=${NFS_ROOT} \\
  --env CACHE_ROOT=${CACHE_ROOT} \\
  --env LOCAL_TMP_ROOT=${LOCAL_TMP_ROOT} \\
  --env MODEL_PATH=${MODEL_PATH} \\
  --env ACTOR_BACKEND=${ACTOR_BACKEND} \\
  --env ROLLOUT_BACKEND=${ROLLOUT_BACKEND} \\
  --env CLUSTER_GPUS=${CLUSTER_GPUS} \\
  --env SCHED_TARGET=${SCHED_TARGET} \\
  --env TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS} \\
  --env SAVE_FREQ_STEPS=${SAVE_FREQ_STEPS} \\
  --env TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} \\
  --env N_SAMPLES=${N_SAMPLES} \\
  --env MAX_NEW_TOKENS=${MAX_NEW_TOKENS} \\
  --env MAX_TOKENS=${MAX_TOKENS} \\
  --env MAX_CONCURRENT_ROLLOUTS=${MAX_CONCURRENT_ROLLOUTS} \\
  --env LR=${LR} \\
  --env EPS_CLIP=${EPS_CLIP} \\
  --env WORKER_SETUP_TIMEOUT=${WORKER_SETUP_TIMEOUT} \\
  --env WORKERS_READY_TIMEOUT=${WORKERS_READY_TIMEOUT} \\
  --env SGLANG_WAIT_WEIGHTS_READY_TIMEOUT=${SGLANG_WAIT_WEIGHTS_READY_TIMEOUT} \\
  --env NCCL_TIMEOUT=${NCCL_TIMEOUT} \\
  --env AWEX_COLOCATE_TIMEOUT_S=${AWEX_COLOCATE_TIMEOUT_S} \\
  --env SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION} \\
  --env SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH} \\
  --env DTE_DELTA_METHOD=${DTE_DELTA_METHOD} \\
  --env DTE_VERIFY_SNAPSHOT=${DTE_VERIFY_SNAPSHOT} \\
  --env PYTHONPATH=${DTE_SRC}:${DTE_AWEX_SRC}:${AREAL_SRC} \\
  --env PYTHONUNBUFFERED=1 \\
  --env PYTHONFAULTHANDLER=1 \\
  --env HYDRA_FULL_ERROR=1 \\
  --env PYTHONNOUSERSITE=1 \\
  --env AREAL_ALLOW_DEFAULT_ADMIN_KEY=1 \\
  "${IMAGE}" \\
  bash "${SCRIPT_PATH}"
EOF

echo "==> topology: colocation"
echo "==> model: ${MODEL_PATH}"
echo "==> actor backend: ${ACTOR_BACKEND}"
echo "==> rollout backend: ${ROLLOUT_BACKEND}"
echo "==> script: ${SCRIPT_PATH}"
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
