#!/bin/bash
# CC-agent Flash-MoE Megatron + SGLang colocated AWEX run.
# Defaults mirror the CC separated baseline, with only required colocation changes.
# Required env vars:
#   AENV_SYSTEM_URL

#SBATCH -J zjw-monolith-swe-cc
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --reservation=swe-rl
#SBATCH --oversubscribe
#SBATCH --gres=gpu:0
#SBATCH -t 7-00:00:00
#SBATCH -o /storage/openpsi/users/zjw531248/logs/monolith-swe-cc-%j.out
#SBATCH -e /storage/openpsi/users/zjw531248/logs/monolith-swe-cc-%j.err
#SBATCH --exclude=slurmd-74,slurmd-97

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -z "${AREAL_DIR:-}" ]; then
  if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/pyproject.toml" ]; then
    AREAL_DIR=${SLURM_SUBMIT_DIR}
  elif [ -f "${SCRIPT_DIR}/../../../pyproject.toml" ]; then
    AREAL_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
  else
    echo "ERROR: AREAL_DIR is required; submit from repo root or set AREAL_DIR explicitly" >&2
    exit 1
  fi
fi
CONFIG_PATH=${CONFIG_PATH:-examples/swe/swe_colocate_cc.yaml}
IMAGE=${AREAL_IMAGE:-/storage/openpsi/images/areal-dev-sglang-20260401.sif}

MODEL_PATH=${MODEL_PATH:-/storage/openpsi/experiments/checkpoints/admin/hcy-ring-sft/0531_flash_moe_bs256_g64_lr5e-5_stepfun_v9_no_agentic_continue_stepfun_ring_swe_v2_scaleswe/default/epoch2epochstep596globalstep1790}
TRAIN_DATA=${TRAIN_DATA:-/storage/openpsi/users/fenghui/projects/AWEAgent_DEV/AWEAgent/src/data/swe_bench_verified_rl.jsonl}
AWEAGENT_ROOT=${AWEAGENT_ROOT:-/storage/openpsi/users/public/projects/AWEAgent}
FLA_DIR=${FLA_DIR:-/storage/openpsi/users/public/projects/flash-linear-attention}
AWEX_ROOT=${AWEX_ROOT:-/storage/openpsi/users/public/projects/asystem-awex/}
AREAL_EXTRA_PYTHONPATH=${AREAL_EXTRA_PYTHONPATH:-/storage/openpsi/users/zjw531248/monolith_pkgs}
SGLANG_PYTHON_ROOT=${SGLANG_PYTHON_ROOT:-/storage/openpsi/users/chucai.dzq/codes/sglang/python}

EXP_NAME=${EXP_NAME:-zjw-monolith-swe-cc}
TRIAL_NAME=${TRIAL_NAME:-aligned-8n-0708}
N_NODES=${N_NODES:-8}
N_GPUS=${N_GPUS:-8}
ENABLE_FP32_LM_HEAD=${ENABLE_FP32_LM_HEAD:-true}
ACTOR_BACKEND=${ACTOR_BACKEND:-megatron:(attn:d2p8t4|ffn:d2p8e4)}
ROLLOUT_BACKEND=${ROLLOUT_BACKEND:-sglang:d16t4p1}
FILEROOT=${FILEROOT:-/storage/openpsi/users/zjw531248/monolith-swe}
NFS_ROOT=${NFS_ROOT:-${FILEROOT}/name_resolve/${EXP_NAME}}
RESERVATION=${RESERVATION:-swe-rl}

AENV_SYSTEM_URL=${AENV_SYSTEM_URL:-http://33.180.184.68}
WANDB_API_KEY=${WANDB_API_KEY:-local-3bca3d5f00a980f3075b3e8ff2e16adc4ef43ffe}
WANDB_BASE_URL=${WANDB_BASE_URL:-http://8.150.1.98:8080}
AWEX_ACTOR_ALLOC_CONF=${AWEX_ACTOR_ALLOC_CONF-expandable_segments:True}

export SBATCH_RESERVATION=${RESERVATION}
export AREAL_SLURM_RESERVATION=${RESERVATION}
export AREAL_RESERVATION=${RESERVATION}
export AREAL_APPTAINER_STAGGER_SECONDS=${AREAL_APPTAINER_STAGGER_SECONDS:-15}
export NCCL_DEBUG=WARN
export NCCL_NET=${NCCL_NET:-IB}
if [ "${NCCL_NET}" = "IB" ]; then
  export NCCL_IB_DISABLE=0
else
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
fi
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_IB_TC=${NCCL_IB_TC:-136}
export NCCL_IB_SL=${NCCL_IB_SL:-5}
export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-8}
export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22}
export NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT:-7}
export NCCL_SET_THREAD_NAME=${NCCL_SET_THREAD_NAME:-1}
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_PXN_LEVEL=0
export AWEX_CHUNK_MB=${AWEX_CHUNK_MB:-2048}
export AWEX_CHUNK_OPS=${AWEX_CHUNK_OPS:-128}

mkdir -p "${FILEROOT}/logs/${EXP_NAME}"
rm -rf "${NFS_ROOT}" 2>/dev/null || true

echo "=== CC colocate run nodes=${N_NODES} trial=${TRIAL_NAME} ==="
echo "=== areal=${AREAL_DIR} model=${MODEL_PATH} aweagent=${AWEAGENT_ROOT} ==="

MAX_DRIVER_RETRIES=${MAX_DRIVER_RETRIES:-1}
attempt=1
while true; do
  echo "=== driver attempt ${attempt}/${MAX_DRIVER_RETRIES} trial=${TRIAL_NAME} $(date) ==="
  set +e
  srun --mpi=pmi2 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=1000M \
    singularity exec --pid --writable-tmpfs \
      --env "CONFIG_PATH=${CONFIG_PATH}" \
      --env "TRIAL_NAME=${TRIAL_NAME}" \
      --env "MODEL_PATH=${MODEL_PATH}" \
      --env "TRAIN_DATA=${TRAIN_DATA}" \
      --env "EXP_NAME=${EXP_NAME}" \
      --env "N_NODES=${N_NODES}" \
      --env "N_GPUS=${N_GPUS}" \
      --env "FILEROOT=${FILEROOT}" \
      --env "NFS_ROOT=${NFS_ROOT}" \
      --env "AWEAGENT_ROOT=${AWEAGENT_ROOT}" \
      --env "AENV_SYSTEM_URL=${AENV_SYSTEM_URL}" \
      --env "SBATCH_RESERVATION=${RESERVATION}" \
      --env "AREAL_SLURM_RESERVATION=${RESERVATION}" \
      --env "AREAL_RESERVATION=${RESERVATION}" \
      --env "AREAL_APPTAINER_STAGGER_SECONDS=${AREAL_APPTAINER_STAGGER_SECONDS}" \
      --env "NCCL_DEBUG=${NCCL_DEBUG}" \
      --env "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}" \
      --env "NCCL_NET=${NCCL_NET}" \
      --env "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}" \
      --env "NCCL_IB_HCA=${NCCL_IB_HCA}" \
      --env "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}" \
      --env "NCCL_IB_TC=${NCCL_IB_TC}" \
      --env "NCCL_IB_SL=${NCCL_IB_SL}" \
      --env "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION}" \
      --env "NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT}" \
      --env "NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT}" \
      --env "NCCL_SET_THREAD_NAME=${NCCL_SET_THREAD_NAME}" \
      --env "NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}" \
      --env "TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
      --env "NCCL_P2P_PXN_LEVEL=${NCCL_P2P_PXN_LEVEL}" \
      --env "AWEX_CHUNK_MB=${AWEX_CHUNK_MB}" \
      --env "AWEX_CHUNK_OPS=${AWEX_CHUNK_OPS}" \
      --env "AWEX_ACTOR_ALLOC_CONF=${AWEX_ACTOR_ALLOC_CONF}" \
      --env "WANDB_API_KEY=${WANDB_API_KEY}" \
      --env "AREAL_ALLOW_DEFAULT_ADMIN_KEY=1" \
      --env "WANDB_BASE_URL=${WANDB_BASE_URL}" \
      --env "AREAL_DIR=${AREAL_DIR}" \
      --env "AREAL_IMAGE=${IMAGE}" \
      --env "FLASH_LINEAR_ATTENTION_ROOT=${FLA_DIR}" \
      --env "AWEX_ROOT=${AWEX_ROOT}" \
      --env "AREAL_EXTRA_PYTHONPATH=${AREAL_EXTRA_PYTHONPATH}" \
      --env "SGLANG_PYTHON_ROOT=${SGLANG_PYTHON_ROOT}" \
      --env "TMPDIR=${TMPDIR:-/tmp}" \
      --env "TEMP=${TEMP:-/tmp}" \
      --env "TMP=${TMP:-/tmp}" \
      --bind /storage:/storage \
      --bind /home:/home \
      --bind /etc/slurm/:/etc/slurm/ \
      --bind /etc/passwd:/etc/passwd:ro \
      --bind /etc/group:/etc/group:ro \
      --bind /etc/munge:/etc/munge:ro \
      --bind /var/run/munge:/var/run/munge \
      --bind /usr/bin/sbatch:/usr/bin/sbatch \
      --bind /usr/bin/srun:/usr/bin/srun \
      --bind /usr/bin/squeue:/usr/bin/squeue \
      --bind /usr/bin/scancel:/usr/bin/scancel \
      --bind /usr/bin/scontrol:/usr/bin/scontrol \
      --bind /usr/lib64/slurm:/usr/lib64/slurm \
      "${IMAGE}" \
      bash -lc "(/usr/sbin/munged 2>/dev/null || true) && \
        cd ${AREAL_DIR} && \
        export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ && \
        true && \
        if [ -n \"\${VIRTUAL_ENV:-}\" ] && [ -f \"\${VIRTUAL_ENV}/bin/activate\" ]; then . \"\${VIRTUAL_ENV}/bin/activate\"; fi && \
        export NCCL_DEBUG=${NCCL_DEBUG} && \
        export NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_NET=${NCCL_NET} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} && \
        export NCCL_IB_HCA=${NCCL_IB_HCA} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX} NCCL_IB_TC=${NCCL_IB_TC} NCCL_IB_SL=${NCCL_IB_SL} && \
        export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION} NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT} NCCL_IB_RETRY_CNT=${NCCL_IB_RETRY_CNT} NCCL_SET_THREAD_NAME=${NCCL_SET_THREAD_NAME} && \
        export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING} TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING} NCCL_P2P_PXN_LEVEL=${NCCL_P2P_PXN_LEVEL} && \
        export PYTHONPATH=${AREAL_EXTRA_PYTHONPATH}:${SGLANG_PYTHON_ROOT}:${AREAL_DIR}:${AWEX_ROOT}:${FLA_DIR}:\${PYTHONPATH:-} && \
        python3 examples/swe/train_swe_rl.py --config ${CONFIG_PATH} \
          experiment_name=${EXP_NAME} \
          trial_name=${TRIAL_NAME} \
          tokenizer_path=${MODEL_PATH} \
          actor.path=${MODEL_PATH} \
          actor.megatron.enable_fp32_lm_head=${ENABLE_FP32_LM_HEAD} \
          \"actor.backend='${ACTOR_BACKEND}'\" \
          \"rollout.backend='${ROLLOUT_BACKEND}'\" \
          cluster.n_nodes=${N_NODES} \
          cluster.n_gpus_per_node=${N_GPUS} \
          cluster.fileroot=${FILEROOT} \
          cluster.name_resolve.nfs_record_root=${NFS_ROOT} \
          train_dataset.path=${TRAIN_DATA} \
          valid_dataset.path=${TRAIN_DATA} \
          stats_logger.wandb.mode=online \
          ++stats_logger.wandb.wandb_api_key=${WANDB_API_KEY} \
          ++stats_logger.wandb.wandb_base_url=${WANDB_BASE_URL} \
          \"++actor.scheduling_spec.0.env_vars.AENV_SYSTEM_URL=${AENV_SYSTEM_URL}\" \
          \"++actor.scheduling_spec.0.env_vars.AREAL_DIR=${AREAL_DIR}\" \
          \"++actor.scheduling_spec.0.env_vars.AWEAGENT_ROOT=${AWEAGENT_ROOT}\" \
          \"++actor.scheduling_spec.0.env_vars.WANDB_BASE_URL=${WANDB_BASE_URL}\" \
          \"++actor.scheduling_spec.0.env_vars.WANDB_API_KEY=${WANDB_API_KEY}\" \
          \"++actor.scheduling_spec.0.env_vars.NCCL_DEBUG=${NCCL_DEBUG}\" \
          \"++actor.scheduling_spec.0.env_vars.NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING}\" \
          \"++actor.scheduling_spec.0.env_vars.TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}\" \
          \"++actor.scheduling_spec.0.env_vars.NCCL_P2P_PXN_LEVEL=${NCCL_P2P_PXN_LEVEL}\""

  exit_code=$?
  set -e
  if [ ${exit_code} -eq 0 ]; then
    break
  fi
  if [ ${attempt} -ge ${MAX_DRIVER_RETRIES} ]; then
    echo "=== driver failed ${MAX_DRIVER_RETRIES} times, giving up ==="
    break
  fi
  attempt=$((attempt + 1))
  echo "=== driver exited ${exit_code}; wait 90s then recover trial ${TRIAL_NAME} ==="
  sleep 90
done

echo "=== Job ${SLURM_JOB_ID:-local} finished $(date), exit=${exit_code} ==="
exit ${exit_code}
