#!/bin/bash
set -uo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR" || exit 1
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

STAMP=${PROFILE_STAMP:-$(date +%Y%m%d_%H%M%S)}
CONFIG=${CONFIG:-examples/profile/qwen3_30b_a3b_sft_profile.yaml}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-30b-a3b-sft-profile}
FILEROOT=${FILEROOT:-/tmp/areal/experiments}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B}
PROFILE_STEPS=${PROFILE_STEPS:-${PROFILE_STEP:-0,1}}
TOTAL_STEPS=${TOTAL_STEPS:-}
PROFILE_KINDS=${PROFILE_KINDS:-kernel,memory}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PROFILE_N_MBS=${PROFILE_N_MBS:-4}
PROFILE_FAKE_SEQ_LEN=${PROFILE_FAKE_SEQ_LEN:-131072}
PROFILE_FAKE_DATASET_SIZE=${PROFILE_FAKE_DATASET_SIZE:-8}
PROFILE_FAKE_LOSS_START_RATIO=${PROFILE_FAKE_LOSS_START_RATIO:-0.5}
AREAL_LOGPROBS_CHUNK_SIZE=${AREAL_LOGPROBS_CHUNK_SIZE:-128}
LM_HEAD_LOSS_CHUNK_SIZE=${LM_HEAD_LOSS_CHUNK_SIZE:-0}
USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER:-true}
MAIN_GRADS_DTYPE=${MAIN_GRADS_DTYPE:-bfloat16}
STOP_ON_FAILURE=${STOP_ON_FAILURE:-1}
RUN_ROOT=${RUN_ROOT:-${ROOT_DIR}/examples/profile/profile_data/${STAMP}_qwen3-30b-a3b_fake128k_sft_profile}
AREAL_LOG_USER=${AREAL_LOG_USER:-$(python -c 'import getpass; print(getpass.getuser())')}

mkdir -p "$RUN_ROOT"

compute_pp_rank0_profile_ranks() {
  python - "$CONFIG" <<'PY'
import sys
from pathlib import Path

from examples.profile.profile_rank_utils import pp_rank0_ranks_from_config

print(pp_rank0_ranks_from_config(Path(sys.argv[1])))
PY
}

if [[ -z "${PROFILE_RANKS+x}" ]]; then
  PROFILE_RANKS="<script-default:pp-rank0>"
  if ! RESOLVED_PROFILE_RANKS=$(compute_pp_rank0_profile_ranks); then
    echo "Failed to compute PP rank0 profile ranks from ${CONFIG}" >&2
    exit 2
  fi
else
  if ! RESOLVED_PROFILE_RANKS=$(python examples/profile/profile_rank_utils.py --config "$CONFIG" --profile-ranks "$PROFILE_RANKS"); then
    echo "Failed to resolve PROFILE_RANKS=${PROFILE_RANKS} from ${CONFIG}" >&2
    exit 2
  fi
fi

monitor_gpu() {
  local out_csv=$1
  echo "timestamp,index,memory.used [MiB]" > "$out_csv"
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits >> "$out_csv" 2>/dev/null || true
    sleep 1
  done
}

count_files() {
  local pattern=$1
  compgen -G "$pattern" >/dev/null || {
    printf "0"
    return
  }
  compgen -G "$pattern" | wc -l
}

step_done() {
  local log=$1
  local total_steps=$2
  [[ -f "$log" ]] || return 1
  ! grep -Eq "Training failed|Traceback|RuntimeError|ChildFailedError" "$log" || return 1
  grep -Eq "Training completes|Train step ${total_steps}/.*done" "$log"
}

should_run_kind() {
  local kind=$1
  [[ ",${PROFILE_KINDS}," == *",${kind},"* ]]
}

run_case() {
  local profile_kind=$1
  local profile_step=$2
  local total_steps=${TOTAL_STEPS:-$((profile_step + 1))}
  local trial_name="qwen3_30b_a3b_fake128k_${profile_kind}_step${profile_step}_${STAMP}"
  local case_dir="${RUN_ROOT}/${trial_name}"
  local launcher_log="${case_dir}/launcher.log"
  local mem_csv="${case_dir}/nvidia_smi.csv"
  local name_resolve_root="${FILEROOT}/name_resolve/${trial_name}"
  local trial_log_dir="${FILEROOT}/logs/${AREAL_LOG_USER}/${EXPERIMENT_NAME}/${trial_name}"
  local trainer_log="${trial_log_dir}/trainer.log"
  local perf_ranks=""
  local memory_ranks=""
  local torch_profiler_profile_memory=false
  local -a profile_args=()

  mkdir -p "$case_dir" "$name_resolve_root"

  case "$profile_kind" in
    kernel)
      perf_ranks="$RESOLVED_PROFILE_RANKS"
      profile_args=(
        perf_tracer.enabled=true
        perf_tracer.experiment_name="${EXPERIMENT_NAME}"
        perf_tracer.trial_name="${trial_name}"
        perf_tracer.fileroot="${FILEROOT}"
        perf_tracer.save_interval=1
        perf_tracer.profile_steps="[${profile_step}]"
      )
      ;;
    memory)
      memory_ranks="$RESOLVED_PROFILE_RANKS"
      profile_args=(
        memory_profiler.profile_steps="[${profile_step}]"
        memory_profiler.max_entries=200000
      )
      ;;
    *)
      echo "Unknown profile kind: ${profile_kind}. Expected kernel or memory." | tee -a "$launcher_log"
      return 2
      ;;
  esac

  echo "===== ${trial_name} start $(date -Is), kind=${profile_kind}, step=${profile_step}, total_steps=${total_steps}, ranks=${RESOLVED_PROFILE_RANKS} =====" | tee -a "$launcher_log"
  monitor_gpu "$mem_csv" &
  local monitor_pid=$!

  AREAL_PERF_TRACER_RANKS="${perf_ranks}" \
  AREAL_MEMORY_PROFILER_RANKS="${memory_ranks}" \
  AREAL_TORCH_PROFILER_PROFILE_MEMORY="${torch_profiler_profile_memory}" \
  AREAL_LOGPROBS_CHUNK_SIZE="${AREAL_LOGPROBS_CHUNK_SIZE}" \
  python -m areal.infra.launcher.local examples/profile/train_sft_profile.py \
    --config "$CONFIG" \
    experiment_name="${EXPERIMENT_NAME}" \
    trial_name="${trial_name}" \
    total_train_steps="${total_steps}" \
    actor.path="${MODEL_PATH}" \
    tokenizer_path="${MODEL_PATH}" \
    train_dataset.batch_size="${TRAIN_BATCH_SIZE}" \
    train_dataset.max_length="${PROFILE_FAKE_SEQ_LEN}" \
    actor.mb_spec.n_mbs="${PROFILE_N_MBS}" \
    actor.mb_spec.n_mbs_divisor="${PROFILE_N_MBS}" \
    actor.megatron.lm_head_loss_chunk_size="${LM_HEAD_LOSS_CHUNK_SIZE}" \
    actor.megatron.use_precision_aware_optimizer="${USE_PRECISION_AWARE_OPTIMIZER}" \
    actor.megatron.main_grads_dtype="${MAIN_GRADS_DTYPE}" \
    profile.fake_seq_len="${PROFILE_FAKE_SEQ_LEN}" \
    profile.fake_dataset_size="${PROFILE_FAKE_DATASET_SIZE}" \
    profile.fake_loss_start_ratio="${PROFILE_FAKE_LOSS_START_RATIO}" \
    cluster.fileroot="${FILEROOT}" \
    cluster.name_resolve.nfs_record_root="${name_resolve_root}" \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    saver.freq_epochs=null \
    saver.freq_steps=null \
    recover.mode=disabled \
    recover.freq_epochs=null \
    recover.freq_steps=null \
    evaluator.freq_epochs=null \
    evaluator.freq_steps=null \
    stats_logger.wandb.mode=disabled \
    "${profile_args[@]}" 2>&1 | tee -a "$launcher_log"
  local launch_status=${PIPESTATUS[0]}
  local case_status=$launch_status

  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true

  if step_done "$trainer_log" "$total_steps"; then
    case_status=0
  fi

  python examples/profile/postprocess_profile.py \
    --log-dir "$trial_log_dir" \
    --run-dir "$case_dir" \
    --profile-kind "$profile_kind" \
    --profile-step "$profile_step" \
    --trainer-log "$trainer_log" \
    --nvidia-smi-csv "$mem_csv" >> "$launcher_log" 2>&1 || true

  local trace_count snapshot_count
  trace_count=$(count_files "${trial_log_dir}/perf_tracer/*/traces-*.jsonl")
  snapshot_count=$(count_files "${trial_log_dir}/memory_snapshots/step_${profile_step}/snapshot_*.pickle")
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$profile_kind" "$profile_step" "$trial_name" "$case_status" "$launch_status" "$trace_count" "$snapshot_count" "$trainer_log" \
    | tee -a "${RUN_ROOT}/summary.tsv"
  echo "===== ${trial_name} end $(date -Is), status=${case_status}, launcher_status=${launch_status} =====" | tee -a "$launcher_log"
  return "$case_status"
}

printf "profile_kind\tprofile_step\ttrial_name\tstatus\tlauncher_status\ttrace_file_count\tmemory_snapshot_count\ttrainer_log\n" > "${RUN_ROOT}/summary.tsv"
echo "Run root: ${RUN_ROOT}" | tee "${RUN_ROOT}/profile_settings.log"
echo "Model: ${MODEL_PATH}" | tee -a "${RUN_ROOT}/profile_settings.log"
echo "Profile steps: ${PROFILE_STEPS}" | tee -a "${RUN_ROOT}/profile_settings.log"
echo "Profile ranks: ${PROFILE_RANKS} -> ${RESOLVED_PROFILE_RANKS}" | tee -a "${RUN_ROOT}/profile_settings.log"
echo "Fake data: seq_len=${PROFILE_FAKE_SEQ_LEN}, dataset_size=${PROFILE_FAKE_DATASET_SIZE}, loss_start_ratio=${PROFILE_FAKE_LOSS_START_RATIO}, batch_size=${TRAIN_BATCH_SIZE}, n_mbs=${PROFILE_N_MBS}" | tee -a "${RUN_ROOT}/profile_settings.log"
echo "Logprobs chunk size: ${AREAL_LOGPROBS_CHUNK_SIZE}" | tee -a "${RUN_ROOT}/profile_settings.log"
echo "LM Head loss chunk size: ${LM_HEAD_LOSS_CHUNK_SIZE}" | tee -a "${RUN_ROOT}/profile_settings.log"
echo "Precision-aware optimizer: ${USE_PRECISION_AWARE_OPTIMIZER}, main grads dtype: ${MAIN_GRADS_DTYPE}" | tee -a "${RUN_ROOT}/profile_settings.log"

overall_status=0
IFS="," read -ra profile_steps <<< "$PROFILE_STEPS"
for raw_step in "${profile_steps[@]}"; do
  profile_step=$(echo "$raw_step" | xargs)
  [[ -n "$profile_step" ]] || continue
  if ! [[ "$profile_step" =~ ^[0-9]+$ ]]; then
    echo "Invalid profile step: ${profile_step}. PROFILE_STEPS must be comma-separated non-negative integers." >&2
    overall_status=2
    continue
  fi
  if should_run_kind kernel; then
    run_case kernel "$profile_step"
    case_status=$?
    if [[ "$case_status" -ne 0 ]]; then
      overall_status=$case_status
      [[ "$STOP_ON_FAILURE" == "1" ]] && break
    fi
  fi
  if should_run_kind memory; then
    run_case memory "$profile_step"
    case_status=$?
    if [[ "$case_status" -ne 0 ]]; then
      overall_status=$case_status
      [[ "$STOP_ON_FAILURE" == "1" ]] && break
    fi
  fi
done

echo "Summary: ${RUN_ROOT}/summary.tsv"
exit "$overall_status"
