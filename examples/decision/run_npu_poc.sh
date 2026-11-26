#!/usr/bin/env bash
# Single-NPU PointerHead decision SFT PoC.
#
# Required:
#   MODEL_PATH   local checkpoint directory or Hugging Face repo id
# And one of:
#   SUITE_PATH   frozen suite directory (sampled to a small JSONL slice)
#   TRAIN_PATH   existing train JSONL (then also set VALID_PATH)
#   DATA_PATH    directory with train.jsonl plus development.jsonl or manifest.json
#
# Optional:
#   VALID_PATH       development / validation JSONL
#   FILEROOT         default /tmp/areal/experiments
#   EPOCHS           default 2
#   STEPS            optional step cap; unset means run the full epoch budget
#   SAVE_STEPS       checkpoint interval when STEPS is set (default: STEPS)
#   EVAL_STEPS       eval interval when STEPS is set (default: STEPS)
#   N_PER_SOURCE     default 32 (only when sampling a suite)
#   DEV_PER_SOURCE   default 4
#   DEVICE           default 0  (ASCEND_RT_VISIBLE_DEVICES)
#
#   MODEL_PATH=/path/to/Qwen3.5-4B-Base \
#   SUITE_PATH=/path/to/decision-suite \
#     bash examples/decision/run_npu_poc.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE:-${ASCEND_RT_VISIBLE_DEVICES:-0}}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
# Local scheduler health-checks the node IP. If that address is missing from
# NO_PROXY, HTTP_PROXY intercepts worker RPC and the run hangs.
HOST_IPS="$(hostname -I 2>/dev/null | xargs | tr ' ' ',')"
HOST_NAME="$(hostname -f 2>/dev/null || hostname 2>/dev/null || true)"
ROUTE_IPS="$(python3 - <<'PY'
import socket
ips = []
def add(ip):
    if ip and ip not in ips:
        ips.append(ip)
try:
    for family, _, _, _, sockaddr in socket.getaddrinfo(socket.gethostname(), None):
        add(sockaddr[0])
except OSError:
    pass
try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        add(sock.getsockname()[0])
except OSError:
    pass
print(",".join(ips))
PY
)"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,::1${HOST_NAME:+,${HOST_NAME}}${HOST_IPS:+,${HOST_IPS}}${ROUTE_IPS:+,${ROUTE_IPS}}"
export no_proxy="$NO_PROXY"

MODEL_PATH="${MODEL_PATH:-}"
SUITE_PATH="${SUITE_PATH:-}"
DATA_PATH="${DATA_PATH:-}"
TRAIN_PATH="${TRAIN_PATH:-}"
VALID_PATH="${VALID_PATH:-}"
FILEROOT="${FILEROOT:-/tmp/areal/experiments}"
NAME_RESOLVE="${NAME_RESOLVE:-${FILEROOT}/name_resolve}"
EPOCHS="${EPOCHS:-2}"
STEPS="${STEPS:-}"
N_PER_SOURCE="${N_PER_SOURCE:-32}"
DEV_PER_SOURCE="${DEV_PER_SOURCE:-4}"
SLICE_DIR="${SLICE_DIR:-${ROOT}/examples/decision/data}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35-decision-sft-npu-poc}"
TRIAL_NAME="${TRIAL_NAME:-poc-e2}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "Set MODEL_PATH to a local Qwen3.5 checkpoint or a Hugging Face repo id." >&2
  exit 1
fi
if [[ "$MODEL_PATH" == /* || "$MODEL_PATH" == .* ]] && [[ ! -e "$MODEL_PATH" ]]; then
  echo "model path not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ -n "$TRAIN_PATH" ]]; then
  if [[ ! -f "$TRAIN_PATH" ]]; then
    echo "train JSONL not found: $TRAIN_PATH" >&2
    exit 1
  fi
  if [[ -z "$VALID_PATH" ]]; then
    echo "Set VALID_PATH when using TRAIN_PATH." >&2
    exit 1
  fi
  if [[ ! -f "$VALID_PATH" ]]; then
    echo "valid JSONL not found: $VALID_PATH" >&2
    exit 1
  fi
elif [[ -n "$SUITE_PATH" ]]; then
  if [[ ! -d "$SUITE_PATH" ]]; then
    echo "decision suite not found: $SUITE_PATH" >&2
    exit 1
  fi
  echo "Sampling ${N_PER_SOURCE} records/source from ${SUITE_PATH}"
  python3 examples/decision/prepare_v7_sample.py \
    --suite "$SUITE_PATH" \
    --out-dir "$SLICE_DIR" \
    --n-per-source "$N_PER_SOURCE" \
    --dev-per-source "$DEV_PER_SOURCE"
  TRAIN_PATH="${SLICE_DIR}/poc-train.jsonl"
  VALID_PATH="${SLICE_DIR}/poc-dev.jsonl"
elif [[ -n "$DATA_PATH" ]]; then
  if [[ ! -d "$DATA_PATH" ]]; then
    echo "data directory not found: $DATA_PATH" >&2
    exit 1
  fi
  if [[ -f "$DATA_PATH/train.jsonl" ]]; then
    TRAIN_PATH="$DATA_PATH/train.jsonl"
    if [[ -f "$DATA_PATH/development.jsonl" ]]; then
      VALID_PATH="$DATA_PATH/development.jsonl"
    elif [[ -f "$DATA_PATH/valid.jsonl" ]]; then
      VALID_PATH="$DATA_PATH/valid.jsonl"
    elif [[ -f "$DATA_PATH/dev.jsonl" ]]; then
      VALID_PATH="$DATA_PATH/dev.jsonl"
    else
      echo "${DATA_PATH} needs development.jsonl (or valid.jsonl / dev.jsonl)" >&2
      exit 1
    fi
  elif [[ -f "$DATA_PATH/manifest.json" ]]; then
    echo "Sampling ${N_PER_SOURCE} records/source from ${DATA_PATH}"
    python3 examples/decision/prepare_v7_sample.py \
      --suite "$DATA_PATH" \
      --out-dir "$SLICE_DIR" \
      --n-per-source "$N_PER_SOURCE" \
      --dev-per-source "$DEV_PER_SOURCE"
    TRAIN_PATH="${SLICE_DIR}/poc-train.jsonl"
    VALID_PATH="${SLICE_DIR}/poc-dev.jsonl"
  else
    echo "${DATA_PATH} needs train.jsonl or manifest.json" >&2
    exit 1
  fi
else
  echo "Set SUITE_PATH, DATA_PATH, or TRAIN_PATH + VALID_PATH to your local decision data." >&2
  exit 1
fi

TRAIN_ARGS=(
  --config examples/decision/qwen3_5_decision_sft_npu_poc.yaml
  "experiment_name=${EXPERIMENT_NAME}"
  "trial_name=${TRIAL_NAME}"
  "scheduler.type=local"
  "cluster.n_nodes=1"
  "cluster.n_gpus_per_node=1"
  "cluster.fileroot=${FILEROOT}"
  "cluster.name_resolve.nfs_record_root=${NAME_RESOLVE}"
  "actor.path=${MODEL_PATH}"
  "tokenizer_path=${MODEL_PATH}"
  "actor.backend=megatron:d1"
  "total_train_epochs=${EPOCHS}"
  "train_dataset.path=${TRAIN_PATH}"
  "valid_dataset.path=${VALID_PATH}"
)
if [[ -n "${STEPS}" ]]; then
  TRAIN_ARGS+=(
    "total_train_steps=${STEPS}"
    "saver.freq_steps=${SAVE_STEPS:-$STEPS}"
    "evaluator.freq_steps=${EVAL_STEPS:-$STEPS}"
  )
  echo "Training up to ${STEPS} steps, ${EPOCHS} epoch cap, NPU ${ASCEND_RT_VISIBLE_DEVICES}"
else
  echo "Training ${EPOCHS} epochs on NPU ${ASCEND_RT_VISIBLE_DEVICES} from ${MODEL_PATH}"
fi
python3 examples/decision/train_decision.py "${TRAIN_ARGS[@]}"

echo "Done. Logs: ${FILEROOT}/logs/${USER:-$(id -un)}/${EXPERIMENT_NAME}/${TRIAL_NAME}"
echo "Checkpoints: ${FILEROOT}/checkpoints/${USER:-$(id -un)}/${EXPERIMENT_NAME}/${TRIAL_NAME}"
