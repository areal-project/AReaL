#!/usr/bin/env bash
# PointerHead decision SFT on one or more Ascend NPUs.
#
# Same recipe as examples/decision/qwen3_5_decision_sft_npu_poc.yaml
# (LoRA 16, lr 5e-5, head_dim 256, max_state 384). PARALLEL picks the
# Megatron layout. World size is DP * TP * PP * CP * EP and must match
# the number of device ids in DEVICES.
#
# Layouts for Qwen3.5-4B-Base (32 layers, 16 query heads, 4 KV heads,
# dense hybrid full-attention + Gated DeltaNet, no MoE experts):
#
#   PARALLEL   backend              NPUs   what it runs
#   single     megatron:d1          1      data parallel 1
#   dp2        megatron:d2          2      data parallel
#   dp4        megatron:d4          4      data parallel
#   tp2 / sp   megatron:d1t2        2      tensor parallel; sequence parallel on
#   tp4        megatron:d1t4        4      tensor parallel; sequence parallel on
#   cp2        megatron:d1c2        2      context parallel
#   cp4        megatron:d1c4        4      context parallel
#   pp2/pp4/vpp                     --      unsupported for PointerHead; see README
#   d4t2       megatron:d4p1t2      8      template in qwen3_5_decision_sft.yaml
#
# Sequence parallel has no letter of its own. MegatronEngine enables it
# whenever tensor parallel is greater than 1. Expert parallel (e) is for
# MoE checkpoints; this 4B model has no experts, so e stays 1.
# TP must divide the 4 KV heads, so 2 and 4 are the sizes to use.
# PP and PP*VPP must divide the 32 layers.
#
#   MODEL_PATH=/path/to/Qwen3.5-4B-Base \
#   SUITE_PATH=/path/to/decision-suite \
#   N_PER_SOURCE=8 DEV_PER_SOURCE=2 \
#   PARALLEL=single DEVICES=0 STEPS=24 EPOCHS=1 \
#     bash examples/decision/run_npu_parallel.sh
#
#   PARALLEL=dp2 DEVICES=0,1 STEPS=2 \
#     bash examples/decision/run_npu_parallel.sh
#
#   PARALLEL=tp2 DEVICES=0,1 STEPS=2 \
#     bash examples/decision/run_npu_parallel.sh
#
#   PARALLEL=d4t2 DEVICES=0,1,2,3,4,5,6,7 STEPS=2 \
#     bash examples/decision/run_npu_parallel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
# Worker health checks use http://<route-ip>:<port>. With HTTP_PROXY set, that
# request leaves the node unless the route IP, hostname, and loopback are in
# NO_PROXY. The rpc_server process inherits this environment.
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
SAVE_STEPS="${SAVE_STEPS:-}"
EVAL_STEPS="${EVAL_STEPS:-}"
N_PER_SOURCE="${N_PER_SOURCE:-32}"
DEV_PER_SOURCE="${DEV_PER_SOURCE:-4}"
BATCH="${BATCH:-4}"
SLICE_DIR="${SLICE_DIR:-${ROOT}/examples/decision/data}"
PARALLEL="${PARALLEL:-single}"
DEVICES="${DEVICES:-}"
MB_TOKENS="${MB_TOKENS:-}"
VPP="${VPP:-}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "Set MODEL_PATH to a local Qwen3.5 checkpoint or a Hugging Face repo id." >&2
  exit 1
fi
if [[ "$MODEL_PATH" == /* || "$MODEL_PATH" == .* ]] && [[ ! -e "$MODEL_PATH" ]]; then
  echo "model path not found: $MODEL_PATH" >&2
  exit 1
fi

case "$PARALLEL" in
  single|d1) BACKEND="megatron:d1" ;;
  dp|dp2) BACKEND="megatron:d2" ;;
  dp4) BACKEND="megatron:d4" ;;
  tp|tp2|sp|sp2) BACKEND="megatron:d1t2" ;;
  tp4) BACKEND="megatron:d1t4" ;;
  cp|cp2) BACKEND="megatron:d1c2" ;;
  cp4) BACKEND="megatron:d1c4" ;;
  pp|pp2) BACKEND="megatron:d1p2" ;;
  pp4) BACKEND="megatron:d1p4" ;;
  vpp) BACKEND="megatron:d1p2"; VPP="${VPP:-2}" ;;
  d4t2|d4p1t2) BACKEND="megatron:d4p1t2" ;;
  megatron:*) BACKEND="$PARALLEL" ;;
  *)
    echo "Unknown PARALLEL=${PARALLEL}." >&2
    echo "Use single, dp2, dp4, tp2, tp4, cp2, cp4, pp2, pp4, vpp, d4t2," >&2
    echo "or a megatron: backend string such as megatron:d2t2." >&2
    exit 1
    ;;
esac

PARSED="$(python3 - "$BACKEND" <<'PY'
import re
import sys

backend = sys.argv[1]
spec = backend.split(":", 1)[1]
if any(token in spec for token in ("attn", "ffn", "|")):
    raise SystemExit(
        "hybrid attn|ffn backends are for MoE folding; this recipe uses dense Qwen3.5-4B"
    )
sizes = {kind: 1 for kind in "dtpce"}
for kind, raw in re.findall(r"([dtpce])([1-9][0-9]*)", spec):
    sizes[kind] *= int(raw)
world = 1
for value in sizes.values():
    world *= value
print(world, sizes["d"], sizes["t"], sizes["p"], sizes["c"], sizes["e"])
PY
)"
read -r WORLD DP TP PP CP EP <<< "$PARSED"

if [[ "$EP" -gt 1 ]]; then
  echo "Expert parallel (${EP}) does not apply to dense Qwen3.5-4B-Base." >&2
  exit 1
fi
if [[ "$TP" -gt 4 ]]; then
  echo "Tensor parallel ${TP} does not divide Qwen3.5-4B's 4 KV heads." >&2
  exit 1
fi
if [[ "$VPP" == "" ]]; then
  VPP=1
fi
if [[ "$VPP" -gt 1 && "$PP" -le 1 ]]; then
  echo "Virtual pipeline parallel requires pipeline parallel greater than 1." >&2
  exit 1
fi
if (( 32 % (PP * VPP) != 0 )); then
  echo "PP=${PP} VPP=${VPP} does not divide Qwen3.5-4B's 32 layers." >&2
  exit 1
fi
if [[ "$PP" -gt 1 ]]; then
  echo "Pipeline parallel is not supported for PointerHead decision SFT." >&2
  echo "See examples/decision/README.md (Pipeline parallel)." >&2
  exit 1
fi

if [[ -z "$DEVICES" ]]; then
  DEVICES="$(seq -s, 0 $((WORLD - 1)))"
fi
IFS=',' read -ra DEVICE_IDS <<< "$DEVICES"
if [[ "${#DEVICE_IDS[@]}" -ne "$WORLD" ]]; then
  echo "${BACKEND} needs ${WORLD} NPUs; DEVICES=${DEVICES} has ${#DEVICE_IDS[@]}." >&2
  exit 1
fi
export ASCEND_RT_VISIBLE_DEVICES="$DEVICES"

if [[ "$BATCH" -lt "$DP" ]]; then
  echo "Raising batch size from ${BATCH} to data-parallel size ${DP}."
  BATCH="$DP"
fi
if [[ "$PP" -gt 1 && -z "$MB_TOKENS" ]]; then
  MB_TOKENS=1024
fi

SAFE_NAME="$(printf '%s' "$PARALLEL" | tr -c 'A-Za-z0-9' '-')"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen35-decision-npu-${SAFE_NAME}}"
TRIAL_NAME="${TRIAL_NAME:-${SAFE_NAME}}"

if [[ -n "$TRAIN_PATH" ]]; then
  if [[ ! -f "$TRAIN_PATH" || ! -f "${VALID_PATH:-}" ]]; then
    echo "Set TRAIN_PATH and VALID_PATH to existing JSONL files." >&2
    exit 1
  fi
elif [[ -n "$SUITE_PATH" ]]; then
  if [[ ! -d "$SUITE_PATH" ]]; then
    echo "decision suite not found: $SUITE_PATH" >&2
    exit 1
  fi
  echo "Sampling ${N_PER_SOURCE} train / ${DEV_PER_SOURCE} dev records per source"
  python3 examples/decision/prepare_v7_sample.py \
    --suite "$SUITE_PATH" \
    --out-dir "$SLICE_DIR" \
    --n-per-source "$N_PER_SOURCE" \
    --dev-per-source "$DEV_PER_SOURCE"
  TRAIN_PATH="${SLICE_DIR}/poc-train.jsonl"
  VALID_PATH="${SLICE_DIR}/poc-dev.jsonl"
elif [[ -n "$DATA_PATH" ]]; then
  if [[ -f "$DATA_PATH/train.jsonl" ]]; then
    TRAIN_PATH="$DATA_PATH/train.jsonl"
    if [[ -f "$DATA_PATH/development.jsonl" ]]; then
      VALID_PATH="$DATA_PATH/development.jsonl"
    elif [[ -f "$DATA_PATH/valid.jsonl" ]]; then
      VALID_PATH="$DATA_PATH/valid.jsonl"
    else
      echo "${DATA_PATH} needs development.jsonl or valid.jsonl" >&2
      exit 1
    fi
  elif [[ -f "$DATA_PATH/manifest.json" ]]; then
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
  echo "Set SUITE_PATH, DATA_PATH, or TRAIN_PATH + VALID_PATH." >&2
  exit 1
fi

echo "PARALLEL=${PARALLEL} backend=${BACKEND} devices=${DEVICES} batch=${BATCH} dp=${DP} tp=${TP} pp=${PP} cp=${CP} vpp=${VPP}"
if [[ "$TP" -gt 1 ]]; then
  echo "Sequence parallel is enabled because tensor parallel is ${TP}."
fi

TRAIN_ARGS=(
  --config examples/decision/qwen3_5_decision_sft_npu_poc.yaml
  "experiment_name=${EXPERIMENT_NAME}"
  "trial_name=${TRIAL_NAME}"
  "scheduler.type=local"
  "cluster.n_nodes=1"
  "cluster.n_gpus_per_node=${WORLD}"
  "cluster.fileroot=${FILEROOT}"
  "cluster.name_resolve.nfs_record_root=${NAME_RESOLVE}"
  "actor.path=${MODEL_PATH}"
  "tokenizer_path=${MODEL_PATH}"
  "actor.backend=${BACKEND}"
  "total_train_epochs=${EPOCHS}"
  "train_dataset.batch_size=${BATCH}"
  "valid_dataset.batch_size=${BATCH}"
  "train_dataset.path=${TRAIN_PATH}"
  "valid_dataset.path=${VALID_PATH}"
)
if [[ -n "$MB_TOKENS" ]]; then
  TRAIN_ARGS+=("actor.mb_spec.max_tokens_per_mb=${MB_TOKENS}")
fi
if [[ "$VPP" -gt 1 ]]; then
  TRAIN_ARGS+=("actor.megatron.virtual_pipeline_parallel_size=${VPP}")
fi
if [[ -n "$STEPS" ]]; then
  TRAIN_ARGS+=(
    "total_train_steps=${STEPS}"
    "saver.freq_steps=${SAVE_STEPS:-$STEPS}"
    "evaluator.freq_steps=${EVAL_STEPS:-$STEPS}"
  )
  echo "Training up to ${STEPS} steps, ${EPOCHS} epoch cap"
else
  echo "Training ${EPOCHS} epochs"
fi

python3 examples/decision/train_decision.py "${TRAIN_ARGS[@]}"

echo "Done. Logs: ${FILEROOT}/logs/${USER:-$(id -un)}/${EXPERIMENT_NAME}/${TRIAL_NAME}"
echo "Checkpoints: ${FILEROOT}/checkpoints/${USER:-$(id -un)}/${EXPERIMENT_NAME}/${TRIAL_NAME}"
