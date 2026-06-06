#!/usr/bin/env bash
set -euo pipefail

# Standalone aggregation runner for new-encoding models using vllm_raw provider.
#
# Like run_aggregation_new_encoding_vllm_direct.sh, but uses the vllm_raw
# provider which manually constructs the prompt with <|im_start|>/<|im_end|>
# markers (identical to the training data format) and sends it via
# requests.post() to /v1/completions — the same approach as
# test_training_data_on_vllm.py.
#
# Usage:
#   bash scripts/run_aggregation_new_encoding_vllm_raw.sh \
#     --result-files /path/to/direct_generation_result1.json [result2.json ...] \
#     --model-path /path/to/new-encoding-model \
#     --model-name qwen3-4b-newenc
#
# Example:
#   bash scripts/run_aggregation_new_encoding_vllm_raw.sh \
#     --result-files results/test_time_compute/imobench_AIME25_rollouts_qwen3-4b/direct_generation_imobench_AIME25_qwen3-4b_20260419.json \
#     --model-path /storage/openpsi/users/zzy/train_new_encoding/Multiverse/ckpts/Qwen3Chunked-20260418_145450 \
#     --model-name qwen3-4b-newenc \
#     --num-agg-runs 8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -----------------------------
# Defaults
# -----------------------------
RESULT_FILES=()
MODEL_PATH=""
MODEL_NAME=""

SERVER_HOST="0.0.0.0"
SERVER_PORT=8000
TENSOR_PARALLEL_SIZE=4
SERVER_STARTUP_TIMEOUT=300
AGG_GPUS="0,1,2,3"

NEW_ENV_ACTIVATE="/storage/openpsi/users/zzy/.zzy-enc-2/bin/activate"
NEW_USE_TRUST_REMOTE_CODE=true
NEW_HF_OVERRIDES='{"architectures": ["Qwen3ChunkedForCausalLM"], "chunk_start_token_id": 151669, "chunk_end_token_id": 151670}'

# Evaluation/grading server defaults
EVAL_MODEL_PATH="/storage/openpsi/models/Qwen__Qwen3-8B"
EVAL_MODEL_NAME="qwen3-8b-eval"
EVAL_PORT=8001
EVAL_TENSOR_PARALLEL_SIZE=4
EVAL_GPUS="4,5,6,7"
EVAL_PROVIDER="vllm_raw"

PAIRWISE_TEMPLATE="aggregation_pairwise_new_encoding_comparison"
MAX_CONCURRENT_AGG=40
NUM_AGG_RUNS=8

RESUME=false
RESUME_FROM=""

SERVER_PID=""
EVAL_SERVER_PID=""

# -----------------------------
# Functions
# -----------------------------
usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_aggregation_new_encoding_vllm_raw.sh [options]

Required:
  --result-files FILE [FILE ...]   One or more generation result JSON files
  --model-path PATH                Path to the new-encoding model
  --model-name NAME                Served model name (e.g. qwen3-4b-newenc)

Aggregation options:
  --num-agg-runs N                 Number of aggregation runs to average (default: 8)
  --max-concurrent-agg N           Max concurrent problems (default: 40)
  --pairwise-template NAME         Pairwise template name
                                   (default: aggregation_pairwise_new_encoding_comparison)

Aggregation server options (new-encoding model, pairwise comparison):
  --host HOST                      Bind host (default: 0.0.0.0)
  --port PORT                      Aggregation server port (default: 8000)
  --tensor-parallel-size N         Tensor parallel size (default: 4)
  --agg-gpus GPUS                  GPU indices for aggregation server (default: 0,1,2,3)
  --server-startup-timeout N       Seconds to wait for readiness (default: 300)

Evaluation server options (grading model):
  --eval-model-path PATH           Path to evaluation model
                                   (default: /storage/openpsi/models/Qwen__Qwen3-8B)
  --eval-model-name NAME           Served eval model name (default: qwen3-8b-eval)
  --eval-port PORT                 Evaluation server port (default: 8001)
  --eval-tensor-parallel-size N    Tensor parallel size for eval (default: 4)
  --eval-gpus GPUS                 GPU indices for eval server (default: 4,5,6,7)
  --eval-provider PROVIDER         Provider for eval server (default: vllm_raw)

New-encoding options:
  --new-env-activate PATH          Activation script for new-encoding env
                                   (default: /storage/openpsi/users/zzy/.zzy-enc-2/bin/activate)
  --new-hf-overrides JSON          HF overrides for the new-encoding server
  --new-no-trust-remote-code       Disable --trust-remote-code

Other:
  --resume                         Enable resume mode
  --resume-from FILE               Resume from specific result file
  -h, --help                       Show this help

Note: This script uses the vllm_raw provider which manually constructs the
prompt with <|im_start|>/<|im_end|> markers (matching the training data format)
and sends it via requests.post() to /v1/completions — identical to the approach
used by test_training_data_on_vllm.py.
EOF
}

cleanup_server() {
  if [[ -n "$SERVER_PID" ]]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Stopping aggregation server (pid=$SERVER_PID)..."
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
  fi
  if [[ -n "$EVAL_SERVER_PID" ]]; then
    if kill -0 "$EVAL_SERVER_PID" 2>/dev/null; then
      echo "Stopping eval server (pid=$EVAL_SERVER_PID)..."
      kill "$EVAL_SERVER_PID" 2>/dev/null || true
      wait "$EVAL_SERVER_PID" 2>/dev/null || true
    fi
    EVAL_SERVER_PID=""
  fi
}

kill_existing_service_on_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Killing existing service(s) on port $port: $pids"
    kill $pids 2>/dev/null || true
    sleep 2
    pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      echo "Force-killing stubborn service(s) on port $port: $pids"
      kill -9 $pids 2>/dev/null || true
      sleep 1
    fi
  fi
}

wait_for_server_ready() {
  local port="$1"
  local pid="$2"
  local start_ts elapsed
  start_ts="$(date +%s)"
  echo "Waiting for server at http://127.0.0.1:${port}/v1/models ..."

  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "Server on port $port is ready."
      return 0
    fi

    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "Error: server exited before becoming ready." >&2
      return 1
    fi

    elapsed=$(( $(date +%s) - start_ts ))
    if (( elapsed >= SERVER_STARTUP_TIMEOUT )); then
      echo "Error: timed out waiting for server after ${SERVER_STARTUP_TIMEOUT}s." >&2
      return 1
    fi
    sleep 2
  done
}

start_new_encoding_server() {
  kill_existing_service_on_port "$SERVER_PORT"

  echo "Starting new-encoding (aggregation) server: $MODEL_NAME -> $MODEL_PATH (GPUs: $AGG_GPUS)"
  (
    cd "$ROOT_DIR"

    if [[ ! -f "$NEW_ENV_ACTIVATE" ]]; then
      echo "Error: new-encoding env activation script not found: $NEW_ENV_ACTIVATE" >&2
      exit 1
    fi

    source "$NEW_ENV_ACTIVATE"

    export CUDA_VISIBLE_DEVICES="$AGG_GPUS"

    cmd=(
      python -m vllm.entrypoints.openai.api_server
      --host "$SERVER_HOST"
      --port "$SERVER_PORT"
      --model "$MODEL_PATH"
      --tokenizer "$MODEL_PATH"
      --served-model-name "$MODEL_NAME"
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
      --hf-overrides "$NEW_HF_OVERRIDES"
    )

    if [[ "$NEW_USE_TRUST_REMOTE_CODE" == true ]]; then
      cmd+=(--trust-remote-code)
    fi

    "${cmd[@]}" &
    local child_pid="$!"

    deactivate >/dev/null 2>&1 || true

    trap 'kill "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true' TERM INT EXIT
    wait "$child_pid"
  ) &
  SERVER_PID=$!

  wait_for_server_ready "$SERVER_PORT" "$SERVER_PID"
}

start_eval_server() {
  kill_existing_service_on_port "$EVAL_PORT"

  echo "Starting eval (grading) server: $EVAL_MODEL_NAME -> $EVAL_MODEL_PATH (GPUs: $EVAL_GPUS)"
  (
    cd "$ROOT_DIR"

    if [[ ! -f "$NEW_ENV_ACTIVATE" ]]; then
      echo "Error: env activation script not found: $NEW_ENV_ACTIVATE" >&2
      exit 1
    fi

    source "$NEW_ENV_ACTIVATE"

    export CUDA_VISIBLE_DEVICES="$EVAL_GPUS"

    cmd=(
      python -m vllm.entrypoints.openai.api_server
      --host "$SERVER_HOST"
      --port "$EVAL_PORT"
      --model "$EVAL_MODEL_PATH"
      --tokenizer "$EVAL_MODEL_PATH"
      --served-model-name "$EVAL_MODEL_NAME"
      --tensor-parallel-size "$EVAL_TENSOR_PARALLEL_SIZE"
    )

    "${cmd[@]}" &
    local child_pid="$!"

    deactivate >/dev/null 2>&1 || true

    trap 'kill "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true' TERM INT EXIT
    wait "$child_pid"
  ) &
  EVAL_SERVER_PID=$!

  wait_for_server_ready "$EVAL_PORT" "$EVAL_SERVER_PID"
}

extract_aggregation_pass1() {
  local json_file="$1"
  python3 - "$json_file" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
val = data.get("aggregate_metrics", {}).get("pass@1")
print("nan" if val is None else f"{float(val):.6f}")
PY
}

average_values() {
  if [[ $# -eq 0 ]]; then
    echo "nan"
    return 0
  fi
  python3 - "$@" <<'PY'
import math, sys
vals = []
for x in sys.argv[1:]:
    try:
        v = float(x)
    except Exception:
        continue
    if not math.isnan(v):
        vals.append(v)
print("nan" if not vals else f"{sum(vals)/len(vals):.6f}")
PY
}

get_latest_file() {
  local dir="$1"
  local pattern="$2"

  if [[ ! -d "$dir" ]]; then
    echo ""
    return 0
  fi

  shopt -s nullglob
  local matches=("$dir"/$pattern)
  shopt -u nullglob

  if [[ ${#matches[@]} -eq 0 ]]; then
    echo ""
    return 0
  fi

  ls -1t "${matches[@]}" | head -n1
}

run_aggregation() {
  local result_file="$1"
  local output_dir="$2"
  local request_model="$3"

  local cmd=(
    python3 -m scripts.run_aggregation_experiment
    --result-files "$result_file"
    --output-dir "$output_dir"
    --model_name "$request_model"
    --pairwise-template "$PAIRWISE_TEMPLATE"
    --max-concurrent "$MAX_CONCURRENT_AGG"
    --provider vllm_raw
    --eval-model-name "$EVAL_MODEL_NAME"
    --eval-api-base "http://127.0.0.1:${EVAL_PORT}/v1"
    --eval-provider "$EVAL_PROVIDER"
  )

  if [[ "$RESUME" == true ]]; then
    cmd+=(--resume)
  fi
  if [[ -n "$RESUME_FROM" ]]; then
    cmd+=(--resume-from "$RESUME_FROM")
  fi

  (
    cd "$ROOT_DIR"
    "${cmd[@]}"
  )
}

# -----------------------------
# Argument parsing
# -----------------------------
trap cleanup_server EXIT INT TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
    --result-files)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        RESULT_FILES+=("$1")
        shift
      done
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --num-agg-runs)
      NUM_AGG_RUNS="$2"
      shift 2
      ;;
    --max-concurrent-agg)
      MAX_CONCURRENT_AGG="$2"
      shift 2
      ;;
    --pairwise-template)
      PAIRWISE_TEMPLATE="$2"
      shift 2
      ;;
    --host)
      SERVER_HOST="$2"
      shift 2
      ;;
    --port)
      SERVER_PORT="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --agg-gpus)
      AGG_GPUS="$2"
      shift 2
      ;;
    --server-startup-timeout)
      SERVER_STARTUP_TIMEOUT="$2"
      shift 2
      ;;
    --eval-model-path)
      EVAL_MODEL_PATH="$2"
      shift 2
      ;;
    --eval-model-name)
      EVAL_MODEL_NAME="$2"
      shift 2
      ;;
    --eval-port)
      EVAL_PORT="$2"
      shift 2
      ;;
    --eval-tensor-parallel-size)
      EVAL_TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --eval-gpus)
      EVAL_GPUS="$2"
      shift 2
      ;;
    --eval-provider)
      EVAL_PROVIDER="$2"
      shift 2
      ;;
    --new-env-activate)
      NEW_ENV_ACTIVATE="$2"
      shift 2
      ;;
    --new-hf-overrides)
      NEW_HF_OVERRIDES="$2"
      shift 2
      ;;
    --new-no-trust-remote-code)
      NEW_USE_TRUST_REMOTE_CODE=false
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# -----------------------------
# Validation
# -----------------------------
if [[ ${#RESULT_FILES[@]} -eq 0 ]]; then
  echo "Error: --result-files is required" >&2
  usage
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  echo "Error: --model-path is required" >&2
  usage
  exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
  echo "Error: --model-name is required" >&2
  usage
  exit 1
fi

if ! [[ "$NUM_AGG_RUNS" =~ ^[0-9]+$ ]] || [[ "$NUM_AGG_RUNS" -le 0 ]]; then
  echo "Error: --num-agg-runs must be a positive integer, got: $NUM_AGG_RUNS" >&2
  exit 1
fi

for rf in "${RESULT_FILES[@]}"; do
  if [[ ! -f "$rf" ]]; then
    echo "Error: result file not found: $rf" >&2
    exit 1
  fi
done

# -----------------------------
# Environment
# -----------------------------
export OPENAI_API_BASE="http://127.0.0.1:${SERVER_PORT}/v1"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="dummy"
fi

# vllm_raw uses the model name directly (no openai/ prefix)
REQUEST_MODEL="$MODEL_NAME"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="$ROOT_DIR/results/test_time_compute/aggregation_new_encoding_vllm_raw"
mkdir -p "$OUTPUT_BASE"
SUMMARY_CSV="$OUTPUT_BASE/summary_${TIMESTAMP}.csv"

echo "result_file,model_name,pairwise_template,aggregation_pairwise_pass_at_1_avg,aggregation_runs,aggregation_output_dir" \
  > "$SUMMARY_CSV"

# -----------------------------
# Print config
# -----------------------------
echo "========================================"
echo "New-encoding aggregation (vllm_raw)"
echo "========================================"
echo "Result files: ${#RESULT_FILES[@]}"
echo "--- Aggregation (pairwise) ---"
echo "  Model: $MODEL_NAME -> $MODEL_PATH"
echo "  Request model: $REQUEST_MODEL"
echo "  Provider: vllm_raw"
echo "  Server: $SERVER_HOST:$SERVER_PORT (GPUs: $AGG_GPUS, TP: $TENSOR_PARALLEL_SIZE)"
echo "--- Evaluation (grading) ---"
echo "  Model: $EVAL_MODEL_NAME -> $EVAL_MODEL_PATH"
echo "  Provider: $EVAL_PROVIDER"
echo "  Server: $SERVER_HOST:$EVAL_PORT (GPUs: $EVAL_GPUS, TP: $EVAL_TENSOR_PARALLEL_SIZE)"
echo "--- Settings ---"
echo "  Pairwise template: $PAIRWISE_TEMPLATE"
echo "  Aggregation runs: $NUM_AGG_RUNS"
echo "  Max concurrent: $MAX_CONCURRENT_AGG"
echo "  Resume: $RESUME"
if [[ -n "$RESUME_FROM" ]]; then
  echo "  Resume from: $RESUME_FROM"
fi
echo "========================================"

# -----------------------------
# Start both servers
# -----------------------------
start_new_encoding_server
start_eval_server

# -----------------------------
# Run aggregation per result file
# -----------------------------
SUMMARY_ROWS=()

for result_file in "${RESULT_FILES[@]}"; do
  FILE_BASENAME="$(basename "$result_file")"
  FILE_LABEL="${FILE_BASENAME%.json}"

  AGG_OUT_DIR="$OUTPUT_BASE/aggregation/${MODEL_NAME}/${FILE_LABEL}"
  mkdir -p "$AGG_OUT_DIR"

  echo
  echo "----------------------------------------"
  echo "Result file: $result_file"
  echo "Output dir: $AGG_OUT_DIR"
  echo "----------------------------------------"

  AGG_RUN_VALUES=()

  for ((run_idx=1; run_idx<=NUM_AGG_RUNS; run_idx++)); do
    echo "  Aggregation run ${run_idx}/${NUM_AGG_RUNS} ..."
    run_aggregation "$result_file" "$AGG_OUT_DIR" "$REQUEST_MODEL"

    AGG_RESULT_FILE="$(get_latest_file "$AGG_OUT_DIR" "aggregation_experiment_pairwise_comparison_*.json")"
    if [[ -z "$AGG_RESULT_FILE" ]]; then
      echo "  Warning: aggregation result not found under $AGG_OUT_DIR — skipping run" >&2
      continue
    fi
    AGG_RUN_PASS1="$(extract_aggregation_pass1 "$AGG_RESULT_FILE")"
    AGG_RUN_VALUES+=("$AGG_RUN_PASS1")
    echo "  Run $run_idx pass@1: $AGG_RUN_PASS1"
  done

  AGG_PASS1_AVG="$(average_values "${AGG_RUN_VALUES[@]}")"

  SUMMARY_ROWS+=("$result_file|$AGG_PASS1_AVG|$AGG_OUT_DIR")
  echo "$result_file,$MODEL_NAME,$PAIRWISE_TEMPLATE,$AGG_PASS1_AVG,$NUM_AGG_RUNS,$AGG_OUT_DIR" >> "$SUMMARY_CSV"

  echo "  Average pass@1 over $NUM_AGG_RUNS runs: $AGG_PASS1_AVG"
done

# -----------------------------
# Summary
# -----------------------------
cleanup_server

echo
echo "========================================"
echo "Results summary"
echo "========================================"
for row in "${SUMMARY_ROWS[@]}"; do
  IFS='|' read -r RF AGG_P1 AGG_DIR <<< "$row"
  echo "file=$(basename "$RF") | model=$MODEL_NAME | pairwise_pass@1_avg(${NUM_AGG_RUNS}runs)=$AGG_P1"
done
echo
echo "Summary CSV: $SUMMARY_CSV"
echo "Done."
