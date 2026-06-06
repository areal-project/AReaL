#!/usr/bin/env bash
set -euo pipefail

# Test-time scaling pipeline for JSONL pairwise datasets.
#
# Starts a new-encoding vLLM server (pairwise comparison) and an eval vLLM
# server (grading), runs the experiment multiple times, and averages pass@1.
#
# Usage:
#   bash TTS_JSONL/run_jsonl_pipeline.sh \
#     --jsonl-file pairwise_from_direct_generation_AIME25.jsonl \
#     --ground-truth-file LLM_Test-time_Scaling/imobench.json \
#     --model-path /path/to/new-encoding-model \
#     --model-name qwen3-4b-newenc
#
# Example:
#   bash TTS_JSONL/run_jsonl_pipeline.sh \
#     --jsonl-file pairwise_from_direct_generation_AIME25.jsonl \
#     --ground-truth-file LLM_Test-time_Scaling/imobench.json \
#     --model-path /storage/openpsi/users/zzy/train_new_encoding/Multiverse/ckpts/Qwen3Chunked-20260418_145450 \
#     --model-name qwen3-4b-newenc \
#     --num-runs 8

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------
# Defaults
# -----------------------------
JSONL_FILE=""
GROUND_TRUTH_FILE=""
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

EVAL_MODEL_PATH="/storage/openpsi/models/Qwen__Qwen3-8B"
EVAL_MODEL_NAME="qwen3-8b-eval"
EVAL_PORT=8001
EVAL_TENSOR_PARALLEL_SIZE=4
EVAL_GPUS="4,5,6,7"

MAX_CONCURRENT=40
NUM_RUNS=8
TEMPERATURE=0.7
MAX_TOKENS=16384
REQUEST_TIMEOUT=600

SERVER_PID=""
EVAL_SERVER_PID=""

# -----------------------------
# Functions
# -----------------------------
usage() {
  cat <<'EOF'
Usage:
  bash TTS_JSONL/run_jsonl_pipeline.sh [options]

Required:
  --jsonl-file FILE              Pairwise JSONL file
  --ground-truth-file FILE       Ground truth JSON (imobench.json or direct_generation JSON)
  --model-path PATH              Path to the new-encoding model
  --model-name NAME              Served model name (e.g. qwen3-4b-newenc)

Experiment options:
  --num-runs N                   Number of runs to average (default: 8)
  --max-concurrent N             Max concurrent requests (default: 40)
  --temperature FLOAT            Sampling temperature (default: 0.7)
  --max-tokens N                 Max tokens for generation (default: 16384)
  --request-timeout N            HTTP request timeout in seconds (default: 600)

Aggregation server options (new-encoding model, pairwise comparison):
  --host HOST                    Bind host (default: 0.0.0.0)
  --port PORT                    Server port (default: 8000)
  --tensor-parallel-size N       Tensor parallel size (default: 4)
  --agg-gpus GPUS                GPU indices (default: 0,1,2,3)
  --server-startup-timeout N     Seconds to wait for readiness (default: 300)

Evaluation server options (grading model):
  --eval-model-path PATH         Path to evaluation model
                                 (default: /storage/openpsi/models/Qwen__Qwen3-8B)
  --eval-model-name NAME         Served eval model name (default: qwen3-8b-eval)
  --eval-port PORT               Evaluation server port (default: 8001)
  --eval-tensor-parallel-size N  Tensor parallel size for eval (default: 4)
  --eval-gpus GPUS               GPU indices for eval server (default: 4,5,6,7)

New-encoding options:
  --new-env-activate PATH        Activation script for new-encoding env
  --new-hf-overrides JSON        HF overrides for the new-encoding server
  --new-no-trust-remote-code     Disable --trust-remote-code

Other:
  -h, --help                     Show this help
EOF
}

cleanup_server() {
  if [[ -n "$SERVER_PID" ]]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Stopping pairwise server (pid=$SERVER_PID)..."
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

  echo "Starting new-encoding (pairwise) server: $MODEL_NAME -> $MODEL_PATH (GPUs: $AGG_GPUS)"
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

extract_pass1() {
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

# -----------------------------
# Argument parsing
# -----------------------------
trap cleanup_server EXIT INT TERM

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jsonl-file)
      JSONL_FILE="$2"
      shift 2
      ;;
    --ground-truth-file)
      GROUND_TRUTH_FILE="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --num-runs)
      NUM_RUNS="$2"
      shift 2
      ;;
    --max-concurrent)
      MAX_CONCURRENT="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --request-timeout)
      REQUEST_TIMEOUT="$2"
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
if [[ -z "$JSONL_FILE" ]]; then
  echo "Error: --jsonl-file is required" >&2
  usage
  exit 1
fi

if [[ ! -f "$JSONL_FILE" ]]; then
  echo "Error: JSONL file not found: $JSONL_FILE" >&2
  exit 1
fi

if [[ -z "$GROUND_TRUTH_FILE" ]]; then
  echo "Error: --ground-truth-file is required" >&2
  usage
  exit 1
fi

if [[ ! -f "$GROUND_TRUTH_FILE" ]]; then
  echo "Error: ground truth file not found: $GROUND_TRUTH_FILE" >&2
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

if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [[ "$NUM_RUNS" -le 0 ]]; then
  echo "Error: --num-runs must be a positive integer, got: $NUM_RUNS" >&2
  exit 1
fi

# -----------------------------
# Environment
# -----------------------------
export OPENAI_API_BASE="http://127.0.0.1:${SERVER_PORT}/v1"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="dummy"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="$ROOT_DIR/results/tts_jsonl"
mkdir -p "$OUTPUT_BASE"
SUMMARY_CSV="$OUTPUT_BASE/summary_${TIMESTAMP}.csv"

echo "jsonl_file,model_name,pass_at_1_avg,num_runs,output_dir" \
  > "$SUMMARY_CSV"

# -----------------------------
# Print config
# -----------------------------
echo "========================================"
echo "JSONL Pairwise Test-Time Scaling"
echo "========================================"
echo "JSONL file: $JSONL_FILE"
echo "Ground truth: $GROUND_TRUTH_FILE"
echo "--- Pairwise comparison ---"
echo "  Model: $MODEL_NAME -> $MODEL_PATH"
echo "  Server: $SERVER_HOST:$SERVER_PORT (GPUs: $AGG_GPUS, TP: $TENSOR_PARALLEL_SIZE)"
echo "--- Evaluation (grading) ---"
echo "  Model: $EVAL_MODEL_NAME -> $EVAL_MODEL_PATH"
echo "  Server: $SERVER_HOST:$EVAL_PORT (GPUs: $EVAL_GPUS, TP: $EVAL_TENSOR_PARALLEL_SIZE)"
echo "--- Settings ---"
echo "  Num runs: $NUM_RUNS"
echo "  Temperature: $TEMPERATURE"
echo "  Max concurrent: $MAX_CONCURRENT"
echo "  Max tokens: $MAX_TOKENS"
echo "========================================"

# -----------------------------
# Start both servers
# -----------------------------
start_new_encoding_server
start_eval_server

# -----------------------------
# Run experiments
# -----------------------------
RUN_OUTPUT_DIR="$OUTPUT_BASE/runs/${MODEL_NAME}/${TIMESTAMP}"
mkdir -p "$RUN_OUTPUT_DIR"

RUN_VALUES=()

for ((run_idx=1; run_idx<=NUM_RUNS; run_idx++)); do
  echo
  echo "----------------------------------------"
  echo "Run ${run_idx}/${NUM_RUNS}"
  echo "----------------------------------------"

  (
    cd "$ROOT_DIR"
    python3 "$SCRIPT_DIR/run_jsonl_experiment.py" \
      --jsonl-file "$JSONL_FILE" \
      --ground-truth-file "$GROUND_TRUTH_FILE" \
      --model-name "$MODEL_NAME" \
      --api-base "http://127.0.0.1:${SERVER_PORT}/v1" \
      --eval-model-name "$EVAL_MODEL_NAME" \
      --eval-api-base "http://127.0.0.1:${EVAL_PORT}/v1" \
      --output-dir "$RUN_OUTPUT_DIR" \
      --temperature "$TEMPERATURE" \
      --max-tokens "$MAX_TOKENS" \
      --max-concurrent "$MAX_CONCURRENT" \
      --request-timeout "$REQUEST_TIMEOUT"
  )

  RESULT_FILE="$(get_latest_file "$RUN_OUTPUT_DIR" "jsonl_pairwise_experiment_*.json")"
  if [[ -z "$RESULT_FILE" ]]; then
    echo "  Warning: result file not found under $RUN_OUTPUT_DIR -- skipping run" >&2
    continue
  fi
  RUN_PASS1="$(extract_pass1 "$RESULT_FILE")"
  RUN_VALUES+=("$RUN_PASS1")
  echo "  Run $run_idx pass@1: $RUN_PASS1"
done

PASS1_AVG="$(average_values "${RUN_VALUES[@]}")"
echo "$JSONL_FILE,$MODEL_NAME,$PASS1_AVG,$NUM_RUNS,$RUN_OUTPUT_DIR" >> "$SUMMARY_CSV"

# -----------------------------
# Summary
# -----------------------------
cleanup_server

echo
echo "========================================"
echo "Results summary"
echo "========================================"
echo "JSONL: $(basename "$JSONL_FILE")"
echo "Model: $MODEL_NAME"
echo "Pass@1 avg (${NUM_RUNS} runs): $PASS1_AVG"
echo
for i in "${!RUN_VALUES[@]}"; do
  echo "  Run $((i+1)): ${RUN_VALUES[$i]}"
done
echo
echo "Summary CSV: $SUMMARY_CSV"
echo "Run outputs: $RUN_OUTPUT_DIR"
echo "Done."
