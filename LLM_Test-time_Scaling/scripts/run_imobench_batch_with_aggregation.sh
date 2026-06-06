#!/usr/bin/env bash
set -euo pipefail

# Batch runner for IMOBench direct generation + pairwise aggregation.
#
# For each model size and each input test file, this script will:
# 1) Run direct generation (no aggregation) with the original-encoding model.
# 2) Run pairwise aggregation with the original-encoding model.
# 3) Run pairwise aggregation with the new-encoding model.
#
# It then prints and saves:
# - no-aggregation accuracies for {size} x {test_file}
# - aggregation accuracies for {size} x {old/new encoding} x {test_file}
#
# Model spec format (configured in this file):
#   size|orig_model_path|orig_model_name|new_model_path|new_model_name
#
# Usage:
#   1) Edit FILES and MODEL_SPECS in the "User configuration" section below.
#   2) Run: bash scripts/run_imobench_batch_with_aggregation.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# -----------------------------
# User configuration
# -----------------------------
# Put your benchmark files here.
FILES=(
  # "/path/to/aime1.json"
  # "/path/to/aime2.json"
)

# Put your models here, one line per size:
# size|orig_model_path|orig_model_name|new_model_path|new_model_name
MODEL_SPECS=(
  # "8b|/storage/models/Qwen__Qwen3-8B|qwen3-8b|/storage/models/Qwen__Qwen3-8B-newenc|qwen3-8b-newenc"
  # "30b|/storage/models/Qwen__Qwen3-30B-A3B|qwen3-30b-a3b|/storage/models/Qwen__Qwen3-30B-A3B-newenc|qwen3-30b-a3b-newenc"
)

VLLM_HOST="0.0.0.0"
VLLM_PORT=8000
TENSOR_PARALLEL_SIZE=8
SERVER_STARTUP_TIMEOUT=300
CONTEXT_LEN=40000

MAX_CONCURRENT_GEN=128
MAX_CONCURRENT_AGG=40
NUM_AGG_RUNS=8

RESUME=false
RESUME_FROM=""

SERVER_PID=""

SUMMARY_ROWS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_imobench_batch_with_aggregation.sh [optional flags]

Configuration:
  Edit FILES and MODEL_SPECS directly in this script.

Optional flags:
  --max-concurrent-gen    Max concurrent problems for generation (default: 128)
  --max-concurrent-agg    Max concurrent problems for aggregation (default: 40)
  --num-agg-runs          Number of aggregation runs to average (default: 8)
  --resume                Enable resume mode for generation + aggregation
  --resume-from FILE      Resume from specific result file where supported
  --host                  vLLM bind host (default: 0.0.0.0)
  --port                  vLLM port (default: 8000)
  --tensor-parallel-size  vLLM tensor parallel size (default: 8)
  --server-startup-timeout Seconds to wait for vLLM readiness (default: 300)
  --context-len           Max model context length (default: 40000)
  -h, --help              Show help
EOF
}

cleanup_server() {
  if [[ -n "$SERVER_PID" ]]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Stopping vLLM server (pid=$SERVER_PID)..."
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
  fi
}

kill_existing_service_on_port() {
  local pids
  pids="$(lsof -ti tcp:"$VLLM_PORT" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Killing existing service(s) on port $VLLM_PORT: $pids"
    kill $pids 2>/dev/null || true
    sleep 2
    pids="$(lsof -ti tcp:"$VLLM_PORT" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      echo "Force-killing stubborn service(s) on port $VLLM_PORT: $pids"
      kill -9 $pids 2>/dev/null || true
      sleep 1
    fi
  fi
}

wait_for_server_ready() {
  local start_ts elapsed
  start_ts="$(date +%s)"
  echo "Waiting for vLLM at http://127.0.0.1:${VLLM_PORT}/v1/models ..."

  while true; do
    if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "vLLM server is ready."
      return 0
    fi

    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Error: vLLM server exited before becoming ready." >&2
      return 1
    fi

    elapsed=$(( $(date +%s) - start_ts ))
    if (( elapsed >= SERVER_STARTUP_TIMEOUT )); then
      echo "Error: timed out waiting for vLLM server after ${SERVER_STARTUP_TIMEOUT}s." >&2
      return 1
    fi
    sleep 2
  done
}

start_server_for_model() {
  local model_name="$1"
  local model_path="$2"

  cleanup_server
  kill_existing_service_on_port

  echo "Starting vLLM server: $model_name -> $model_path"
  (
    cd "$ROOT_DIR"
    vllm serve "$model_path" \
      --host "$VLLM_HOST" \
      --port "$VLLM_PORT" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
      --max-model-len "$CONTEXT_LEN" \
      --served-model-name "$model_name"
  ) &
  SERVER_PID=$!

  wait_for_server_ready
}

sanitize_name() {
  local raw="$1"
  echo "$raw" | sed 's/[^A-Za-z0-9_-]/_/g'
}

ensure_openai_model_prefix() {
  local model_name="$1"
  if [[ "$model_name" == openai/* ]]; then
    echo "$model_name"
  else
    echo "openai/$model_name"
  fi
}

get_latest_file() {
  local dir="$1"
  local pattern="$2"
  local latest=""

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

  latest="$(ls -1t "${matches[@]}" | head -n1)"
  echo "$latest"
}

extract_generation_pass1() {
  local json_file="$1"
  python3 - "$json_file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

val = data.get("metrics", {}).get("pass@1")
if val is None:
    print("nan")
else:
    print(f"{float(val):.6f}")
PY
}

extract_aggregation_pass1() {
  local json_file="$1"
  python3 - "$json_file" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

val = data.get("aggregate_metrics", {}).get("pass@1")
if val is None:
    print("nan")
else:
    print(f"{float(val):.6f}")
PY
}

average_values() {
  if [[ $# -eq 0 ]]; then
    echo "nan"
    return 0
  fi

  python3 - "$@" <<'PY'
import math
import sys

vals = []
for x in sys.argv[1:]:
    try:
        v = float(x)
    except Exception:
        continue
    if math.isnan(v):
        continue
    vals.append(v)

if not vals:
    print("nan")
else:
    print(f"{sum(vals) / len(vals):.6f}")
PY
}

run_generation() {
  local file="$1"
  local orig_name="$2"
  local orig_request_model="$3"

  local cmd=(
    python3 "$ROOT_DIR/scripts/run_imobench_experiment.py"
    --input-files "$file"
    --model-name "$orig_name"
    --model-path "$orig_request_model"
    --max-concurrent "$MAX_CONCURRENT_GEN"
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

run_aggregation() {
  local result_file="$1"
  local output_dir="$2"
  local request_model="$3"

  local cmd=(
    python3 -m scripts.run_aggregation_experiment
    --result-files "$result_file"
    --output-dir "$output_dir"
    --model_name "$request_model"
    --max-concurrent "$MAX_CONCURRENT_AGG"
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

trap cleanup_server EXIT INT TERM

# Parse optional args (all inputs are configured in this file)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-concurrent-gen)
      MAX_CONCURRENT_GEN="$2"
      shift 2
      ;;
    --max-concurrent-agg)
      MAX_CONCURRENT_AGG="$2"
      shift 2
      ;;
    --num-agg-runs)
      NUM_AGG_RUNS="$2"
      shift 2
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --resume-from)
      RESUME_FROM="$2"
      shift 2
      ;;
    --host)
      VLLM_HOST="$2"
      shift 2
      ;;
    --port)
      VLLM_PORT="$2"
      shift 2
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --server-startup-timeout)
      SERVER_STARTUP_TIMEOUT="$2"
      shift 2
      ;;
    --context-len)
      CONTEXT_LEN="$2"
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

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "Error: FILES is empty. Edit this script and set FILES in the user configuration section." >&2
  usage
  exit 1
fi

if [[ ${#MODEL_SPECS[@]} -eq 0 ]]; then
  echo "Error: MODEL_SPECS is empty. Edit this script and set MODEL_SPECS in the user configuration section." >&2
  usage
  exit 1
fi

if ! [[ "$NUM_AGG_RUNS" =~ ^[0-9]+$ ]] || [[ "$NUM_AGG_RUNS" -le 0 ]]; then
  echo "Error: NUM_AGG_RUNS must be a positive integer, got: $NUM_AGG_RUNS" >&2
  exit 1
fi

for file in "${FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Error: input file not found: $file" >&2
    exit 1
  fi
done

export OPENAI_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
export LLM_CONTEXT_LIMIT_TOKENS="$CONTEXT_LEN"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="dummy"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_DIR="$ROOT_DIR/results/test_time_compute/batch_with_aggregation"
mkdir -p "$SUMMARY_DIR"
SUMMARY_CSV="$SUMMARY_DIR/summary_${TIMESTAMP}.csv"

echo "size,test_file,encoding,generation_pass_at_1,aggregation_pairwise_pass_at_1_avg,aggregation_runs,generation_result_file,aggregation_output_dir" > "$SUMMARY_CSV"

echo "========================================"
echo "IMOBench generation + aggregation batch"
echo "========================================"
echo "Files: ${#FILES[@]}"
echo "Model specs: ${#MODEL_SPECS[@]}"
echo "Gen max concurrent: $MAX_CONCURRENT_GEN"
echo "Agg max concurrent: $MAX_CONCURRENT_AGG"
echo "Aggregation runs per setting: $NUM_AGG_RUNS"
echo "vLLM host: $VLLM_HOST"
echo "vLLM port: $VLLM_PORT"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "Server startup timeout: $SERVER_STARTUP_TIMEOUT"
echo "Context length: $CONTEXT_LEN"
echo "Resume mode: $RESUME"
if [[ -n "$RESUME_FROM" ]]; then
  echo "Resume from: $RESUME_FROM"
fi
echo "========================================"

for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r SIZE ORIG_PATH ORIG_NAME NEW_PATH NEW_NAME <<< "$spec"

  if [[ -z "${SIZE:-}" || -z "${ORIG_PATH:-}" || -z "${ORIG_NAME:-}" || -z "${NEW_PATH:-}" || -z "${NEW_NAME:-}" ]]; then
    echo "Error: invalid model spec '$spec'" >&2
    echo "Expected: size|orig_model_path|orig_model_name|new_model_path|new_model_name" >&2
    exit 1
  fi

  ORIG_REQUEST_MODEL="$(ensure_openai_model_prefix "$ORIG_NAME")"
  NEW_REQUEST_MODEL="$(ensure_openai_model_prefix "$NEW_NAME")"
  ORIG_ALIAS_SANITIZED="$(sanitize_name "$ORIG_NAME")"

  echo
  echo "========================================"
  echo "Model size: $SIZE"
  echo "Original model: $ORIG_NAME -> $ORIG_PATH"
  echo "New-encoding model: $NEW_NAME -> $NEW_PATH"
  echo "========================================"

  for file in "${FILES[@]}"; do
    FILE_STEM="$(basename "$file")"
    FILE_STEM="${FILE_STEM%.*}"
    BENCH_LABEL="$(sanitize_name "$FILE_STEM")"
    BENCH_NAME="imobench_${BENCH_LABEL}"

    GEN_OUTPUT_DIR="$ROOT_DIR/results/test_time_compute/${BENCH_NAME}_rollouts_${ORIG_ALIAS_SANITIZED}"

    echo
    echo "----------------------------------------"
    echo "Size: $SIZE | File: $file"
    echo "Benchmark label: $BENCH_LABEL"
    echo "Generation output dir: $GEN_OUTPUT_DIR"
    echo "----------------------------------------"

    # 1) Start original model server
    start_server_for_model "$ORIG_NAME" "$ORIG_PATH"

    # 2) Run direct generation (no aggregation)
    run_generation "$file" "$ORIG_NAME" "$ORIG_REQUEST_MODEL"

    GEN_RESULT_FILE="$(get_latest_file "$GEN_OUTPUT_DIR" "direct_generation_${BENCH_NAME}_*.json")"
    if [[ -z "$GEN_RESULT_FILE" ]]; then
      echo "Error: cannot find direct_generation result under $GEN_OUTPUT_DIR" >&2
      exit 1
    fi

    GEN_PASS1="$(extract_generation_pass1 "$GEN_RESULT_FILE")"

    # 3) Aggregation with original encoding (same size), repeated NUM_AGG_RUNS times
    AGG_OLD_OUT_DIR="$SUMMARY_DIR/aggregation/${SIZE}/${BENCH_LABEL}/old"
    mkdir -p "$AGG_OLD_OUT_DIR"
    AGG_OLD_RUN_VALUES=()
    for ((run_idx=1; run_idx<=NUM_AGG_RUNS; run_idx++)); do
      echo "Running old-encoding aggregation ${run_idx}/${NUM_AGG_RUNS} ..."
      run_aggregation "$GEN_RESULT_FILE" "$AGG_OLD_OUT_DIR" "$ORIG_REQUEST_MODEL"

      AGG_OLD_RESULT_FILE="$(get_latest_file "$AGG_OLD_OUT_DIR" "aggregation_experiment_pairwise_comparison_${BENCH_NAME}_*.json")"
      if [[ -z "$AGG_OLD_RESULT_FILE" ]]; then
        echo "Error: cannot find old-encoding aggregation result under $AGG_OLD_OUT_DIR" >&2
        exit 1
      fi
      AGG_OLD_RUN_PASS1="$(extract_aggregation_pass1 "$AGG_OLD_RESULT_FILE")"
      AGG_OLD_RUN_VALUES+=("$AGG_OLD_RUN_PASS1")
    done
    AGG_OLD_PASS1_AVG="$(average_values "${AGG_OLD_RUN_VALUES[@]}")"

    # 4) Switch to new-encoding server and run aggregation with same size
    start_server_for_model "$NEW_NAME" "$NEW_PATH"

    AGG_NEW_OUT_DIR="$SUMMARY_DIR/aggregation/${SIZE}/${BENCH_LABEL}/new"
    mkdir -p "$AGG_NEW_OUT_DIR"
    AGG_NEW_RUN_VALUES=()
    for ((run_idx=1; run_idx<=NUM_AGG_RUNS; run_idx++)); do
      echo "Running new-encoding aggregation ${run_idx}/${NUM_AGG_RUNS} ..."
      run_aggregation "$GEN_RESULT_FILE" "$AGG_NEW_OUT_DIR" "$NEW_REQUEST_MODEL"

      AGG_NEW_RESULT_FILE="$(get_latest_file "$AGG_NEW_OUT_DIR" "aggregation_experiment_pairwise_comparison_${BENCH_NAME}_*.json")"
      if [[ -z "$AGG_NEW_RESULT_FILE" ]]; then
        echo "Error: cannot find new-encoding aggregation result under $AGG_NEW_OUT_DIR" >&2
        exit 1
      fi
      AGG_NEW_RUN_PASS1="$(extract_aggregation_pass1 "$AGG_NEW_RESULT_FILE")"
      AGG_NEW_RUN_VALUES+=("$AGG_NEW_RUN_PASS1")
    done
    AGG_NEW_PASS1_AVG="$(average_values "${AGG_NEW_RUN_VALUES[@]}")"

    SUMMARY_ROWS+=("$SIZE|$file|$GEN_PASS1|old|$AGG_OLD_PASS1_AVG|$GEN_RESULT_FILE|$AGG_OLD_OUT_DIR")
    SUMMARY_ROWS+=("$SIZE|$file|$GEN_PASS1|new|$AGG_NEW_PASS1_AVG|$GEN_RESULT_FILE|$AGG_NEW_OUT_DIR")

    echo "$SIZE,$file,old,$GEN_PASS1,$AGG_OLD_PASS1_AVG,$NUM_AGG_RUNS,$GEN_RESULT_FILE,$AGG_OLD_OUT_DIR" >> "$SUMMARY_CSV"
    echo "$SIZE,$file,new,$GEN_PASS1,$AGG_NEW_PASS1_AVG,$NUM_AGG_RUNS,$GEN_RESULT_FILE,$AGG_NEW_OUT_DIR" >> "$SUMMARY_CSV"

    cleanup_server
  done
done

echo
echo "========================================"
echo "No aggregation accuracies: {size} x {test_file}"
echo "========================================"
for row in "${SUMMARY_ROWS[@]}"; do
  IFS='|' read -r SIZE FILE GEN_PASS1 ENCODING _ _ _ <<< "$row"
  if [[ "$ENCODING" == "old" ]]; then
    echo "size=$SIZE | file=$FILE | direct_generation_pass@1=$GEN_PASS1"
  fi
done

echo
echo "========================================"
echo "Aggregation accuracies: {size} x {old/new encoding} x {test_file}"
echo "========================================"
for row in "${SUMMARY_ROWS[@]}"; do
  IFS='|' read -r SIZE FILE _ ENCODING AGG_PASS1 _ _ <<< "$row"
  echo "size=$SIZE | encoding=$ENCODING | file=$FILE | pairwise_aggregation_pass@1_avg_over_${NUM_AGG_RUNS}_runs=$AGG_PASS1"
done

echo
echo "Summary CSV saved to: $SUMMARY_CSV"
echo "All runs completed."
