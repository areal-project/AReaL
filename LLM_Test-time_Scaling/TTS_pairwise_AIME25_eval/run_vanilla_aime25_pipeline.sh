#!/usr/bin/env bash
set -euo pipefail

# Vanilla AIME25 evaluation pipeline using the stock Qwen3-4B model.
#
# Same 3-stage pipeline as run_pairwise_aime25_pipeline.sh but starts vLLM
# servers WITHOUT --hf-overrides, --trust-remote-code, or a custom env.
#
# Usage:
#   bash TTS_pairwise_AIME25_eval/run_vanilla_aime25_pipeline.sh \
#     --jsonl-file pairwise_from_direct_generation_AIME25.jsonl \
#     --ground-truth-file LLM_Test-time_Scaling/imobench.json
#
# Optional overrides:
#   --model-path PATH   (default: /storage/openpsi/models/Qwen__Qwen3-4B)
#   --model-name NAME   (default: qwen3-4b-vanilla)
#   --num-runs N        (default: 8)
#   --num-gpus N        (default: 8)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -----------------------------
# Defaults
# -----------------------------
JSONL_FILE=""
GROUND_TRUTH_FILE=""
MODEL_PATH="/storage/openpsi/models/Qwen__Qwen3-4B"
MODEL_NAME="qwen3-4b-vanilla"

SERVER_HOST="0.0.0.0"
BASE_PORT=8000
TENSOR_PARALLEL_SIZE=1
SERVER_STARTUP_TIMEOUT=300
NUM_GPUS=8

NUM_RUNS=8
SAMPLE_SIZE=0
TEMPERATURE=0.6
MAX_TOKENS=16384
SEED=42
CONCURRENCY=1

VANILLA_ENV_ACTIVATE="/storage/openpsi/users/zzy/.zzy-enc-2/bin/activate"

SERVER_PIDS=()

# -----------------------------
# Functions
# -----------------------------
usage() {
  cat <<'EOF'
Usage:
  bash TTS_pairwise_AIME25_eval/run_vanilla_aime25_pipeline.sh [options]

Required:
  --jsonl-file FILE              Pairwise JSONL file
  --ground-truth-file FILE       Ground truth JSON (imobench.json)

Model options:
  --model-path PATH              Path to vanilla model (default: /storage/openpsi/models/Qwen__Qwen3-4B)
  --model-name NAME              Served model name (default: qwen3-4b-vanilla)

Experiment options:
  --num-runs N                   Number of runs to average (default: 8)
  --num-gpus N                   Number of GPUs / parallel servers (default: 8)
  --sample-size N                Number of lines to process per run (0 = all, default: 0)
  --temperature FLOAT            Sampling temperature (default: 0.6)
  --max-tokens N                 Max tokens for generation (default: 16384)
  --seed N                       Random seed for line sampling (default: 42)
  --concurrency N                Concurrent requests per server (default: 1)

Server options:
  --host HOST                    Bind host (default: 0.0.0.0)
  --base-port PORT               First server port (default: 8000)
  --tensor-parallel-size N       Tensor parallel size per server (default: 1)
  --server-startup-timeout N     Seconds to wait for each server (default: 300)

Other:
  -h, --help                     Show this help
EOF
}

cleanup_servers() {
  echo "Cleaning up servers..."
  for pid in "${SERVER_PIDS[@]}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "  Stopping server (pid=$pid)..."
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  SERVER_PIDS=()
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

  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      echo "  Server on port $port is ready."
      return 0
    fi

    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "Error: server on port $port exited before becoming ready." >&2
      return 1
    fi

    elapsed=$(( $(date +%s) - start_ts ))
    if (( elapsed >= SERVER_STARTUP_TIMEOUT )); then
      echo "Error: timed out waiting for server on port $port after ${SERVER_STARTUP_TIMEOUT}s." >&2
      return 1
    fi
    sleep 2
  done
}

start_single_server() {
  local gpu_idx="$1"
  local port="$2"

  kill_existing_service_on_port "$port"

  echo "Starting vanilla server on GPU $gpu_idx, port $port: $MODEL_NAME -> $MODEL_PATH"
  (
    cd "$ROOT_DIR"

    if [[ -f "$VANILLA_ENV_ACTIVATE" ]]; then
      source "$VANILLA_ENV_ACTIVATE"
    fi

    export CUDA_VISIBLE_DEVICES="$gpu_idx"

    python -m vllm.entrypoints.openai.api_server \
      --host "$SERVER_HOST" \
      --port "$port" \
      --model "$MODEL_PATH" \
      --tokenizer "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" &
    local child_pid="$!"

    trap 'kill "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true' TERM INT EXIT
    wait "$child_pid"
  ) &
  SERVER_PIDS+=($!)
}

start_all_servers() {
  echo "Starting $NUM_GPUS vanilla vLLM servers (one per GPU, TP=$TENSOR_PARALLEL_SIZE)..."
  for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
    local port=$((BASE_PORT + gpu_idx))
    start_single_server "$gpu_idx" "$port"
  done

  echo "Waiting for all $NUM_GPUS servers to become ready..."
  for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
    local port=$((BASE_PORT + gpu_idx))
    local pid="${SERVER_PIDS[$gpu_idx]}"
    wait_for_server_ready "$port" "$pid"
  done
  echo "All $NUM_GPUS servers are ready."
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

# -----------------------------
# Argument parsing
# -----------------------------
trap cleanup_servers EXIT INT TERM

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
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --sample-size)
      SAMPLE_SIZE="$2"
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
    --seed)
      SEED="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --host)
      SERVER_HOST="$2"
      shift 2
      ;;
    --base-port)
      BASE_PORT="$2"
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

if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [[ "$NUM_RUNS" -le 0 ]]; then
  echo "Error: --num-runs must be a positive integer, got: $NUM_RUNS" >&2
  exit 1
fi

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [[ "$NUM_GPUS" -le 0 ]]; then
  echo "Error: --num-gpus must be a positive integer, got: $NUM_GPUS" >&2
  exit 1
fi

# -----------------------------
# Environment
# -----------------------------
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="dummy"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="$ROOT_DIR/results/tts_pairwise_aime25_eval"
mkdir -p "$OUTPUT_BASE"
SUMMARY_CSV="$OUTPUT_BASE/summary_vanilla_${TIMESTAMP}.csv"

echo "jsonl_file,model_name,gt_accuracy_avg,tournament_pass1_avg,num_runs,sample_size,seed,output_dir" \
  > "$SUMMARY_CSV"

# -----------------------------
# Print config
# -----------------------------
echo "========================================"
echo "Vanilla Pairwise AIME25 Eval"
echo "========================================"
echo "JSONL file: $JSONL_FILE"
echo "Ground truth: $GROUND_TRUTH_FILE"
echo "--- Model ---"
echo "  Model: $MODEL_NAME -> $MODEL_PATH"
echo "  Servers: $NUM_GPUS x (TP=$TENSOR_PARALLEL_SIZE), ports ${BASE_PORT}..$(( BASE_PORT + NUM_GPUS - 1 ))"
echo "  No --hf-overrides, no --trust-remote-code"
echo "--- Settings ---"
echo "  Num runs: $NUM_RUNS"
echo "  Num GPUs: $NUM_GPUS"
echo "  Sample size: $SAMPLE_SIZE"
echo "  Temperature: $TEMPERATURE"
echo "  Max tokens: $MAX_TOKENS"
echo "  Seed: $SEED"
echo "  Concurrency per server: $CONCURRENCY"
echo "========================================"

# -----------------------------
# Start all servers
# -----------------------------
start_all_servers

# Build comma-separated list of all server API bases
ALL_API_BASES=""
for ((gpu_idx=0; gpu_idx<NUM_GPUS; gpu_idx++)); do
  port=$((BASE_PORT + gpu_idx))
  if [[ -n "$ALL_API_BASES" ]]; then
    ALL_API_BASES="${ALL_API_BASES},"
  fi
  ALL_API_BASES="${ALL_API_BASES}http://127.0.0.1:${port}/v1"
done

# -----------------------------
# Run experiments in parallel
# -----------------------------
# -----------------------------
# Run experiments sequentially (each run uses ALL servers)
# -----------------------------
RUN_OUTPUT_DIR="$OUTPUT_BASE/runs/${MODEL_NAME}/${TIMESTAMP}"
mkdir -p "$RUN_OUTPUT_DIR"

TMPDIR_RESULTS="$(mktemp -d)"
trap 'cleanup_servers; rm -rf "$TMPDIR_RESULTS"' EXIT INT TERM

FAILED=0

for ((run_idx=1; run_idx<=NUM_RUNS; run_idx++)); do
  echo "[Run ${run_idx}/${NUM_RUNS}] Starting (distributing across $NUM_GPUS servers)"

  EVAL_OUTPUT="$RUN_OUTPUT_DIR/eval_results_run${run_idx}.jsonl"
  GRADED_OUTPUT="$RUN_OUTPUT_DIR/graded_results_run${run_idx}.json"
  TOURNEY_OUTPUT="$RUN_OUTPUT_DIR/tournament_results_run${run_idx}.json"

  # Stage 1: Aggregation
  echo "[Run ${run_idx}] Stage 1: Aggregation (test_training_data_eval.py)"
  (
    cd "$ROOT_DIR"
    python3 "$SCRIPT_DIR/test_training_data_eval.py" \
      --dataset "$JSONL_FILE" \
      --api-base "$ALL_API_BASES" \
      --model-name "$MODEL_NAME" \
      --n "$SAMPLE_SIZE" \
      --max-tokens "$MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --seed "$SEED" \
      --concurrency "$NUM_GPUS" \
      --output "$EVAL_OUTPUT"
  ) || { echo "[Run ${run_idx}] Stage 1 failed"; FAILED=$((FAILED + 1)); continue; }

  if [[ ! -f "$EVAL_OUTPUT" ]]; then
    echo "[Run ${run_idx}] Warning: aggregation output not found -- skipping"
    echo "nan" > "$TMPDIR_RESULTS/run_${run_idx}_gt.txt"
    echo "nan" > "$TMPDIR_RESULTS/run_${run_idx}_tourney.txt"
    continue
  fi

  # Stage 2: Grading against ground truth
  echo "[Run ${run_idx}] Stage 2: Grading against AIME25 ground truth"
  (
    cd "$ROOT_DIR"
    python3 "$SCRIPT_DIR/grade_against_aime25.py" \
      --eval-jsonl "$EVAL_OUTPUT" \
      --pairwise-jsonl "$JSONL_FILE" \
      --ground-truth "$GROUND_TRUTH_FILE" \
      --output "$GRADED_OUTPUT"
  )

  if [[ -f "$GRADED_OUTPUT" ]]; then
    python3 - "$GRADED_OUTPUT" <<'PY' > "$TMPDIR_RESULTS/run_${run_idx}_gt.txt"
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
val = data.get("aggregate_metrics", {}).get("pass@1")
print("nan" if val is None else f"{float(val):.6f}")
PY
    echo "[Run ${run_idx}] GT accuracy: $(cat "$TMPDIR_RESULTS/run_${run_idx}_gt.txt")"
  else
    echo "nan" > "$TMPDIR_RESULTS/run_${run_idx}_gt.txt"
  fi

  # Stage 3: Tournament aggregation
  echo "[Run ${run_idx}] Stage 3: Tournament aggregation"
  (
    cd "$ROOT_DIR"
    python3 "$SCRIPT_DIR/tournament_aggregate.py" \
      --eval-jsonl "$EVAL_OUTPUT" \
      --pairwise-jsonl "$JSONL_FILE" \
      --ground-truth "$GROUND_TRUTH_FILE" \
      --output "$TOURNEY_OUTPUT"
  )

  if [[ -f "$TOURNEY_OUTPUT" ]]; then
    python3 - "$TOURNEY_OUTPUT" <<'PY' > "$TMPDIR_RESULTS/run_${run_idx}_tourney.txt"
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
val = data.get("aggregate_metrics", {}).get("pass@1")
print("nan" if val is None else f"{float(val):.6f}")
PY
    echo "[Run ${run_idx}] Tournament pass@1: $(cat "$TMPDIR_RESULTS/run_${run_idx}_tourney.txt")"
  else
    echo "nan" > "$TMPDIR_RESULTS/run_${run_idx}_tourney.txt"
  fi

  echo "[Run ${run_idx}] Done."
done

if [[ $FAILED -gt 0 ]]; then
  echo "Warning: $FAILED run(s) failed." >&2
fi

# -----------------------------
# Collect results
# -----------------------------
RUN_VALUES=()
TOURNEY_VALUES=()

for ((run_idx=1; run_idx<=NUM_RUNS; run_idx++)); do
  gt_file="$TMPDIR_RESULTS/run_${run_idx}_gt.txt"
  tourney_file="$TMPDIR_RESULTS/run_${run_idx}_tourney.txt"

  if [[ -f "$gt_file" ]]; then
    RUN_VALUES+=("$(cat "$gt_file")")
  else
    RUN_VALUES+=("nan")
  fi

  if [[ -f "$tourney_file" ]]; then
    TOURNEY_VALUES+=("$(cat "$tourney_file")")
  else
    TOURNEY_VALUES+=("nan")
  fi
done

PASS1_AVG="$(average_values "${RUN_VALUES[@]}")"
TOURNEY_AVG="$(average_values "${TOURNEY_VALUES[@]}")"
echo "$JSONL_FILE,$MODEL_NAME,$PASS1_AVG,$TOURNEY_AVG,$NUM_RUNS,$SAMPLE_SIZE,$SEED,$RUN_OUTPUT_DIR" >> "$SUMMARY_CSV"

# -----------------------------
# Summary
# -----------------------------
cleanup_servers

echo
echo "========================================"
echo "Results summary (vanilla)"
echo "========================================"
echo "JSONL: $(basename "$JSONL_FILE")"
echo "Model: $MODEL_NAME"
echo
echo "--- Per-pair GT accuracy (Stage 2) ---"
echo "  Avg (${NUM_RUNS} runs): $PASS1_AVG"
for i in "${!RUN_VALUES[@]}"; do
  echo "  Run $((i+1)): ${RUN_VALUES[$i]}"
done
echo
echo "--- Tournament pass@1 (Stage 3) ---"
echo "  Avg (${NUM_RUNS} runs): $TOURNEY_AVG"
for i in "${!TOURNEY_VALUES[@]}"; do
  echo "  Run $((i+1)): ${TOURNEY_VALUES[$i]}"
done
echo
echo "Summary CSV: $SUMMARY_CSV"
echo "Run outputs: $RUN_OUTPUT_DIR"
echo "Done."
