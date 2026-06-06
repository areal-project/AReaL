#!/usr/bin/env bash
set -euo pipefail

# Batch runner for IMOBench-compatible experiments via SGLang.
#
# This mirrors scripts/run_imobench_batch.sh but uses:
#   scripts/run_imobench_sglang_eval.py
# and runs one SGLang server per model.
#
# Usage example:
#   bash scripts/run_imobench_batch_sglang.sh \
#     --files /path/to/aime1.json /path/to/aime2.json \
#     --models qwen3-30b-a3b=openai/Qwen__Qwen3-30B-A3B another=openai/custom-model

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MAX_CONCURRENT=16
N_SAMPLES=8
SAMPLE_CONCURRENCY=8
TEMPERATURE=0.6
TOP_P=0.95
TOP_K=20
MAX_TOKENS=35000
OUTPUT_DIR="results/test_time_compute"
EXPERIMENT_NAME="direct_generation"

AUTO_ADD_REQUIRED_MODELS=false
SGLANG_HOST="0.0.0.0"
SGLANG_PORT=30000
SERVER_STARTUP_TIMEOUT=300
TENSOR_PARALLEL_SIZE=8
SGLANG_ATTENTION_BACKEND="triton"
TOKENIZER_OVERRIDE=""
ADDITIONAL_SERVER_ARGS=""

FILES=()
MODEL_ITEMS=()
MODEL_ORDER=()

SERVER_PID=""

cleanup_server() {
  if [[ -n "$SERVER_PID" ]]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Stopping SGLang server (pid=$SERVER_PID)..."
      kill "$SERVER_PID" 2>/dev/null || true
      wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=""
  fi
}

kill_existing_service_on_port() {
  local pids
  pids="$(lsof -ti tcp:"$SGLANG_PORT" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Killing existing service(s) on port $SGLANG_PORT: $pids"
    kill $pids 2>/dev/null || true
    sleep 2
    pids="$(lsof -ti tcp:"$SGLANG_PORT" 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      echo "Force-killing stubborn service(s) on port $SGLANG_PORT: $pids"
      kill -9 $pids 2>/dev/null || true
      sleep 1
    fi
  fi
}

wait_for_server_ready() {
  local start_ts elapsed
  start_ts="$(date +%s)"
  echo "Waiting for SGLang server readiness at http://127.0.0.1:${SGLANG_PORT}/v1/models ..."

  while true; do
    if curl -fsS "http://127.0.0.1:${SGLANG_PORT}/v1/models" >/dev/null 2>&1; then
      echo "SGLang server is ready."
      return 0
    fi

    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Error: SGLang server process exited before becoming ready." >&2
      return 1
    fi

    elapsed=$(( $(date +%s) - start_ts ))
    if (( elapsed >= SERVER_STARTUP_TIMEOUT )); then
      echo "Error: timed out waiting for SGLang server after ${SERVER_STARTUP_TIMEOUT}s." >&2
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

  echo "Starting SGLang server for model: $model_name -> $model_path"
  (
    cd "$ROOT_DIR"

    cmd=(
      python3 -m sglang.launch_server
      --model "$model_path"
      --host "$SGLANG_HOST"
      --port "$SGLANG_PORT"
      --attention-backend "$SGLANG_ATTENTION_BACKEND"
      --tp "$TENSOR_PARALLEL_SIZE"
    )

    if [[ -n "$ADDITIONAL_SERVER_ARGS" ]]; then
      # shellcheck disable=SC2206
      extra_args=($ADDITIONAL_SERVER_ARGS)
      cmd+=("${extra_args[@]}")
    fi

    "${cmd[@]}"
  ) &
  SERVER_PID=$!

  wait_for_server_ready
}

trap cleanup_server EXIT INT TERM

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_imobench_batch_sglang.sh --files <file1> [file2 ...] --models <name=path> [name=path ...] [options]

Required:
  --files                    One or more IMOBench-format JSON files
  --models                   One or more model mappings as name=path

Options:
  --max-concurrent           Max concurrent problems (default: 16)
  --n-samples                Number of samples per problem (default: 8)
  --sample-concurrency       Parallel samples per problem (default: 8)
  --temperature              Sampling temperature (default: 0.6)
  --top-p                    Top-p (default: 0.95)
  --top-k                    Top-k (default: 20)
  --max-tokens               Max completion tokens (default: 35000)
  --output-dir               Output base dir (default: results/test_time_compute)
  --experiment-name          Experiment name (default: direct_generation)
  --tokenizer                Optional tokenizer override passed to eval script

  --host                     SGLang bind host (default: 0.0.0.0)
  --port                     SGLang port (default: 30000)
  --tensor-parallel-size     SGLang tensor parallel size (default: 8)
  --server-startup-timeout   Wait timeout in seconds (default: 300)
  --attention-backend        SGLang attention backend (default: triton)
  --additional-server-args   Extra args appended to sglang.launch_server

  --auto-add-required-models Auto-add qwen3-8b and qwen3-30b-a3b if missing
  -h, --help                 Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --files)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        FILES+=("$1")
        shift
      done
      ;;
    --models)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        MODEL_ITEMS+=("$1")
        shift
      done
      ;;
    --max-concurrent)
      MAX_CONCURRENT="$2"
      shift 2
      ;;
    --n-samples)
      N_SAMPLES="$2"
      shift 2
      ;;
    --sample-concurrency)
      SAMPLE_CONCURRENCY="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --top-k)
      TOP_K="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --experiment-name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --tokenizer)
      TOKENIZER_OVERRIDE="$2"
      shift 2
      ;;
    --host)
      SGLANG_HOST="$2"
      shift 2
      ;;
    --port)
      SGLANG_PORT="$2"
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
    --attention-backend)
      SGLANG_ATTENTION_BACKEND="$2"
      shift 2
      ;;
    --additional-server-args)
      ADDITIONAL_SERVER_ARGS="$2"
      shift 2
      ;;
    --auto-add-required-models)
      AUTO_ADD_REQUIRED_MODELS=true
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

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "Error: --files is required" >&2
  usage
  exit 1
fi

for file in "${FILES[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Error: input file not found: $file" >&2
    exit 1
  fi
done

declare -A MODELS
for item in "${MODEL_ITEMS[@]}"; do
  if [[ "$item" != *=* ]]; then
    echo "Error: invalid --models item '$item' (expected name=path)" >&2
    exit 1
  fi
  name="${item%%=*}"
  path="${item#*=}"
  if [[ -z "$name" || -z "$path" ]]; then
    echo "Error: invalid --models item '$item' (empty name/path)" >&2
    exit 1
  fi
  if [[ -z "${MODELS[$name]:-}" ]]; then
    MODEL_ORDER+=("$name")
  fi
  MODELS["$name"]="$path"
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODEL_ORDER+=("qwen3-8b")
  MODELS["qwen3-8b"]="openai/qwen3-8b"
  MODEL_ORDER+=("qwen3-30b-a3b")
  MODELS["qwen3-30b-a3b"]="openai/Qwen__Qwen3-30B-A3B"
elif [[ "$AUTO_ADD_REQUIRED_MODELS" == true ]]; then
  if [[ -z "${MODELS[qwen3-8b]:-}" ]]; then
    MODEL_ORDER+=("qwen3-8b")
    MODELS["qwen3-8b"]="openai/qwen3-8b"
  fi
  if [[ -z "${MODELS[qwen3-30b-a3b]:-}" ]]; then
    MODEL_ORDER+=("qwen3-30b-a3b")
    MODELS["qwen3-30b-a3b"]="openai/Qwen__Qwen3-30B-A3B"
  fi
fi

echo "========================================"
echo "IMOBench SGLang batch runner"
echo "========================================"
echo "Files: ${#FILES[@]}"
echo "Models: ${#MODELS[@]}"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Samples per problem: $N_SAMPLES"
echo "Sample concurrency: $SAMPLE_CONCURRENCY"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Top-k: $TOP_K"
echo "Max tokens: $MAX_TOKENS"
echo "SGLang host: $SGLANG_HOST"
echo "SGLang port: $SGLANG_PORT"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "Server startup timeout (s): $SERVER_STARTUP_TIMEOUT"
echo "Output dir: $OUTPUT_DIR"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Auto-add required models: $AUTO_ADD_REQUIRED_MODELS"
if [[ -n "$TOKENIZER_OVERRIDE" ]]; then
  echo "Tokenizer override: $TOKENIZER_OVERRIDE"
fi
if [[ -n "$ADDITIONAL_SERVER_ARGS" ]]; then
  echo "Additional server args: $ADDITIONAL_SERVER_ARGS"
fi
echo "========================================"

for name in "${MODEL_ORDER[@]}"; do
  model_path="${MODELS[$name]}"

  echo
  echo "========================================"
  echo "Preparing model server: $name -> $model_path"
  echo "========================================"
  start_server_for_model "$name" "$model_path"

  for file in "${FILES[@]}"; do
    echo
    echo "----------------------------------------"
    echo "Running file: $file"
    echo "Model name: $name"
    echo "Model path: $model_path"
    echo "----------------------------------------"

    cmd=(
      python3 "$ROOT_DIR/scripts/run_imobench_sglang_eval.py"
      --input-file "$file"
      --model-name "$name"
      --model-path "$model_path"
      --output-dir "$OUTPUT_DIR"
      --experiment-name "$EXPERIMENT_NAME"
      --n-samples "$N_SAMPLES"
      --max-concurrent "$MAX_CONCURRENT"
      --sample-concurrency "$SAMPLE_CONCURRENCY"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --top-k "$TOP_K"
      --max-tokens "$MAX_TOKENS"
    )

    if [[ -n "$TOKENIZER_OVERRIDE" ]]; then
      cmd+=(--tokenizer "$TOKENIZER_OVERRIDE")
    fi

    (
      cd "$ROOT_DIR"
      SGLANG_API_BASE="http://127.0.0.1:${SGLANG_PORT}/v1" "${cmd[@]}"
    )
  done

  cleanup_server
done

echo
echo "All SGLang batch runs completed."
