#!/usr/bin/env bash
set -euo pipefail

# Batch runner for IMOBench-compatible experiments.
#
# Usage example:
#   bash scripts/run_imobench_batch.sh \
#     --files /path/to/aime1.json /path/to/aime2.json \
#     --models qwen3-30b-a3b=openai/qwen3-30b-a3b mys-model=openai/custom-model
#
# Optional flags:
#   --max-concurrent N
#   --resume
#   --resume-from /path/to/results.json
#   --auto-add-required-models
#
# Notes:
# - By default, --models is used as-is.
# - Use --auto-add-required-models to auto-add qwen3-8b and qwen3-30b-a3b when missing.
# - Each run calls scripts/run_imobench_experiment.py with exactly one input file.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MAX_CONCURRENT=128
RESUME=false
RESUME_FROM=""
AUTO_ADD_REQUIRED_MODELS=false
VLLM_HOST="0.0.0.0"
VLLM_PORT=8000
TENSOR_PARALLEL_SIZE=8
SERVER_STARTUP_TIMEOUT=300
CONTEXT_LEN=40000

FILES=()
MODEL_ITEMS=()
MODEL_ORDER=()

SERVER_PID=""

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
  echo "Waiting for vLLM server readiness at http://127.0.0.1:${VLLM_PORT}/v1/models ..."

  while true; do
    if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "vLLM server is ready."
      return 0
    fi

    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Error: vLLM server process exited before becoming ready." >&2
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

  echo "Starting vLLM server for model: $model_name -> $model_path"
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

trap cleanup_server EXIT INT TERM

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_imobench_batch.sh --files <file1> [file2 ...] --models <name=path> [name=path ...] [options]

Required:
  --files           One or more IMOBench-format JSON files (e.g., AIME files)
  --models          One or more model mappings as name=path

Options:
  --max-concurrent  Maximum concurrent problems per run (default: 128)
  --resume          Auto-resume each run from latest results
  --resume-from     Resume each run from a specific results file
  --host            vLLM bind host (default: 0.0.0.0)
  --port            vLLM port (default: 8000)
  --tensor-parallel-size
                    vLLM tensor parallel size (default: 8)
  --server-startup-timeout
                    Seconds to wait for server readiness (default: 300)
  --context-len     Max model context length for vLLM and request budgeting (default: 40000)
  --auto-add-required-models
                    Auto-add qwen3-8b and qwen3-30b-a3b if missing
  -h, --help        Show this help message
EOF
}

# Parse args
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

# Validate files exist
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

# If no --models are provided, use required defaults.
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODEL_ORDER+=("qwen3-8b")
  MODELS["qwen3-8b"]="openai/qwen3-8b"
  MODEL_ORDER+=("qwen3-30b-a3b")
  MODELS["qwen3-30b-a3b"]="openai/qwen3-30b-a3b"
elif [[ "$AUTO_ADD_REQUIRED_MODELS" == true ]]; then
  # Optional auto-add of required models.
  if [[ -z "${MODELS[qwen3-8b]:-}" ]]; then
    MODEL_ORDER+=("qwen3-8b")
    MODELS["qwen3-8b"]="openai/qwen3-8b"
  fi
  if [[ -z "${MODELS[qwen3-30b-a3b]:-}" ]]; then
    MODEL_ORDER+=("qwen3-30b-a3b")
    MODELS["qwen3-30b-a3b"]="openai/qwen3-30b-a3b"
  fi
fi

echo "========================================"
echo "IMOBench batch runner"
echo "========================================"
echo "Files: ${#FILES[@]}"
echo "Models: ${#MODELS[@]}"
echo "Max concurrent: $MAX_CONCURRENT"
echo "Resume mode: $RESUME"
echo "vLLM host: $VLLM_HOST"
echo "vLLM port: $VLLM_PORT"
echo "Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "Server startup timeout (s): $SERVER_STARTUP_TIMEOUT"
echo "Context length: $CONTEXT_LEN"
echo "Auto-add required models: $AUTO_ADD_REQUIRED_MODELS"
if [[ -n "$RESUME_FROM" ]]; then
  echo "Resume from: $RESUME_FROM"
fi
echo "========================================"

export OPENAI_API_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
export LLM_CONTEXT_LIMIT_TOKENS="$CONTEXT_LEN"
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="dummy"
fi

for name in "${MODEL_ORDER[@]}"; do
  server_model_path="${MODELS[$name]}"
  request_model_path="${name}"

  echo
  echo "========================================"
  echo "Preparing model server: $name -> $server_model_path"
  echo "========================================"
  start_server_for_model "$name" "$server_model_path"

  for file in "${FILES[@]}"; do

    echo
    echo "----------------------------------------"
    echo "Running file: $file"
    echo "Server model path: $server_model_path"
    echo "Request model: $request_model_path"
    echo "----------------------------------------"

    cmd=(
      python3 "$ROOT_DIR/scripts/run_imobench_experiment_vllm_direct.py"
      --input-files "$file"
      --model-name "$name"
      --model-path "$request_model_path"
      --max-concurrent "$MAX_CONCURRENT"
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
  done

  cleanup_server
done

echo
echo "All batch runs completed."
