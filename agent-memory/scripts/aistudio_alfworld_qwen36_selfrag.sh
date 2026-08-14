#!/bin/bash
# ============================================================================
# AIStudio: Qwen3.6 ALFWorld — Self-RAG only (3 GPUs)
#
# GPU layout:
#   GPU 0: Qwen3-Embedding-8B (embed)
#   GPU 1: Qwen3.6 vLLM WITH reasoning-parser (for act + critique)
#   GPU 2: spare
# ============================================================================
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RUN_STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
LOGFILE="/storage/openpsi/users/yl/agent-memory/MemRL/logs/aistudio_qwen36_selfrag_${RUN_STAMP}.log"

EMBED_PORT=""
LLM_PORT=""
export MEMRL_RUN_ID="qwen36_selfrag_v1"
VLLM_TMPDIR="/tmp"
ALFWORLD_TMPDIR="/storage/openpsi/users/yl/agent-memory/.tmp/q36_selfrag/${RUN_STAMP}_$$"
SELF_RAG_CKPT="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_selfrag_qwen36_qwen36_selfrag_v1/local_cache"
export MEMRL_ALFWORLD_LLM_CONCURRENCY="${MEMRL_ALFWORLD_LLM_CONCURRENCY:-8}"
export MEMRL_LLM_CLIENT_TIMEOUT_S="${MEMRL_LLM_CLIENT_TIMEOUT_S:-600}"
export MEMRL_LLM_MAX_RETRIES="${MEMRL_LLM_MAX_RETRIES:-1}"
mkdir -p "$ALFWORLD_TMPDIR"

# --- Logging ---
mkdir -p $(dirname $LOGFILE)
exec > >(tee -a $LOGFILE) 2>&1

echo "=========================================="
echo "AIStudio: Qwen3.6 ALFWorld — Self-RAG (critique-based)"
echo "Start: $(date)"
echo "=========================================="

# --- Environment setup ---
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
export PYTHONPATH=$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

# --- Install dependencies ---
cd $MEMRL_DIR
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 - <<'PYIMPORT'
import pathlib, memrl, memos
actual = pathlib.Path(memrl.__file__).resolve()
expected = pathlib.Path.cwd().joinpath("memrl").resolve()
if expected not in actual.parents:
    raise SystemExit(f"memrl imported from unexpected path: {actual}")
print(f"[OK] source memrl imported from {actual}; memos available")
PYIMPORT

SELF_RAG_SNAPSHOT="${SELF_RAG_CKPT}/snapshot/8"
python3 - "$SELF_RAG_SNAPSHOT" <<'PYCHECKPOINT'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
cache = root / "local_cache"
required = ["dict_memory.json", "mem_cache.json", "q_cache.json", "query_embeddings.json", "cum_state.json"]
missing = [str(cache / x) for x in required if not (cache / x).is_file() or (cache / x).stat().st_size == 0]
if missing:
    raise SystemExit("Self-RAG resume checkpoint incomplete: " + ", ".join(missing))
state = json.loads((cache / "cum_state.json").read_text())
print(f"[CHECKPOINT] Self-RAG local-cache-only snapshot valid; global_step={state.get('global_step')}")
PYCHECKPOINT

SERVER_PIDS=()
cleanup() {
    status=$?
    echo "[INFO] cleanup status=$status at $(date)"
    if ((${#SERVER_PIDS[@]})); then kill "${SERVER_PIDS[@]}" 2>/dev/null || true; fi
    wait 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

PORT_BASE=$(python3 - <<'PYPORT'
import os, random, socket
rng=random.Random(f"{os.environ.get('HOSTNAME','')}:{os.getpid()}")
ports=list(range(20000,45000,2)); rng.shuffle(ports)
for base in ports:
    ss=[]
    try:
        for port in (base,base+1):
            x=socket.socket(); x.bind(('0.0.0.0',port)); ss.append(x)
    except OSError: pass
    else:
        print(base); raise SystemExit
    finally:
        for x in ss: x.close()
raise SystemExit('no free two-port block')
PYPORT
)
EMBED_PORT=$PORT_BASE
LLM_PORT=$((PORT_BASE+1))
echo "[PORTS] embed=$EMBED_PORT llm=$LLM_PORT"

wait_for_server() {
    local name=$1 port=$2 pid=$3 expected=$4 timeout=$5 response stable=0
    for ((i=1;i<=timeout;i++)); do
        kill -0 "$pid" 2>/dev/null || { echo "[ERROR] $name pid=$pid exited"; return 1; }
        response=$(curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)
        if MODEL_RESPONSE="$response" EXPECTED_MODEL="$expected" python3 - <<'PYMODEL'
import json,os,sys
try: ids={x.get('id') for x in json.loads(os.environ['MODEL_RESPONSE']).get('data',[])}
except Exception: sys.exit(1)
sys.exit(0 if os.environ['EXPECTED_MODEL'] in ids else 1)
PYMODEL
        then stable=$((stable+1)); ((stable>=5)) && { echo "[INFO] $name ready pid=$pid"; return 0; }
        else stable=0; fi
        sleep 1
    done
    return 1
}

# --- Launch Embedding server (GPU 0) ---
TMPDIR="$VLLM_TMPDIR" TMP="$VLLM_TMPDIR" TEMP="$VLLM_TMPDIR" \
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port $EMBED_PORT \
    --max-model-len 8192 --trust-remote-code \
    --convert embed &
PID_EMBED=$!; SERVER_PIDS+=("$PID_EMBED")

# --- Launch LLM with reasoning-parser (GPU 1) ---
TMPDIR="$VLLM_TMPDIR" TMP="$VLLM_TMPDIR" TEMP="$VLLM_TMPDIR" \
CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
    --host 127.0.0.1 --port $LLM_PORT \
    --max-model-len 32768 --trust-remote-code \
    --reasoning-parser qwen3 &
PID_LLM=$!; SERVER_PIDS+=("$PID_LLM")

# --- Wait for servers and verify ownership/model identity ---
wait_for_server "embedding" "$EMBED_PORT" "$PID_EMBED" "Qwen/Qwen3-Embedding-8B" 1200
wait_for_server "Qwen3.6" "$LLM_PORT" "$PID_LLM" "Qwen3.6-35B-A3B" 1800

# ============================================================================
# Run Self-RAG experiment
# ============================================================================
CFG_PATH="$ALFWORLD_TMPDIR/alf_selfrag_qwen36_${RUN_STAMP}_$$.yaml"
cat > "$CFG_PATH" << CFGEOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${LLM_PORT}/v1/
  model: Qwen3.6-35B-A3B
  temperature: 0
  max_tokens: 4096
embedding:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${EMBED_PORT}/v1/
  model: Qwen/Qwen3-Embedding-8B
  max_text_len: 8196
  dimension: 4096
memory:
  build_strategy: trajectory
  retrieve_strategy: query
  update_strategy: adjustment
  k_retrieve: 3
  max_keywords: 5
  add_similarity_threshold: 0.9
  memory_budget_tokens: 0
  sim_norm_mean: 0.5187
  sim_norm_std: 0.1203
environment:
  alfworld_config_path: configs/envs/alfworld.yaml
  alfworld_env_type: AlfredTWEnv
experiment:
  random_seed: 42
  enable_value_driven: false
  experiment_name: alfworld_selfrag_qwen36
  mode: train
  num_sections: 10
  batch_size: 128
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: null
  baseline_k: 10
  output_dir: /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld
  max_steps: 30
  max_recent_turns: 20
  strip_thinking: true
  max_trajectory_len: 6000
  max_history_response_chars: 4000
  force_think: false
  save_trajectories: true
  save_memories: true
  ckpt_resume_enabled: true
  ckpt_resume_path: ${SELF_RAG_CKPT}
  ckpt_resume_epoch: 8
  n_eval_runs: 4
  eval_temperature: 0.2
rl_config:
  epsilon: 0
  tau: 0.0
  alpha: 0.3
  gamma: 0.0
  q_init_pos: 0
  q_init_neg: 0
  success_reward: 1.0
  failure_reward: -1.0
  topk: 3
  novelty_threshold: 0.85
  recency_boost: 0.0
  reward_merge_gain: 0.1
  q_min_threshold: -10
  weight_sim: 1.0
  weight_q: 0.0
CFGEOF

echo "[RESUME] Self-RAG clean checkpoint: ${SELF_RAG_CKPT}/snapshot/8"
echo "[RESUME] concurrency=${MEMRL_ALFWORLD_LLM_CONCURRENCY} timeout=${MEMRL_LLM_CLIENT_TIMEOUT_S}s retries=${MEMRL_LLM_MAX_RETRIES}"
echo "[EXP] Self-RAG (critique) starting..."
TMPDIR="$ALFWORLD_TMPDIR" TMP="$ALFWORLD_TMPDIR" TEMP="$ALFWORLD_TMPDIR" \
python3 scripts/run_alfworld_qwen36_resume.py \
    --config "$CFG_PATH" \
    --selfrag --skip_initial_eval

echo "[EXP] Self-RAG exit=$? at $(date)"
echo "=========================================="
echo "Self-RAG experiment complete. End: $(date)"
echo "=========================================="
