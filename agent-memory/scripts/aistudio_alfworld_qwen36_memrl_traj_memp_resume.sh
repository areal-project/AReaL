#!/bin/bash
# AIStudio: ALFWorld Qwen3.6 — fresh MemRL(trajectory) + MemP continuation.
# GPU 0: shared Qwen3-Embedding-8B; GPU 1: MemRL-trajectory; GPU 2: MemP resume.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RUN_STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
LOGFILE="$MEMRL_DIR/logs/aistudio_qwen36_memrl_traj_memp_resume_${RUN_STAMP}.log"
MEMRL_TRAJ_RUN_ID="qwen36_memrl_traj_v2"
MEMP_RESUME_RUN_ID="qwen36_memp_resume_s10b20"
MEMP_RESUME_SOURCE="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_memp_qwen36_qwen36_v9/local_cache"
MEMP_RESUME_SNAPSHOT="${MEMP_RESUME_SOURCE}/snapshot/s10_b20"

mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================================"
echo "AIStudio: ALFWorld Qwen3.6 MemRL-trajectory + Mem0"
echo "MemRL trajectory run: $MEMRL_TRAJ_RUN_ID (fresh)"
echo "MemP continuation run: $MEMP_RESUME_RUN_ID"
echo "MemP source snapshot:  $MEMP_RESUME_SNAPSHOT"
echo "Start: $(date)"
echo "============================================================"

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
USER_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
export PYTHONPATH="$MEMRL_DIR:$VENV_SP:$USER_SP:${PYTHONPATH:-}"

cd "$MEMRL_DIR"
PIP_BIN="$(command -v pip || command -v pip3 || true)"
if [[ -z "$PIP_BIN" ]]; then
    echo "[FATAL] neither pip nor pip3 is available"
    exit 1
fi
# Import from mounted source to avoid stale shared-workspace wheel artifacts.
python3 - <<'PYIMPORT'
import pathlib, memrl
actual = pathlib.Path(memrl.__file__).resolve()
expected = pathlib.Path.cwd().joinpath("memrl").resolve()
if expected not in actual.parents:
    raise SystemExit(f"memrl imported from unexpected path: {actual}")
print(f"[PREFLIGHT] source memrl import: {actual}")
PYIMPORT
"$PIP_BIN" install --upgrade --target "$VENV_SP" \
    -i https://pypi.antfin-inc.com/simple/ \
    "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld
python3 - <<'PY'
import hdbscan, memos, memrl
print("[PREFLIGHT] hdbscan:", hdbscan.__file__)
print("[PREFLIGHT] memrl:", memrl.__file__)
print("[PREFLIGHT] memos:", memos.__file__)
PY

SERVER_PIDS=()
EXP_PIDS=()
cleanup() {
    status=$?
    echo "[INFO] cleanup status=$status at $(date)"
    if ((${#EXP_PIDS[@]})); then kill "${EXP_PIDS[@]}" 2>/dev/null || true; fi
    if ((${#SERVER_PIDS[@]})); then kill "${SERVER_PIDS[@]}" 2>/dev/null || true; fi
    wait 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

# host_network=True makes ports global to the host. Pick a fresh consecutive
# block and verify that every server we start owns its endpoint.
PORT_BASE=$(RUN_STAMP="$RUN_STAMP" python3 - <<'PY'
import os, random, socket, sys
rng = random.Random(f"{os.environ.get('HOSTNAME','')}:{os.getpid()}:{os.environ['RUN_STAMP']}")
starts = list(range(20000, 44998, 3))
rng.shuffle(starts)
for base in starts:
    sockets = []
    try:
        for port in range(base, base + 3):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            sock.bind(("0.0.0.0", port))
            sockets.append(sock)
    except OSError:
        pass
    else:
        print(base)
        sys.exit(0)
    finally:
        for sock in sockets:
            sock.close()
raise SystemExit("no free four-port block found")
PY
)
EMBED_PORT=$PORT_BASE
MEMRL_LLM_PORT=$((PORT_BASE + 1))
MEMP_LLM_PORT=$((PORT_BASE + 2))
printf '[PORTS] embed=%s memrl_traj=%s memp=%s\n' \
    "$EMBED_PORT" "$MEMRL_LLM_PORT" "$MEMP_LLM_PORT"

wait_for_server() {
    local name=$1 port=$2 pid=$3 expected_model=$4 timeout=$5
    local response stable=0
    echo "[WAIT] $name pid=$pid port=$port model=$expected_model"
    for ((i=1; i<=timeout; i++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[FATAL] $name process $pid exited before readiness"
            wait "$pid" 2>/dev/null || true
            return 1
        fi
        response=$(curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)
        if MODEL_RESPONSE="$response" EXPECTED_MODEL="$expected_model" python3 - <<'PY'
import json, os, sys
try:
    ids = {x.get("id") for x in json.loads(os.environ["MODEL_RESPONSE"]).get("data", [])}
except Exception:
    sys.exit(1)
sys.exit(0 if os.environ["EXPECTED_MODEL"] in ids else 1)
PY
        then
            stable=$((stable + 1))
            if ((stable >= 5)); then
                echo "[READY] $name is owned by pid=$pid"
                return 0
            fi
        else
            stable=0
        fi
        sleep 1
    done
    echo "[FATAL] timeout waiting for $name"
    return 1
}

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port "$EMBED_PORT" --max-model-len 8192 \
    --trust-remote-code --convert embed &
PID_EMBED=$!; SERVER_PIDS+=("$PID_EMBED")

CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model "$QWEN36_PATH" --served-model-name Qwen3.6-35B-A3B \
    --host 127.0.0.1 --port "$MEMRL_LLM_PORT" --max-model-len 32768 \
    --trust-remote-code --reasoning-parser qwen3 &
PID_MEMRL_LLM=$!; SERVER_PIDS+=("$PID_MEMRL_LLM")

CUDA_VISIBLE_DEVICES=2 python3 -m vllm.entrypoints.openai.api_server \
    --model "$QWEN36_PATH" --served-model-name Qwen3.6-35B-A3B \
    --host 127.0.0.1 --port "$MEMP_LLM_PORT" --max-model-len 32768 \
    --trust-remote-code --reasoning-parser qwen3 &
PID_MEMP_LLM=$!; SERVER_PIDS+=("$PID_MEMP_LLM")

wait_for_server embed "$EMBED_PORT" "$PID_EMBED" Qwen/Qwen3-Embedding-8B 1200
wait_for_server memrl_action "$MEMRL_LLM_PORT" "$PID_MEMRL_LLM" Qwen3.6-35B-A3B 1800
wait_for_server memp_action "$MEMP_LLM_PORT" "$PID_MEMP_LLM" Qwen3.6-35B-A3B 1800

if [[ ! -f "$MEMP_RESUME_SNAPSHOT/local_cache/cum_state.json" || ! -d "$MEMP_RESUME_SNAPSHOT/cube" ]]; then
    echo "[FATAL] MemP resume snapshot is incomplete: $MEMP_RESUME_SNAPSHOT"
    exit 1
fi
echo "[PREFLIGHT] MemP resume checkpoint verified: $MEMP_RESUME_SNAPSHOT"

write_config() {
    local path=$1 llm_port=$2 experiment=$3 build_strategy=$4 enable_vd=$5 tau=$6 weight_sim=$7 weight_q=$8
    cat > "$path" <<CFG
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${llm_port}/v1/
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
  build_strategy: ${build_strategy}
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
  enable_value_driven: ${enable_vd}
  experiment_name: ${experiment}
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
  ckpt_resume_enabled: false
  ckpt_resume_path: ""
  ckpt_resume_epoch: null
  n_eval_runs: 4
  eval_temperature: 0.2
rl_config:
  epsilon: 0
  tau: ${tau}
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
  weight_sim: ${weight_sim}
  weight_q: ${weight_q}
CFG
}

MEMRL_CFG="/tmp/alfworld_memrl_traj_qwen36_${RUN_STAMP}_$$.yaml"
MEMP_CFG="/tmp/alfworld_memp_qwen36_resume_${RUN_STAMP}_$$.yaml"
write_config "$MEMRL_CFG" "$MEMRL_LLM_PORT" alfworld_memrl_traj_qwen36 trajectory true 0.62 0.5 0.5
write_config "$MEMP_CFG" "$MEMP_LLM_PORT" alfworld_memp_qwen36_resume_s10b20 proceduralization false 0.0 1.0 0.0
python3 - "$MEMP_CFG" "$MEMP_RESUME_SOURCE" <<'PYCFG'
from pathlib import Path
import sys
p=Path(sys.argv[1])
s=p.read_text()
s=s.replace("  ckpt_resume_enabled: false\n  ckpt_resume_path: \"\"", f"  ckpt_resume_enabled: true\n  ckpt_resume_path: {sys.argv[2]}")
p.write_text(s)
PYCFG

(
    echo "[EXP] MemRL trajectory start: run=$MEMRL_TRAJ_RUN_ID config=$MEMRL_CFG"
    MEMRL_RUN_ID="$MEMRL_TRAJ_RUN_ID" python3 run/run_alfworld.py \
        --config "$MEMRL_CFG" --skip_initial_eval
) &
PID_MEMRL=$!; EXP_PIDS+=("$PID_MEMRL")

(
    echo "[EXP] MemP continuation start: run=$MEMP_RESUME_RUN_ID source=$MEMP_RESUME_SNAPSHOT config=$MEMP_CFG"
    MEMRL_RUN_ID="$MEMP_RESUME_RUN_ID" python3 run/run_alfworld.py \
        --config "$MEMP_CFG" --skip_initial_eval
) &
PID_MEMP=$!; EXP_PIDS+=("$PID_MEMP")

echo "[INFO] experiments launched: memrl_traj=$PID_MEMRL memp_resume=$PID_MEMP"
failed=0
for entry in "MemRL-trajectory:$PID_MEMRL" "MemP-resume:$PID_MEMP"; do
    name=${entry%%:*}; pid=${entry##*:}
    if wait "$pid"; then
        echo "[DONE] $name exit=0 at $(date)"
    else
        rc=$?
        echo "[ERROR] $name exit=$rc at $(date)"
        failed=1
    fi
done
if ((failed)); then exit 1; fi

echo "[DONE] MemRL-trajectory and MemP continuation complete at $(date)"
