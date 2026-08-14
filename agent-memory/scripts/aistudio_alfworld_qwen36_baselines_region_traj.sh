#!/bin/bash
# AIStudio: resume Qwen3.6 Region+FS trajectory and baselines
# (MemRL/MemP/RAG) from their existing checkpoints in the same 5-GPU job.
# GPU 0: shared embedding; GPUs 1-4: Region, MemRL, MemP, RAG.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RUN_STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
LOGFILE="$MEMRL_DIR/logs/aistudio_qwen36_baselines_region_traj_${RUN_STAMP}.log"

# host_network=True makes ports host-global. Select a fresh five-port block at
# runtime instead of reusing common fixed ports shared by unrelated jobs.
EMBED_PORT=""
LLM_PORT_1=""
LLM_PORT_2=""
LLM_PORT_3=""
LLM_PORT_4=""
REGION_RUN_ID="qwen36_region_traj_v5"
BASELINE_RUN_ID="qwen36_v9"

mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "AIStudio: Region+FS trajectory + baselines resume"
echo "Region run ID: $REGION_RUN_ID (auto-resume)"
echo "Baseline run ID: $BASELINE_RUN_ID (auto-resume)"
echo "Start: $(date)"
echo "=========================================="

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
# Keep vLLM/ZMQ on the node-local /tmp: ZMQ IPC sockets are unsupported on
# CPFS and Unix socket paths are length-limited.  Only ALFWorld/TextWorld child
# processes receive the CPFS temp directory below.
VLLM_TMPDIR="/tmp"
ALFWORLD_TMPDIR="/storage/openpsi/users/yl/agent-memory/.tmp/q36/${RUN_STAMP}_$$"
mkdir -p "$ALFWORLD_TMPDIR"
echo "[PREFLIGHT] vLLM TMPDIR=$VLLM_TMPDIR"
echo "[PREFLIGHT] ALFWorld TMPDIR=$ALFWORLD_TMPDIR"
TMPDIR="$VLLM_TMPDIR" TMP="$VLLM_TMPDIR" TEMP="$VLLM_TMPDIR" python3 - <<'PYTMP'
import tempfile
actual = tempfile.gettempdir()
assert actual == "/tmp", actual
candidate = f"{actual}/" + "x" * 64
assert len(candidate.encode()) < 107, (len(candidate.encode()), candidate)
print("[PREFLIGHT] vLLM tempfile dir:", actual)
print("[PREFLIGHT] ZMQ candidate path bytes:", len(candidate.encode()))
PYTMP
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
USER_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
export PYTHONPATH="$MEMRL_DIR:$VENV_SP:$USER_SP:${PYTHONPATH:-}"

cd "$MEMRL_DIR"
PIP_BIN="$(command -v pip || command -v pip3 || true)"
if [[ -z "$PIP_BIN" ]]; then
    echo "[ERROR] neither pip nor pip3 is available in the runtime image"
    exit 1
fi
echo "[PREFLIGHT] python executable: $(command -v python3)"
echo "[PREFLIGHT] pip executable: $PIP_BIN"
"$PIP_BIN" install --no-deps --upgrade --target "$VENV_SP" \
    -i https://pypi.antfin-inc.com/simple/ .
"$PIP_BIN" install --upgrade --target "$VENV_SP" \
    -i https://pypi.antfin-inc.com/simple/ \
    mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld

python3 - <<'PY'
import sys
import hdbscan
import memrl
import memos
print("[PREFLIGHT] python:", sys.executable)
print("[PREFLIGHT] hdbscan:", hdbscan.__file__)
print("[PREFLIGHT] memrl:", memrl.__file__)
print("[PREFLIGHT] memos:", memos.__file__)
PY

SERVER_PIDS=()
EXP_PIDS=()
cleanup() {
    status=$?
    echo "[INFO] cleanup (status=$status) at $(date)"
    if ((${#EXP_PIDS[@]})); then
        kill "${EXP_PIDS[@]}" 2>/dev/null || true
    fi
    if ((${#SERVER_PIDS[@]})); then
        kill "${SERVER_PIDS[@]}" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM

# Find five consecutive ports that are all bindable on the host. The randomized
# scan start reduces races between concurrently starting host-network jobs.
PORT_BASE=$(python3 - <<'PYPORT'
import os
import random
import socket
import sys

seed = f"{os.environ.get('HOSTNAME', '')}:{os.getpid()}:{os.environ.get('RUN_STAMP', '')}"
rng = random.Random(seed)
starts = list(range(20000, 44996, 5))
rng.shuffle(starts)
for base in starts:
    sockets = []
    try:
        for port in range(base, base + 5):
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
raise SystemExit("no free five-port block found in 20000-44999")
PYPORT
)
EMBED_PORT=$PORT_BASE
LLM_PORT_1=$((PORT_BASE + 1))
LLM_PORT_2=$((PORT_BASE + 2))
LLM_PORT_3=$((PORT_BASE + 3))
LLM_PORT_4=$((PORT_BASE + 4))
export RUN_STAMP
printf '[PORTS] selected host-global block: embed=%s llm=%s,%s,%s,%s\n' \
    "$EMBED_PORT" "$LLM_PORT_1" "$LLM_PORT_2" "$LLM_PORT_3" "$LLM_PORT_4"

# Do not accept an arbitrary service already listening on the requested port.
# The process we launched must remain alive, and its endpoint must report the
# expected served model for several consecutive probes.
wait_for_server() {
    local name=$1 port=$2 pid=$3 expected_model=$4 timeout=$5
    local response stable=0
    echo "[INFO] Waiting for $name pid=$pid port=$port model=$expected_model..."
    for ((i=1; i<=timeout; i++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[ERROR] $name process $pid exited before readiness on port $port"
            wait "$pid" 2>/dev/null || true
            return 1
        fi
        response=$(curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null || true)
        if MODEL_RESPONSE="$response" EXPECTED_MODEL="$expected_model" python3 - <<'PYMODEL'
import json
import os
import sys
try:
    payload = json.loads(os.environ.get("MODEL_RESPONSE", ""))
    ids = {item.get("id") for item in payload.get("data", []) if isinstance(item, dict)}
except Exception:
    sys.exit(1)
sys.exit(0 if os.environ["EXPECTED_MODEL"] in ids else 1)
PYMODEL
        then
            stable=$((stable + 1))
            if ((stable >= 5)); then
                echo "[INFO] $name ready and owned by live pid=$pid on port $port (model=$expected_model)"
                return 0
            fi
        else
            stable=0
        fi
        sleep 1
    done
    echo "[ERROR] $name timeout after ${timeout}s (pid=$pid port=$port model=$expected_model)"
    return 1
}

TMPDIR="$VLLM_TMPDIR" TMP="$VLLM_TMPDIR" TEMP="$VLLM_TMPDIR" \
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port "$EMBED_PORT" \
    --max-model-len 8192 --trust-remote-code --convert embed &
PID_EMBED=$!
SERVER_PIDS+=("$PID_EMBED")

for gpu in 1 2 3 4; do
    port_var="LLM_PORT_${gpu}"
    port="${!port_var}"
    TMPDIR="$VLLM_TMPDIR" TMP="$VLLM_TMPDIR" TEMP="$VLLM_TMPDIR" \
    CUDA_VISIBLE_DEVICES="$gpu" python3 -m vllm.entrypoints.openai.api_server \
        --model "$QWEN36_PATH" --served-model-name Qwen3.6-35B-A3B \
        --host 127.0.0.1 --port "$port" \
        --max-model-len 32768 --trust-remote-code \
        --reasoning-parser qwen3 &
    printf -v "PID_LLM_${gpu}" '%s' "$!"
    SERVER_PIDS+=("$!")
done

wait_for_server "embedding server" "$EMBED_PORT" "$PID_EMBED" \
    "Qwen/Qwen3-Embedding-8B" 1200
for gpu in 1 2 3 4; do
    port_var="LLM_PORT_${gpu}"
    pid_var="PID_LLM_${gpu}"
    wait_for_server "Qwen3.6 GPU $gpu" "${!port_var}" "${!pid_var}" \
        "Qwen3.6-35B-A3B" 1800
done

gen_config() {
    local llm_port=$1 experiment_name=$2 build_strategy=$3 enable_vd=$4
    local cfg_path="$ALFWORLD_TMPDIR/alf_${experiment_name}_${RUN_STAMP}_$$.yaml"
    cat > "$cfg_path" <<CFGEOF
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
  experiment_name: ${experiment_name}
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
  tau: 0.62
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
  weight_sim: 0.5
  weight_q: 0.5
CFGEOF
    echo "$cfg_path"
}

# Region retains its v5 run ID and experiment name. Its checkpoint directory
# already contains snapshots, so AlfworldRunner auto-resumes the latest one.
(
    CFG=$(gen_config "$LLM_PORT_1" "alfworld_region_traj_qwen36" trajectory true)
    echo "[EXP] Region+FS trajectory resume: run=$REGION_RUN_ID config=$CFG"
    TMPDIR="$ALFWORLD_TMPDIR" TMP="$ALFWORLD_TMPDIR" TEMP="$ALFWORLD_TMPDIR" \
    MEMRL_RUN_ID="$REGION_RUN_ID" python3 run/run_alfworld.py \
        --config "$CFG" \
        --region --region_gating_mode additive \
        --region_utility_mode beta \
        --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
        --val_lambda_max 0.05 --no_z_norm \
        --explore_schedule '0,2,2,1,1,1,1,0,0,0' \
        --failure_summary_n_slots 2 \
        --skip_initial_eval
) &
PID_REGION=$!; EXP_PIDS+=("$PID_REGION")

# Baselines retain their original experiment names + qwen36_v9 run ID. Their
# ck_dir already contains snapshots, so AlfworldRunner's auto-resume restores
# the latest snapshot even though ckpt_resume_enabled is false in the config.
(
    CFG=$(gen_config "$LLM_PORT_2" "alfworld_memrl_qwen36" proceduralization true)
    echo "[EXP] MemRL resume: run=$BASELINE_RUN_ID config=$CFG"
    TMPDIR="$ALFWORLD_TMPDIR" TMP="$ALFWORLD_TMPDIR" TEMP="$ALFWORLD_TMPDIR" \
    MEMRL_RUN_ID="$BASELINE_RUN_ID" python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
) &
PID_MEMRL=$!; EXP_PIDS+=("$PID_MEMRL")

(
    CFG=$(gen_config "$LLM_PORT_3" "alfworld_memp_qwen36" proceduralization false)
    sed -i 's/tau: 0.62/tau: 0.0/' "$CFG"
    sed -i 's/weight_sim: 0.5/weight_sim: 1.0/' "$CFG"
    sed -i 's/weight_q: 0.5/weight_q: 0.0/' "$CFG"
    echo "[EXP] MemP resume: run=$BASELINE_RUN_ID config=$CFG"
    TMPDIR="$ALFWORLD_TMPDIR" TMP="$ALFWORLD_TMPDIR" TEMP="$ALFWORLD_TMPDIR" \
    MEMRL_RUN_ID="$BASELINE_RUN_ID" python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
) &
PID_MEMP=$!; EXP_PIDS+=("$PID_MEMP")

(
    CFG=$(gen_config "$LLM_PORT_4" "alfworld_rag_pure_qwen36" trajectory false)
    sed -i 's/tau: 0.62/tau: 0.0/' "$CFG"
    sed -i 's/weight_sim: 0.5/weight_sim: 1.0/' "$CFG"
    sed -i 's/weight_q: 0.5/weight_q: 0.0/' "$CFG"
    echo "[EXP] RAG resume: run=$BASELINE_RUN_ID config=$CFG"
    TMPDIR="$ALFWORLD_TMPDIR" TMP="$ALFWORLD_TMPDIR" TEMP="$ALFWORLD_TMPDIR" \
    MEMRL_RUN_ID="$BASELINE_RUN_ID" python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval
) &
PID_RAG=$!; EXP_PIDS+=("$PID_RAG")

echo "[INFO] launched: region=$PID_REGION memrl=$PID_MEMRL memp=$PID_MEMP rag=$PID_RAG"
failed=0
for entry in "Region:$PID_REGION" "MemRL:$PID_MEMRL" "MemP:$PID_MEMP" "RAG:$PID_RAG"; do
    name=${entry%%:*}
    pid=${entry##*:}
    if wait "$pid"; then
        echo "[DONE] $name exit=0 at $(date)"
    else
        rc=$?
        echo "[ERROR] $name exit=$rc at $(date)"
        failed=1
    fi
done

if ((failed)); then
    echo "[ERROR] one or more experiments failed"
    exit 1
fi

echo "=========================================="
echo "All experiments complete. End: $(date)"
echo "=========================================="
