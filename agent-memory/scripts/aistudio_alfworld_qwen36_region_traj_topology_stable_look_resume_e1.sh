#!/bin/bash
# AIStudio: Qwen3.6 ALFWorld — Region+FS trajectory memory only — topology-stability fallback-look continuation (Region+FS/additive retained).
# GPU 0: embedding server; GPU 1: Qwen3.6 vLLM + Region+FS experiment.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RUN_STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
LOGFILE="$MEMRL_DIR/logs/aistudio_qwen36_region_traj_${RUN_STAMP}.log"

EMBED_PORT=9000
LLM_PORT=9101
# A new ID makes this an empty checkpoint namespace; do not reuse v5/v3.
export MEMRL_RUN_ID="qwen36_region_traj_v10_topology_stable_look_resume_e1"
# Region-only override: safe no-op for missing/invalid batched actions.
export MEMRL_ALFWORLD_FALLBACK_ACTION="look"
export MEMRL_ALFWORLD_LLM_CONCURRENCY="${MEMRL_ALFWORLD_LLM_CONCURRENCY:-8}"
export MEMRL_LLM_CLIENT_TIMEOUT_S="${MEMRL_LLM_CLIENT_TIMEOUT_S:-600}"
export MEMRL_LLM_MAX_RETRIES="${MEMRL_LLM_MAX_RETRIES:-1}"

mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "AIStudio: Qwen3.6 ALFWorld — Region+FS trajectory-only"
echo "Run ID: $MEMRL_RUN_ID (fallback=look continuation from completed E1)"
echo "[CONFIG] Region-only fallback action: $MEMRL_ALFWORLD_FALLBACK_ACTION"
echo "Start: $(date)"
echo "=========================================="

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
USER_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
export PYTHONPATH="$MEMRL_DIR:$VENV_SP:$USER_SP:${PYTHONPATH:-}"

cd "$MEMRL_DIR"
# The vLLM image's runtime python may not bundle `python3 -m pip`, while its
# standalone pip installs Python 3.12 packages into the requested target.
PIP_BIN="$(command -v pip || command -v pip3 || true)"
if [[ -z "$PIP_BIN" ]]; then
    echo "[ERROR] neither pip nor pip3 is available in the runtime image"
    exit 1
fi
echo "[PREFLIGHT] python executable: $(command -v python3)"
echo "[PREFLIGHT] pip executable: $PIP_BIN"
# Import memrl directly from this mounted source tree.  Installing `.` rebuilds
# a wheel in a shared/reused workspace and can leave a stale dist-info directory.
# Dependencies are still installed into the runtime target below.
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
    mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld

# Fail before loading models if the exact runtime cannot see Region dependencies.
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
cleanup() {
    status=$?
    echo "[INFO] cleanup (status=$status) at $(date)"
    if ((${#SERVER_PIDS[@]})); then
        kill "${SERVER_PIDS[@]}" 2>/dev/null || true
        wait "${SERVER_PIDS[@]}" 2>/dev/null || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port "$EMBED_PORT" \
    --max-model-len 8192 --trust-remote-code \
    --convert embed &
SERVER_PIDS+=("$!")

CUDA_VISIBLE_DEVICES=1 python3 -m vllm.entrypoints.openai.api_server \
    --model "$QWEN36_PATH" --served-model-name Qwen3.6-35B-A3B \
    --host 127.0.0.1 --port "$LLM_PORT" \
    --max-model-len 32768 --trust-remote-code \
    --reasoning-parser qwen3 &
SERVER_PIDS+=("$!")

wait_for_server() {
    local name=$1 port=$2 timeout=$3
    echo "[INFO] Waiting for $name on port $port..."
    for ((i=1; i<=timeout; i++)); do
        if curl -fsS "http://localhost:${port}/v1/models" 2>/dev/null | grep -q 'model'; then
            echo "[INFO] $name ready on port $port"
            return 0
        fi
        sleep 1
    done
    echo "[ERROR] $name timeout after ${timeout}s"
    return 1
}

wait_for_server "embedding server" "$EMBED_PORT" 1200
wait_for_server "Qwen3.6 server" "$LLM_PORT" 1800

CFG="/tmp/alfworld_region_traj_qwen36_${RUN_STAMP}_$$.yaml"
cat > "$CFG" <<CFGEOF
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
  enable_value_driven: true
  experiment_name: alfworld_region_traj_qwen36
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
  # Read the completed topology-stable E1 checkpoint only; write continuation output separately.
  ckpt_resume_enabled: true
  ckpt_resume_path: "/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_region_traj_qwen36_qwen36_region_traj_v9_topology_stable/local_cache/snapshot/1"
  ckpt_resume_epoch: null
  batch_checkpoint_interval: 5
  batch_checkpoint_keep: 3
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

echo "[EXP] Region+FS topology-stability fallback-look continuation from E1 starting with config $CFG"
python3 run/run_alfworld.py \
    --config "$CFG" \
    --region --region_gating_mode additive \
    --region_utility_mode beta \
    --region_cluster_init_step 3000 \
    --region_merge_interval 3553 \
    --region_disable_mid_epoch_topology \
    --region_topology_cooldown_sections 1 \
    --shrinkage_confidence_k 3.0 \
    --propagation_eta 0.12 \
    --val_lambda_max 0.05 \
    --no_z_norm \
    --explore_schedule '0,2,2,1,1,1,1,0,0,0' \
    --failure_summary_n_slots 2 \
    --skip_initial_eval

echo "[DONE] Region+FS trajectory completed at $(date)"
