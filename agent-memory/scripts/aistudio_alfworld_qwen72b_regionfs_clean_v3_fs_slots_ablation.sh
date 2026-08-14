#!/bin/bash
# Qwen2.5-72B ALFWorld Region+FS clean-v3 final-checkpoint evaluation ablation.
# Compares failure-summary slots 0 (Region-only), 1 (current full method), 2.
# This is evaluation-only: it must be run only after the clean-v3 resume writes
# a complete final snapshot/10 and FINAL RESULTS.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
: "${SNAPSHOT_PATH:?Set SNAPSHOT_PATH to the completed clean-v3 resume local_cache/snapshot/10}"

LOGFILE="$MEMRL_DIR/logs/aistudio_qwen72b_regionfs_clean_v3_fs_slots_ablation_${RUN_TAG}.log"
EMBED_PORT="${EMBED_PORT:-19080}"
LLM_PORT="${LLM_PORT:-19280}"
NCCL_PORT="${NCCL_PORT:-29680}"
WORK_TMP_BASE="/storage/openpsi/users/yl/agent-memory/.tmp"
WORK_TMP="$WORK_TMP_BASE/qwen72b_regionfs_fs_slots_ablation_${RUN_TAG}_$$"
mkdir -p "$WORK_TMP"
chmod 700 "$WORK_TMP"

mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1

[[ -f "$SNAPSHOT_PATH/snapshot_meta.json" ]]
[[ -f "$SNAPSHOT_PATH/local_cache/cum_state.json" ]]
[[ -f "$SNAPSHOT_PATH/local_cache/region_manager.json" ]]
[[ -f "$SNAPSHOT_PATH/cube/textual_memory.json" ]]

echo "============================================================"
echo "Qwen2.5-72B Region+FS clean-v3 FS-slots ablation (eval only)"
echo "Start: $(date)"
echo "Snapshot: $SNAPSHOT_PATH"
echo "Cells: FS slots = 0 (Region-only), 1 (full method), 2"
echo "Fixed: tau=0.60, ws=0.45, wq=0.55, lambda_max=0.45"
echo "============================================================"
df -h "$WORK_TMP" || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd "$MEMRL_DIR"
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld --target "$VENV_SP" \
    -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5

# CPFS does not support SGLang's Unix-domain ZMQ IPC socket. Explicitly keep
# server startup on node-local /tmp; redirect only ALFWorld/TextWorld afterward.
unset TMPDIR TMP TEMP
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
    --model-path "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port "$EMBED_PORT" --context-length 8192 \
    --trust-remote-code --is-embedding &
EMBED_PID=$!
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server \
    --model-path "$QWEN72B_PATH" --served-model-name Qwen2.5-72B-Instruct \
    --tp 2 --host 127.0.0.1 --port "$LLM_PORT" --context-length 32768 \
    --nccl-port "$NCCL_PORT" --trust-remote-code &
LLM_PID=$!

cleanup() {
    status=$?
    kill "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
    wait "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
    rm -rf -- "$WORK_TMP"
    exit "$status"
}
trap cleanup EXIT INT TERM

for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/v1/models" 2>/dev/null | grep -q model && { echo '[INFO] Embed ready'; break; }
    [[ "$i" -eq 1200 ]] && { echo '[ERROR] Embed timeout'; exit 1; }
    sleep 1
done
for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT}/v1/models" 2>/dev/null | grep -q model && { echo '[INFO] LLM ready'; break; }
    [[ "$i" -eq 1800 ]] && { echo '[ERROR] LLM timeout'; exit 1; }
    sleep 1
done

# Fast Downward uses Python tempfile to copy libdownward.so. Shared storage has
# sufficient quota; apply it only after SGLang already owns its local IPC paths.
export TMPDIR="$WORK_TMP"
export TMP="$WORK_TMP"
export TEMP="$WORK_TMP"
python3 - <<'PY'
import os, tempfile
assert tempfile.gettempdir() == os.environ['TMPDIR']
with tempfile.NamedTemporaryFile(dir=os.environ['TMPDIR'], delete=True) as f:
    f.write(b'ok')
    f.flush()
print('[TMP-PREFLIGHT] ALFWorld/TextWorld tempfile=' + tempfile.gettempdir())
PY

for FS_SLOTS in 0 1 2; do
    CELL="fs${FS_SLOTS}"
    CFG="$WORK_TMP/${CELL}.yaml"
    cat > "$CFG" <<CFGEOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${LLM_PORT}/v1/
  model: Qwen2.5-72B-Instruct
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
  experiment_name: alfworld_regionfs_qwen72b_clean_v3_ablation_${CELL}_${RUN_TAG}
  mode: test
  num_sections: 10
  batch_size: 128
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  output_dir: /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_ablation
  max_steps: 30
  save_trajectories: true
  save_memories: false
  ckpt_resume_enabled: true
  ckpt_resume_path: ${SNAPSHOT_PATH}
  ckpt_resume_epoch: null
  n_eval_runs: 4
  eval_temperature: 0.2
rl_config:
  epsilon: 0
  tau: 0.60
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
  weight_sim: 0.45
  weight_q: 0.55
CFGEOF
    echo "[ABLATION] $CELL: failure_summary_n_slots=$FS_SLOTS"
    export MEMRL_RUN_ID="qwen72b-regionfs-clean-v3-ablation-${CELL}-${RUN_TAG//_/-}"
    python3 run/run_alfworld.py --config "$CFG" \
        --region --region_gating_mode additive --region_utility_mode beta \
        --shrinkage_confidence_k 2.5 --propagation_eta 0.12 \
        --val_lambda_max 0.45 --no_z_norm \
        --failure_summary_n_slots "$FS_SLOTS"
    echo "[ABLATION] $CELL complete at $(date)"
done

echo "[DONE] FS-slot ablation complete at $(date)"
