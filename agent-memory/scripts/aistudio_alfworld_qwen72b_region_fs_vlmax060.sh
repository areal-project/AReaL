#!/bin/bash
# Qwen2.5-72B ALFWorld Region+FS eval-only strict control on the repaired E10 snapshot.
# Uses the original val_lambda_max=0.60. All other repaired-grid parameters are fixed.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_region_qwen72b_v2_traj_20260714-101803/local_cache/snapshot/10}"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
LOGFILE="$MEMRL_DIR/logs/aistudio_qwen72b_region_fs_vlmax060_${RUN_TAG}.log"
EMBED_PORT="${EMBED_PORT:-19060}"
LLM_PORT="${LLM_PORT:-19260}"

mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages

cd "$MEMRL_DIR"
if [[ ! -f "$SNAPSHOT_PATH/local_cache/region_manager.json" ]]; then
    echo "[ERROR] Missing RegionManager snapshot: $SNAPSHOT_PATH/local_cache/region_manager.json"
    exit 1
fi

echo "============================================================"
echo "Qwen72B ALFWorld Region+FS strict E10 control"
echo "Snapshot: $SNAPSHOT_PATH"
echo "val_lambda_max: 0.60"
echo "Start: $(date)"
echo "============================================================"

pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld --target "$VENV_SP" \
    -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
    --model-path "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port "$EMBED_PORT" --context-length 8192 \
    --trust-remote-code --is-embedding &
EMBED_PID=$!

CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server \
    --model-path "$QWEN72B_PATH" --served-model-name Qwen2.5-72B-Instruct \
    --tp 2 --host 127.0.0.1 --port "$LLM_PORT" --trust-remote-code \
    --context-length 32768 --nccl-port 29660 &
LLM_PID=$!

cleanup() {
    kill "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
    wait "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
}
trap cleanup EXIT

for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/v1/models" 2>/dev/null | grep -q model && break
    [[ "$i" -eq 1200 ]] && echo "[ERROR] Embedding server timeout" && exit 1
    sleep 1
done
for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT}/v1/models" 2>/dev/null | grep -q model && break
    [[ "$i" -eq 1800 ]] && echo "[ERROR] LLM server timeout" && exit 1
    sleep 1
done

echo "[INFO] Both servers ready. Starting strict eval-only control."

for VLMAX in 0.60; do
    CELL_TAG="vlmax${VLMAX/./p}"
    CONFIG="/tmp/alf_qwen72b_region_fs_${CELL_TAG}_$$.yaml"
    cat > "$CONFIG" <<CFGEOF
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
  load_from_checkpoint: true
  checkpoint_path: ${SNAPSHOT_PATH}
environment:
  alfworld_config_path: configs/envs/alfworld.yaml
  alfworld_env_type: AlfredTWEnv
experiment:
  random_seed: 42
  enable_value_driven: true
  experiment_name: alfworld_region_fs_qwen72b_e10_${CELL_TAG}_${RUN_TAG}
  mode: test
  num_sections: 10
  batch_size: 128
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: null
  baseline_k: 10
  output_dir: /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_grid
  max_steps: 30
  save_trajectories: true
  save_memories: false
  ckpt_resume_enabled: false
  ckpt_resume_path: ""
  ckpt_resume_epoch: null
  n_eval_runs: 1
rl_config:
  epsilon: 0
  tau: 0.58
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
  weight_sim: 0.35
  weight_q: 0.65
CFGEOF

    echo "============================================================"
    echo "[CONTROL] $CELL_TAG: val_lambda_max=$VLMAX, Region=ON, FS slots=2"
    echo "[CONTROL] Fixed: tau=0.58, wq=0.65, ws=0.35, shrinkage_k=2.5, no_z_norm"
    echo "============================================================"
    export MEMRL_RUN_ID="qwen72b-region-fs-e10-${CELL_TAG}-${RUN_TAG//_/-}"
    python3 run/run_alfworld.py \
        --config "$CONFIG" \
        --region --region_gating_mode additive --region_utility_mode beta \
        --shrinkage_confidence_k 2.5 --propagation_eta 0.12 \
        --val_lambda_max "$VLMAX" --no_z_norm \
        --failure_summary_n_slots 2
    rm -f "$CONFIG"
    echo "[CONTROL] Completed $CELL_TAG at $(date)"
done

echo "[DONE] Region+FS vlmax=0.60 control completed at $(date)"
