#!/bin/bash
# Qwen2.5-72B ALFWorld Region+FS parameter ablation: three full-training cells
# in one AIS job, sharing one embedding server and one Qwen72B server.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG_SAFE="${RUN_TAG//_/-}"

EMBED_PORT="${EMBED_PORT:-19090}"
LLM_PORT="${LLM_PORT:-19290}"
NCCL_PORT="${NCCL_PORT:-29690}"
OUTPUT_ROOT="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_regionfs_param_ablation_20260721"
LOG_ROOT="$MEMRL_DIR/logs/alfworld_regionfs_qwen72b_param_ablation_3cells_${RUN_TAG}"
DRIVER_LOG="$LOG_ROOT/driver.log"
WORK_TMP_BASE="/storage/openpsi/users/yl/agent-memory/.tmp"
WORK_TMP="$WORK_TMP_BASE/qwen72b_regionfs_param_ablation_3cells_${RUN_TAG}_$$"

mkdir -p "$LOG_ROOT" "$WORK_TMP" "$OUTPUT_ROOT"
chmod 700 "$WORK_TMP"
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "================================================================"
echo "Qwen2.5-72B Region+FS parameter ablation: 3 cells / 1 AIS job"
echo "Start: $(date --iso-8601=seconds)"
echo "Run tag: $RUN_TAG"
echo "Shared servers: embedding GPU0; Qwen72B TP=2 GPU1-2"
echo "Cell A: eta=0.03 (single-variable propagation ablation)"
echo "Cell B: shrinkage_k=5.0 (single-variable shrinkage ablation)"
echo "Cell C: tau=0.62, ws/wq=0.50/0.50, eta=0.03, k=5.0 (combo)"
echo "All cells: full 10 epochs from empty memory; FS slots=1"
echo "Logs: $LOG_ROOT"
echo "Outputs: $OUTPUT_ROOT"
echo "================================================================"

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
python3 -c "import memrl; import memos; print('[OK] memrl + memos imported')"

# SGLang uses Unix-domain ZMQ IPC under the process temp directory. CPFS does
# not support that socket mode, so keep server startup on node-local /tmp.
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
    trap - EXIT INT TERM
    echo "[CLEANUP] status=$status at $(date --iso-8601=seconds)"
    kill "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
    wait "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
    rm -rf -- "$WORK_TMP"
    exit "$status"
}
trap cleanup EXIT INT TERM

wait_for_server() {
    local name="$1" port="$2" max_wait="$3"
    echo "[INFO] Waiting for $name on port $port..."
    for i in $(seq 1 "$max_wait"); do
        if curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q model; then
            echo "[INFO] $name ready after ${i}s"
            return 0
        fi
        if ! kill -0 "$EMBED_PID" 2>/dev/null || ! kill -0 "$LLM_PID" 2>/dev/null; then
            echo "[ERROR] A shared server exited while waiting for readiness"
            return 1
        fi
        sleep 1
    done
    echo "[ERROR] $name readiness timeout after ${max_wait}s"
    return 1
}

wait_for_server Embed "$EMBED_PORT" 1200
wait_for_server LLM "$LLM_PORT" 1800

# Fast Downward copies libdownward.so through Python tempfile. Redirect only
# ALFWorld/TextWorld temp files after SGLang has established local IPC sockets.
export TMPDIR="$WORK_TMP"
export TMP="$WORK_TMP"
export TEMP="$WORK_TMP"
python3 - <<'PY'
import os
import tempfile
assert tempfile.gettempdir() == os.environ["TMPDIR"]
with tempfile.NamedTemporaryFile(dir=os.environ["TMPDIR"], delete=True) as f:
    f.write(b"ok")
    f.flush()
print("[TMP-PREFLIGHT] ALFWorld/TextWorld tempfile=" + tempfile.gettempdir())
PY

write_config() {
    local cfg="$1" exp_name="$2" tau="$3" weight_sim="$4" weight_q="$5"
    cat > "$cfg" <<CFGEOF
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
  experiment_name: ${exp_name}
  mode: train
  num_sections: 10
  batch_size: 128
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: null
  baseline_k: 10
  output_dir: ${OUTPUT_ROOT}
  max_steps: 30
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
CFGEOF
}

run_cell() {
    local cell="$1" label="$2" eta="$3" shrink_k="$4" tau="$5" weight_sim="$6" weight_q="$7"
    local exp_name="alfworld_regionfs_qwen72b_${label}_fs1_${RUN_TAG}"
    local cfg="$WORK_TMP/${cell}.yaml"
    local cell_log="$LOG_ROOT/${cell}_${label}.log"

    write_config "$cfg" "$exp_name" "$tau" "$weight_sim" "$weight_q"
    export MEMRL_RUN_ID="qwen72b-regionfs-${label}-fs1-${RUN_TAG_SAFE}"

    {
        echo "============================================================"
        echo "[CELL-START] $cell / $label at $(date --iso-8601=seconds)"
        echo "[CELL-CONFIG] eta=$eta shrinkage_k=$shrink_k tau=$tau ws=$weight_sim wq=$weight_q fs_slots=1"
        echo "[CELL-CONFIG] experiment_name=$exp_name"
        echo "[CELL-CONFIG] MEMRL_RUN_ID=$MEMRL_RUN_ID"
        echo "[CELL-CONFIG] config=$cfg"
        echo "============================================================"
        python3 run/run_alfworld.py \
            --config "$cfg" \
            --region --region_gating_mode additive \
            --region_utility_mode beta \
            --shrinkage_confidence_k "$shrink_k" \
            --propagation_eta "$eta" \
            --val_lambda_max 0.45 --no_z_norm \
            --explore_schedule '0,1,1,1,0,0,0,0,0,0' \
            --failure_summary_n_slots 1
        echo "[CELL-DONE] $cell / $label exit=0 at $(date --iso-8601=seconds)"
    } 2>&1 | tee -a "$cell_log"
}

# A/B are strict one-parameter changes from clean-v3 control. C is a combined
# candidate configuration and is intentionally not labelled single-variable.
run_cell cell_a eta0p03       0.03 2.5 0.60 0.45 0.55
run_cell cell_b shrinkk5      0.12 5.0 0.60 0.45 0.55
run_cell cell_c combo_t062    0.03 5.0 0.62 0.50 0.50

echo "[ALL-DONE] All three Region+FS cells completed at $(date --iso-8601=seconds)"
