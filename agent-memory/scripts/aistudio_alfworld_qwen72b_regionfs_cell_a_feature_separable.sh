#!/bin/bash
# Qwen2.5-72B ALFWorld Region+FS Cell A feature-separability ablation after split-posterior fix.
# One full 10-section experiment from empty memory; no resume.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG_SAFE="${RUN_TAG//_/-}"

EMBED_PORT="${EMBED_PORT:-19090}"
LLM_PORT="${LLM_PORT:-19290}"
NCCL_PORT="${NCCL_PORT:-29690}"
OUTPUT_ROOT="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_regionfs_cell_a_feature_separable_20260725"
LOG_ROOT="$MEMRL_DIR/logs/alfworld_regionfs_qwen72b_cell_a_feature_separable_${RUN_TAG}"
DRIVER_LOG="$LOG_ROOT/driver.log"
WORK_TMP_BASE="/storage/openpsi/users/yl/agent-memory/.tmp"
WORK_TMP="$WORK_TMP_BASE/qwen72b_regionfs_cell_a_feature_separable_${RUN_TAG}_$$"

mkdir -p "$LOG_ROOT" "$WORK_TMP" "$OUTPUT_ROOT"
chmod 700 "$WORK_TMP"
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "================================================================"
echo "Qwen2.5-72B Region+FS Cell A feature-separability-only ablation"
echo "Start: $(date --iso-8601=seconds)"
echo "Run tag: $RUN_TAG"
echo "Shared servers: embedding GPU0; Qwen72B TP=2 GPU1-2"
echo "Cell A fixed: eta=0.03, shrinkage_k=2.5, tau=0.60, ws/wq=0.45/0.55, FS slots=1"
echo "Split fix: source-conserving soft evidence migration; no q*count reconstruction"
echo "Only added variable group: anchor-first raw-dominant Region geometry; topology-stability schedule retained"
echo "Logs: $LOG_ROOT"
echo "Outputs: $OUTPUT_ROOT"
echo "================================================================"

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export MEMRL_FEATURE_SEPARABILITY_SOURCE="$MEMRL_DIR"
export PYTHONPATH="$MEMRL_DIR/scripts/region_feature_separability_overlay:$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages

cd "$MEMRL_DIR"
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld --target "$VENV_SP" \
    -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 -c "import memrl; import memos; print('[OK] memrl + memos imported')"
python3 - <<'PYFIX'
import inspect
from memrl.service.region_manager import RegionManager
src = inspect.getsource(RegionManager)
required = [
    "_split_posterior_states",
    "_split_child_routing_weights",
    "region_source_success_by_region",
    "region_source_total_by_region",
    "parent.prior_alpha_by_subtask",
    "parent.prior_beta_by_subtask",
    "soft_source_conserving",
]
for token in required:
    assert token in src, f"source-conserving split fix missing token: {token}"
assert "self.subtask_q[m].get(st, 0.5) * c" not in src, "legacy q*count pseudo-evidence split logic is still active"
assert "q_val * float(q_count)" not in src, "legacy q*count pseudo-evidence split logic is still active"
assert inspect.signature(RegionManager).parameters["region_split_evidence_migration_mode"].default == "soft_source_conserving"
print("[SPLIT-FIX-PREFLIGHT] PASS: parent priors inherited; source-attributed soft evidence is sibling-rerouted and conserved; no q*count reconstruction")
PYFIX
OVERLAY_DIR="$MEMRL_DIR/scripts/region_feature_separability_overlay"
test -f "$OVERLAY_DIR/run_alfworld.py"
test -f "$OVERLAY_DIR/memrl/run/alfworld_rl_runner.py"
python3 - "$OVERLAY_DIR" <<'PYTOPO'
import sys
from pathlib import Path
overlay = Path(sys.argv[1])
cli = (overlay / "run_alfworld.py").read_text()
runner = (overlay / "memrl/run/alfworld_rl_runner.py").read_text()
for token in [
    "region_cluster_init_step", "region_merge_interval",
    "region_disable_mid_epoch_topology", "region_topology_cooldown_sections",
]:
    assert token in cli, f"missing topology CLI control: {token}"
assert "_MID_EPOCH_TOPOLOGY" in runner
assert "skipped region split/merge due to" in runner
assert "Initial clustering changes topology too" in runner
feature_mgr = overlay / "memrl/service/region_manager.py"
assert feature_mgr.is_file(), "feature RegionManager overlay missing"
feature_src = feature_mgr.read_text()
for token in [
    "feature_separability_mode", "anchor_blended", "_feature_anchor_mask",
    "_feature_blended_distance_matrix", "_assign_to_anchor_clusters",
    "Feature-separability clustering",
]:
    assert token in feature_src, f"feature overlay missing {token}"
print("[FEATURE-SEPARABILITY-PREFLIGHT] PASS: anchor_blended raw_w=0.70; min_dims=2; min_count=3; overlap_penalty=0.15; stable topology retained")
PYTOPO

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
        echo "[CELL-CONFIG] split_evidence_migration=soft_source_conserving"
        echo "[CELL-CONFIG] topology=init_step=3000,mid_epoch=false,merge_interval=3553,cooldown_sections=1; feature=anchor_blended,raw_w=0.70,min_dims=2,min_count=3,penalty=0.15"
        echo "[CELL-CONFIG] experiment_name=$exp_name"
        echo "[CELL-CONFIG] MEMRL_RUN_ID=$MEMRL_RUN_ID"
        echo "[CELL-CONFIG] config=$cfg"
        echo "============================================================"
        python3 "$MEMRL_DIR/scripts/region_feature_separability_overlay/run_alfworld.py" \
            --config "$cfg" \
            --region --region_gating_mode additive \
            --region_utility_mode beta \
            --region_split_evidence_migration_mode soft_source_conserving \
            --region_cluster_init_step 3000 \
            --region_merge_interval 3553 \
            --region_disable_mid_epoch_topology \
            --region_topology_cooldown_sections 1 \
            --region_feature_separability_mode anchor_blended \
            --region_feature_raw_distance_weight 0.70 \
            --region_feature_min_observed_dims 2 \
            --region_feature_min_total_q_count 3 \
            --region_feature_low_overlap_penalty 0.15 \
            --shrinkage_confidence_k "$shrink_k" \
            --propagation_eta "$eta" \
            --val_lambda_max 0.45 --no_z_norm \
            --explore_schedule '0,1,1,1,0,0,0,0,0,0' \
            --failure_summary_n_slots 1
        echo "[CELL-DONE] $cell / $label exit=0 at $(date --iso-8601=seconds)"
    } 2>&1 | tee -a "$cell_log"
}

# Cell A rerun: exact same hyperparameters as the old-split Cell A, but from
# empty memory with the corrected split posterior migration.
run_cell cell_a eta0p03_feature_separable 0.03 2.5 0.60 0.45 0.55

echo "[ALL-DONE] Region+FS Cell A feature-separability ablation completed at $(date --iso-8601=seconds)"
