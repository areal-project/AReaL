#!/bin/bash
# Resume Qwen2.5-72B ALFWorld Region+FS clean v3 from healthy S10 batch-10 checkpoint.
# GPU 0: embedding; GPU 1-2: Qwen2.5-72B TP=2.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RESUME_CKPT="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_regionfs_qwen72b_clean_v3_20260717-230550/local_cache/snapshot/s10_b10"
TS="${1:-$(date +%Y%m%d_%H%M%S)}"
LOGFILE="$MEMRL_DIR/logs/aistudio_qwen72b_regionfs_clean_v3_resume_s10b10_${TS}.log"

EMBED_PORT="${EMBED_PORT:-19070}"
LLM_PORT_1="${LLM_PORT:-19270}"
NCCL_PORT="${NCCL_PORT:-29670}"

export MEMRL_RUN_ID="qwen72b-regionfs-clean-v3-resume-s10b10-${TS//_/-}"

# TextWorld/Fast Downward copies libdownward.so through tempfile.  Do not use the
# node's quota-limited /tmp; keep this run's temporary files on shared storage.
WORK_TMP_BASE="/storage/openpsi/users/yl/agent-memory/.tmp"
WORK_TMP="$WORK_TMP_BASE/qwen72b_regionfs_resume_s10b10_${TS}_$$"
mkdir -p "$WORK_TMP"
chmod 700 "$WORK_TMP"
CFG="$WORK_TMP/alf_regionfs_resume_s10b10.yaml"

mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "AIStudio: Qwen2.5-72B Region+FS clean v3 resume S10-b10"
echo "Start: $(date)"
echo "Resume checkpoint: $RESUME_CKPT"
echo "ALFWorld TMPDIR (set after servers start): $WORK_TMP"
echo "MEMRL_RUN_ID: $MEMRL_RUN_ID"
echo "=========================================="

test -f "$RESUME_CKPT/snapshot_meta.json"
test -f "$RESUME_CKPT/local_cache/cum_state.json"
test -f "$RESUME_CKPT/cube/textual_memory.json"
df -h "$WORK_TMP" || true
df -i "$WORK_TMP" || true
WORK_TMP_FOR_PREFLIGHT="$WORK_TMP" python3 - <<'PY'
import os, tempfile
p = os.environ["WORK_TMP_FOR_PREFLIGHT"]
with tempfile.NamedTemporaryFile(prefix="memrl_tmp_preflight_", delete=True, dir=p) as f:
    f.write(b"ok")
    f.flush()
print(f"[TMP-PREFLIGHT] shared temp dir={p}; write=ok")
PY

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
export PYTHONPATH="$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"

cd "$MEMRL_DIR"
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
    concurrent-log-handler textworld alfworld \
    --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 -c "import memrl; import memos; print('[OK] memrl + memos imported')"
WORK_TMP_FOR_PREFLIGHT="$WORK_TMP" python3 - <<'PY'
import glob, os, shutil
matches = glob.glob('/AReaL/.venv/lib/python3.12/site-packages/**/fast_downward/libdownward.so', recursive=True)
if not matches:
    raise SystemExit('[TMP-PREFLIGHT] libdownward.so not found')
dst = os.path.join(os.environ['WORK_TMP_FOR_PREFLIGHT'], 'libdownward_preflight.so')
shutil.copy2(matches[0], dst)
print(f'[TMP-PREFLIGHT] copied {matches[0]} -> {dst} ({os.path.getsize(dst)} bytes)')
os.unlink(dst)
PY

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
    --model-path "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port "$EMBED_PORT" \
    --context-length 8192 --trust-remote-code --is-embedding &
EMBED_PID=$!

CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server \
    --model-path "$QWEN72B_PATH" --served-model-name Qwen2.5-72B-Instruct \
    --tp 2 --host 127.0.0.1 --port "$LLM_PORT_1" \
    --trust-remote-code --context-length 32768 --nccl-port "$NCCL_PORT" &
LLM_PID=$!

cleanup() {
    status=$?
    kill "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
    wait "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
    if [[ "$WORK_TMP" == "$WORK_TMP_BASE"/qwen72b_regionfs_resume_s10b10_* ]]; then
        rm -rf -- "$WORK_TMP"
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

echo "[INFO] Waiting for Embed..."
for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] Embed ready!" && break
    [ "$i" -eq 1200 ] && echo "[ERROR] Embed timeout" && exit 1
    sleep 1
done
echo "[INFO] Waiting for LLM..."
for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT_1}/v1/models" 2>/dev/null | grep -q "model" && echo "[INFO] LLM ready!" && break
    [ "$i" -eq 1800 ] && echo "[ERROR] LLM timeout" && exit 1
    sleep 1
done

# SGLang uses tempfile for Unix-domain ZMQ sockets; CPFS does not support those
# sockets. Start servers with the node-local default, then redirect only the
# ALFWorld/TextWorld runner to shared storage before it creates environments.
export TMPDIR="$WORK_TMP"
export TMP="$WORK_TMP"
export TEMP="$WORK_TMP"
echo "[TMP] ALFWorld/TextWorld TMPDIR=$TMPDIR"
python3 - <<'PY'
import os, tempfile
assert tempfile.gettempdir() == os.environ['TMPDIR']
print('[TMP] runner tempfile.gettempdir()=' + tempfile.gettempdir())
PY

cat > "$CFG" <<CFGEOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${LLM_PORT_1}/v1/
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
  experiment_name: alfworld_regionfs_qwen72b_clean_v3_resume_s10b10
  mode: train
  num_sections: 10
  batch_size: 128
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: null
  baseline_k: 10
  output_dir: /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld
  max_steps: 30
  save_trajectories: true
  save_memories: true
  ckpt_resume_enabled: true
  ckpt_resume_path: ${RESUME_CKPT}
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

echo "[EXP] 72B Region+FS clean v3 resume starting..."
echo "[EXP] Expected resume: section=10 batch=11; business log under logs/alfworld_regionfs_qwen72b_clean_v3_resume_s10b10/"
python3 run/run_alfworld.py \
    --config "$CFG" --skip_initial_eval \
    --region --region_gating_mode additive \
    --region_utility_mode beta \
    --shrinkage_confidence_k 2.5 --propagation_eta 0.12 \
    --val_lambda_max 0.45 --no_z_norm \
    --explore_schedule '0,1,1,1,0,0,0,0,0,0' \
    --failure_summary_n_slots 1
rc=$?
echo "[EXP] 72B Region+FS clean v3 resume exit=$rc at $(date)"
exit "$rc"
