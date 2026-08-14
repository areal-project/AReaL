#!/bin/bash
# Resume Qwen2.5-72B ALFWorld RAG from the last fully valid checkpoint (end S4).
# S5 is rerun in full so its section-level train denominator remains exactly 3553;
# resuming from a mid-S5 batch checkpoint would omit earlier S5 trajectories.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RESUME_CKPT="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_rag_qwen72b_qwen72b_rag_v1/local_cache/snapshot/4"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG_SAFE="${RUN_TAG//_/-}"
LOGFILE="$MEMRL_DIR/logs/aistudio_qwen72b_rag_resume_s5_full_${RUN_TAG}.log"
BUSINESS_LOG_DIR="$MEMRL_DIR/logs/alfworld_rag_qwen72b_resume_s5_full_${RUN_TAG}"

EMBED_PORT="${EMBED_PORT:-19120}"
LLM_PORT="${LLM_PORT:-19320}"
WORK_TMP_BASE="/storage/openpsi/users/yl/agent-memory/.tmp"
WORK_TMP="$WORK_TMP_BASE/qwen72b_rag_resume_s5_full_${RUN_TAG}_$$"
CFG="$WORK_TMP/alf_rag_resume_s5_full.yaml"

mkdir -p "$WORK_TMP" "$BUSINESS_LOG_DIR" "$(dirname "$LOGFILE")"
chmod 700 "$WORK_TMP"
exec > >(tee -a "$LOGFILE") 2>&1

export MEMRL_RUN_ID="qwen72b-rag-resume-s5-full-${RUN_TAG_SAFE}"
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages

printf '%s\n' \
  '============================================================' \
  'Qwen2.5-72B ALFWorld RAG accurate resume' \
  "Start: $(date --iso-8601=seconds)" \
  "Resume checkpoint: $RESUME_CKPT (completed S4)" \
  'Expected resume point: section 5, batch 1 (full S5 rerun)' \
  "Supervisor log: $LOGFILE" \
  "Business log directory: $BUSINESS_LOG_DIR" \
  "Runner TMPDIR after server startup: $WORK_TMP" \
  "MEMRL_RUN_ID: $MEMRL_RUN_ID" \
  '============================================================'

test -f "$RESUME_CKPT/snapshot_meta.json"
test -f "$RESUME_CKPT/local_cache/cum_state.json"
test -f "$RESUME_CKPT/cube/textual_memory.json"

cd "$MEMRL_DIR"
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
  concurrent-log-handler textworld alfworld --target "$VENV_SP" \
  -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 -c "import memrl; import memos; print('[OK] memrl + memos imported')"

# Keep vLLM startup on node-local temp. Redirect only the ALFWorld/TextWorld
# runner after servers are ready; Fast Downward copies libdownward.so via tempfile.
unset TMPDIR TMP TEMP
CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
  --model "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
  --host 127.0.0.1 --port "$EMBED_PORT" --max-model-len 8192 \
  --trust-remote-code --convert embed &
EMBED_PID=$!

CUDA_VISIBLE_DEVICES=1,2 python3 -m vllm.entrypoints.openai.api_server \
  --model "$QWEN72B_PATH" --served-model-name Qwen2.5-72B-Instruct \
  --host 127.0.0.1 --port "$LLM_PORT" --max-model-len 32768 \
  --trust-remote-code --tensor-parallel-size 2 &
LLM_PID=$!

cleanup() {
  status=$?
  trap - EXIT INT TERM
  echo "[CLEANUP] status=$status at $(date --iso-8601=seconds)"
  kill "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
  wait "$LLM_PID" "$EMBED_PID" 2>/dev/null || true
  if [[ "$WORK_TMP" == "$WORK_TMP_BASE"/qwen72b_rag_resume_s5_full_* ]]; then
    rm -rf -- "$WORK_TMP"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local name="$1" port="$2" pid="$3" max_wait="$4"
  echo "[INFO] Waiting for $name on port $port..."
  for i in $(seq 1 "$max_wait"); do
    kill -0 "$pid" 2>/dev/null || { echo "[ERROR] $name exited before readiness"; return 1; }
    if curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q model; then
      echo "[INFO] $name ready after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] $name readiness timeout after ${max_wait}s"
  return 1
}
wait_for_server Embed "$EMBED_PORT" "$EMBED_PID" 1200
wait_for_server LLM "$LLM_PORT" "$LLM_PID" 1800

export TMPDIR="$WORK_TMP"
export TMP="$WORK_TMP"
export TEMP="$WORK_TMP"
python3 - <<'PY'
import glob, os, shutil, tempfile
assert tempfile.gettempdir() == os.environ['TMPDIR'], (tempfile.gettempdir(), os.environ['TMPDIR'])
with tempfile.NamedTemporaryFile(prefix='memrl_tmp_preflight_', dir=os.environ['TMPDIR'], delete=True) as f:
    f.write(b'ok'); f.flush()
matches = glob.glob('/AReaL/.venv/lib/python3.12/site-packages/**/fast_downward/libdownward.so', recursive=True)
if not matches:
    matches = glob.glob('/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages/**/fast_downward/libdownward.so', recursive=True)
if not matches:
    raise SystemExit('[TMP-PREFLIGHT] libdownward.so not found')
dst = os.path.join(os.environ['TMPDIR'], 'libdownward_preflight.so')
shutil.copy2(matches[0], dst)
print(f'[TMP-PREFLIGHT] tempfile={tempfile.gettempdir()}; copied libdownward.so ({os.path.getsize(dst)} bytes); ok')
os.unlink(dst)
PY

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
  enable_value_driven: false
  experiment_name: alfworld_rag_qwen72b_resume_s5_full_${RUN_TAG}
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

# run_alfworld creates its own timestamped business log. Tee a second explicit
# per-job business log as well, so supervisor and experiment output stay separate.
BUSINESS_LOG="$BUSINESS_LOG_DIR/rag_resume_s5_full.log"
echo "[EXP] 72B RAG resume starting at $(date --iso-8601=seconds)" | tee -a "$BUSINESS_LOG"
echo '[EXP] Correctness policy: load completed S4 and rerun all 3553 S5 samples.' | tee -a "$BUSINESS_LOG"
set +e
python3 run/run_alfworld.py --config "$CFG" --skip_initial_eval 2>&1 | tee -a "$BUSINESS_LOG"
rc=${PIPESTATUS[0]}
set -e
echo "[EXP] 72B RAG resume exit=$rc at $(date --iso-8601=seconds)" | tee -a "$BUSINESS_LOG"
exit "$rc"
