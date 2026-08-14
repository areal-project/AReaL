#!/bin/bash
# Qwen2.5-72B ALFWorld zero-shot holdout: trajectory MemRL + Self-RAG.
# Historical-comparable protocol: hold out alf/pick_and_place_simple, evaluate
# on train+valid held-out games (n=825), ten total training sections (continuation from completed S5), k_retrieve=5.
# GPU 0: shared embedding; GPU 1-2: trajectory MemRL; GPU 3-4: Self-RAG.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG_SAFE="${RUN_TAG//_/-}"
EMBED_PORT="${EMBED_PORT:-19090}"
MEMRL_PORT="${MEMRL_PORT:-19290}"
SELFRAG_PORT="${SELFRAG_PORT:-19390}"
NCCL_PORT_MEMRL="${NCCL_PORT_MEMRL:-29690}"
NCCL_PORT_SELFRAG="${NCCL_PORT_SELFRAG:-29790}"
HOLDOUT_SUBTASK="alf/pick_and_place_simple"
HOLDOUT_POOLS="train,valid"
HOLDOUT_N=825
OUTPUT_ROOT="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_holdout_qwen72b_traj_selfrag_20260729"
LOG_ROOT="$MEMRL_DIR/logs/alfworld_holdout_qwen72b_traj_selfrag_${RUN_TAG}"
DRIVER_LOG="$LOG_ROOT/driver.log"
WORK_TMP_BASE="/storage/openpsi/users/yl/agent-memory/.tmp"
WORK_TMP="$WORK_TMP_BASE/qwen72b_holdout_traj_selfrag_${RUN_TAG}_$$"

mkdir -p "$LOG_ROOT" "$WORK_TMP" "$OUTPUT_ROOT"
chmod 700 "$WORK_TMP"
exec > >(tee -a "$DRIVER_LOG") 2>&1

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages

printf '%s\n' \
  '================================================================' \
  'Qwen2.5-72B ALFWorld holdout: trajectory MemRL + Self-RAG' \
  "Start: $(date --iso-8601=seconds)" \
  "Holdout: $HOLDOUT_SUBTASK; eval pools: $HOLDOUT_POOLS; expected n=$HOLDOUT_N" \
  'Protocol: continue completed S5 checkpoints through S10; independent holdout eval after every section' \
  'GPU0 embedding; GPU1-2 trajectory MemRL; GPU3-4 Self-RAG' \
  "Logs: $LOG_ROOT" \
  "Output root: $OUTPUT_ROOT" \
  '================================================================'

cd "$MEMRL_DIR"
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai "chonkie==1.2.1" tensorboard hdbscan pandas tqdm \
  concurrent-log-handler textworld alfworld --target "$VENV_SP" \
  -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
python3 -c "import memrl; import memos; print('[OK] memrl + memos imported')"

unset TMPDIR TMP TEMP
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
  --host 127.0.0.1 --port "$EMBED_PORT" --context-length 8192 \
  --trust-remote-code --is-embedding &
EMBED_PID=$!
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server \
  --model-path "$QWEN72B_PATH" --served-model-name Qwen2.5-72B-Instruct \
  --tp 2 --host 127.0.0.1 --port "$MEMRL_PORT" --context-length 32768 \
  --nccl-port "$NCCL_PORT_MEMRL" --trust-remote-code &
MEMRL_PID=$!
CUDA_VISIBLE_DEVICES=3,4 python3 -m sglang.launch_server \
  --model-path "$QWEN72B_PATH" --served-model-name Qwen2.5-72B-Instruct \
  --tp 2 --host 127.0.0.1 --port "$SELFRAG_PORT" --context-length 32768 \
  --nccl-port "$NCCL_PORT_SELFRAG" --trust-remote-code &
SELFRAG_PID=$!

cleanup() {
  status=$?
  trap - EXIT INT TERM
  echo "[CLEANUP] status=$status at $(date --iso-8601=seconds)"
  kill "$SELFRAG_PID" "$MEMRL_PID" "$EMBED_PID" 2>/dev/null || true
  wait "$SELFRAG_PID" "$MEMRL_PID" "$EMBED_PID" 2>/dev/null || true
  rm -rf -- "$WORK_TMP"
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_server() {
  local name="$1" port="$2" pid="$3" limit="$4"
  echo "[INFO] Waiting for $name on port $port..."
  for i in $(seq 1 "$limit"); do
    kill -0 "$pid" 2>/dev/null || { echo "[ERROR] $name exited before readiness"; return 1; }
    curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q model && { echo "[INFO] $name ready after ${i}s"; return 0; }
    sleep 1
  done
  echo "[ERROR] $name readiness timeout"; return 1
}
wait_server Embed "$EMBED_PORT" "$EMBED_PID" 1200
wait_server Trajectory-MemRL "$MEMRL_PORT" "$MEMRL_PID" 1800
wait_server Self-RAG "$SELFRAG_PORT" "$SELFRAG_PID" 1800

export TMPDIR="$WORK_TMP" TMP="$WORK_TMP" TEMP="$WORK_TMP"
python3 - <<'PYTMP'
import os, tempfile
assert tempfile.gettempdir() == os.environ['TMPDIR']
with tempfile.NamedTemporaryFile(dir=os.environ['TMPDIR'], delete=True) as f:
    f.write(b'ok'); f.flush()
print('[TMP-PREFLIGHT] holdout tempfile=' + tempfile.gettempdir())
PYTMP

write_cfg() {
  local cfg="$1" exp="$2" port="$3" vd="$4" strategy="$5" k="$6" tau="$7" ws="$8" wq="$9"
  cat > "$cfg" <<CFGEOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${port}/v1/
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
  build_strategy: ${strategy}
  retrieve_strategy: query
  update_strategy: adjustment
  k_retrieve: ${k}
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
  enable_value_driven: ${vd}
  experiment_name: ${exp}
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
  bon: 0
  valid_interval: 1
  test_interval: 1
  holdout_subtask: ${HOLDOUT_SUBTASK}
  holdout_eval_pools: ${HOLDOUT_POOLS}
  ckpt_resume_enabled: false
  ckpt_resume_path: ""
  ckpt_resume_epoch: null
  batch_checkpoint_interval: 10
  batch_checkpoint_keep: 1
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
  weight_sim: ${ws}
  weight_q: ${wq}
CFGEOF
}

MEMRL_CFG="$WORK_TMP/traj_memrl_holdout.yaml"
SELFRAG_CFG="$WORK_TMP/selfrag_holdout.yaml"
MEMRL_EXP="alfworld_holdout_pick_and_place_simple_qwen72b_memrl_traj_${RUN_TAG}"
SELFRAG_EXP="alfworld_holdout_pick_and_place_simple_qwen72b_selfrag_traj_${RUN_TAG}"
write_cfg "$MEMRL_CFG" "$MEMRL_EXP" "$MEMRL_PORT" true trajectory 5 0.62 0.5 0.5
write_cfg "$SELFRAG_CFG" "$SELFRAG_EXP" "$SELFRAG_PORT" false trajectory 5 0.0 1.0 0.0

preflight_holdout() {
  local cfg="$1" label="$2"
  python3 - "$cfg" "$HOLDOUT_SUBTASK" "$HOLDOUT_N" "$label" <<'PYHOLD'
import sys, yaml
from memrl.run.alfworld_rl_runner import AlfworldRunner
cfg, subtask, expected_n, label = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
d = yaml.safe_load(open(cfg))
assert d['experiment']['holdout_subtask'] == subtask
assert d['experiment']['holdout_eval_pools'] == 'train,valid'
assert d['experiment']['num_sections'] == 10
assert d['memory']['build_strategy'] == 'trajectory'
print(f'[HOLDOUT-PREFLIGHT] {label}: subtask={subtask}, eval_pools=train,valid, expected_n={expected_n}, sections=10, build=trajectory')
PYHOLD
}
preflight_holdout "$MEMRL_CFG" trajectory-memrl
preflight_holdout "$SELFRAG_CFG" selfrag

run_arm() {
  local label cfg exp extra log
  label="$1"
  cfg="$2"
  exp="$3"
  extra="$4"
  log="$LOG_ROOT/${label}.log"
  export MEMRL_RUN_ID="qwen72b-holdout-${label}-${RUN_TAG_SAFE}"
  (
    echo "[ARM-START] $label at $(date --iso-8601=seconds)"
    echo "[ARM-CONFIG] exp=$exp holdout=$HOLDOUT_SUBTASK pools=$HOLDOUT_POOLS n=$HOLDOUT_N"
    echo "[ARM-CONFIG] cfg=$cfg run_id=$MEMRL_RUN_ID"
    python3 run/run_alfworld.py --config "$cfg" \
      --holdout_subtask "$HOLDOUT_SUBTASK" \
      --holdout_eval_pools "$HOLDOUT_POOLS" \
      --skip_initial_eval $extra
    echo "[ARM-DONE] $label exit=0 at $(date --iso-8601=seconds)"
  ) 2>&1 | tee -a "$log"
}

run_arm trajectory_memrl "$MEMRL_CFG" "$MEMRL_EXP" "" &
PID_TRAJ=$!
run_arm selfrag "$SELFRAG_CFG" "$SELFRAG_EXP" "--selfrag" &
PID_SELFRAG=$!

set +e
wait "$PID_TRAJ"; RC_TRAJ=$?
wait "$PID_SELFRAG"; RC_SELFRAG=$?
set -e
echo "[ALL-DONE] trajectory_memrl_rc=$RC_TRAJ selfrag_rc=$RC_SELFRAG"
[[ "$RC_TRAJ" -eq 0 && "$RC_SELFRAG" -eq 0 ]]
