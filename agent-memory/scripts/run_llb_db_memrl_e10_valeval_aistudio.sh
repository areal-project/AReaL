#!/usr/bin/env bash
# Read-only validation of the completed LLB-DB MemRL gpt-4.1-mini E10 snapshot.
# The permanent snapshot is copied before use; all restore/output/markers are /tmp only.
set -euo pipefail

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SOURCE_SNAPSHOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect/exp_llb_db_memrl_haiku_v2reflect_v2reflect-dbpattern-gpt41mini-20260715/snapshot/10
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | sed 's/^gpulingjun//; s/\..*$//')
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="${PROJECT_DIR}/logs/llb_db_memrl_e10_valeval_${HOST_SHORT}_${TS}.log"
exec > >(tee -a "$LOGFILE") 2>&1

cd "$PROJECT_DIR"
echo '=========================================='
echo 'LLB-DB MemRL E10 read-only validation (gpt-4.1-mini)'
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Source snapshot (read-only): $SOURCE_SNAPSHOT"
echo "Log: $LOGFILE"
echo '=========================================='

[[ -f "$SOURCE_SNAPSHOT/snapshot_meta.json" ]] || { echo 'ERROR: E10 snapshot_meta.json missing' >&2; exit 2; }
[[ -d "$SOURCE_SNAPSHOT/cube" && -d "$SOURCE_SNAPSHOT/qdrant" ]] || { echo 'ERROR: E10 snapshot incomplete' >&2; exit 2; }

export PYTHONPATH="${PROJECT_DIR}:${LOCAL_SP}:${PYTHONPATH:-}"
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=auto
export MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=0.5
export MEMRL_LLM_MIN_INTERVAL=0.8
export MEMRL_LLB_REFLECTION_PROMPT=v2
export MEMRL_LLB_SCRIPT_DETAIL=db_pattern
export MEMRL_LLM_MODEL=gpt-4.1-mini-2025-04-14
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
printf '%s\n' '[INFO] Installing runtime deps...'
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm concurrent-log-handler mysql-connector-python \
  --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/
python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"

# This entire work tree is node-local and disposable. No /storage checkpoint is
# modified, removed, moved, or used as an output directory.
WORKDIR=$(mktemp -d /tmp/llb_db_memrl_e10_valeval.XXXXXX)
trap 'rm -rf "$WORKDIR"' EXIT
LOCAL_SNAPSHOT="$WORKDIR/snapshot_10"
LOCAL_OUTPUT="$WORKDIR/output"
LOCAL_CONFIG="$WORKDIR/llb_db_memrl_e10_eval.yaml"
printf '%s\n' '[INFO] Copying E10 snapshot to node-local temporary storage...'
cp -a "$SOURCE_SNAPSHOT" "$LOCAL_SNAPSHOT"

cat > "$LOCAL_CONFIG" <<EOF
llm:
  provider: "openai"
  api_key: "runtime-injected"
  base_url: "https://matrixllm.alipay.com/v1/"
  api_version: null
  model: "gpt-4.1-mini-2025-04-14"
  temperature: 0.0
  max_tokens: 10240
embedding:
  provider: "openai"
  api_key: "runtime-injected"
  base_url: "https://matrixllm.alipay.com/v1/"
  api_version: null
  model: "text-embedding-3-large"
  dimension: 3072
  max_text_len: 8196
memory:
  build_strategy: "proceduralization"
  retrieve_strategy: "query"
  update_strategy: "adjustment"
  k_retrieve: 10
  max_keywords: 8
  confidence_threshold: 0.0
  memory_confidence: 100.0
  add_similarity_threshold: 0.99
  mos_config_path: "configs/mos_config.json"
  user_id: "llb_db_memrl_e10_readonly_eval"
  load_from_checkpoint: true
  checkpoint_path: "$LOCAL_SNAPSHOT"
  sim_norm_mean: 0.2747681439
  sim_norm_std: 0.1127030626
environment:
  alfworld_config_path: "configs/base_config.yaml"
  alfworld_env_type: "AlfredTWEnv"
experiment:
  experiment_name: "llb_db_memrl_e10_readonly_eval"
  llb_use_z_score_normalization: true
  llb_q_floor: 0.0
  llb_dedup_by_task_id: false
  llb_reflection_prompt: "v2"
  algorithm: "rl"
  val_before_train: false
  enable_value_driven: true
  random_seed: 42
  mode: "train"
  task: "db"
  split_file: "data/llb/db_train.json"
  valid_file: "data/llb/db_val.json"
  num_sections: 10
  batch_size: 5
  max_steps: 15
  valid_interval: 1
  test_interval: 1
  eval_runs: 1
  eval_temperature: 0.0
  ckpt_save_every_n_batches: 0
  ckpt_max_keep: 1
  dataset_ratio: 1.0
  few_shot_path: "data/alfworld/alfworld_examples.json"
  bon: 0
  output_dir: "$LOCAL_OUTPUT"
  save_trajectories: false
  save_memories: false
  enable_logging: true
  log_level: "INFO"
rl_config:
  epsilon: 0.01
  tau: 0.35
  alpha: 0.3
  gamma: 0.0
  q_init_pos: 0.0
  q_init_neg: 0.0
  q_floor: null
  success_reward: 1.0
  failure_reward: 0.0
  sim_threshold: 0.369
  topk: 5
  novelty_threshold: 0.85
  recency_boost: 0.0
  reward_merge_gain: 0.1
  q_min_threshold: -0.8
  weight_sim: 0.5
  weight_q: 0.5
EOF

printf '%s\n' '[INFO] Starting E10 validation only; training loop is empty after explicit E10 resume.'
python3 scripts/run_llb_with_rotated_matrix_credentials.py \
  --config "$LOCAL_CONFIG" \
  --output_dir "$LOCAL_OUTPUT" \
  --resume_eval_section 10

echo '=========================================='
echo "End time: $(date)"
echo 'E10_READONLY_EVAL_COMPLETE'
echo '=========================================='
