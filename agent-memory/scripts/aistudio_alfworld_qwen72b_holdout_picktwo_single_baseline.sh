#!/bin/bash
# One fresh pick-two baseline per 3xH200 job: rag | memp | proc_memrl.
set -euo pipefail
METHOD="${1:?method required: rag|memp|proc_memrl}"
TAG="${2:-$(date +%Y%m%d_%H%M%S)}"; SAFE="${TAG//_/-}"
case "$METHOD" in
  rag) BUILD=trajectory; VD=false; TAU=0.0; WS=1.0; WQ=0.0 ;;
  memp) BUILD=proceduralization; VD=false; TAU=0.0; WS=1.0; WQ=0.0 ;;
  proc_memrl) BUILD=proceduralization; VD=true; TAU=0.62; WS=0.5; WQ=0.5 ;;
  *) echo "unknown method: $METHOD" >&2; exit 2 ;;
esac
MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
MODEL=/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct
EMBED=/storage/openpsi/models/Qwen3-Embedding-8B
read -r EP LP NP < <(python3 - <<'PY'
import socket
ports=[]
for _ in range(3):
 s=socket.socket(); s.bind(('127.0.0.1',0)); ports.append(s.getsockname()[1]); s.close()
print(*ports)
PY
)
OUT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_holdout_qwen72b_picktwo_${METHOD}_20260807
LOG="$MEMRL_DIR/logs/alfworld_holdout_picktwo_qwen72b_${METHOD}_$TAG"
WORK_TMP=/storage/openpsi/users/yl/agent-memory/.tmp/q72b_picktwo_${METHOD}_${TAG}_$$
mkdir -p "$OUT" "$LOG" "$WORK_TMP"; chmod 700 "$WORK_TMP"; exec > >(tee -a "$LOG/driver.log") 2>&1
echo "[PORTS] embed=$EP llm=$LP nccl=$NP"
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MEMRL_ALFWORLD_LLM_CONCURRENCY=16 MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES=16
cd "$MEMRL_DIR"
python3 - <<'PY'
import memrl, memos, textworld, alfworld, pandas, tqdm
from torch.utils.tensorboard import SummaryWriter
print('[PREFLIGHT] memrl=', memrl.__file__)
PY
unset TMPDIR TMP TEMP
PIDS=()
cleanup(){ rc=$?; trap - EXIT INT TERM; ((${#PIDS[@]})) && kill "${PIDS[@]}" 2>/dev/null || true; rm -rf -- "$WORK_TMP"; echo "[CLEANUP] exit=$rc $(date -Is)"; exit "$rc"; }; trap cleanup EXIT INT TERM
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path "$EMBED" --served-model-name Qwen/Qwen3-Embedding-8B --host 127.0.0.1 --port "$EP" --context-length 8192 --trust-remote-code --is-embedding & PIDS+=("$!")
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name Qwen2.5-72B-Instruct --tp 2 --host 127.0.0.1 --port "$LP" --context-length 32768 --trust-remote-code --nccl-port "$NP" & PIDS+=("$!")
wait_server(){ n=$1 p=$2; for i in $(seq 1 1800); do curl -fsS "http://127.0.0.1:$p/v1/models" 2>/dev/null | grep -q model && { echo "[READY] $n ${i}s"; return; }; sleep 1; done; return 1; }
wait_server embed "$EP"; wait_server llm "$LP"
export TMPDIR="$WORK_TMP" TMP="$WORK_TMP" TEMP="$WORK_TMP"
CFG="$WORK_TMP/$METHOD.yaml"
cat > "$CFG" <<YAML
llm: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${LP}/v1/", model: Qwen2.5-72B-Instruct, temperature: 0, max_tokens: 4096}
embedding: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${EP}/v1/", model: Qwen/Qwen3-Embedding-8B, max_text_len: 8196, dimension: 4096}
memory: {build_strategy: ${BUILD}, retrieve_strategy: query, update_strategy: adjustment, k_retrieve: 3, max_keywords: 5, add_similarity_threshold: 0.9, memory_budget_tokens: 0, sim_norm_mean: 0.5187, sim_norm_std: 0.1203}
environment: {alfworld_config_path: configs/envs/alfworld.yaml, alfworld_env_type: AlfredTWEnv}
experiment:
  random_seed: 42
  enable_value_driven: ${VD}
  experiment_name: alfworld_holdout_picktwo_qwen72b_${METHOD}_${TAG}
  mode: train
  num_sections: 10
  batch_size: 128
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: null
  output_dir: ${OUT}
  max_steps: 30
  save_trajectories: true
  save_memories: true
  valid_interval: 1
  test_interval: 1
  holdout_subtask: alf/pick_two_obj_and_place
  holdout_eval_pools: train,valid
  ckpt_resume_enabled: false
  ckpt_resume_path: ""
  ckpt_resume_epoch: null
  batch_checkpoint_interval: 10
  batch_checkpoint_keep: 1
  n_eval_runs: 1
rl_config: {epsilon: 0, tau: ${TAU}, alpha: 0.3, gamma: 0.0, q_init_pos: 0, q_init_neg: 0, success_reward: 1.0, failure_reward: -1.0, topk: 3, novelty_threshold: 0.85, recency_boost: 0.0, reward_merge_gain: 0.1, q_min_threshold: -10, weight_sim: ${WS}, weight_q: ${WQ}}
YAML
export MEMRL_RUN_ID="qwen72b-picktwo-${METHOD}-${SAFE}"
python3 run/run_alfworld.py --config "$CFG" --holdout_subtask alf/pick_two_obj_and_place --holdout_eval_pools train,valid --skip_initial_eval 2>&1 | tee -a "$LOG/$METHOD.log"
echo "[ALL-DONE] $METHOD=0 $(date -Is)"
