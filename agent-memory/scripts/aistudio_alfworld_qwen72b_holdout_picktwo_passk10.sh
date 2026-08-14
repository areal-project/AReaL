#!/bin/bash
# Pick-two holdout Pass@10 (837 fixed games); no memory training.
set -euo pipefail
MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
OVERLAY="$MEMRL_DIR/scripts/picktwo_passk_overlay"
MODEL=/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct
EMBED=/storage/openpsi/models/Qwen3-Embedding-8B
TAG="${1:-$(date +%Y%m%d_%H%M%S)}"; SAFE="${TAG//_/-}"
EP=${EMBED_PORT:-25190}; LP=${LLM_PORT:-25290}; NP=${NCCL_PORT:-35690}
OUT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_holdout_qwen72b_picktwo_passk10_20260806
LOG="$MEMRL_DIR/logs/alfworld_holdout_picktwo_qwen72b_passk10_$TAG"
WORK_TMP=/storage/openpsi/users/yl/agent-memory/.tmp/q72b_picktwo_passk_${TAG}_$$
mkdir -p "$OUT" "$LOG" "$WORK_TMP"; chmod 700 "$WORK_TMP"; exec > >(tee -a "$LOG/driver.log") 2>&1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$OVERLAY:$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "$MEMRL_DIR"
python3 - <<'PY'
import sys
sys.path.insert(0, '/storage/openpsi/users/yl/agent-memory/MemRL/scripts/picktwo_passk_overlay')
import memrl
print('[PREFLIGHT] passk overlay=', memrl.__file__)
assert 'picktwo_passk_overlay' in memrl.__file__
PY
unset TMPDIR TMP TEMP
PIDS=()
cleanup(){ rc=$?; trap - EXIT INT TERM; ((${#PIDS[@]})) && kill "${PIDS[@]}" 2>/dev/null || true; rm -rf -- "$WORK_TMP"; echo "[CLEANUP] exit=$rc $(date -Is)"; exit "$rc"; }; trap cleanup EXIT INT TERM
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path "$EMBED" --served-model-name Qwen/Qwen3-Embedding-8B --host 127.0.0.1 --port "$EP" --context-length 8192 --trust-remote-code --is-embedding & PIDS+=("$!")
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name Qwen2.5-72B-Instruct --tp 2 --host 127.0.0.1 --port "$LP" --context-length 32768 --trust-remote-code --nccl-port "$NP" & PIDS+=("$!")
wait_server(){ n=$1 p=$2; for i in $(seq 1 1800); do curl -fsS "http://127.0.0.1:$p/v1/models" 2>/dev/null | grep -q model && { echo "[READY] $n ${i}s"; return; }; sleep 1; done; return 1; }
wait_server embed "$EP"; wait_server llm "$LP"
export TMPDIR="$WORK_TMP" TMP="$WORK_TMP" TEMP="$WORK_TMP"
CFG="$WORK_TMP/passk.yaml"
cat > "$CFG" <<YAML
llm: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${LP}/v1/", model: Qwen2.5-72B-Instruct, temperature: 0, max_tokens: 4096}
embedding: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${EP}/v1/", model: Qwen/Qwen3-Embedding-8B, max_text_len: 8196, dimension: 4096}
memory: {build_strategy: proceduralization, retrieve_strategy: query, update_strategy: adjustment, k_retrieve: 0, max_keywords: 5, add_similarity_threshold: 0.9, memory_budget_tokens: 0, sim_norm_mean: 0.5187, sim_norm_std: 0.1203}
environment: {alfworld_config_path: configs/envs/alfworld.yaml, alfworld_env_type: AlfredTWEnv}
experiment:
  random_seed: 43
  enable_value_driven: false
  experiment_name: alfworld_holdout_picktwo_passk10_qwen72b_${TAG}
  mode: train
  num_sections: 10
  batch_size: 128
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: passk
  baseline_k: 10
  output_dir: ${OUT}
  max_steps: 30
  save_trajectories: false
  save_memories: false
  holdout_subtask: alf/pick_two_obj_and_place
  holdout_eval_pools: train,valid
  ckpt_resume_enabled: false
  ckpt_resume_path: ""
  ckpt_resume_epoch: null
rl_config: {epsilon: 0, tau: 0.62, alpha: 0.3, gamma: 0.0, q_init_pos: 0, q_init_neg: 0, success_reward: 1.0, failure_reward: -1.0, topk: 3, novelty_threshold: 0.85, recency_boost: 0.0, reward_merge_gain: 0.1, q_min_threshold: -10, weight_sim: 0.5, weight_q: 0.5}
YAML
export MEMRL_RUN_ID="qwen72b-picktwo-passk10-${SAFE}"
python3 "$OVERLAY/run/run_alfworld.py" --config "$CFG" --holdout_subtask alf/pick_two_obj_and_place --holdout_eval_pools train,valid --skip_initial_eval 2>&1 | tee -a "$LOG/passk.log"
echo "[ALL-DONE] passk=0 $(date -Is)"
