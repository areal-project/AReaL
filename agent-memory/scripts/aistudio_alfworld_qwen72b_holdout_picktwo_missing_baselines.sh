#!/bin/bash
# Pick-two missing 72B baselines: RAG + MemP + proceduralization MemRL.
# GPU0 shared embedding; GPU1-2 RAG; GPU3-4 MemP; GPU5-6 proc-MemRL.
set -euo pipefail
MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
MODEL=/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct
EMBED=/storage/openpsi/models/Qwen3-Embedding-8B
TAG="${1:-$(date +%Y%m%d_%H%M%S)}"; SAFE="${TAG//_/-}"
EP=${EMBED_PORT:-23190}; RAGP=${RAG_PORT:-23290}; MEMPP=${MEMP_PORT:-23390}; PROCP=${PROC_PORT:-23490}
OUT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_holdout_qwen72b_picktwo_missing_baselines_20260806
LOG="$MEMRL_DIR/logs/alfworld_holdout_picktwo_qwen72b_missing_baselines_$TAG"
WORK_TMP=/storage/openpsi/users/yl/agent-memory/.tmp/q72b_picktwo_missing_${TAG}_$$
mkdir -p "$OUT" "$LOG" "$WORK_TMP"; chmod 700 "$WORK_TMP"; exec > >(tee -a "$LOG/driver.log") 2>&1
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
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name Qwen2.5-72B-Instruct --tp 2 --host 127.0.0.1 --port "$RAGP" --context-length 32768 --trust-remote-code --nccl-port 33690 & PIDS+=("$!")
CUDA_VISIBLE_DEVICES=3,4 python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name Qwen2.5-72B-Instruct --tp 2 --host 127.0.0.1 --port "$MEMPP" --context-length 32768 --trust-remote-code --nccl-port 33790 & PIDS+=("$!")
CUDA_VISIBLE_DEVICES=5,6 python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name Qwen2.5-72B-Instruct --tp 2 --host 127.0.0.1 --port "$PROCP" --context-length 32768 --trust-remote-code --nccl-port 33890 & PIDS+=("$!")
wait_server(){ n=$1 p=$2; for i in $(seq 1 1800); do curl -fsS "http://127.0.0.1:$p/v1/models" 2>/dev/null | grep -q model && { echo "[READY] $n ${i}s"; return; }; sleep 1; done; return 1; }
wait_server embed "$EP"; wait_server rag "$RAGP"; wait_server memp "$MEMPP"; wait_server proc_memrl "$PROCP"
export TMPDIR="$WORK_TMP" TMP="$WORK_TMP" TEMP="$WORK_TMP"
write_cfg(){ cfg=$1 exp=$2 port=$3 build=$4 vd=$5 tau=$6 ws=$7 wq=$8; cat > "$cfg" <<YAML
llm: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${port}/v1/", model: Qwen2.5-72B-Instruct, temperature: 0, max_tokens: 4096}
embedding: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${EP}/v1/", model: Qwen/Qwen3-Embedding-8B, max_text_len: 8196, dimension: 4096}
memory: {build_strategy: ${build}, retrieve_strategy: query, update_strategy: adjustment, k_retrieve: 3, max_keywords: 5, add_similarity_threshold: 0.9, memory_budget_tokens: 0, sim_norm_mean: 0.5187, sim_norm_std: 0.1203}
environment: {alfworld_config_path: configs/envs/alfworld.yaml, alfworld_env_type: AlfredTWEnv}
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
rl_config: {epsilon: 0, tau: ${tau}, alpha: 0.3, gamma: 0.0, q_init_pos: 0, q_init_neg: 0, success_reward: 1.0, failure_reward: -1.0, topk: 3, novelty_threshold: 0.85, recency_boost: 0.0, reward_merge_gain: 0.1, q_min_threshold: -10, weight_sim: ${ws}, weight_q: ${wq}}
YAML
}
write_cfg "$WORK_TMP/rag.yaml" "alfworld_holdout_picktwo_qwen72b_rag_${TAG}" "$RAGP" trajectory false 0.0 1.0 0.0
write_cfg "$WORK_TMP/memp.yaml" "alfworld_holdout_picktwo_qwen72b_memp_${TAG}" "$MEMPP" proceduralization false 0.0 1.0 0.0
write_cfg "$WORK_TMP/proc_memrl.yaml" "alfworld_holdout_picktwo_qwen72b_proc_memrl_${TAG}" "$PROCP" proceduralization true 0.62 0.5 0.5
run_arm(){ label=$1 cfg=$2; export MEMRL_RUN_ID="qwen72b-picktwo-${label}-${SAFE}"; echo "[ARM] $label run_id=$MEMRL_RUN_ID"; python3 run/run_alfworld.py --config "$cfg" --holdout_subtask alf/pick_two_obj_and_place --holdout_eval_pools train,valid --skip_initial_eval 2>&1 | tee -a "$LOG/$label.log"; }
run_arm rag "$WORK_TMP/rag.yaml" & A=$!
run_arm memp "$WORK_TMP/memp.yaml" & B=$!
run_arm proc_memrl "$WORK_TMP/proc_memrl.yaml" & C=$!
set +e; wait $A; ra=$?; wait $B; rb=$?; wait $C; rc=$?; set -e
echo "[ALL-DONE] rag=$ra memp=$rb proc_memrl=$rc"; [[ $ra -eq 0 && $rb -eq 0 && $rc -eq 0 ]]
