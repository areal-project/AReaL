#!/bin/bash
# Hard ALFWorld holdout continuation: pick_and_place_simple Region+FS, fresh 10-epoch run.
# GPU0 embed; GPU1-2 Region.
set -euo pipefail
MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
MODEL=/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct
EMBED=/storage/openpsi/models/Qwen3-Embedding-8B
TAG="${1:-$(date +%Y%m%d_%H%M%S)}"; SAFE="${TAG//_/-}"
EP="${EMBED_PORT:-19590}"; RP="${REGION_PORT:-19690}"; NCCL_REGION="${NCCL_REGION:-29990}"
OUT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_holdout_qwen72b_pickplace_regionfs_20260802
LOG="$MEMRL_DIR/logs/alfworld_holdout_pickplace_qwen72b_regionfs_$TAG"
WORK_TMP=/storage/openpsi/users/yl/agent-memory/.tmp/q72b_picktwo_3arm_${TAG}_$$
mkdir -p "$OUT" "$LOG" "$WORK_TMP"; chmod 700 "$WORK_TMP"; exec > >(tee -a "$LOG/driver.log") 2>&1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MEMRL_ALFWORLD_LLM_CONCURRENCY=16 MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES=16
cd "$MEMRL_DIR"
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install -e . --no-deps --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -3
pip install mem0ai 'chonkie==1.2.1' tensorboard hdbscan pandas tqdm concurrent-log-handler textworld alfworld --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ 2>&1 | tail -5
unset TMPDIR TMP TEMP
PIDS=()
cleanup(){ rc=$?; trap - EXIT INT TERM; ((${#PIDS[@]})) && kill "${PIDS[@]}" 2>/dev/null || true; rm -rf -- "$WORK_TMP"; echo "[CLEANUP] exit=$rc $(date -Is)"; exit "$rc"; }; trap cleanup EXIT INT TERM
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path "$EMBED" --served-model-name Qwen/Qwen3-Embedding-8B --host 127.0.0.1 --port $EP --context-length 8192 --trust-remote-code --is-embedding & PIDS+=("$!")
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name Qwen2.5-72B-Instruct --tp 2 --host 127.0.0.1 --port $RP --context-length 32768 --trust-remote-code --nccl-port "$NCCL_REGION" & PIDS+=("$!")
wait_server(){ n=$1 p=$2; for i in $(seq 1 1800); do curl -fsS http://127.0.0.1:$p/v1/models 2>/dev/null | grep -q model && { echo "[READY] $n ${i}s"; return; }; sleep 1; done; return 1; }
wait_server embed $EP; wait_server region $RP
export TMPDIR="$WORK_TMP" TMP="$WORK_TMP" TEMP="$WORK_TMP"
write_cfg(){ cfg=$1 exp=$2 port=$3 vd=$4; cat > "$cfg" <<EOF
llm: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${port}/v1/", model: Qwen2.5-72B-Instruct, temperature: 0, max_tokens: 4096}
embedding: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${EP}/v1/", model: Qwen/Qwen3-Embedding-8B, max_text_len: 8196, dimension: 4096}
memory: {build_strategy: trajectory, retrieve_strategy: query, update_strategy: adjustment, k_retrieve: 5, max_keywords: 5, add_similarity_threshold: 0.9, memory_budget_tokens: 0, sim_norm_mean: 0.5187, sim_norm_std: 0.1203}
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
  holdout_subtask: alf/pick_and_place_simple
  holdout_eval_pools: train,valid
  ckpt_resume_enabled: false
  ckpt_resume_path: ""
  ckpt_resume_epoch: null
  batch_checkpoint_interval: 10
  batch_checkpoint_keep: 1
  n_eval_runs: 1
rl_config: {epsilon: 0, tau: 0.62, alpha: 0.3, gamma: 0.0, q_init_pos: 0, q_init_neg: 0, success_reward: 1.0, failure_reward: -1.0, topk: 3, novelty_threshold: 0.85, recency_boost: 0.0, reward_merge_gain: 0.1, q_min_threshold: -10, weight_sim: 0.5, weight_q: 0.5}
EOF
}
write_cfg "$WORK_TMP/region.yaml" "alfworld_holdout_pickplace_qwen72b_regionfs_traj_${TAG}" $RP true
run_arm(){ label=$1 cfg=$2 extra=$3; export MEMRL_RUN_ID="qwen72b-pickplace-${label}-${SAFE}"; echo "[ARM] $label run_id=$MEMRL_RUN_ID"; python3 run/run_alfworld.py --config "$cfg" --holdout_subtask alf/pick_and_place_simple --holdout_eval_pools train,valid --skip_initial_eval $extra 2>&1 | tee -a "$LOG/$label.log"; }
run_arm region "$WORK_TMP/region.yaml" "--region --region_gating_mode additive --region_utility_mode beta --region_split_evidence_migration_mode soft_source_conserving --region_cluster_init_step 3000 --region_merge_interval 2763 --region_disable_mid_epoch_topology --region_topology_cooldown_sections 1 --shrinkage_confidence_k 2.5 --propagation_eta 0.03 --failure_summary_n_slots 1" & C=$!
set +e; wait $C; rc=$?; set -e
echo "[ALL-DONE] region=$rc"; [[ $rc -eq 0 ]]
