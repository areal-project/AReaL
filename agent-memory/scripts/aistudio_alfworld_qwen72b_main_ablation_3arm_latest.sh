#!/bin/bash
# Latest region-dev: forced RFS, Category-Q, Capability No-RFS. Shared embedding, 7x H200.
set -euo pipefail
MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
MODEL=/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct
EMBED=/storage/openpsi/models/Qwen3-Embedding-8B
TAG="${1:-$(date +%Y%m%d_%H%M%S)}"; SAFE="${TAG//_/-}"
read -r EP FP CP AP NC1 NC2 NC3 < <(python3 - <<'PYPORTS'
import socket
p=[]
for _ in range(7):
 s=socket.socket();s.bind(('127.0.0.1',0));p.append(s.getsockname()[1]);s.close()
print(*p)
PYPORTS
)
OUT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_qwen72b_main_ablation_3arm_latest_20260811
LOG="$MEMRL_DIR/logs/alfworld_qwen72b_main_ablation_3arm_latest_$TAG"
WORK_TMP=/storage/openpsi/users/yl/agent-memory/.tmp/q72b_main_ablation_3arm_latest_${TAG}_$$
mkdir -p "$OUT" "$LOG" "$WORK_TMP";chmod 700 "$WORK_TMP";exec > >(tee -a "$LOG/driver.log") 2>&1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
DEPS="$WORK_TMP/site-packages";mkdir -p "$DEPS";PIP_BIN="$(command -v pip || command -v pip3)"
"$PIP_BIN" install --disable-pip-version-check --no-deps --target "$DEPS" 'hdbscan==0.8.40' -i https://pypi.antfin-inc.com/simple/
export PYTHONPATH="$DEPS:$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MEMRL_ALFWORLD_LLM_CONCURRENCY=16 MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES=16
cd "$MEMRL_DIR"
python3 - <<'PY'
import memrl,hdbscan
from memrl.service.region_manager import RegionManager
r=RegionManager(task_hierarchy={},cluster_space='capability');assert r.cluster_space=='capability';assert '/scripts/' not in memrl.__file__
print('[PREFLIGHT] latest source',memrl.__file__,hdbscan.__file__)
PY
unset TMPDIR TMP TEMP
PIDS=();cleanup(){ rc=$?;trap - EXIT INT TERM;((${#PIDS[@]}))&&kill "${PIDS[@]}" 2>/dev/null||true;rm -rf -- "$WORK_TMP";exit "$rc";};trap cleanup EXIT INT TERM
wait_server(){ n=$1 p=$2 pid=$3;for i in $(seq 1 1800);do kill -0 "$pid" 2>/dev/null||return 1;curl -fsS "http://127.0.0.1:$p/v1/models" 2>/dev/null|grep -q model&&{ echo "[READY] $n ${i}s";return;};sleep 1;done;return 1;}
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path "$EMBED" --served-model-name Qwen/Qwen3-Embedding-8B --host 127.0.0.1 --port "$EP" --context-length 8192 --trust-remote-code --is-embedding & E=$!;PIDS+=("$E");wait_server embed "$EP" "$E"
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name Qwen2.5-72B-Instruct --tp 2 --host 127.0.0.1 --port "$FP" --context-length 32768 --trust-remote-code --nccl-port "$NC1" & F=$!;PIDS+=("$F");wait_server forced "$FP" "$F"
CUDA_VISIBLE_DEVICES=3,4 python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name Qwen2.5-72B-Instruct --tp 2 --host 127.0.0.1 --port "$CP" --context-length 32768 --trust-remote-code --nccl-port "$NC2" & C=$!;PIDS+=("$C");wait_server category "$CP" "$C"
CUDA_VISIBLE_DEVICES=5,6 python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name Qwen2.5-72B-Instruct --tp 2 --host 127.0.0.1 --port "$AP" --context-length 32768 --trust-remote-code --nccl-port "$NC3" & A=$!;PIDS+=("$A");wait_server capability "$AP" "$A"
export TMPDIR="$WORK_TMP" TMP="$WORK_TMP" TEMP="$WORK_TMP"
write_cfg(){ cfg=$1 exp=$2 port=$3;cat > "$cfg" <<YAML
llm: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${port}/v1/", model: Qwen2.5-72B-Instruct, temperature: 0, max_tokens: 4096}
embedding: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${EP}/v1/", model: Qwen/Qwen3-Embedding-8B, max_text_len: 8196, dimension: 4096}
memory: {build_strategy: trajectory, retrieve_strategy: query, update_strategy: adjustment, k_retrieve: 3, max_keywords: 5, add_similarity_threshold: 0.9, memory_budget_tokens: 0, sim_norm_mean: 0.5187, sim_norm_std: 0.1203}
environment: {alfworld_config_path: configs/envs/alfworld.yaml, alfworld_env_type: AlfredTWEnv}
experiment: {random_seed: 42, enable_value_driven: true, experiment_name: "${exp}", mode: train, num_sections: 10, batch_size: 128, dataset_ratio: 1.0, few_shot_path: data/alfworld/alfworld_examples.json, baseline_mode: null, output_dir: "${OUT}", max_steps: 30, save_trajectories: true, save_memories: true, valid_interval: 1, test_interval: 1, ckpt_resume_enabled: false, ckpt_resume_path: "", ckpt_resume_epoch: null, batch_checkpoint_interval: 10, batch_checkpoint_keep: 1, n_eval_runs: 4, eval_temperature: 0.2}
rl_config: {epsilon: 0, tau: 0.60, alpha: 0.3, gamma: 0.0, q_init_pos: 0, q_init_neg: 0, success_reward: 1.0, failure_reward: -1.0, topk: 3, novelty_threshold: 0.85, recency_boost: 0.0, reward_merge_gain: 0.1, q_min_threshold: -10, weight_sim: 0.45, weight_q: 0.55}
YAML
}
write_cfg "$WORK_TMP/f.yaml" "alfworld_qwen72b_full_forced_latest_${TAG}" "$FP";write_cfg "$WORK_TMP/c.yaml" "alfworld_qwen72b_category_q_latest_${TAG}" "$CP";write_cfg "$WORK_TMP/a.yaml" "alfworld_qwen72b_capability_norfs_latest_${TAG}" "$AP"
common=(--region --region_gating_mode additive --region_cluster_space capability --region_utility_mode beta --region_split_evidence_migration_mode soft_source_conserving --region_cluster_init_step 3000 --region_merge_interval 3553 --region_disable_mid_epoch_topology --region_topology_cooldown_sections 1 --shrinkage_confidence_k 2.5 --propagation_eta 0.03 --val_lambda_max 0.45 --no_z_norm --explore_schedule '0,1,1,1,0,0,0,0,0,0' --skip_initial_eval)
run_arm(){ label=$1 cfg=$2 mode=$3 extra=$4;export MEMRL_RUN_ID="qwen72b-main-ablation-latest-${label}-${SAFE}";echo "[ARM] $label latest_source=true";python3 run/run_alfworld.py --config "$cfg" "${common[@]}" --region_value_mode "$mode" $extra 2>&1|tee -a "$LOG/$label.log";}
run_arm forced_rfs "$WORK_TMP/f.yaml" shrinkage "--failure_summary_n_slots 1 --failure_summary_force_recall" & P1=$!;run_arm category_q "$WORK_TMP/c.yaml" category_q "" & P2=$!;run_arm capability_norfs "$WORK_TMP/a.yaml" shrinkage "" & P3=$!
set +e;wait $P1;r1=$?;wait $P2;r2=$?;wait $P3;r3=$?;set -e;echo "[ALL-DONE] $r1/$r2/$r3";[[ $r1 -eq 0 && $r2 -eq 0 && $r3 -eq 0 ]]
