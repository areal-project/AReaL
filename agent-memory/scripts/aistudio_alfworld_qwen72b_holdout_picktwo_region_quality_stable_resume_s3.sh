#!/bin/bash
# Continuation of pick-two low-frequency Region from complete Section 3 checkpoint; runs E4-E10.
# GPU0 embedding; GPU1-2 Qwen72B. Two logs: driver.log + region.log.
set -euo pipefail
MEMRL_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
OVERLAY="$MEMRL_DIR/scripts/region_quality_fixed_overlay"
MODEL=/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct
EMBED=/storage/openpsi/models/Qwen3-Embedding-8B
TAG="${1:-$(date +%Y%m%d_%H%M%S)}"; SAFE="${TAG//_/-}"
read -r EP RP NCCL_REGION < <(python3 - <<'PYPORTS'
import socket
ports=[]
for _ in range(3):
    s=socket.socket(); s.bind(('127.0.0.1',0)); ports.append(s.getsockname()[1]); s.close()
print(*ports)
PYPORTS
)
OUT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_holdout_qwen72b_picktwo_region_quality_stable_resume_s3_20260808
RESUME_ROOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_holdout_qwen72b_picktwo_region_quality_stable_20260807/alfworld/exp_alfworld_holdout_picktwo_qwen72b_region_quality_stable_20260807_214636_qwen72b-picktwo-region-quality-stable-20260807-214636/local_cache
LOG="$MEMRL_DIR/logs/alfworld_holdout_picktwo_qwen72b_region_quality_stable_resume_s3_$TAG"
WORK_TMP=/storage/openpsi/users/yl/agent-memory/.tmp/q72b_picktwo_region_quality_stable_resume_s3_${TAG}_$$
mkdir -p "$OUT" "$LOG" "$WORK_TMP"; chmod 700 "$WORK_TMP"
exec > >(tee -a "$LOG/driver.log") 2>&1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$OVERLAY:$MEMRL_DIR:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MEMRL_ALFWORLD_LLM_CONCURRENCY=16 MEMRL_ALFWORLD_DEFERRED_REPAIR_MAX_GAMES=16
DEPS_SITE="$WORK_TMP/site-packages"
mkdir -p "$DEPS_SITE"
PIP_BIN="$(command -v pip || command -v pip3 || true)"
if [[ -z "$PIP_BIN" ]]; then
  echo "[ERROR] neither pip nor pip3 is available in the runtime image"
  exit 1
fi
echo "[DEPS] pip=$PIP_BIN; installing hdbscan==0.8.40 into isolated $DEPS_SITE"
"$PIP_BIN" install --disable-pip-version-check --no-deps --target "$DEPS_SITE" 'hdbscan==0.8.40' -i https://pypi.antfin-inc.com/simple/
export PYTHONPATH="$DEPS_SITE:$PYTHONPATH"
cd "$MEMRL_DIR"
python3 - <<'PYDEPS'
import sys
sys.path.insert(0, '/storage/openpsi/users/yl/agent-memory/MemRL/scripts/region_quality_fixed_overlay')
import memrl, hdbscan
from hdbscan import HDBSCAN
print('[OK] dependency preflight memrl overlay:', memrl.__file__)
print('[OK] hdbscan preflight:', hdbscan.__file__, HDBSCAN)
assert 'region_quality_fixed_overlay' in memrl.__file__, memrl.__file__
assert 'site-packages/hdbscan' in hdbscan.__file__, hdbscan.__file__
PYDEPS
unset TMPDIR TMP TEMP
PIDS=()
cleanup(){ rc=$?; trap - EXIT INT TERM; ((${#PIDS[@]})) && kill "${PIDS[@]}" 2>/dev/null || true; rm -rf -- "$WORK_TMP"; echo "[CLEANUP] exit=$rc $(date -Is)"; exit "$rc"; }
trap cleanup EXIT INT TERM
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path "$EMBED" --served-model-name Qwen/Qwen3-Embedding-8B --host 127.0.0.1 --port "$EP" --context-length 8192 --trust-remote-code --is-embedding & PIDS+=("$!")
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name Qwen2.5-72B-Instruct --tp 2 --host 127.0.0.1 --port "$RP" --context-length 32768 --trust-remote-code --nccl-port "$NCCL_REGION" & PIDS+=("$!")
wait_server(){ n=$1 p=$2; for i in $(seq 1 1800); do curl -fsS "http://127.0.0.1:$p/v1/models" 2>/dev/null | grep -q model && { echo "[READY] $n ${i}s"; return; }; sleep 1; done; return 1; }
wait_server embed "$EP"; wait_server region "$RP"
export TMPDIR="$WORK_TMP" TMP="$WORK_TMP" TEMP="$WORK_TMP"
CFG="$WORK_TMP/region_quality_fixed.yaml"
cat > "$CFG" <<YAML
llm: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${RP}/v1/", model: Qwen2.5-72B-Instruct, temperature: 0, max_tokens: 4096}
embedding: {provider: openai, api_key: EMPTY, base_url: "http://localhost:${EP}/v1/", model: Qwen/Qwen3-Embedding-8B, max_text_len: 8196, dimension: 4096}
memory: {build_strategy: trajectory, retrieve_strategy: query, update_strategy: adjustment, k_retrieve: 5, max_keywords: 5, add_similarity_threshold: 0.9, memory_budget_tokens: 0, sim_norm_mean: 0.5187, sim_norm_std: 0.1203}
environment: {alfworld_config_path: configs/envs/alfworld.yaml, alfworld_env_type: AlfredTWEnv}
experiment:
  random_seed: 42
  enable_value_driven: true
  experiment_name: alfworld_holdout_picktwo_qwen72b_region_quality_stable_resume_s3_${TAG}
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
  ckpt_resume_enabled: true
  ckpt_resume_path: "${RESUME_ROOT}"
  ckpt_resume_epoch: 3
  batch_checkpoint_interval: 10
  batch_checkpoint_keep: 1
  n_eval_runs: 1
rl_config: {epsilon: 0, tau: 0.62, alpha: 0.3, gamma: 0.0, q_init_pos: 0, q_init_neg: 0, success_reward: 1.0, failure_reward: -1.0, topk: 3, novelty_threshold: 0.85, recency_boost: 0.0, reward_merge_gain: 0.1, q_min_threshold: -10, weight_sim: 0.5, weight_q: 0.5}
YAML
export MEMRL_RUN_ID="qwen72b-picktwo-region-quality-stable-resume-s3-${SAFE}"
echo "[RESUME-PREFLIGHT] source=${RESUME_ROOT}/snapshot/3 expected_checkpoint_id=3 start_section=4"
python3 - <<PYRESUME
import json, pathlib
p=pathlib.Path("${RESUME_ROOT}/snapshot/3")
m=json.loads((p/"snapshot_meta.json").read_text())
assert m.get("checkpoint_id")==3, m
for rel in ("cube/textual_memory.json","local_cache/cum_state.json","local_cache/region_manager.json","local_cache/global_q_cache.json"):
    q=p/rel
    assert q.is_file() and q.stat().st_size>10, q
print("[OK] complete Section 3 resume checkpoint:", p)
PYRESUME
echo "[QUALITY-CONFIG] fresh=false resume_s3=true holdout=pick_two train_expected=2740 holdout_expected=837 init_step=2500 precluster=soft_source_backfill:1.0 freeze_after_initial=false low_frequency_topology=true cooldown=1 temperature=0.05 top_n=2 propagation=0.01/8/0.65 mem_cache=30000 FS_slots=1 tau=0.62 ws/wq=0.5/0.5 keep=1"
python3 "$OVERLAY/run/run_alfworld.py" \
  --config "$CFG" \
  --holdout_subtask alf/pick_two_obj_and_place \
  --holdout_eval_pools train,valid \
  --skip_initial_eval \
  --region \
  --region_gating_mode additive \
  --region_utility_mode beta \
  --region_split_evidence_migration_mode soft_source_conserving \
  --region_precluster_evidence_mode soft_source_backfill \
  --region_precluster_evidence_scale 1.0 \
  --region_cluster_init_step 2500 \
  --region_merge_interval 2740 \
  --region_disable_mid_epoch_topology \
  --region_topology_cooldown_sections 1 \
  --region_temperature 0.05 \
  --shrinkage_top_n 2 \
  --shrinkage_confidence_k 2.5 \
  --propagation_eta 0.01 \
  --propagation_k 8 \
  --propagation_sim_min 0.65 \
  --mem_cache_max_size 30000 \
  --failure_summary_n_slots 1 \
  2>&1 | tee -a "$LOG/region.log"
echo "[ALL-DONE] region=0 $(date -Is)"
