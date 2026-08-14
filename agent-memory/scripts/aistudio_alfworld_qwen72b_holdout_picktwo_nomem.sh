#!/bin/bash
# True no-memory single-pass holdout eval on hard pick_two_obj_and_place (n=837).
set -euo pipefail
D=/storage/openpsi/users/yl/agent-memory/MemRL; MODEL=/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct; EMB=/storage/openpsi/models/Qwen3-Embedding-8B
TAG="${1:-$(date +%Y%m%d_%H%M%S)}"; OUT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_holdout_qwen72b_picktwo_nomem_20260730; LOG=$D/logs/aistudio_qwen72b_holdout_picktwo_nomem_$TAG.log; WORK_TMP=/storage/openpsi/users/yl/agent-memory/.tmp/q72b_picktwo_nomem_${TAG}_$$
mkdir -p "$OUT" "$(dirname "$LOG")" "$WORK_TMP"; exec > >(tee -a "$LOG") 2>&1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 MEMRL_ALFWORLD_LLM_CONCURRENCY=24
export PYTHONPATH="$D:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "$D"
OVERLAY="$WORK_TMP/overlay"
mkdir -p "$OVERLAY"
# Copy the complete package node-locally so Python cannot mix the patched runner
# with the repository package. This avoids both CPFS writes and namespace shadowing.
cp -a "$D/memrl" "$OVERLAY/memrl"
cp "$D/scripts/holdout_nomem_overlay/memrl/run/alfworld_rl_runner.py" "$OVERLAY/memrl/run/alfworld_rl_runner.py"
export PYTHONPATH="$OVERLAY:$D:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "$WORK_TMP"
python3 - <<'PYOVERLAY'
import memrl.run.alfworld_rl_runner as r
assert '/overlay/memrl/run/' in r.__file__, r.__file__
print('[OK] no-memory holdout overlay:', r.__file__)
PYOVERLAY
unset TMPDIR TMP TEMP
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path "$EMB" --served-model-name Qwen/Qwen3-Embedding-8B --host 127.0.0.1 --port 19090 --context-length 8192 --trust-remote-code --is-embedding & E=$!
CUDA_VISIBLE_DEVICES=1,2 python3 -m sglang.launch_server --model-path "$MODEL" --served-model-name Qwen2.5-72B-Instruct --tp 2 --host 127.0.0.1 --port 19290 --context-length 32768 --trust-remote-code --nccl-port 29690 & L=$!
cleanup(){ rc=$?; kill $L $E 2>/dev/null || true; rm -rf "$WORK_TMP"; echo "[CLEANUP] exit=$rc"; }; trap cleanup EXIT INT TERM
for p in 19090 19290; do for i in $(seq 1 1800); do curl -fsS http://127.0.0.1:$p/v1/models 2>/dev/null | grep -q model && break; sleep 1; done; done
export TMPDIR="$WORK_TMP" TMP="$WORK_TMP" TEMP="$WORK_TMP" MEMRL_RUN_ID="qwen72b-picktwo-nomem-${TAG//_/-}"
cat > "$WORK_TMP/nomem.yaml" <<EOF
llm: {provider: openai, api_key: EMPTY, base_url: "http://localhost:19290/v1/", model: Qwen2.5-72B-Instruct, temperature: 0, max_tokens: 4096}
embedding: {provider: openai, api_key: EMPTY, base_url: "http://localhost:19090/v1/", model: Qwen/Qwen3-Embedding-8B, max_text_len: 8196, dimension: 4096}
memory: {build_strategy: trajectory, retrieve_strategy: query, update_strategy: adjustment, k_retrieve: 0, max_keywords: 5, add_similarity_threshold: 0.9, memory_budget_tokens: 0, sim_norm_mean: 0.5187, sim_norm_std: 0.1203}
environment: {alfworld_config_path: configs/envs/alfworld.yaml, alfworld_env_type: AlfredTWEnv}
experiment: {random_seed: 42, enable_value_driven: false, experiment_name: alfworld_holdout_picktwo_qwen72b_nomem_${TAG}, mode: test, num_sections: 1, batch_size: 128, dataset_ratio: 1.0, few_shot_path: data/alfworld/alfworld_examples.json, baseline_mode: null, output_dir: ${OUT}, max_steps: 30, save_trajectories: true, save_memories: false, valid_interval: 1, test_interval: 1, holdout_subtask: alf/pick_two_obj_and_place, holdout_eval_pools: "train,valid", n_eval_runs: 1}
rl_config: {epsilon: 0, tau: 0.0, alpha: 0.3, gamma: 0.0, q_init_pos: 0, q_init_neg: 0, success_reward: 1.0, failure_reward: -1.0, topk: 3, novelty_threshold: 0.85, recency_boost: 0.0, reward_merge_gain: 0.1, q_min_threshold: -10, weight_sim: 1.0, weight_q: 0.0}
EOF
D="$D" CFG="$WORK_TMP/nomem.yaml" python3 - <<'PYRUN'
import os, runpy, sys
root = os.environ['D']
overlay = os.environ['PYTHONPATH'].split(':', 1)[0]
# Import and pin the patched module before returning to the repository cwd.
sys.path.insert(0, overlay)
import memrl.run.alfworld_rl_runner as patched
assert '/overlay/memrl/run/' in patched.__file__, patched.__file__
os.chdir(root)
sys.argv = [
    os.path.join(root, 'run/run_alfworld.py'),
    '--config', os.environ['CFG'],
    '--holdout_subtask', 'alf/pick_two_obj_and_place',
    '--holdout_eval_pools', 'train,valid',
]
runpy.run_path(sys.argv[0], run_name='__main__')
PYRUN
