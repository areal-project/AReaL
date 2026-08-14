#!/bin/bash
#SBATCH --job-name=yl-alf-qwen36-passk-resume
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=slurmd-16
#SBATCH --output=logs/alf_qwen36_passk_resume_%j.log
#SBATCH --error=logs/alf_qwen36_passk_resume_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
VLLM_IMG="/storage/openpsi/images/areal-vllm0202-torch211.sif"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
QWEN36_PORT=8500
EMBED_PORT=8201

echo "=========================================="
echo "ALFWorld Qwen3.6: pass@10 resume (8 rounds)"
echo "SLURM CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Job $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage \
    $VLLM_IMG bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3
python -m vllm.entrypoints.openai.api_server \
    --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 1 --port $QWEN36_PORT --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 --seed 42
" &
VLLM_BG_PID=$!

for i in $(seq 1 1800); do curl -s "http://localhost:${QWEN36_PORT}/health" > /dev/null 2>&1 && echo "[INFO] LLM ready!" && break; kill -0 $VLLM_BG_PID 2>/dev/null || { echo "[ERROR] died"; exit 1; }; sleep 1; done
for i in $(seq 1 600); do curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo "[INFO] Embed ready!" && break; sleep 1; done

singularity exec --no-home --writable-tmpfs --bind /storage:/storage $RUNNER_IMG bash -c '
cd /storage/openpsi/users/yl/agent-memory/MemRL
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai "chonkie==1.2.1" tensorboard hdbscan textworld alfworld --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

cat > /tmp/passk.yaml << EOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:8500/v1/
  model: Qwen3.6-35B-A3B
  temperature: 0
  max_tokens: 4096
embedding:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:8201/v1/
  model: Qwen/Qwen3-Embedding-8B
  max_text_len: 8196
  dimension: 4096
memory:
  build_strategy: proceduralization
  retrieve_strategy: query
  update_strategy: adjustment
  k_retrieve: 0
  max_keywords: 5
  add_similarity_threshold: 0.9
  memory_budget_tokens: 0
  sim_norm_mean: 0.5187
  sim_norm_std: 0.1203
environment:
  alfworld_config_path: configs/envs/alfworld.yaml
  alfworld_env_type: AlfredTWEnv
experiment:
  random_seed: 43
  enable_value_driven: false
  experiment_name: alfworld_passk8_qwen36_resume
  mode: train
  num_sections: 8
  batch_size: 32
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: passk
  baseline_k: 10
  output_dir: /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld
  max_steps: 30
  save_trajectories: false
  save_memories: false
  ckpt_resume_enabled: false
  ckpt_resume_path: ""
  ckpt_resume_epoch: null
rl_config:
  epsilon: 0
  tau: 0.62
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
  weight_sim: 0.5
  weight_q: 0.5
EOF
python run/run_alfworld.py --config /tmp/passk.yaml
echo "[pass@k resume] exit=$? at $(date)"
'
kill $VLLM_BG_PID 2>/dev/null; wait $VLLM_BG_PID 2>/dev/null
echo "End: $(date)"
