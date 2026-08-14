#!/bin/bash
#SBATCH --job-name=yl-alf-qwen36-all
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --gres=gpu:4
#SBATCH --nodelist=slurmd-16
#SBATCH --output=logs/alf_qwen36_all_%j.log
#SBATCH --error=logs/alf_qwen36_all_%j.log

# Qwen3.6-35B-A3B ALFWorld: ALL experiments in one job
# GPU 0: Embed | GPU 1: Qwen3.6 (memrl) | GPU 2: Qwen3.6 (region) | GPU 3: Qwen3.6 (baselines)
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
VLLM_IMG="/storage/openpsi/images/areal-vllm0202-torch211.sif"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"

EMBED_PORT=8001
MEMRL_PORT=8200
REGION_PORT=8300
BASELINE_PORT=8400

echo "=========================================="
echo "ALFWorld Qwen3.6: ALL experiments (memrl + region + baselines)"
echo "Job $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

# --- Phase A: Start all vLLM servers ---
singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage \
    $VLLM_IMG bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/storage/openpsi/users/yl/agent-memory/.cache/triton

# Start servers sequentially to avoid /tmp Triton compilation overflow
# Embed on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --port $EMBED_PORT --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --seed 42 &
EMBED_PID=\$!
sleep 5

# Qwen3.6 for memrl on GPU 1 (start first, let Triton compile and cache)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 1 --port $MEMRL_PORT --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 --seed 42 &
LLM1_PID=\$!

# Wait for first LLM to be ready (Triton compilation done)
for i in \$(seq 1 1800); do
    curl -s http://localhost:${MEMRL_PORT}/health > /dev/null 2>&1 && echo '[INFO] LLM1 ready, starting LLM2+LLM3...' && break
    kill -0 \$LLM1_PID 2>/dev/null || { echo '[ERROR] LLM1 died'; exit 1; }
    sleep 1
done

# Now start LLM2 and LLM3 (Triton cache is warm)
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 1 --port $REGION_PORT --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 --seed 42 &

CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 1 --port $BASELINE_PORT --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 --seed 42 &

wait
" &
VLLM_BG_PID=$!

# Wait for all servers
echo "[INFO] Waiting for servers..."
for port in $EMBED_PORT $MEMRL_PORT $REGION_PORT $BASELINE_PORT; do
    for i in $(seq 1 1800); do
        curl -s "http://localhost:${port}/health" > /dev/null 2>&1 && echo "[INFO] Port $port ready!" && break
        kill -0 $VLLM_BG_PID 2>/dev/null || { echo "[ERROR] vLLM container died"; exit 1; }
        [ "$i" -eq 1800 ] && echo "[ERROR] Port $port timeout" && kill $VLLM_BG_PID 2>/dev/null && exit 1
        sleep 1
    done
done

echo "=========================================="
echo "All 4 servers ready. Starting 3 experiments in parallel..."
echo "=========================================="

# --- Phase B: Run 3 experiments in parallel ---
singularity exec --no-home --writable-tmpfs --bind /storage:/storage \
    $RUNNER_IMG bash -c "
cd $MEMRL_DIR
find . -name '*.pyc' -delete 2>/dev/null; find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan textworld alfworld --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# Generate configs with correct ports
NOMEM_CFG=/tmp/alf_qwen36_nomem_\$\$.yaml
MEMRL_CFG=/tmp/alf_qwen36_memrl_\$\$.yaml
REGION_CFG=/tmp/alf_qwen36_region_\$\$.yaml
BASELINE_CFG=/tmp/alf_qwen36_baseline_\$\$.yaml

sed 's|localhost:8100|localhost:${MEMRL_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g' \
    configs/rl_alf_config.qwen36_nomem.yaml > \"\$NOMEM_CFG\"
sed 's|localhost:8100|localhost:${MEMRL_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g' \
    configs/rl_alf_config.qwen36_memrl.yaml > \"\$MEMRL_CFG\"
sed 's|localhost:8100|localhost:${REGION_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g' \
    configs/rl_alf_config.qwen36_memrl.yaml > \"\$REGION_CFG\"
sed 's|localhost:8100|localhost:${BASELINE_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g; s|alfworld_memrl_qwen36|alfworld_passk10_qwen36|; s|enable_value_driven: true|enable_value_driven: false|; s|k_retrieve: 5|k_retrieve: 0|; s|num_sections: 10|num_sections: 10|' \
    configs/rl_alf_config.qwen36_memrl.yaml > \"\$BASELINE_CFG\"

echo '=========================================='
echo 'Launching 3 parallel experiments...'
echo '=========================================='

# Experiment 1: no-mem → memrl
(
    echo '[EXP1] no-mem eval (train set)'
    python run/run_alfworld.py --config \"\$NOMEM_CFG\" --eval_train
    echo \"[EXP1 no-mem] exit=\$? at \$(date)\"
    echo '[EXP1] memrl (10 sections)'
    python run/run_alfworld.py --config \"\$MEMRL_CFG\" --skip_initial_eval
    echo \"[EXP1 memrl] exit=\$? at \$(date)\"
) &
EXP1_PID=\$!

# Experiment 2: region+FS
(
    echo '[EXP2] region+FS (10 sections)'
    python run/run_alfworld.py \
        --config \"\$REGION_CFG\" \
        --region --region_gating_mode additive \
        --region_utility_mode beta \
        --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
        --val_lambda_max 0.05 --no_z_norm \
        --explore_schedule '0,2,2,1,1,1,1,0,0,0' \
        --failure_summary_n_slots 2 \
        --skip_initial_eval
    echo \"[EXP2 region] exit=\$? at \$(date)\"
) &
EXP2_PID=\$!

# Experiment 3: baselines (pass@10 → RAG → MemP)
(
    echo '[EXP3] pass@10'
    # pass@10 config: no memory, baseline_mode=passk
    sed -i 's|enable_value_driven: false|enable_value_driven: false|; s|alfworld_passk10_qwen36|alfworld_passk10_qwen36|' \"\$BASELINE_CFG\"
    cat > /tmp/alf_qwen36_passk_\$\$.yaml << PCFG
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${BASELINE_PORT}/v1/
  model: Qwen3.6-35B-A3B
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
  random_seed: 42
  enable_value_driven: false
  experiment_name: alfworld_passk10_qwen36
  mode: train
  num_sections: 10
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
  ckpt_resume_path: ''
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
PCFG
    python run/run_alfworld.py --config /tmp/alf_qwen36_passk_\$\$.yaml
    echo \"[EXP3 pass@10] exit=\$? at \$(date)\"

    echo '[EXP3] RAG (k=3, no Q-value)'
    cat > /tmp/alf_qwen36_rag_\$\$.yaml << RCFG
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${BASELINE_PORT}/v1/
  model: Qwen3.6-35B-A3B
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
  build_strategy: proceduralization
  retrieve_strategy: query
  update_strategy: adjustment
  k_retrieve: 3
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
  enable_value_driven: false
  experiment_name: alfworld_rag_qwen36
  mode: train
  num_sections: 10
  batch_size: 32
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: null
  baseline_k: 10
  output_dir: /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld
  max_steps: 30
  save_trajectories: true
  save_memories: true
  ckpt_resume_enabled: false
  ckpt_resume_path: ''
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
RCFG
    python run/run_alfworld.py --config /tmp/alf_qwen36_rag_\$\$.yaml --skip_initial_eval
    echo \"[EXP3 RAG] exit=\$? at \$(date)\"

    echo '[EXP3] MemP (procedural, no Q, k=3)'
    cat > /tmp/alf_qwen36_memp_\$\$.yaml << MCFG
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${BASELINE_PORT}/v1/
  model: Qwen3.6-35B-A3B
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
  build_strategy: proceduralization
  retrieve_strategy: query
  update_strategy: adjustment
  k_retrieve: 3
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
  enable_value_driven: false
  experiment_name: alfworld_memp_qwen36
  mode: train
  num_sections: 10
  batch_size: 32
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: null
  baseline_k: 10
  output_dir: /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld
  max_steps: 30
  save_trajectories: true
  save_memories: true
  ckpt_resume_enabled: false
  ckpt_resume_path: ''
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
MCFG
    python run/run_alfworld.py --config /tmp/alf_qwen36_memp_\$\$.yaml --skip_initial_eval
    echo \"[EXP3 MemP] exit=\$? at \$(date)\"
) &
EXP3_PID=\$!

# Wait for all experiments
wait \$EXP1_PID; echo \"[EXP1 done] at \$(date)\"
wait \$EXP2_PID; echo \"[EXP2 done] at \$(date)\"
wait \$EXP3_PID; echo \"[EXP3 done] at \$(date)\"

echo '=========================================='
echo \"ALL EXPERIMENTS DONE at \$(date)\"
echo '=========================================='
"

kill $VLLM_BG_PID 2>/dev/null; wait $VLLM_BG_PID 2>/dev/null
echo "End: $(date)"
