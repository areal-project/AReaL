#!/bin/bash
#SBATCH --job-name=cross_bcb_exp
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sbatch_cross_bcb_%j.out
#SBATCH --error=logs/sbatch_cross_bcb_%j.err

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "=========================================="
echo "Cross-Benchmark Memory Transfer Experiment"
echo "Source: BCB"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Run inside singularity container
singularity exec --nv \
    --bind /storage:/storage \
    --pwd ${PROJECT_DIR} \
    $SINGULARITY_IMAGE \
    bash -c "
        set -e
        cd ${PROJECT_DIR}

        echo 'Python version:'
        python --version
        which python

        echo ''
        echo 'Installing dependencies...'
        pip install chonkie pyyaml pydantic tenacity openai --quiet 2>&1 | tail -5
        pip install -e . --quiet 2>&1 | tail -3 || true

        echo ''
        echo '===== Phase 1: Training on BCB ====='
        echo ''

        mkdir -p scripts/generated_configs

        # Create config for BCB training
        cat > scripts/generated_configs/bcb_train_config.yaml << 'YAML_EOF'
llm:
  provider: openai
  api_key: PLACEHOLDER_API_KEY
  base_url: PLACEHOLDER_BASE_URL
  model: gpt-4o-2024-11-20
  temperature: 0.0
  max_tokens: 12800

embedding:
  provider: openai
  api_key: PLACEHOLDER_API_KEY
  base_url: PLACEHOLDER_BASE_URL
  model: text-embedding-3-small
  max_text_len: 8196

memory:
  build_strategy: proceduralization
  retrieve_strategy: query
  update_strategy: adjustment
  k_retrieve: 5
  max_keywords: 8
  confidence_threshold: 0.0
  memory_confidence: 100.0
  add_similarity_threshold: 0.9
  mos_config_path: configs/mos_config.json
  user_id: cross_bcb_exp
  sim_norm_mean: 0.31
  sim_norm_std: 0.10

environment:
  alfworld_config_path: configs/envs/alfworld.yaml
  alfworld_env_type: AlfredTWEnv

experiment:
  experiment_name: cross_bcb_train
  algorithm: rl
  val_before_train: false
  enable_value_driven: true
  random_seed: 42
  mode: train
  bcb_run_validation: false
  output_dir: ./results
  save_trajectories: true
  save_memories: true
  enable_logging: true
  log_level: INFO

rl_config:
  epsilon: 0.1
  tau: 0.35
  alpha: 0.3
  gamma: 0.0
  q_init_pos: 0.0
  q_init_neg: 0.0
  success_reward: 1.0
  failure_reward: -1.0
  sim_threshold: 0.5
  topk: 5
  novelty_threshold: 0.85
  recency_boost: 0.0
  reward_merge_gain: 0.1
  q_min_threshold: -0.8
  weight_sim: 0.5
  weight_q: 0.5
YAML_EOF

        # Replace placeholders with actual values
        sed -i \"s|PLACEHOLDER_API_KEY|${LLM_API_KEY}|g\" scripts/generated_configs/bcb_train_config.yaml
        sed -i \"s|PLACEHOLDER_BASE_URL|${LLM_BASE_URL}|g\" scripts/generated_configs/bcb_train_config.yaml

        echo 'Config created. Starting BCB training...'
        echo ''

        python run/run_bcb.py \
            --config scripts/generated_configs/bcb_train_config.yaml \
            --subset hard \
            --epochs 3 \
            --train_ratio 0.7 \
            --seed 42 2>&1

        echo ''
        echo '===== Phase 1 Complete ====='
        echo ''

        # Show results
        echo 'BCB Results:'
        ls -la results/bigcodebench_eval/*/memory/ 2>/dev/null || echo 'No BCB results found'

        # Find and show metrics
        for f in results/bigcodebench_eval/*/memory/*/metrics*.json; do
            if [ -f \"\$f\" ]; then
                echo ''
                echo \"Metrics from \$f:\"
                cat \"\$f\" 2>/dev/null | head -50
            fi
        done
    "

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
