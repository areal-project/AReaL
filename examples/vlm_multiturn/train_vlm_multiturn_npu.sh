set -x
set -e

# Hydra configuration
export HYDRA_FULL_ERROR=1

# System limits
ulimit -n 65535

# Update these paths for your environment

CKPTS_DIR=/path/to/areal_log/
MODEL_PATH="/path/to/model_folder"
MODEL_NAME="Qwen3-VL-4B-Instruct"

# Config directory (relative to AReaL root)
CONFIG_PATH="examples/vlm_multiturn"

# Project and experiment names
PROJECT_NAME="vlm_multiturn"
EXPERIMENT_NAME_PRE="${MODEL_NAME}_multiturn_agentic"
TIME_RANDOM=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME=${EXPERIMENT_NAME_PRE}-${TIME_RANDOM}
OUTPUT_LOG_FILE="${CKPTS_DIR}/${EXPERIMENT_NAME}.log"

# Create output directory
mkdir -p "${CKPTS_DIR}"

# Multi-turn Configuration
# ============================================================================

MAX_TURNS=4
TURN_DISCOUNT=0.95

# ============================================================================
# Training Execution
# ============================================================================

# Run training using AReaL's launcher
export USE_OPTIMIZED_MODEL=0
python3 -m areal.launcher.local \
    examples/vlm_multiturn/vlm_multiturn_grpo.py --config examples/vlm_multiturn/vlm_multiturn_grpo.yaml \
    tokenizer_path="${MODEL_PATH}/${MODEL_NAME}" \
    actor.path="${MODEL_PATH}/${MODEL_NAME}" \
    actor.optimizer.lr=1e-6 \
    gconfig.n_samples=4 \
    gconfig.max_new_tokens=1024 \
    gconfig.temperature=0.8 \
    gconfig.top_p=0.95 \
    train_dataset.batch_size=32 \
    valid_dataset.batch_size=32 \
    experiment_name=${EXPERIMENT_NAME} \
    trial_name=trial1 \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=8 \
    cluster.fileroot=${CKPTS_DIR} \
    max_turns=${MAX_TURNS} \
    turn_discount=${TURN_DISCOUNT} \
    total_train_epochs=10 \
    saver.freq_epochs=1 \
    "$@" 2>&1 | tee ${OUTPUT_LOG_FILE}

echo "Training completed. Logs saved to: ${OUTPUT_LOG_FILE}"
