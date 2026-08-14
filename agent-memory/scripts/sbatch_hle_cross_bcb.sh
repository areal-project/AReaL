#!/bin/bash
#SBATCH --job-name=yl-hle-cross-bcb
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --exclude=slurmd-23
#SBATCH --output=logs/hle_cross_bcb_%j.log
#SBATCH --error=logs/hle_cross_bcb_%j.log

# Cross-benchmark experiment: BCB-trained memory → HLE eval
# Tests whether code generation memories transfer to scientific QA
#
# Usage:
#   BCB_CKPT=/path/to/bcb/epoch5/snapshot/step_752 sbatch scripts/sbatch_hle_cross_bcb.sh
#
# If BCB_CKPT not set, uses the latest bs=8 baseline checkpoint

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

# Default BCB checkpoint: bs=8 baseline latest epoch
BCB_CKPT="${BCB_CKPT:-${MEMRL_DIR}/results/deepseek_memrl_old_bs8}"

echo "=========================================="
echo "HLE Cross-Benchmark Eval (BCB memory → HLE)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "BCB checkpoint: $BCB_CKPT"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Starting on \$(hostname)'

echo '[INFO] Installing dependencies...'
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# Generate temp config with dynamic ports and checkpoint path
TEMP_CONFIG=/tmp/hle_cross_\$\$.yaml
sed \"s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g; s|checkpoint_path: \\\"\\\"|checkpoint_path: \\\"${BCB_CKPT}\\\"|g\" configs/rl_hle_config.cross_bcb.yaml > \$TEMP_CONFIG
echo \"[INFO] Using ports: LLM=${LLM_PORT}, Embed=${EMBED_PORT}\"
echo \"[INFO] BCB checkpoint: ${BCB_CKPT}\"

# --- Start embedding vLLM on GPU 4 ---
echo '[INFO] Starting embedding vLLM (Qwen3-Embedding-8B) on GPU 4...'
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model ${EMBED_MODEL_PATH} \
    --served-model-name Qwen/Qwen3-Embedding-8B \
    --port ${EMBED_PORT} \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --trust-remote-code \
    --disable-log-requests \
    --seed 42 \
    &
EMBED_PID=\$!

echo '[INFO] Waiting for embedding vLLM to be ready...'
for i in \$(seq 1 1200); do
    if curl -s http://localhost:${EMBED_PORT}/health > /dev/null 2>&1; then
        echo '[INFO] Embedding vLLM is ready!'
        break
    fi
    if [ \$i -eq 1200 ]; then
        echo '[ERROR] Embedding vLLM failed to start after 1200s'
        kill \$EMBED_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# --- Start LLM vLLM on GPUs 0-3 (TP=4) ---
echo '[INFO] Starting LLM vLLM (DeepSeek-R1-32B, TP=4) on GPUs 0-3...'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 4 \
    --port ${LLM_PORT} \
    --trust-remote-code \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.80 \
    --disable-log-requests \
    --seed 42 \
    --disable-frontend-multiprocessing \
    &
LLM_PID=\$!

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo '[INFO] Waiting for LLM vLLM server to be ready...'
for i in \$(seq 1 900); do
    if curl -s http://localhost:${LLM_PORT}/health > /dev/null 2>&1; then
        echo '[INFO] LLM vLLM server is ready!'
        break
    fi
    if [ \$i -eq 900 ]; then
        echo '[ERROR] LLM vLLM server failed to start after 900s'
        kill \$EMBED_PID \$LLM_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# --- Phase 1: HLE no-memory baseline ---
echo ''
echo '=========================================='
echo '[HLE-BASELINE] Running HLE without memory (baseline)'
echo '=========================================='

python run/run_hle.py \
    --config \$TEMP_CONFIG \
    --num_valid 0 \
    --num_train 0

HLE_BASE_EXIT=\$?
echo \"[INFO] HLE baseline exited with code: \$HLE_BASE_EXIT\"

# --- Phase 2: HLE with BCB cross-benchmark memory ---
echo ''
echo '=========================================='
echo '[HLE-CROSS] Running HLE with BCB memory (cross-benchmark transfer)'
echo 'Checkpoint: ${BCB_CKPT}'
echo '=========================================='

# Modify config to enable checkpoint loading
TEMP_CONFIG_CROSS=/tmp/hle_cross_mem_\$\$.yaml
cp \$TEMP_CONFIG \$TEMP_CONFIG_CROSS
# The checkpoint_path is already set in the config

python run/run_hle.py \
    --config \$TEMP_CONFIG_CROSS \
    --num_valid 0 \
    --num_train 0

HLE_CROSS_EXIT=\$?
echo \"[INFO] HLE cross-benchmark exited with code: \$HLE_CROSS_EXIT\"

# Cleanup vLLM servers
kill \$EMBED_PID \$LLM_PID 2>/dev/null
wait \$EMBED_PID 2>/dev/null
wait \$LLM_PID 2>/dev/null
echo '[INFO] vLLM servers stopped.'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
