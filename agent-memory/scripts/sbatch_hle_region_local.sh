#!/bin/bash
#SBATCH --job-name=yl-hle-region-q235
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --gres=gpu:8
#SBATCH --exclude=slurmd-24
#SBATCH --output=logs/hle_region_q235_%j.log
#SBATCH --error=logs/hle_region_q235_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen3-VL-235B-A22B-Thinking"
LLM_PORT=8000

echo "=========================================="
echo "HLE Region: Qwen3-VL-235B-A22B-Thinking (TP=8, full node) + API embedding"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# --- Start LLM vLLM on all allocated GPUs (TP=8), port 8000 ---
echo '[INFO] Starting LLM vLLM (Qwen3-VL-235B-A22B-Thinking, TP=8)...'
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name Qwen/Qwen3-VL-235B-A22B-Thinking \
    --tensor-parallel-size 8 \
    --port ${LLM_PORT} \
    --trust-remote-code \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.92 \
    --enable-expert-parallel \
    --limit-mm-per-prompt '{\"image\": 4}' \
    --disable-log-requests \
    --disable-frontend-multiprocessing \
    &
LLM_PID=\$!
echo \"[INFO] LLM vLLM PID: \$LLM_PID\"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Wait for LLM vLLM to be ready (235B load is slow)
echo '[INFO] Waiting for LLM vLLM server to be ready...'
for i in \$(seq 1 5400); do
    if curl -s http://localhost:${LLM_PORT}/health > /dev/null 2>&1; then
        echo '[INFO] LLM vLLM server is ready!'
        break
    fi
    if [ \$i -eq 5400 ]; then
        echo '[ERROR] LLM vLLM server failed to start after 5400s'
        kill \$LLM_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# --- Run HLE Region experiment ---
echo ''
echo '=========================================='
echo '[HLE-REGION] Starting HLE MemRL Region (Qwen3-235B, 10 sections)'
echo '=========================================='

python run/run_hle_region.py \
    --config configs/rl_hle_config.region_local.yaml \
    --train data/hle/hle_test.parquet \
    --region_gating_mode additive \
    --region_retrieve_mode global \
    --k_global 30 \
    --k_local 10 \
    --shrinkage_top_n 3 \
    --shrinkage_lambda_max 0.6 \
    --region_utility_mode ema \
    --region_smoothing_C 0.5 \
    --region_cluster_init_step 300 \
    --region_merge_interval 200 \
    --explore_schedule '0,4,3,2,2,1,1,1,1,0'

HLE_EXIT=\$?
echo \"[INFO] HLE Region exited with code: \$HLE_EXIT\"

# Cleanup vLLM server
kill \$LLM_PID 2>/dev/null
wait \$LLM_PID 2>/dev/null
echo '[INFO] vLLM server stopped.'
echo \"[INFO] Final exit code: \$HLE_EXIT\"
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
