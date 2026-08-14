#!/bin/bash
#SBATCH --job-name=yl-alf-qwen36
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=slurmd-16
#SBATCH --output=logs/alf_qwen36_%j.log
#SBATCH --error=logs/alf_qwen36_%j.log

# Qwen3.6-35B-A3B ALFWorld: no-mem eval → memrl 10 sections
# Dual-image: vLLM serve in vllm0202 image, runner in areal-latest image
# GPU 0: Qwen3.6 TP=1 | GPU 1: Embed (Qwen3-Embedding-8B)
# Embed stays up for DSv4 job to share later.
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
VLLM_IMG="/storage/openpsi/images/areal-vllm0202-torch211.sif"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"

QWEN36_PORT=8200
EMBED_PORT=8001

echo "=========================================="
echo "ALFWorld Qwen3.6-35B-A3B: no-mem + memrl (dual-image)"
echo "GPU 0: Qwen3.6 | GPU 1: Embed (shared)"
echo "Job $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

# --- Phase A: Start vLLM servers (background) ---
singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage \
    $VLLM_IMG bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

# Start Embed on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --port $EMBED_PORT --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --seed 42 &
EMBED_PID=\$!

# Start Qwen3.6 on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 1 --port $QWEN36_PORT --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 \
    --seed 42 &
LLM_PID=\$!

wait \$LLM_PID \$EMBED_PID
" &
VLLM_BG_PID=$!

# Wait for Embed
echo "[INFO] Waiting for Embed (port $EMBED_PORT)..."
for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo "[INFO] Embed ready!" && break
    kill -0 $VLLM_BG_PID 2>/dev/null || { echo "[ERROR] vLLM container died"; exit 1; }
    [ "$i" -eq 1200 ] && echo "[ERROR] Embed timeout" && kill $VLLM_BG_PID 2>/dev/null && exit 1
    sleep 1
done

# Wait for Qwen3.6
echo "[INFO] Waiting for Qwen3.6 (port $QWEN36_PORT)..."
for i in $(seq 1 1800); do
    curl -s "http://localhost:${QWEN36_PORT}/health" > /dev/null 2>&1 && echo "[INFO] Qwen3.6 ready!" && break
    kill -0 $VLLM_BG_PID 2>/dev/null || { echo "[ERROR] vLLM container died"; exit 1; }
    [ "$i" -eq 1800 ] && echo "[ERROR] Qwen3.6 timeout" && kill $VLLM_BG_PID 2>/dev/null && exit 1
    sleep 1
done

# --- Phase B: Run experiments in areal-latest image ---
singularity exec --no-home --writable-tmpfs --bind /storage:/storage \
    $RUNNER_IMG bash -c "
cd $MEMRL_DIR
echo '[INFO] Installing runner deps...'
find . -name '*.pyc' -delete 2>/dev/null; find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan textworld alfworld --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# Temp configs with correct ports
NOMEM_CFG=/tmp/alf_qwen36_nomem_\$\$.yaml
MEMRL_CFG=/tmp/alf_qwen36_memrl_\$\$.yaml
sed 's|localhost:8100|localhost:${QWEN36_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g' \
    configs/rl_alf_config.qwen36_nomem.yaml > \"\$NOMEM_CFG\"
sed 's|localhost:8100|localhost:${QWEN36_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g' \
    configs/rl_alf_config.qwen36_memrl.yaml > \"\$MEMRL_CFG\"

echo '=========================================='
echo 'Servers ready. Starting experiments...'
echo '=========================================='

# Phase 1: no-mem eval
echo '[Qwen3.6] Phase 1: no-mem eval (train set)'
python run/run_alfworld.py --config \"\$NOMEM_CFG\" --eval_train
echo \"[Qwen3.6 no-mem] exit=\$? at \$(date)\"

# Phase 2: memrl 10 sections
echo '[Qwen3.6] Phase 2: memrl (10 sections)'
python run/run_alfworld.py --config \"\$MEMRL_CFG\" --skip_initial_eval
echo \"[Qwen3.6 memrl] exit=\$? at \$(date)\"

echo '=========================================='
echo \"All done. End: \$(date)\"
echo '=========================================='
"

# Cleanup - keep embed alive? No, job ends = all processes die.
kill $VLLM_BG_PID 2>/dev/null; wait $VLLM_BG_PID 2>/dev/null
echo "End: $(date)"
