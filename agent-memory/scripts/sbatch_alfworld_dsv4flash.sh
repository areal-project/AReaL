#!/bin/bash
#SBATCH --job-name=yl-alf-dsv4flash
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --gres=gpu:4
#SBATCH --nodelist=slurmd-16
#SBATCH --output=logs/alf_dsv4flash_%j.log
#SBATCH --error=logs/alf_dsv4flash_%j.log

# DSv4-Flash ALFWorld: no-mem eval → memrl 10 sections
# Dual-image: vLLM serve in vllm0202 image (+ transformers upgrade), runner in areal-latest
# GPU: 0-3 DSv4-Flash TP=4 | Embed shared from Qwen3.6 job (port 8001)
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
VLLM_IMG="/storage/openpsi/images/areal-vllm0202-torch211.sif"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
DSV4_PATH="/storage/openpsi/models/deepseek-v4-flash"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"

DSV4_PORT=8000
EMBED_PORT=8001

echo "=========================================="
echo "ALFWorld DSv4-Flash: no-mem + memrl (vLLM + transformers 4.57.1)"
echo "Job $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

# --- Phase A: Start vLLM servers (background) ---
singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage \
    $VLLM_IMG bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

echo '[INFO] Upgrading transformers and vllm for V4 support...'
pip install transformers==4.57.1 2>&1 | tail -5
pip install 'vllm>=0.20.0' 2>&1 | tail -15
echo '[INFO] versions after upgrade:'
python -c 'import transformers,vllm; print(f\"transformers={transformers.__version__}, vllm={vllm.__version__}\")' 2>&1

export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

# Start DSv4-Flash on GPU 0-3 (no embed - shared from Qwen3.6 job)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model $DSV4_PATH --served-model-name deepseek-v4-flash \
    --tensor-parallel-size 4 --port $DSV4_PORT --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --tokenizer-mode deepseek_v4 --enable-expert-parallel \
    --kv-cache-dtype fp8 --block-size 256 \
    --seed 42 --disable-frontend-multiprocessing &
LLM_PID=\$!

wait \$LLM_PID
" &
VLLM_BG_PID=$!

# Wait for shared Embed (from Qwen3.6 job, port 8001)
echo "[INFO] Waiting for shared Embed (port $EMBED_PORT from Qwen3.6 job)..."
for i in $(seq 1 3600); do
    curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo "[INFO] Shared Embed ready!" && break
    [ "$i" -eq 3600 ] && echo "[ERROR] Embed never came up" && kill $VLLM_BG_PID 2>/dev/null && exit 1
    sleep 1
done

# Wait for DSv4-Flash (MoE loading can take 15-30 min)
echo "[INFO] Waiting for DSv4-Flash (port $DSV4_PORT, may take 15-30 min)..."
for i in $(seq 1 3600); do
    curl -s "http://localhost:${DSV4_PORT}/health" > /dev/null 2>&1 && echo "[INFO] DSv4-Flash ready!" && break
    kill -0 $VLLM_BG_PID 2>/dev/null || { echo "[ERROR] vLLM container died"; exit 1; }
    [ "$i" -eq 3600 ] && echo "[ERROR] DSv4-Flash timeout (60min)" && kill $VLLM_BG_PID 2>/dev/null && exit 1
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

echo '=========================================='
echo 'Servers ready. Starting experiments...'
echo '=========================================='

# Phase 1: no-mem eval
echo '[DSv4] Phase 1: no-mem eval (train set)'
python run/run_alfworld.py --config configs/rl_alf_config.dsv4flash_nomem.yaml --eval_train
echo \"[DSv4 no-mem] exit=\$? at \$(date)\"

# Phase 2: memrl 10 sections
echo '[DSv4] Phase 2: memrl (10 sections)'
python run/run_alfworld.py --config configs/rl_alf_config.dsv4flash_memrl.yaml --skip_initial_eval
echo \"[DSv4 memrl] exit=\$? at \$(date)\"

echo '=========================================='
echo \"All done. End: \$(date)\"
echo '=========================================='
"

# Cleanup
kill $VLLM_BG_PID 2>/dev/null; wait $VLLM_BG_PID 2>/dev/null
echo "End: $(date)"
