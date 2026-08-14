#!/bin/bash
#SBATCH --job-name=cross_webshop_alf_v7
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_webshop_alf_v7_%j.out
#SBATCH --error=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_webshop_alf_v7_%j.err

set -ex

echo "[INFO] Starting job on $(hostname)"

cd /storage/openpsi/users/yl/agent-memory/MemRL

# Use singularity container
singularity exec --nv --bind /storage:/storage /storage/openpsi/images/areal-latest.sif bash << 'CONTAINER_SCRIPT'
set -ex

cd /storage/openpsi/users/yl/agent-memory/MemRL

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

# Install dependencies
pip install -e . -q 2>/dev/null || true
pip install 'litellm[proxy]' -q

# Start LiteLLM proxy
LITELLM_PORT=$((40000 + RANDOM % 1000))
cp /storage/openpsi/users/yl/.claude/config.yaml /tmp/litellm_cross_webshop.yaml

echo "[INFO] Starting LiteLLM on port ${LITELLM_PORT}..."
python3 -m litellm.proxy.proxy_cli --config /tmp/litellm_cross_webshop.yaml --port ${LITELLM_PORT} --host 127.0.0.1 &
LITELLM_PID=$!
LITELLM_URL="http://127.0.0.1:${LITELLM_PORT}"

# Wait for LiteLLM to be ready
for i in $(seq 1 60); do
    if curl -s "${LITELLM_URL}/health" > /dev/null 2>&1; then
        echo "[INFO] LiteLLM ready at ${LITELLM_URL}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "[ERROR] LiteLLM failed to start"
        exit 1
    fi
    sleep 2
done

echo "[INFO] Starting WebShop→ALF cross-benchmark experiment..."

# Run the cross-benchmark experiment
python scripts/run_cross_benchmark_experiment.py \
    --source webshop \
    --targets alf \
    --api_key "sk-placeholder" \
    --base_url "${LITELLM_URL}" \
    --model "gpt-4o-2024-11-20" \
    --embedding_model "text-embedding-3-small" \
    --epochs 10 \
    --batch_size 5 \
    --name "cross_webshop_alf_v7" \
    --mode local

echo "[INFO] Stopping LiteLLM..."
kill ${LITELLM_PID} 2>/dev/null || true

echo "[INFO] WebShop→ALF experiment completed"
CONTAINER_SCRIPT

echo "[INFO] Job finished"
