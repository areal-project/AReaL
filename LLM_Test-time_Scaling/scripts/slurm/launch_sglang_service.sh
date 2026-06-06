#!/bin/bash
#SBATCH --job-name=tts_sglang-service
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=10000M
#SBATCH --chdir=path-to-results/LLM_Test-time_Scaling
#SBATCH --output=logs/sglang_%j_%N.out
#SBATCH --error=logs/sglang_%j_%N.err
#SBATCH --time=10000:00:00

# Usage: sbatch launch_sglang_service.sh <MODEL_PATH> <PORT> <SERVICE_REGISTRY_FILE> <ADDITIONAL_SGLANG_ARGS>
# Example: sbatch launch_sglang_service.sh gpt-oss-120b 8000 services.txt "--tensor-parallel-size 4"

set -e

MODEL_PATH=${1:-"gpt-oss-120b"}
PORT=${2:-8000}
SERVICE_REGISTRY=${3:-"services.txt"}
ADDITIONAL_SGLANG_ARGS=${4:-""}

# Create logs directory if it doesn't exist
mkdir -p logs

# Get host IP (try multiple methods)
get_host_ip() {
    # Method 1: Try hostname -I (most reliable on compute nodes)
    HOST_IP=$(hostname -I | awk '{print $1}')

    if [ -z "$HOST_IP" ]; then
        # Method 2: Try getting from specific interface
        HOST_IP=$(ip addr show | grep 'inet ' | grep -v '127.0.0.1' | head -1 | awk '{print $2}' | cut -d'/' -f1)
    fi

    if [ -z "$HOST_IP" ]; then
        # Method 3: Fallback to hostname resolution
        HOST_IP=$(hostname -i)
    fi

    echo "$HOST_IP"
}

HOST_IP=$(get_host_ip)
HOSTNAME=$(hostname)
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "=========================================="
echo "SGlang Service Launch"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Host IP: $HOST_IP"
echo "Port: $PORT"
echo "Model: $MODEL_PATH"
echo "Additional args: $ADDITIONAL_SGLANG_ARGS"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Record service info to registry file
echo "$SLURM_JOB_ID|$HOSTNAME|$HOST_IP|$PORT|$MODEL_PATH|$TIMESTAMP|running" >> "$SERVICE_REGISTRY"

echo "Service registered to: $SERVICE_REGISTRY"

# Launch sglang service inside singularity container
echo "Launching SGlang service..."

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    /storage/openpsi/images/areal-latest.sif \
    bash -c "
        export SGLANG_USE_AITER=0
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

        echo 'Starting SGlang server...'
        python3 -m sglang.launch_server \
            --model $MODEL_PATH \
            --attention-backend triton \
            --host $HOST_IP \
            --port $PORT \
            --log-level info \
            $ADDITIONAL_SGLANG_ARGS
    "

# If service exits, update registry
EXIT_CODE=$?
TIMESTAMP_END=$(date '+%Y-%m-%d %H:%M:%S')

echo "Service stopped at $TIMESTAMP_END with exit code $EXIT_CODE"

# Update service status in registry
sed -i "s/^$SLURM_JOB_ID|.*|running$/$SLURM_JOB_ID|$HOSTNAME|$HOST_IP|$PORT|$MODEL_PATH|$TIMESTAMP|stopped|$TIMESTAMP_END|$EXIT_CODE/" "$SERVICE_REGISTRY"

exit $EXIT_CODE
