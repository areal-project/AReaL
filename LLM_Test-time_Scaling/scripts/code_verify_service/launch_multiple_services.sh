#!/bin/bash
# Script to launch multiple code verify services on SLURM cluster using srun
# Each service runs in a singularity container via srun
# Usage: bash launch_multiple_services.sh <NUM_NODES> <START_PORT> [DATA_DIR] [SERVICE_REGISTRY]

set -e

NUM_NODES=${1:-4}
START_PORT=${2:-8000}
DATA_DIR=${3:-"path-to-results/llm_test_time_scaling/data/local_data/lcb_testcases/data"}
SERVICE_REGISTRY=${4:-"services.txt"}

# Create necessary directories
mkdir -p logs
mkdir -p slurm_jobs

# Initialize or clear service registry
echo "# Service Registry - $(date)" > "$SERVICE_REGISTRY"
echo "# Format: JOB_ID|HOSTNAME|HOST_IP|PORT|DATA_DIR|START_TIME|STATUS|END_TIME|EXIT_CODE" >> "$SERVICE_REGISTRY"
echo "# =====================================" >> "$SERVICE_REGISTRY"

echo "=========================================="
echo "Launching Multiple Code Verify Services"
echo "=========================================="
echo "Number of nodes: $NUM_NODES"
echo "Starting port: $START_PORT"
echo "Data directory: $DATA_DIR"
echo "Service registry: $SERVICE_REGISTRY"
echo "=========================================="

JOB_IDS=()
PIDS=()

# Launch services on multiple nodes using srun
for i in $(seq 1 $NUM_NODES); do
    PORT=$((START_PORT + i - 1))

    echo "[$i/$NUM_NODES] Starting service on port $PORT..."

    # Use srun to launch service in background
    # Each service runs in a singularity container via srun
    # The launch_service.sh script runs INSIDE the container (does not call singularity itself)
    # CPU 64, GPU 0, Memory per CPU: 10000M
    srun --job-name=tts_cpu_eval \
         --mpi=pmi2 \
         --chdir=path-to-results/LLM_Test-time_Scaling \
         --ntasks=1 \
         --cpus-per-task=64 \
         --mem-per-cpu=10000M \
         --gres=gpu:0 \
         singularity exec --pid --env=ENV_VAR=env_var --nv \
             --bind /storage:/storage \
             /storage/openpsi/images/areal-v0.3.3-sglang-v0.5.2-vllm-v0.10.2-v3.sif \
             bash scripts/code_verify_service/launch_service.sh "$PORT" "$DATA_DIR" "$SERVICE_REGISTRY" &
    
    PID=$!
    PIDS+=($PID)
    echo "  Started with PID $PID on port $PORT"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 2
done

echo ""
echo "=========================================="
echo "All services launched!"
echo "=========================================="
echo "PIDs: ${PIDS[@]}"
echo ""
echo "Monitor services with:"
echo "  cat $SERVICE_REGISTRY"
echo ""
echo "Check service health:"
echo "  curl http://<HOST_IP>:<PORT>/health"
echo ""
echo "Stop services by killing PIDs:"
echo "  kill ${PIDS[@]}"
echo "=========================================="

# Save PIDs for later reference
echo "${PIDS[@]}" > slurm_jobs/latest_service_pids.txt
echo "$(date '+%Y%m%d_%H%M%S')|${PIDS[@]}" >> slurm_jobs/service_history.txt
