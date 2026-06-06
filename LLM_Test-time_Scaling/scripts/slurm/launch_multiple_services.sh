#!/bin/bash
# Script to launch multiple SGlang services on SLURM cluster with specified nodes
# Usage: bash launch_multiple_services.sh <MODEL_PATH> <START_PORT> <SERVICE_REGISTRY> <ADDITIONAL_SGLANG_ARGS>
# Example: bash launch_multiple_services.sh /path/to/model 8000 services.txt "--tensor-parallel-size 4"

set -e

MODEL_PATH=${1:-"gpt-oss-120b"}
START_PORT=${2:-8000}
SERVICE_REGISTRY=${3:-"services.txt"}
ADDITIONAL_SGLANG_ARGS=${4:-"--tensor-parallel-size 8"}

# Define node list: slurmd-0(8), slurmd-16(8), ..., slurmd-112(8)
# Format: node_name (number in parentheses indicates GPUs, already set in launch_sglang_service.sh)
NODES=(
    "slurmd-0"
    "slurmd-16"
    "slurmd-24"
    "slurmd-36"
    "slurmd-39"
    "slurmd-47"
    "slurmd-50"
    "slurmd-51"
    # "slurmd-58"
    # "slurmd-59"
    # "slurmd-61"
    # "slurmd-64"
    # "slurmd-87"
    # "slurmd-88"
    # "slurmd-95"
    # "slurmd-102"
    # "slurmd-103"
    # "slurmd-104"
    # "slurmd-112"
)

NUM_NODES=${#NODES[@]}

# Create necessary directories
mkdir -p logs
mkdir -p slurm_jobs

# Initialize or clear service registry
echo "# Service Registry - $(date)" > "$SERVICE_REGISTRY"
echo "# Format: JOB_ID|HOSTNAME|HOST_IP|PORT|MODEL_PATH|START_TIME|STATUS|END_TIME|EXIT_CODE" >> "$SERVICE_REGISTRY"
echo "# =====================================" >> "$SERVICE_REGISTRY"

echo "=========================================="
echo "Launching Multiple SGlang Services"
echo "=========================================="
echo "Number of nodes: $NUM_NODES"
echo "Nodes: ${NODES[*]}"
echo "Model: $MODEL_PATH"
echo "Starting port: $START_PORT"
echo "Service registry: $SERVICE_REGISTRY"
echo "Additional SGlang args: $ADDITIONAL_SGLANG_ARGS"
echo "=========================================="

JOB_IDS=()

# Launch services on specified nodes
for i in $(seq 0 $((NUM_NODES - 1))); do
    NODE=${NODES[$i]}
    PORT=$((START_PORT + i))

    echo "[$((i + 1))/$NUM_NODES] Submitting job for node $NODE on port $PORT..."

    JOB_ID=$(sbatch --parsable --nodelist="$NODE" scripts/slurm/launch_sglang_service.sh "$MODEL_PATH" "$PORT" "$SERVICE_REGISTRY" "$ADDITIONAL_SGLANG_ARGS")

    JOB_IDS+=($JOB_ID)

    echo "  Submitted: Job ID $JOB_ID on node $NODE"

    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo "Job IDs: ${JOB_IDS[@]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View service registry:"
echo "  cat $SERVICE_REGISTRY"
echo ""
echo "Cancel all jobs:"
echo "  scancel ${JOB_IDS[@]}"
echo "=========================================="

# Save job IDs for later reference
echo "${JOB_IDS[@]}" > slurm_jobs/latest_job_ids.txt
echo "$(date '+%Y%m%d_%H%M%S')|${JOB_IDS[@]}" >> slurm_jobs/job_history.txt
