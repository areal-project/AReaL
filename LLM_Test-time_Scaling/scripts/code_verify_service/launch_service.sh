#!/bin/bash
# Script to launch code verify service on SLURM cluster
# This script is designed to run INSIDE a singularity container via srun
# Usage (from host):
#   srun ... singularity exec ... bash code_verify_service/launch_service.sh [PORT] [DATA_DIR] [SERVICE_REGISTRY]
# 
# Or directly (if already in container):
#   bash launch_service.sh [PORT] [DATA_DIR] [SERVICE_REGISTRY]

set -e

PORT=${1:-8000}
DATA_DIR=${2:-"/storage/openpsi/users/yl/AReaL/LLM_Test-time_Scaling/data/lcb_testcases"}
SERVICE_REGISTRY=${3:-"services.txt"}

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
PROJECT_ROOT="/storage/openpsi/users/yl/AReaL/LLM_Test-time_Scaling"

echo "=========================================="
echo "Code Verify Service Launch"
echo "=========================================="
echo "Node: $HOSTNAME"
echo "Host IP: $HOST_IP"
echo "Port: $PORT"
echo "Data Directory: $DATA_DIR"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Record service info to registry file (if provided)
if [ -n "$SERVICE_REGISTRY" ]; then
    # Create registry file if it doesn't exist
    if [ ! -f "$SERVICE_REGISTRY" ]; then
        echo "# Service Registry - $(date)" > "$SERVICE_REGISTRY"
        echo "# Format: JOB_ID|HOSTNAME|HOST_IP|PORT|DATA_DIR|START_TIME|STATUS|END_TIME|EXIT_CODE" >> "$SERVICE_REGISTRY"
        echo "# =====================================" >> "$SERVICE_REGISTRY"
    fi
    
    # Use SLURM_JOB_ID if available, otherwise use a placeholder
    JOB_ID=${SLURM_JOB_ID:-"$(date +%s)"}
    echo "$JOB_ID|$HOSTNAME|$HOST_IP|$PORT|$DATA_DIR|$TIMESTAMP|running" >> "$SERVICE_REGISTRY"
    echo "Service registered to: $SERVICE_REGISTRY"
fi

# Launch service (script is already running inside singularity container via srun)
echo "Launching Code Verify Service..."

cd "$PROJECT_ROOT"
export DATA_DIR="$DATA_DIR"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PYTHONPATH"

# Install dependencies if requirements.txt exists
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    # Try pip3 first, fallback to pip
    if command -v pip3 &> /dev/null; then
        pip3 install -q -r "$PROJECT_ROOT/requirements.txt" || {
            echo "Warning: Some dependencies may have failed to install, continuing anyway..."
        }
    elif command -v pip &> /dev/null; then
        pip install -q -r "$PROJECT_ROOT/requirements.txt" || {
            echo "Warning: Some dependencies may have failed to install, continuing anyway..."
        }
    else
        echo "Warning: pip/pip3 not found, skipping dependency installation"
    fi
fi

echo "Starting Code Verify Service on $HOST_IP:$PORT..."
python3 -m scripts.code_verify_service.server \
    --host "$HOST_IP" \
    --port "$PORT" \
    --workers 1

# If service exits, update registry
EXIT_CODE=$?
TIMESTAMP_END=$(date '+%Y-%m-%d %H:%M:%S')

echo "Service stopped at $TIMESTAMP_END with exit code $EXIT_CODE"

# Update service status in registry (if provided)
if [ -n "$SERVICE_REGISTRY" ] && [ -f "$SERVICE_REGISTRY" ]; then
    if [ -n "$SLURM_JOB_ID" ]; then
        JOB_ID="$SLURM_JOB_ID"
    else
        # Try to find the job ID from the registry
        JOB_ID=$(grep "$HOSTNAME|$HOST_IP|$PORT" "$SERVICE_REGISTRY" | tail -1 | cut -d'|' -f1)
    fi
    if [ -n "$JOB_ID" ]; then
        sed -i "s/^$JOB_ID|.*|running$/$JOB_ID|$HOSTNAME|$HOST_IP|$PORT|$DATA_DIR|$TIMESTAMP|stopped|$TIMESTAMP_END|$EXIT_CODE/" "$SERVICE_REGISTRY"
    fi
fi

exit $EXIT_CODE
