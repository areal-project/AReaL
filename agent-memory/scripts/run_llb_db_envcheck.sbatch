#!/bin/bash
#SBATCH --job-name=yl-llb-db-envcheck
#SBATCH --partition=all
#SBATCH --nodelist=slurmd-3
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/llb_db_envcheck_%j.log
#SBATCH --error=logs/llb_db_envcheck_%j.log

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "=========================================="
echo "LLB DB Bench — Environment Compatibility Check"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p ${PROJECT_DIR}/logs

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMAGE \
    bash -c "
export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH
cd ${PROJECT_DIR}

echo '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed'

echo '[INFO] Installing Python dependencies...'
pip install mysql-connector-python --quiet 2>/dev/null

echo '[INFO] Running environment compatibility check...'
python scripts/debug_db_env_compat.py

echo '[INFO] Done!'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
