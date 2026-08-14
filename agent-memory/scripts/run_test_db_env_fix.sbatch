#!/bin/bash
#SBATCH --job-name=yl-llb-db-envfix-test
#SBATCH --partition=all
#SBATCH --nodelist=slurmd-3
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/llb_db_envfix_test_%j.log
#SBATCH --error=logs/llb_db_envfix_test_%j.log

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "=========================================="
echo "LLB DB — Environment Fix Validation Test"
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
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=auto
cd ${PROJECT_DIR}

echo '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1

echo '[INFO] Installing Python dependencies...'
pip install mysql-connector-python docker --quiet 2>/dev/null

echo '[INFO] Running environment fix validation...'
python scripts/test_db_env_fix.py

echo '[INFO] Done!'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
