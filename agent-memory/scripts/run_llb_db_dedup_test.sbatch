#!/bin/bash
#SBATCH --job-name=yl-llb-db-dedup-test
#SBATCH --partition=all
#SBATCH --nodelist=slurmd-3
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/llb_db_dedup_test_%j.log
#SBATCH --error=logs/llb_db_dedup_test_%j.log

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "=========================================="
echo "MemRL LLB DB — Dedup Fix Test (3 epochs)"
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
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface
cd ${PROJECT_DIR}

echo '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1

echo '[INFO] Installing Python dependencies...'
pip install -e . --quiet 2>/dev/null
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard docker mysql-connector-python --quiet 2>/dev/null

echo '[INFO] Running MemRL LLB DB (3 epochs, dedup fix)...'
python run/run_llb.py --config configs/rl_llb_db_memrl_test3e.yaml

echo '[INFO] Done!'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
