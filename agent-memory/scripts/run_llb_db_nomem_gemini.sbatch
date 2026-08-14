#!/bin/bash
#SBATCH --job-name=yl-llb-db-nomem-gemini
#SBATCH --partition=all
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/llb_db_nomem_gemini_%j.log
#SBATCH --error=logs/llb_db_nomem_gemini_%j.log

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "=========================================="
echo "MemRL - LLB DB Bench (No Memory Baseline, gemini-2.5-flash)"
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
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed'

echo '[INFO] Installing Python dependencies...'
pip install -e . --quiet || echo 'Warning: pip install -e . failed'
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard docker mysql-connector-python --quiet || echo 'Warning: pip install deps failed'

echo 'Python: ' && which python
echo 'mysqld: ' && (which mysqld || which mariadbd || find /usr/sbin -name mysqld -o -name mariadbd 2>/dev/null | head -1)

echo '[INFO] Running LLB DB No-Memory Baseline (gemini-2.5-flash)...'
python run/run_llb.py --config configs/rl_llb_db_nomem_gemini.yaml

echo '[INFO] Done!'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
