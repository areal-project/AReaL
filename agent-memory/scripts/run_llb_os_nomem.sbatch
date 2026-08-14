#!/bin/bash
#SBATCH --job-name=yl-llb-os-nomem
#SBATCH --partition=all
#SBATCH --nodelist=slurmd-16
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/llb_os_nomem_%j.log
#SBATCH --error=logs/llb_os_nomem_%j.log

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/areal-latest.sif

echo "=========================================="
echo "MemRL - LLB OS No-Memory Eval"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p ${PROJECT_DIR}/logs

singularity exec \
    --writable-tmpfs \
    --bind /storage/openpsi/users/yl:/storage/openpsi/users/yl \
    --bind /storage/openpsi/experiments:/storage/openpsi/experiments \
    $SINGULARITY_IMAGE \
    bash -c "
export PYTHONUSERBASE=/storage/openpsi/users/yl/agent-memory/.local
export PATH=\$PYTHONUSERBASE/bin:\$PATH
export PYTHONPATH=${PROJECT_DIR}:\$PYTHONPATH
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MEMRL_LLB_REQUEST_INTERVAL=1.0
export MEMRL_LLM_MIN_INTERVAL=0.5

cd ${PROJECT_DIR}

pip install docker mysql-connector-python memoryos chonkie==1.2.1 qdrant-client --quiet 2>/dev/null || true
yum install -y mariadb-server --quiet 2>/dev/null || true

echo 'Running LLB OS no-memory eval...'
python run/run_llb.py --config configs/rl_llb_os_nomem.yaml
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
