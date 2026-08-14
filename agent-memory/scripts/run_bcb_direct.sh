#!/bin/bash
# BCB Experiment Runner - Direct API call (no LiteLLM proxy)
# Usage: bash scripts/run_bcb_direct.sh

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd $MEMRL_DIR
mkdir -p logs

LOG_FILE="logs/bcb_direct_${TIMESTAMP}.log"

echo "=========================================="
echo "Submitting BCB experiment (direct API)"
echo "Log file: $LOG_FILE"
echo "=========================================="

srun --mpi=pmi2 \
    --job-name=yl-memrl-bcb \
    --ntasks=1 \
    --gres=gpu:1 \
    --chdir=$MEMRL_DIR \
    --cpus-per-task=16 \
    --mem=64G \
    --time=48:00:00 \
    singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Starting BCB on \$(hostname)'
echo '[INFO] Installing dependencies...'
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true
echo '[INFO] Starting BCB experiment...'
python run/run_bcb.py --config configs/rl_bcb_config.local.yaml --split instruct --subset full --epochs 10
echo '[INFO] BCB experiment completed!'
" 2>&1 | tee $LOG_FILE

echo "BCB experiment completed! Log saved to $LOG_FILE"
