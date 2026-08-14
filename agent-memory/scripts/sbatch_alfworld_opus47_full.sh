#!/bin/bash
#SBATCH --job-name=yl-alf-opus47
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/alf_opus47_%j.log
#SBATCH --error=logs/alf_opus47_%j.log

# ALFWorld claude-opus-4-7 (matrixllm API, throttled, no GPU).
# Phase 1: no-mem on TRAIN set (--eval_train) -> paper Table1 runtime no-mem baseline.
#          (val/valid_seen no-mem already measured = 82.86%, job 932447.)
# Phase 2: MemRL full train, 10 sections, in-distribution.
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "ALFWorld opus-4-7: no-mem(train) + MemRL | Job $SLURM_JOB_ID | $(date)"
echo "=========================================="

run_in_runner() {
    singularity exec --no-home --writable-tmpfs \
        --bind /storage:/storage \
        $RUNNER_IMG \
        bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Installing dependencies...'
find . -name '*.pyc' -delete 2>/dev/null; find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan textworld alfworld --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export MEMRL_LLM_MIN_INTERVAL=${MEMRL_LLM_MIN_INTERVAL:-0.15}
echo \"[INFO] MEMRL_LLM_MIN_INTERVAL=\$MEMRL_LLM_MIN_INTERVAL\"
$1
"
}

# --- Phase 1: no-mem on train set ---
echo ''
echo '[1/2] opus-4-7 no-mem (train set, --eval_train)'
run_in_runner "python run/run_alfworld.py --config configs/rl_alf_config.opus47_nomem_trainset.yaml --eval_train"
echo "[no-mem train] exit=$? at $(date)"

# --- Phase 2: MemRL full train, 10 sections ---
echo ''
echo '[2/2] opus-4-7 MemRL (full train, 10 sections, in-distribution)'
run_in_runner "python run/run_alfworld.py --config configs/rl_alf_config.opus47_memrl.yaml"
echo "[memrl] exit=$? at $(date)"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
