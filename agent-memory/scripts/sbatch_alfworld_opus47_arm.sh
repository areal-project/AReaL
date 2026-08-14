#!/bin/bash
#SBATCH --job-name=yl-alf-opus47-arm
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=logs/alf_opus47_%x_%j.log
#SBATCH --error=logs/alf_opus47_%x_%j.log

# ALFWorld opus-4-7 single ARM (matrixllm API, throttled, no GPU).
# Pass ARM via --export: nomem | memrl | region
#   sbatch --job-name=opus47-memrl --export=ALL,ARM=memrl,MEMRL_LLM_MIN_INTERVAL=0.2 this.sh
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
: "${ARM:?must pass ARM=nomem|memrl|region}"

case "$ARM" in
  nomem)  CFG=rl_alf_config.opus47_nomem_trainset.yaml; EXTRA="--eval_train" ;;
  memrl)  CFG=rl_alf_config.opus47_memrl.yaml;          EXTRA="--skip_initial_eval" ;;
  region) CFG=rl_alf_config.opus47_region.yaml
          EXTRA="--region --region_gating_mode additive --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --val_lambda_max 0.15 --failure_summary_n_slots 2 --skip_initial_eval" ;;
  *) echo "bad ARM=$ARM"; exit 1 ;;
esac

echo "=========================================="
echo "ALFWorld opus-4-7 ARM=$ARM | CFG=$CFG | EXTRA=$EXTRA"
echo "Job $SLURM_JOB_ID | $(date)"
echo "=========================================="

singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $RUNNER_IMG \
    bash -c "
cd ${MEMRL_DIR}
export TMPDIR=/dev/shm/alf_opus47_\${SLURM_JOB_ID:-manual}
export TEMP=\$TMPDIR
export TMP=\$TMPDIR
mkdir -p \$TMPDIR
echo '[INFO] Installing dependencies...'
find . -name '*.pyc' -delete 2>/dev/null; find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan textworld alfworld --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export MEMRL_LLM_MIN_INTERVAL=${MEMRL_LLM_MIN_INTERVAL:-0.2}
export MEMRL_EMBED_MIN_INTERVAL=${MEMRL_EMBED_MIN_INTERVAL:-1.0}
echo \"[INFO] MEMRL_LLM_MIN_INTERVAL=\$MEMRL_LLM_MIN_INTERVAL\"
echo \"[RUN] ARM=$ARM\"
python run/run_alfworld.py --config configs/${CFG} ${EXTRA}
echo \"[opus47 $ARM] exit=\$? at \$(date)\"
"
echo "End: $(date)"
