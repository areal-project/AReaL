#!/bin/bash
# Submit 4 experiments to 2 machines (2 experiments per machine)
# Each experiment uses 4 GPUs

echo "=========================================="
echo "Submitting 4 Old Config Experiments"
echo "Time: $(date)"
echo "=========================================="

# Machine 1: GPU 0-3 and GPU 4-7
# Experiment 1: Old Baseline (bs=1)
# Experiment 2: Old Region Post (bs=1)

echo ""
echo "[1/4] Submitting Old Config Baseline (bs=1)..."
JOB1=$(sbatch --parsable scripts/sbatch_bcb_deepseek_old_4gpu.sh)
echo "Job ID: $JOB1"
echo "Log: logs/bcb_ds_old_4gpu_${JOB1}.log"

echo ""
echo "[2/4] Submitting Old Config + Region Post (bs=1)..."
JOB2=$(sbatch --parsable scripts/sbatch_bcb_deepseek_old_region_post_4gpu.sh)
echo "Job ID: $JOB2"
echo "Log: logs/bcb_ds_old_region_post_4gpu_${JOB2}.log"

# Machine 2: GPU 0-3 and GPU 4-7
# Experiment 3: Old Region Additive OLD (bs=1)
# Experiment 4: Old Region Additive NEW (bs=8)

echo ""
echo "[3/4] Submitting Old Config + Additive OLD (bs=1)..."
JOB3=$(sbatch --parsable scripts/sbatch_bcb_deepseek_old_region_additive_old_4gpu.sh)
echo "Job ID: $JOB3"
echo "Log: logs/bcb_ds_old_region_additive_old_4gpu_${JOB3}.log"

echo ""
echo "[4/4] Submitting Old Config + Additive NEW (bs=8)..."
JOB4=$(sbatch --parsable scripts/sbatch_bcb_deepseek_old_region_additive_new_b8_4gpu.sh)
echo "Job ID: $JOB4"
echo "Log: logs/bcb_ds_old_region_additive_new_b8_4gpu_${JOB4}.log"

echo ""
echo "=========================================="
echo "All 4 experiments submitted!"
echo "=========================================="
echo ""
echo "Job IDs:"
echo "  Baseline:        $JOB1"
echo "  Region Post:     $JOB2"
echo "  Additive OLD:    $JOB3"
echo "  Additive NEW b8: $JOB4"
echo ""
echo "Monitor with:"
echo "  squeue | grep yl"
echo ""
echo "Check progress:"
echo "  tail -f logs/bcb_ds_old_4gpu_${JOB1}.log | grep epoch"
echo "  tail -f logs/bcb_ds_old_region_post_4gpu_${JOB2}.log | grep epoch"
echo "  tail -f logs/bcb_ds_old_region_additive_old_4gpu_${JOB3}.log | grep epoch"
echo "  tail -f logs/bcb_ds_old_region_additive_new_b8_4gpu_${JOB4}.log | grep epoch"
echo ""
echo "Extract results:"
echo "  grep -E '(798/798|342/342) pass=' logs/bcb_ds_old_4gpu_${JOB1}.log | grep -oE 'epoch [0-9]+ (train|val) (798/798|342/342) pass=[0-9]+' | sort -t' ' -k2n -k3"
echo ""
