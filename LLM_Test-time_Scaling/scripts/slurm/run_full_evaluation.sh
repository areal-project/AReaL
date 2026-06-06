#!/bin/bash
#SBATCH --job-name=full-eval
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000M
#SBATCH --chdir=working-dir
#SBATCH --output=logs/full_eval_%j.out
#SBATCH --error=logs/full_eval_%j.err
#SBATCH --time=24:00:00

# Complete evaluation pipeline on SLURM
# Usage: sbatch scripts/slurm/run_full_evaluation.sh <MODEL_PATH> <NUM_SERVICES> [BENCHMARKS]
# Example: sbatch scripts/slurm/run_full_evaluation.sh gpt-oss-120b 4 all

set -e

MODEL_PATH=${1:-"gpt-oss-120b"}
NUM_SERVICES=${2:-4}
BENCHMARKS=${3:-"all"}
START_PORT=${4:-8000}
OUTPUT_DIR=${5:-"results"}

echo "=========================================="
echo "Full Evaluation Pipeline"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Number of services: $NUM_SERVICES"
echo "Benchmarks: $BENCHMARKS"
echo "Start port: $START_PORT"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Activate environment if needed
# source /path/to/venv/bin/activate

# Run the full evaluation pipeline
python scripts/run_full_evaluation.py \
    --model-path "$MODEL_PATH" \
    --num-services "$NUM_SERVICES" \
    --start-port "$START_PORT" \
    --benchmarks $BENCHMARKS \
    --output-dir "$OUTPUT_DIR" \
    --timeout 600

echo "=========================================="
echo "Evaluation pipeline completed"
echo "=========================================="

