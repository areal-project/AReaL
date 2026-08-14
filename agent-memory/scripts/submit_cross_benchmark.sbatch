#!/bin/bash
#SBATCH --job-name=cross_bench_exp
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sbatch_cross_bench_%j.out
#SBATCH --error=logs/sbatch_cross_bench_%j.err

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
SINGULARITY_IMAGE=/storage/openpsi/images/pytorch280-1210-v1.sif

echo "=========================================="
echo "Cross-Benchmark Memory Transfer Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Run inside singularity container
singularity exec --nv \
    --bind /storage:/storage \
    $SINGULARITY_IMAGE \
    bash -c "
        cd ${PROJECT_DIR}

        echo 'Python: ' && which python
        echo 'Python version: ' && python --version

        # Install dependencies if needed
        pip install pyyaml pydantic openai tenacity --quiet 2>/dev/null
        pip install -e . --quiet 2>/dev/null || true

        echo ''
        echo 'Running experiment...'
        echo ''

        python scripts/run_cross_benchmark_experiment.py \
            --source llb \
            --targets hle bcb \
            --api_key '${LLM_API_KEY}' \
            --base_url '${LLM_BASE_URL}' \
            --model 'gpt-4o-2024-11-20' \
            --embedding_model 'text-embedding-3-small' \
            --epochs 5 \
            --batch_size 3 \
            --llb_task os \
            --mode local \
            --name 'cross_llb_hle_bcb'
    "

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
