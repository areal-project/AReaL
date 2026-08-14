#!/bin/bash
# MemRL Experiment Runner Scripts
# Usage: bash scripts/run_experiments.sh [bcb|alf|hle|llb_os|llb_db|all]

set -e

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"

cd $MEMRL_DIR

run_bcb() {
    echo "=========================================="
    echo "Running BigCodeBench (gpt-4o)"
    echo "Target: Last Epoch SR 0.595 / Cumulative SR 0.627"
    echo "=========================================="

    singularity exec $SINGULARITY_IMG python run/run_bcb.py \
        --config configs/rl_bcb_config.local.yaml \
        --split instruct \
        --subset full \
        --epochs 10 \
        2>&1 | tee logs/bcb_gpt4o_$(date +%Y%m%d_%H%M%S).log
}

run_alf() {
    echo "=========================================="
    echo "Running ALFWorld (gpt-5-mini)"
    echo "Target: Last Epoch SR 0.949 / Cumulative SR 0.981"
    echo "=========================================="

    singularity exec $SINGULARITY_IMG python run/run_alfworld.py \
        --config configs/rl_alf_config.local.yaml \
        2>&1 | tee logs/alf_gpt5mini_$(date +%Y%m%d_%H%M%S).log
}

run_hle() {
    echo "=========================================="
    echo "Running HLE (gemini-3-pro-image-preview)"
    echo "Target: Last Epoch SR 0.570 / Cumulative SR 0.606"
    echo "=========================================="

    singularity exec $SINGULARITY_IMG python run/run_hle.py \
        --config configs/rl_hle_config.local.yaml \
        --train data/hle/hle_test.parquet \
        2>&1 | tee logs/hle_gemini3_$(date +%Y%m%d_%H%M%S).log
}

run_llb_os() {
    echo "=========================================="
    echo "Running LLB OS Task (gpt-4o)"
    echo "Target: Last Epoch SR 0.788 / Cumulative SR 0.804"
    echo "=========================================="

    singularity exec $SINGULARITY_IMG python run/run_llb.py \
        --config configs/rl_llb_os_config.local.yaml \
        2>&1 | tee logs/llb_os_gpt4o_$(date +%Y%m%d_%H%M%S).log
}

run_llb_db() {
    echo "=========================================="
    echo "Running LLB DB Task (gpt-4o)"
    echo "Target: Last Epoch SR 0.960 / Cumulative SR 0.972"
    echo "=========================================="

    singularity exec $SINGULARITY_IMG python run/run_llb.py \
        --config configs/rl_llb_db_config.local.yaml \
        2>&1 | tee logs/llb_db_gpt4o_$(date +%Y%m%d_%H%M%S).log
}

# Parse argument
case "${1:-all}" in
    bcb)
        run_bcb
        ;;
    alf)
        run_alf
        ;;
    hle)
        run_hle
        ;;
    llb_os)
        run_llb_os
        ;;
    llb_db)
        run_llb_db
        ;;
    all)
        run_bcb
        run_alf
        run_hle
        run_llb_os
        run_llb_db
        ;;
    *)
        echo "Usage: $0 [bcb|alf|hle|llb_os|llb_db|all]"
        exit 1
        ;;
esac

echo "Experiment completed!"
