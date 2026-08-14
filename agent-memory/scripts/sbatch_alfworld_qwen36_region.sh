#!/bin/bash
#SBATCH --job-name=yl-alf-qwen36-region
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --gres=gpu:2
#SBATCH --nodelist=slurmd-16
#SBATCH --output=logs/alf_qwen36_region_%j.log
#SBATCH --error=logs/alf_qwen36_region_%j.log

# Qwen3.6 ALFWorld: Region+FS (10 sections)
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
VLLM_IMG="/storage/openpsi/images/areal-vllm0202-torch211.sif"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
QWEN36_PATH="/storage/openpsi/models/Qwen__Qwen3.6-35B-A3B"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"

QWEN36_PORT=8300
EMBED_PORT=8301

echo "=========================================="
echo "ALFWorld Qwen3.6: Region+FS (10 sections)"
echo "SLURM CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Job $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

# singularity --nv on this cluster mounts ALL GPUs regardless of host CUDA_VISIBLE_DEVICES.
# Container sees all 8 GPUs with physical IDs. Use SLURM-assigned IDs directly.
SLURM_GPUS=$CUDA_VISIBLE_DEVICES
GPU0=$(echo $SLURM_GPUS | cut -d, -f1)
GPU1=$(echo $SLURM_GPUS | cut -d, -f2)
echo "Using physical GPUs: LLM=$GPU0 Embed=$GPU1"

singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage \
    $VLLM_IMG bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1

CUDA_VISIBLE_DEVICES=$GPU1 python -m vllm.entrypoints.openai.api_server \
    --model $EMBED_PATH --served-model-name Qwen/Qwen3-Embedding-8B \
    --port $EMBED_PORT --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --seed 42 &

CUDA_VISIBLE_DEVICES=$GPU0 python -m vllm.entrypoints.openai.api_server \
    --model $QWEN36_PATH --served-model-name Qwen3.6-35B-A3B \
    --tensor-parallel-size 1 --port $QWEN36_PORT --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --reasoning-parser qwen3 --seed 42 &

wait
" &
VLLM_BG_PID=$!

echo "[INFO] Waiting for Embed (port $EMBED_PORT)..."
for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo "[INFO] Embed ready!" && break
    kill -0 $VLLM_BG_PID 2>/dev/null || { echo "[ERROR] vLLM died"; exit 1; }
    [ "$i" -eq 1200 ] && echo "[ERROR] Embed timeout" && kill $VLLM_BG_PID 2>/dev/null && exit 1
    sleep 1
done

echo "[INFO] Waiting for Qwen3.6 (port $QWEN36_PORT)..."
for i in $(seq 1 1800); do
    curl -s "http://localhost:${QWEN36_PORT}/health" > /dev/null 2>&1 && echo "[INFO] Qwen3.6 ready!" && break
    kill -0 $VLLM_BG_PID 2>/dev/null || { echo "[ERROR] vLLM died"; exit 1; }
    [ "$i" -eq 1800 ] && echo "[ERROR] timeout" && kill $VLLM_BG_PID 2>/dev/null && exit 1
    sleep 1
done

singularity exec --no-home --writable-tmpfs --bind /storage:/storage \
    $RUNNER_IMG bash -c "
cd $MEMRL_DIR
find . -name '*.pyc' -delete 2>/dev/null; find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan textworld alfworld --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

CFG=/tmp/alf_qwen36_region_\$\$.yaml
sed 's|localhost:8100|localhost:${QWEN36_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g' \
    configs/rl_alf_config.qwen36_memrl.yaml > \"\$CFG\"
# Enable resume from S5 complete checkpoint + 3x eval
sed -i 's|ckpt_resume_enabled: false|ckpt_resume_enabled: true|' \"\$CFG\"
sed -i 's|ckpt_resume_path: \"\"|ckpt_resume_path: \"/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_memrl_qwen36_20260627-112247/local_cache\"|' \"\$CFG\"
# Add n_eval_runs: 3 after num_sections line
sed -i '/num_sections:/a\\  n_eval_runs: 4' \"\$CFG\"
sed -i '/n_eval_runs:/a\\  eval_temperature: 0.2' \"\$CFG\"

echo '[Qwen3.6] Region+FS (10 sections)'
python run/run_alfworld.py \
    --config \"\$CFG\" \
    --region --region_gating_mode additive \
    --region_utility_mode beta \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
    --val_lambda_max 0.05 --no_z_norm \
    --explore_schedule '0,2,2,1,1,1,1,0,0,0' \
    --failure_summary_n_slots 2 \
    --skip_initial_eval
echo \"[Qwen3.6 region] exit=\$? at \$(date)\"
"

kill $VLLM_BG_PID 2>/dev/null; wait $VLLM_BG_PID 2>/dev/null
echo "End: $(date)"
