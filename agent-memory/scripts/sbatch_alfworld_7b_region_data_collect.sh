#!/bin/bash
#SBATCH --job-name=yl-alf-7b-region-dc
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=slurmd-16
#SBATCH --output=logs/alf_7b_region_dc_%j.log
#SBATCH --error=logs/alf_7b_region_dc_%j.log

# Qwen2.5-7B region+FS data collection (1 section, full train 3553 games).
# Collects trajectories for downstream GRPO training.
# LLM: local vLLM 7B on 1 GPU (port 8100).
# Embedding: REUSES existing Qwen3-Embedding-8B on localhost:8177 (from pick_heat job).
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen__Qwen2.5-7B-Instruct"
LLM_PORT=8100

INNER_SCRIPT=$(mktemp /tmp/alf_7b_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"
cd "$MEMRL_DIR"
find . -name "*.pyc" -delete 2>/dev/null; find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan vllm textworld alfworld --quiet 2>/dev/null
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# Start 7B LLM (1 GPU, pick the free one — CUDA_VISIBLE_DEVICES auto from gres)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name Qwen2.5-7B-Instruct \
    --port "$LLM_PORT" --trust-remote-code \
    --max-model-len 8192 --gpu-memory-utilization 0.90 \
    --disable-log-requests --seed 42 &
LLM_PID=$!

export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
for i in $(seq 1 600); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] 7B LLM ready!' && break
    kill -0 $LLM_PID 2>/dev/null || { echo "[ERROR] 7B LLM died"; exit 1; }
    [ "$i" -eq 600 ] && { echo '[ERROR] 7B LLM timeout'; exit 1; }
    sleep 1
done

# Verify embedding (already running from pick_heat job)
curl -s "http://localhost:8177/health" > /dev/null 2>&1 && echo '[INFO] Embedding (8177) reachable' || echo '[WARN] Embedding not reachable!'

echo "=========================================="
echo "Qwen2.5-7B region+FS data collection (1 section, 3553 train)"
echo "Start: $(date)"
echo "=========================================="
python run/run_alfworld.py \
    --config configs/rl_alf_config.qwen7b_region_data_collect.yaml \
    --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --val_lambda_max 0.15 \
    --skip_initial_eval \
    --failure_summary_n_slots 2
echo "[7b region data collect] exit=$? at $(date)"

kill $LLM_PID 2>/dev/null; wait $LLM_PID 2>/dev/null
echo "[INFO] Done."
INNEREOF
chmod +x "$INNER_SCRIPT"
singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage \
    $SINGULARITY_IMG bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT"
rm -f "$INNER_SCRIPT"
