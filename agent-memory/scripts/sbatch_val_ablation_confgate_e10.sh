#!/bin/bash
#SBATCH --job-name=yl-bcb-valablE10
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/bcb_val_abl_e10_%j.log
#SBATCH --error=logs/bcb_val_abl_e10_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

# Best checkpoint from region experiment 927120
CHECKPOINT="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench/training/deepseek_region_confgate_5gpu/bigcodebench_eval/instruct_full/region/20260603_155337_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region/epoch10/snapshot/10"

echo "=========================================="
echo "Val Ablation: Confgate E10 checkpoint (19 regions)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/val_ablation_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash

MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"; EMBED_MODEL_PATH="$4"; EMBED_PORT="$5"; CHECKPOINT="$6"
cd "$MEMRL_DIR"

# Graceful shutdown for vLLM children so cancel/preempt doesn't leave orphans.
EMBED_PID=""
LLM_PID=""
TEMP_CONFIG=""
cleanup() {
    [ -n "$LLM_PID" ]   && kill "$LLM_PID"   2>/dev/null || true
    [ -n "$EMBED_PID" ] && kill "$EMBED_PID" 2>/dev/null || true
    [ -n "$LLM_PID" ]   && wait "$LLM_PID"   2>/dev/null || true
    [ -n "$EMBED_PID" ] && wait "$EMBED_PID" 2>/dev/null || true
    [ -n "$TEMP_CONFIG" ] && rm -f "$TEMP_CONFIG" 2>/dev/null || true
}
# Disarm EXIT trap inside signal handler so cleanup only runs once on cancel/preempt.
trap 'trap - EXIT; cleanup; exit 143' SIGINT SIGTERM
trap cleanup EXIT

echo '[INFO] Installing dependencies...'
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>/dev/null || true
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -5 || true

python -c 'import memrl; print(memrl.__file__)'
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# Temp config with correct ports
TEMP_CONFIG=/tmp/val_ablation_config_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_bcb_config.deepseek_old_region_bs8.yaml > "$TEMP_CONFIG"

# --- Start embedding vLLM on GPU 4 ---
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --port "$EMBED_PORT" --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &
EMBED_PID=$!

for i in $(seq 1 600); do
    curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo '[INFO] Embedding ready' && break
    [ "$i" -eq 600 ] && echo '[ERROR] Embedding failed' && kill $EMBED_PID 2>/dev/null && exit 1
    sleep 1
done

# --- Start LLM vLLM (TP=4, GPUs 0-3) ---
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 4 --port "$LLM_PORT" --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
LLM_PID=$!

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for i in $(seq 1 900); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM ready' && break
    [ "$i" -eq 900 ] && echo '[ERROR] LLM failed' && kill $LLM_PID 2>/dev/null && exit 1
    sleep 1
done

# --- Run ablations ---
echo '=========================================='
echo "Running val ablations on checkpoint: $CHECKPOINT"
echo '=========================================='

python run/run_bcb_region.py \
    --config "$TEMP_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --val_ablations baseline,posthoc_succ,posthoc_succ_lmax05,prerank_succ,prerank_succ_lmax05,outcome_b01,outcome_b03,outcome_b05 \
    --ablation_output_dir results/val_ablation_confgate_e10 \
    --split instruct --subset full \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --strip_think \
    --region_gating_mode additive \
    --region_utility_mode beta \
    --region_temperature 0.1 \
    --shrinkage_top_n 1 \
    --region_min_cluster_size 15 \
    --region_smoothing_C 0.5 \
    --shrinkage_confidence_k 3.0 \
    --val_lambda_max 0.15

EXIT_CODE=$?
echo "[INFO] Val ablation exited with code: $EXIT_CODE"

# vLLM cleanup handled by trap; just propagate the real Python exit code.
echo '[INFO] Done.'
exit $EXIT_CODE
INNEREOF

chmod +x "$INNER_SCRIPT"

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT" "$CHECKPOINT"
RC=$?

rm -f "$INNER_SCRIPT"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $RC"
echo "=========================================="
exit $RC
