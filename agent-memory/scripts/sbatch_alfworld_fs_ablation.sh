#!/bin/bash
#SBATCH --job-name=yl-alf-fs-ablation
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/alf_fs_ablation_%j.log
#SBATCH --error=logs/alf_fs_ablation_%j.log

# Failure Summary Ablation: inline top-k FS (various k) vs region-only vs region+FS
# Runs on pick_and_place holdout (most stable subtask, n=825).
# Pass ARM via --export: inline_k3 | inline_k5 | inline_k10 | inline_k20 | region_only
#   sbatch --job-name=yl-alf-fs-abl-k5 --export=ALL,ABL_ARM=inline_k5 this.sh
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen2.5-72B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

: "${ABL_ARM:?must pass ABL_ARM=inline_k3|inline_k5|inline_k10|inline_k20|global|region_only}"

# Map arm to CLI flags
case "$ABL_ARM" in
  inline_k3)   FS_FLAGS="--failure_summary_n_slots 2 --failure_summary_mode inline --failure_summary_k 3"; REGION_FLAGS="" ;;
  inline_k5)   FS_FLAGS="--failure_summary_n_slots 2 --failure_summary_mode inline --failure_summary_k 5"; REGION_FLAGS="" ;;
  inline_k10)  FS_FLAGS="--failure_summary_n_slots 2 --failure_summary_mode inline --failure_summary_k 10"; REGION_FLAGS="" ;;
  inline_k20)  FS_FLAGS="--failure_summary_n_slots 2 --failure_summary_mode inline --failure_summary_k 20"; REGION_FLAGS="" ;;
  global)      FS_FLAGS="--failure_summary_n_slots 2 --failure_summary_mode global"; REGION_FLAGS="" ;;
  region_only) FS_FLAGS="--failure_summary_n_slots 0"; REGION_FLAGS="--region --region_gating_mode additive --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --val_lambda_max 0.15" ;;
  *) echo "bad ABL_ARM=$ABL_ARM"; exit 1 ;;
esac

INNER_SCRIPT=$(mktemp /tmp/alf_abl_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"; EMBED_MODEL_PATH="$4"; EMBED_PORT="$5"; FS_FLAGS="$6"; ABL_ARM="$7"; REGION_FLAGS="$8"
cd "$MEMRL_DIR"
find . -name "*.pyc" -delete 2>/dev/null; find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan vllm textworld alfworld --quiet 2>/dev/null
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# Use the base config (pick_and_place holdout, 4 sections) and swap ports
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g; s|_fs_ablation_4section|_fs_ablation_${ABL_ARM}_4section|" \
    "configs/rl_alf_config.qwen72b_holdout_pickplace_fs_ablation_4section.yaml" > "/tmp/alf_abl_$$.yaml"

CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --port "$EMBED_PORT" --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &
EMBED_PID=$!
for i in $(seq 1 1200); do curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && break; sleep 1; done

export NCCL_ASYNC_ERROR_HANDLING=1; export NCCL_IB_TIMEOUT=22
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --served-model-name Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 --port "$LLM_PORT" --trust-remote-code \
    --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
LLM_PID=$!
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM ready!' && break
    kill -0 $LLM_PID 2>/dev/null || { echo "[ERROR] LLM died"; kill "$EMBED_PID" 2>/dev/null; exit 1; }
    [ "$i" -eq 1800 ] && { echo '[ERROR] LLM timeout'; exit 1; }
    sleep 1
done

echo "=========================================="
echo "Failure Summary Ablation: ABL_ARM=$ABL_ARM | FS_FLAGS=$FS_FLAGS"
echo "pick_and_place holdout, 4 sections"
echo "Start: $(date)"
echo "=========================================="
python run/run_alfworld.py \
    --config "/tmp/alf_abl_$$.yaml" \
    $REGION_FLAGS \
    --holdout_subtask alf/pick_and_place_simple \
    --holdout_eval_pools train,valid \
    --skip_initial_eval \
    $FS_FLAGS
echo "[fs ablation $ABL_ARM] exit=$? at $(date)"

kill $LLM_PID $EMBED_PID 2>/dev/null; wait $LLM_PID $EMBED_PID 2>/dev/null
echo "[INFO] Done."
INNEREOF
chmod +x "$INNER_SCRIPT"
singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage \
    $SINGULARITY_IMG bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT" "$FS_FLAGS" "$ABL_ARM" "$REGION_FLAGS"
rm -f "$INNER_SCRIPT"
