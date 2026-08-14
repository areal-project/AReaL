#!/bin/bash
#SBATCH --job-name=yl-alf-rfs-all
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/alf_rfs_all_%j.log
#SBATCH --error=logs/alf_rfs_all_%j.log

# Region failure summary holdout: serial execution
#   Phase 1: pick_two S3 eval (resume S3 ckpt)
#   Phase 2: pick_two S4 eval (resume S4 ckpt)
#   Phase 3: look_at 4 sections from scratch

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen2.5-72B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

INNER_SCRIPT=$(mktemp /tmp/alf_rfs_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"; EMBED_MODEL_PATH="$4"; EMBED_PORT="$5"
cd "$MEMRL_DIR"
git checkout region-dev --quiet 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null; find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
rm -rf *.egg-info 2>/dev/null; pip uninstall -y memrl 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan vllm textworld alfworld --quiet 2>/dev/null
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# Prepare configs
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    "configs/rl_alf_config.qwen72b_holdout_two_region_failure_summary_4section.yaml" > "/tmp/alf_two_rfs_$$.yaml"
# S3 eval config: add ckpt_resume_epoch: 3
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    "configs/rl_alf_config.qwen72b_holdout_two_region_failure_summary_4section.yaml" | \
    sed '/ckpt_resume_enabled/a\  ckpt_resume_epoch: 3' > "/tmp/alf_two_rfs_s3_$$.yaml"
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    "configs/rl_alf_config.qwen72b_holdout_look_region_failure_summary_4section.yaml" > "/tmp/alf_look_rfs_$$.yaml"

# Start embedding server
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --port "$EMBED_PORT" --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &
EMBED_PID=$!
for i in $(seq 1 1200); do curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && break; sleep 1; done

# Start LLM server
export NCCL_ASYNC_ERROR_HANDLING=1; export NCCL_IB_TIMEOUT=22
LLM_PID_FILE=/tmp/vllm_llm_pid_$$
start_llm() {
    [ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && sleep 2
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" --served-model-name Qwen2.5-72B-Instruct \
        --tensor-parallel-size 4 --port "$LLM_PORT" --trust-remote-code \
        --max-model-len 32768 --gpu-memory-utilization 0.90 \
        --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
    echo $! > "$LLM_PID_FILE"
}
start_llm
export HF_HUB_OFFLINE=1; export TRANSFORMERS_OFFLINE=1
for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM ready!' && break
    kill -0 "$(cat $LLM_PID_FILE)" 2>/dev/null || { echo "[ERROR] LLM died"; kill "$EMBED_PID" 2>/dev/null; exit 1; }
    [ "$i" -eq 1800 ] && { echo '[ERROR] LLM timeout'; exit 1; }
    sleep 1
done

# Watchdog
WATCHDOG_KEEP_RUNNING=/tmp/watchdog_keep_$$
touch "$WATCHDOG_KEEP_RUNNING"
(
    fails=0
    while [ -f "$WATCHDOG_KEEP_RUNNING" ]; do
        if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then fails=0
        else fails=$((fails+1)); [ "$fails" -ge 3 ] && { start_llm; sleep 300; fails=0; }; fi
        sleep 60
    done
) &
WD_PID=$!

# ============================================================
# Phase 1: pick_two S3 eval (resume from S3 checkpoint)
# ============================================================
echo "=========================================="
echo "[PHASE 1/3] pick_two_obj_and_place — S3 eval-only"
echo "Start: $(date)"
echo "=========================================="
python run/run_alfworld.py \
    --config "/tmp/alf_two_rfs_s3_$$.yaml" \
    --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --val_lambda_max 0.15 \
    --holdout_subtask alf/pick_two_obj_and_place \
    --holdout_eval_pools train,valid \
    --skip_initial_eval \
    --failure_summary_n_slots 2 \
    --failure_summary_path analysis/region_failure_summaries.json
echo "[PHASE 1] exit=$? at $(date)"

# ============================================================
# Phase 2: pick_two S4 eval (resume from S4 checkpoint)
# ============================================================
echo "=========================================="
echo "[PHASE 2/3] pick_two_obj_and_place — S4 eval-only"
echo "Start: $(date)"
echo "=========================================="
python run/run_alfworld.py \
    --config "/tmp/alf_two_rfs_$$.yaml" \
    --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --val_lambda_max 0.15 \
    --holdout_subtask alf/pick_two_obj_and_place \
    --holdout_eval_pools train,valid \
    --skip_initial_eval \
    --failure_summary_n_slots 2 \
    --failure_summary_path analysis/region_failure_summaries.json
echo "[PHASE 2] exit=$? at $(date)"

# ============================================================
# Phase 3: look_at 4 sections from scratch
# ============================================================
echo "=========================================="
echo "[PHASE 3/3] look_at_obj_in_light — 4 sections from scratch"
echo "Start: $(date)"
echo "=========================================="
python run/run_alfworld.py \
    --config "/tmp/alf_look_rfs_$$.yaml" \
    --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 --val_lambda_max 0.15 \
    --holdout_subtask alf/look_at_obj_in_light \
    --holdout_eval_pools train,valid \
    --skip_initial_eval \
    --failure_summary_n_slots 2 \
    --failure_summary_path analysis/region_failure_summaries.json
echo "[PHASE 3] exit=$? at $(date)"

rm -f "$WATCHDOG_KEEP_RUNNING"
kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
[ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && rm -f "$LLM_PID_FILE"
kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null
echo "[INFO] All phases done."
INNEREOF
chmod +x "$INNER_SCRIPT"
singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage \
    $SINGULARITY_IMG bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT"
rm -f "$INNER_SCRIPT"
