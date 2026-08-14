#!/bin/bash
#SBATCH --job-name=yl-alf-quota-abl
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/alf_quota_ablation_%j.log
#SBATCH --error=logs/alf_quota_ablation_%j.log

# v5 Quota Rerank ablation — 3 cells, eval-only on 928122 S7 snapshot.
# All 3 cells share: vlmax=0.05 + --no_z_norm + additive + shrinkage_k=3.0
# (= best config from 929089/929158 Phase 1)
# Only variable: --region_retrieve_mode
#
# Cell A: global         (baseline = current behavior, should match 929089 vlmax=0.05 OOD 55.22%)
# Cell B: quota_fixed    (strict quota=3, sim_floor=0.4)
# Cell C: quota_adaptive (quota=3 + gates: sim_floor=0.5, utility_margin=0.15)
#
# Expected runtime: ~3.5h total (30min vLLM warmup + ~50min per cell × 3)
# Decision: if C ≥ B > A on OOD with apply_rate >50%, ship to train run

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen2.5-72B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "ALFWorld QUOTA RERANK ABLATION (v5 §14)"
echo "3 cells on 928122 S7 snapshot:"
echo "  A. global         (baseline, should match 929089 vlmax=0.05)"
echo "  B. quota_fixed    (quota_max=3, sim_floor=0.4)"
echo "  C. quota_adaptive (quota_max=3, sim_floor=0.5, util_margin=0.15)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/alf_quota_abl_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash

MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"; EMBED_MODEL_PATH="$4"; EMBED_PORT="$5"
cd "$MEMRL_DIR"
echo "[INFO] Starting on $(hostname)"

git checkout region-dev --quiet 2>/dev/null || true
# git pull disabled: always use local working-tree code

find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true
pip install textworld alfworld --quiet 2>/dev/null || true

python -c "import memrl; print('[VERIFY] memrl loaded from:', memrl.__file__)"

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

# --- Start embedding vLLM on GPU 4 ---
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --port "$EMBED_PORT" --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &
EMBED_PID=$!

for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo '[INFO] Embedding vLLM is ready!' && break
    [ "$i" -eq 1200 ] && echo '[ERROR] Embedding vLLM failed' && kill $EMBED_PID 2>/dev/null && exit 1
    sleep 1
done

# --- Start LLM vLLM (Qwen2.5-72B-Instruct, TP=4, GPUs 0-3) ---
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
LLM_PID_FILE=/tmp/vllm_llm_pid_$$

start_llm() {
    if [ -f "$LLM_PID_FILE" ]; then
        local old_pid=$(cat "$LLM_PID_FILE")
        if kill -0 "$old_pid" 2>/dev/null; then
            kill "$old_pid" 2>/dev/null || true
            sleep 2
        fi
    fi
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" --served-model-name Qwen2.5-72B-Instruct \
        --tensor-parallel-size 4 --port "$LLM_PORT" --trust-remote-code \
        --max-model-len 32768 --gpu-memory-utilization 0.90 \
        --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
    echo $! > "$LLM_PID_FILE"
}

start_llm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for i in $(seq 1 1800); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM vLLM server is ready!' && break
    if ! kill -0 "$(cat $LLM_PID_FILE)" 2>/dev/null; then
        echo "[ERROR] LLM vLLM process died. Aborting."
        kill "$EMBED_PID" 2>/dev/null
        exit 1
    fi
    if [ "$i" -eq 1800 ]; then
        echo '[ERROR] LLM vLLM failed within 1800s.'
        kill "$(cat $LLM_PID_FILE)" 2>/dev/null
        kill "$EMBED_PID" 2>/dev/null
        exit 1
    fi
    sleep 1
done

WATCHDOG_KEEP_RUNNING=/tmp/watchdog_keep_$$
touch "$WATCHDOG_KEEP_RUNNING"
(
    fails=0
    while [ -f "$WATCHDOG_KEEP_RUNNING" ]; do
        if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
            fails=0
        else
            fails=$((fails+1))
            if [ "$fails" -ge 3 ]; then
                start_llm
                for w in $(seq 1 300); do
                    if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
                        fails=0; break
                    fi
                    sleep 1
                done
            fi
        fi
        sleep 60
    done
) &
WD_PID=$!

# ============ 3-cell quota ablation ============
# Common base: rl_alf_config.qwen72b_region_abl_vlmax0.05_wq0.5.yaml
# (vlmax=0.05, wq=0.5, eval-only on 928122 S7 snapshot)
# Common CLI flags (set once, reused in each cell)
BASE_CONFIG="configs/rl_alf_config.qwen72b_region_abl_vlmax0.05_wq0.5.yaml"
COMMON_FLAGS="--region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
    --val_lambda_max 0.05 --no_z_norm"

run_cell() {
    local name="$1"; shift
    local extra_flags="$@"
    local temp_cfg=/tmp/alf_quota_${name}_$$.yaml
    sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g; s|alfworld_region_abl_vlmax0.05_wq0.5|alfworld_quota_${name}|g" \
        "$BASE_CONFIG" > "$temp_cfg"
    echo ""
    echo "=========================================="
    echo "[QUOTA-CELL] ${name}: ${extra_flags}"
    echo "=========================================="
    python run/run_alfworld.py \
        --config "$temp_cfg" \
        $COMMON_FLAGS \
        $extra_flags
    echo "[INFO] Cell ${name} done."
}

# Cell A: global (baseline, no quota)
run_cell "A_global" "--region_retrieve_mode global"

# Cell B: quota_fixed (strict quota=3, sim_floor=0.4)
run_cell "B_fixed" "--region_retrieve_mode quota_fixed --quota_max 3 --quota_min_sim_floor 0.4"

# Cell C: quota_adaptive (quota=3, sim_floor=0.5, util_margin=0.15)
run_cell "C_adaptive" "--region_retrieve_mode quota_adaptive --quota_max 3 --quota_min_sim_floor 0.5 --quota_utility_margin 0.15"

# Cleanup
rm -f "$WATCHDOG_KEEP_RUNNING"
kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
[ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && rm -f "$LLM_PID_FILE"
kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null
echo '[INFO] All 3 quota cells complete.'
INNEREOF

chmod +x "$INNER_SCRIPT"

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT"

rm -f "$INNER_SCRIPT"

echo "End time: $(date)"
echo "Exit code: $?"
