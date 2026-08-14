#!/bin/bash
#SBATCH --job-name=yl-alf-main-failsum
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/alf_main_failsum_%j.log
#SBATCH --error=logs/alf_main_failsum_%j.log

# Main exp: region failure summary (dynamic, no external file)
# Config B: vlmax=0.05 + --no_z_norm + soft explore + failure_summary_n_slots=2
# Baseline comparison: 928122 S8 OOD 54.48% (vlmax=0.15, znorm ON, no failure summary)

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/Qwen2.5-72B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "ALFWorld Main Failure Summary (vlmax=0.05 + nozn + soft + failure_summary=2)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/alf_main_failsum_XXXXXX.sh)
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

TEMP_CONFIG=/tmp/alf_main_failsum_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_alf_config.qwen72b_region_10section_softexplore_vlmax005_nozn.yaml > "$TEMP_CONFIG"

CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model "$EMBED_MODEL_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --port "$EMBED_PORT" --max-model-len 8192 --gpu-memory-utilization 0.30 \
    --trust-remote-code --disable-log-requests --seed 42 &
EMBED_PID=$!
for i in $(seq 1 1200); do
    curl -s "http://localhost:${EMBED_PORT}/health" > /dev/null 2>&1 && echo '[INFO] Embedding vLLM ready' && break
    [ "$i" -eq 1200 ] && echo '[ERROR] Embedding vLLM failed' && kill $EMBED_PID 2>/dev/null && exit 1
    sleep 1
done

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
LLM_PID_FILE=/tmp/vllm_llm_pid_$$
start_llm() {
    if [ -f "$LLM_PID_FILE" ]; then
        local old_pid=$(cat "$LLM_PID_FILE")
        kill -0 "$old_pid" 2>/dev/null && kill "$old_pid" 2>/dev/null && sleep 2
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
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM vLLM ready' && break
    if ! kill -0 "$(cat $LLM_PID_FILE)" 2>/dev/null; then
        echo "[ERROR] LLM died"; kill "$EMBED_PID" 2>/dev/null; exit 1
    fi
    [ "$i" -eq 1800 ] && echo '[ERROR] LLM timeout' && kill "$(cat $LLM_PID_FILE)" "$EMBED_PID" 2>/dev/null && exit 1
    sleep 1
done

WATCHDOG_KEEP=/tmp/wd_keep_$$; touch "$WATCHDOG_KEEP"
(
    fails=0
    while [ -f "$WATCHDOG_KEEP" ]; do
        if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then fails=0
        else
            fails=$((fails+1))
            [ "$fails" -ge 3 ] && start_llm && for w in $(seq 1 300); do curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && fails=0 && break; sleep 1; done
        fi
        sleep 60
    done
) &
WD_PID=$!

python run/run_alfworld.py \
    --config "$TEMP_CONFIG" \
    --region --region_gating_mode additive \
    --shrinkage_confidence_k 3.0 --propagation_eta 0.12 \
    --val_lambda_max 0.05 \
    --no_z_norm \
    --explore_schedule "0,2,2,1,1,1,1,0,0,0" \
    --skip_initial_eval \
    --failure_summary_n_slots 2 &
MAIN_PID=$!
wait "$MAIN_PID"; MAIN_EXIT=$?

rm -f "$WATCHDOG_KEEP"
kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
[ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && rm -f "$LLM_PID_FILE"
kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null
echo "[DONE] exit=$MAIN_EXIT"
INNEREOF

chmod +x "$INNER_SCRIPT"
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT"
rm -f "$INNER_SCRIPT"
echo "End: $(date)  Exit: $?"
