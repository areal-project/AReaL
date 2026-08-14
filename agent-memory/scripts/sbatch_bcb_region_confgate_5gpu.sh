#!/bin/bash
#SBATCH --job-name=yl-bcb-region-confgate-add
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/bcb_region_confgate_%j.log
#SBATCH --error=logs/bcb_region_confgate_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

echo "=========================================="
echo "BCB Region: Domain Subtask + Propagation + Exploration"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "=========================================="

# Write inner script to temp file to avoid nested quoting issues
INNER_SCRIPT=$(mktemp /tmp/bcb_region_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash

MEMRL_DIR="$1"; MODEL_PATH="$2"; EMBED_MODEL_PATH="$3"; LLM_PORT="$4"; EMBED_PORT="$5"
cd "$MEMRL_DIR"
echo "[INFO] Starting on $(hostname)"

echo '[INFO] Switching to region-dev branch...'
git checkout region-dev --quiet 2>/dev/null || true
# git pull disabled: always use local working-tree code

echo '[INFO] Installing dependencies...'
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
python -c "import memrl; print('[VERIFY] memrl loaded from:', memrl.__file__)"
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true

echo '[INFO] Installing BCB tool dependencies...'
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>/dev/null || true
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -5 || true
pip install faker statsmodels django openpyxl python-docx xlwt sendgrid scikit-image pyquery \
    geopandas geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays folium pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein natsort librosa Flask-WTF flask-restful --quiet 2>&1 | tail -3 || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

TEMP_CONFIG=/tmp/region_domain_config_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" configs/rl_bcb_config.deepseek_old_region_bs8.yaml > "$TEMP_CONFIG"
echo "[INFO] Using ports: LLM=${LLM_PORT}, Embed=${EMBED_PORT}"

# --- Embedding vLLM ---
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

# --- LLM vLLM with watchdog ---
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
        --model "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --tensor-parallel-size 4 --port "$LLM_PORT" --trust-remote-code \
        --max-model-len 131072 --gpu-memory-utilization 0.80 \
        --disable-log-requests --seed 42 --disable-frontend-multiprocessing &
    echo $! > "$LLM_PID_FILE"
    echo "[WATCHDOG] Started vLLM pid=$(cat $LLM_PID_FILE) at $(date)"
}

start_llm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for i in $(seq 1 900); do
    curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1 && echo '[INFO] LLM vLLM server is ready!' && break
    [ "$i" -eq 900 ] && echo '[ERROR] LLM vLLM failed' && kill $EMBED_PID $(cat $LLM_PID_FILE) 2>/dev/null && exit 1
    sleep 1
done

# --- Run experiment ---
echo '=========================================='
echo '[REGION-CONFGATE] confidence-gated lambda + val_lmax=0.15 + eta=0.12 Domain subtask (7 types) + propagation + exploration'
echo '=========================================='

python run/run_bcb_region.py \
    --config "$TEMP_CONFIG" --split instruct --subset full --epochs 10 \
    --checkpoint_interval 50 --max_checkpoints 3 --eval_timeout 240 \
    --untrusted_hard_timeout 300 --strip_think --task_cluster_k 0 \
    --region_gating_mode additive --region_utility_mode beta \
    --region_temperature 0.1 --shrinkage_top_n 1 --region_min_cluster_size 15 \
    --region_smoothing_C 0.5 --propagation_eta 0.12 --propagation_k 30 \
    --propagation_sim_min 0.40 --explore_schedule 0,4,3,2,2,1,1,1,1,0 \
    --explore_success_ratio 0.7 --shrinkage_confidence_k 3.0 --val_lambda_max 0.15 \
    --output_dir ./results/deepseek_region_confgate_5gpu &
MAIN_PID=$!

# --- Watchdog ---
(
    fails=0
    while kill -0 "$MAIN_PID" 2>/dev/null; do
        if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
            fails=0
        else
            fails=$((fails+1))
            echo "[WATCHDOG] health check failed ($fails/3) at $(date)"
            if [ "$fails" -ge 3 ]; then
                echo "[WATCHDOG] restarting vLLM at $(date)"
                start_llm
                restarted=0
                for w in $(seq 1 300); do
                    if curl -s "http://localhost:${LLM_PORT}/health" > /dev/null 2>&1; then
                        echo "[WATCHDOG] vLLM restarted at $(date)"
                        restarted=1; break
                    fi
                    sleep 1
                done
                [ "$restarted" -eq 1 ] && fails=0 || echo "[WATCHDOG] restart FAILED at $(date)"
            fi
        fi
        sleep 60
    done
) &
WD_PID=$!

wait "$MAIN_PID"; EXIT_CODE=$?
echo "[INFO] Exited with code: $EXIT_CODE"

kill "$WD_PID" 2>/dev/null; wait "$WD_PID" 2>/dev/null
[ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && rm -f "$LLM_PID_FILE"
kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null
echo '[INFO] Done.'
INNEREOF

chmod +x "$INNER_SCRIPT"

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$EMBED_MODEL_PATH" "$LLM_PORT" "$EMBED_PORT"

rm -f "$INNER_SCRIPT"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
