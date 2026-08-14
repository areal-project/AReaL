#!/bin/bash
#SBATCH --job-name=yl-webshop-nomem
#SBATCH --partition=all
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/webshop_nomem_%j.log
#SBATCH --error=logs/webshop_nomem_%j.log

# WebShop: no-memory baseline (sanity check + comparison baseline)
# Short run: 5 sections, no region, no failure_summary, no retrieval

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
MODEL_PATH="/storage/openpsi/models/Qwen2.5-72B-Instruct"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"

echo "=========================================="
echo "WebShop No-Memory Baseline"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

INNER_SCRIPT=$(mktemp /tmp/webshop_nomem_XXXXXX.sh)
cat > "$INNER_SCRIPT" << 'INNEREOF'
#!/bin/bash
MEMRL_DIR="$1"; MODEL_PATH="$2"; LLM_PORT="$3"; EMBED_MODEL_PATH="$4"; EMBED_PORT="$5"
cd "$MEMRL_DIR"
echo "[INFO] Starting WebShop no-mem baseline on $(hostname)"

find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
rm -rf *.egg-info 2>/dev/null || true
pip uninstall -y memrl 2>/dev/null || true
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true
# WebShop deps: Java + python packages
apt-get update -qq 2>/dev/null
apt install -y openjdk-21-jdk --fix-missing 2>&1 | tail -3
apt --fix-broken -y install 2>&1 | tail -3
pip install flask beautifulsoup4 rich gym cleantext thefuzz rank-bm25 spacy 2>&1 | tail -5
# pyserini pulls torch 2.12 which breaks vLLM; install with --no-deps
pip install pyserini --no-deps 2>&1 | tail -3
pip install pyjnius 2>&1 | tail -3
# spacy model from local whl (avoid slow download)
pip install /storage/openpsi/users/yl/LaMer/webshop_spacy_models/en_core_web_sm-3.8.0-py3-none-any.whl 2>&1 | tail -3
pip install /storage/openpsi/users/yl/LaMer/webshop_spacy_models/en_core_web_lg-3.8.0-py3-none-any.whl 2>&1 | tail -3
java -version 2>&1 || echo "[ERROR] Java still not found"
python -c "import memrl; print('[VERIFY] memrl loaded from:', memrl.__file__)"

# Sanity check critical WebShop deps before starting vLLM (fail fast)
python -c "
import gym, flask, bs4, cleantext, spacy, rank_bm25
from thefuzz import fuzz
from pyserini.search.lucene import LuceneSearcher
import sys; sys.path.insert(0, 'memrl/envs/webshop')
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
print('[VERIFY] All WebShop deps OK')
" || { echo "[ERROR] WebShop deps check failed, aborting"; exit 1; }

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

TEMP_CONFIG=/tmp/webshop_nomem_$$.yaml
sed "s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g" \
    configs/rl_webshop_config.qwen72b_region_failsummary.yaml > "$TEMP_CONFIG"

# Override experiment name for baseline
python3 - "$TEMP_CONFIG" << 'PYEOF'
import sys, yaml
cfg_path = sys.argv[1]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg["experiment"]["experiment_name"] = "webshop_nomem_baseline_qwen72b_5section"
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
print("[INJECT] experiment_name =", cfg["experiment"]["experiment_name"])
PYEOF

# Embedding server (still needed for memory service init, even if unused)
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

# LLM server
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

# Watchdog
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

# Run NO-MEMORY baseline (no --region, no --failure_summary_n_slots)
python run/run_webshop.py \
    --config "$TEMP_CONFIG" \
    --skip_initial_eval &
MAIN_PID=$!
wait "$MAIN_PID"; MAIN_EXIT=$?

rm -f "$WATCHDOG_KEEP"; wait "$WD_PID" 2>/dev/null
[ -f "$LLM_PID_FILE" ] && kill "$(cat $LLM_PID_FILE)" 2>/dev/null && rm -f "$LLM_PID_FILE"
kill "$EMBED_PID" 2>/dev/null; wait "$EMBED_PID" 2>/dev/null
echo "[DONE] exit=$MAIN_EXIT"
INNEREOF

chmod +x "$INNER_SCRIPT"
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash "$INNER_SCRIPT" "$MEMRL_DIR" "$MODEL_PATH" "$LLM_PORT" "$EMBED_MODEL_PATH" "$EMBED_PORT"

EXIT_CODE=$?
rm -f "$INNER_SCRIPT"
echo "End: $(date)  Exit: $EXIT_CODE"
exit $EXIT_CODE
