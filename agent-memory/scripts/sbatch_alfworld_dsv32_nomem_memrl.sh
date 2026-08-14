#!/bin/bash
#SBATCH --job-name=yl-alf-dsv32
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --gres=gpu:8
#SBATCH --exclude=slurmd-24,slurmd-16
#SBATCH --output=logs/alf_dsv32_%j.log
#SBATCH --error=logs/alf_dsv32_%j.log

# ALFWorld no-mem + MemRL via local DeepSeek-V3.2 (SGLang TP=8, thinking).
# Serving recipe: docs/model_use/sglang_serving_deepseek_v32.md.
# Dual-image: SGLang serves (no mem pkgs), runner image runs experiment.
MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SGLANG_IMG="/storage/openpsi/images/sglang-v0.5.10.sif"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-v3.2"
LLM_PORT=30000

echo "=========================================="
echo "ALFWorld no-mem + MemRL serial: deepseek-v3.2 via SGLang (TP=8, thinking)"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

fuser -k ${LLM_PORT}/tcp 2>/dev/null || true
sleep 5

# ============================================================
# 1) Start SGLang server (native DeepSeek-V3.2 DSA/NSA), background.
# ============================================================
echo "[INFO] Launching SGLang server..."
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SGLANG_IMG \
    bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --served-model-name deepseek/deepseek-v3.2 \
    --tp 8 \
    --trust-remote-code \
    --host 127.0.0.1 --port ${LLM_PORT} \
    --context-length 65536 \
    --reasoning-parser deepseek-v3 \
    --enforce-disable-flashinfer-allreduce-fusion
" &
SGLANG_PID=$!
echo "[INFO] SGLang server PID: $SGLANG_PID"

# ============================================================
# 2) Wait for server with a REAL chat probe (not just /health).
# ============================================================
echo "[INFO] Waiting for SGLang server (real chat probe, up to 90min)..."
for i in $(seq 1 5400); do
    PROBE=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://localhost:${LLM_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"deepseek/deepseek-v3.2","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
    if [ "$PROBE" = "200" ]; then
        echo "[INFO] SGLang server ready (chat probe 200)!"
        break
    fi
    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "[ERROR] SGLang server process died during startup."
        exit 1
    fi
    if [ $i -eq 5400 ]; then
        echo "[ERROR] SGLang server failed to start after 5400s (last probe: $PROBE)"
        kill $SGLANG_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# ============================================================
# 3) Run experiments in the RUNNER image (mem packages here).
# ============================================================
run_in_runner() {
    singularity exec --nv --no-home --writable-tmpfs \
        --bind /storage:/storage \
        $RUNNER_IMG \
        bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Installing runner dependencies...'
find . -name '*.pyc' -delete 2>/dev/null; find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
pip install --no-cache-dir -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan textworld alfworld --quiet 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
$1
"
}

# --- Baseline 1: no-memory (single pass) ---
echo ''
echo '[1/2] ALFWorld no-memory (dsv32)'
run_in_runner "python run/run_alfworld.py --config configs/rl_alf_config.dsv32_nomem.yaml"
NOMEM_EXIT=$?
echo "[INFO] no-mem exited: $NOMEM_EXIT"

# --- Baseline 2: MemRL (memory, no region, 10 sections) ---
echo ''
echo '[2/2] ALFWorld MemRL (dsv32, in-distribution full train, 10 sections)'
run_in_runner "python run/run_alfworld.py --config configs/rl_alf_config.dsv32_memrl.yaml"
MEMRL_EXIT=$?
echo "[INFO] MemRL exited: $MEMRL_EXIT"

kill $SGLANG_PID 2>/dev/null
wait $SGLANG_PID 2>/dev/null
echo "[INFO] SGLang stopped. no-mem: $NOMEM_EXIT | MemRL: $MEMRL_EXIT"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
