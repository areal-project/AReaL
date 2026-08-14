#!/bin/bash
#SBATCH --job-name=yl-hle-region-fs
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --gres=gpu:8
#SBATCH --exclude=slurmd-24,slurmd-16
#SBATCH --output=logs/hle_region_fs_%j.log
#SBATCH --error=logs/hle_region_fs_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SGLANG_IMG="/storage/openpsi/images/sglang-v0.5.10.sif"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-v3.2"
LLM_PORT=30000

echo "=========================================="
echo "HLE Region + Failure Summary: deepseek-v3.2 via SGLang"
echo "SGLang image: $SGLANG_IMG (TP=8, native DSA/NSA)"
echo "Runner image: $RUNNER_IMG"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

# Free port if a stale server from a prior cancelled job holds it
fuser -k ${LLM_PORT}/tcp 2>/dev/null || true
sleep 5

# ============================================================
# 1) Start SGLang server (native DeepSeek-V3.2 DSA support), background.
# ============================================================
echo "[INFO] Launching SGLang server (image: $SGLANG_IMG)..."
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
    --mem-fraction-static 0.85 \
    --enforce-disable-flashinfer-allreduce-fusion
" &
VLLM_PID=$!
echo "[INFO] SGLang server PID: $VLLM_PID"

# ============================================================
# 2) Wait for the server with a REAL chat probe (not just /health).
# ============================================================
echo "[INFO] Waiting for SGLang server to be ready (real chat probe)..."
for i in $(seq 1 5400); do
    PROBE=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://localhost:${LLM_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"deepseek/deepseek-v3.2","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
    if [ "$PROBE" = "200" ]; then
        echo "[INFO] SGLang server is ready (chat probe 200)!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[ERROR] SGLang server process died during startup."
        exit 1
    fi
    if [ $i -eq 5400 ]; then
        echo "[ERROR] SGLang server failed to start after 5400s (last probe: $PROBE)"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

# ============================================================
# 3) Run the Region+FS experiment in the RUNNER image (mem packages here).
# ============================================================
echo ''
echo '[HLE-REGION-FS] Starting HLE Region + Failure Summary (deepseek-v3.2, 10 sections)'
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $RUNNER_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Installing runner dependencies...'
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python run/run_hle_region.py \
    --config configs/rl_hle_config.region_local.yaml \
    --train data/hle/hle_test.parquet \
    --text_only \
    --judge_model gpt-4o-2024-11-20 \
    --judge_base_url https://matrixllm.alipay.com/v1/ \
    --judge_api_key sk-43dd5f664179406d92fec42a9364f8a5 \
    --region_gating_mode additive \
    --region_retrieve_mode global \
    --k_global 30 \
    --k_local 10 \
    --shrinkage_top_n 3 \
    --shrinkage_lambda_max 0.6 \
    --region_utility_mode ema \
    --region_smoothing_C 0.5 \
    --region_cluster_init_step 300 \
    --region_merge_interval 200 \
    --explore_schedule '0,4,3,2,2,1,1,1,1,0' \
    --failure_summary_n_slots 2
"
HLE_EXIT=$?
echo "[INFO] HLE Region+FS exited with code: $HLE_EXIT"

# Cleanup SGLang server
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "[INFO] SGLang server stopped. Region+FS exit: $HLE_EXIT"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
