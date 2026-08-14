#!/bin/bash
#SBATCH --job-name=yl-dsv3-serve-test
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=800G
#SBATCH --gres=gpu:8
#SBATCH --exclude=slurmd-24,slurmd-16
#SBATCH --output=logs/dsv3_serve_test_%j.log
#SBATCH --error=logs/dsv3_serve_test_%j.log

# Standalone serve-only test: DeepSeek-V3 (original instruct, 671B FP8) on vLLM 0.17.0.
# Goal: confirm the model loads and answers a chat probe BEFORE wiring the BCB runner.
# Per docs/model_use/deepseekv3.md: V3 is standard MLA (NOT V3.2 sparse attn), so
# TP8+EP is a supported config and the TP=8 head-padding warning does NOT apply.

VLLM_IMG="/storage/openpsi/images/areal-dev-vllm-20260429.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-V3"
LLM_PORT=8000
SERVED_NAME="deepseek-ai/DeepSeek-V3"

echo "=========================================="
echo "DeepSeek-V3 vLLM serve-only test (TP8+EP)"
echo "vLLM image: $VLLM_IMG"
echo "Model:      $MODEL_PATH"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

fuser -k ${LLM_PORT}/tcp 2>/dev/null || true
sleep 5

echo "[INFO] Launching vLLM server (DeepSeek-V3, TP8 + Expert Parallel)..."
singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $VLLM_IMG \
    bash -c "
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name ${SERVED_NAME} \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --port ${LLM_PORT} \
    --trust-remote-code \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
" &
VLLM_PID=$!
echo "[INFO] vLLM server PID: $VLLM_PID"

# FP8 MoE startup is slow (weights load + kernel warmup); allow up to 90 min.
echo "[INFO] Waiting for vLLM (real chat probe, up to 90min for FP8 startup)..."
READY=0
for i in $(seq 1 5400); do
    PROBE=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://localhost:${LLM_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"${SERVED_NAME}\",\"max_tokens\":4,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" 2>/dev/null)
    if [ "$PROBE" = "200" ]; then
        echo "[INFO] vLLM is ready after ${i}s (chat probe 200)!"
        READY=1
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[ERROR] vLLM process died during startup (after ${i}s)."
        exit 1
    fi
    if [ $((i % 60)) -eq 0 ]; then
        echo "[INFO]   ...still waiting (${i}s, last probe HTTP $PROBE)"
    fi
    sleep 1
done

if [ "$READY" != "1" ]; then
    echo "[ERROR] vLLM failed to become ready within 5400s."
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ============================================================
# Functional checks: a real code-generation prompt + token count.
# ============================================================
echo ''
echo '=== [TEST 1] Simple chat completion ==='
curl -s -X POST http://localhost:${LLM_PORT}/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_NAME}\",\"max_tokens\":64,\"temperature\":0.0,\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: OK\"}]}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print('content:', repr(r['choices'][0]['message']['content'])); print('usage:', r.get('usage'))" 2>&1

echo ''
echo '=== [TEST 2] BCB-style code generation prompt ==='
curl -s -X POST http://localhost:${LLM_PORT}/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_NAME}\",\"max_tokens\":512,\"temperature\":0.0,\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function task_func(nums) that returns the sum of squares of a list of integers. Return only the code in a python code block.\"}]}" \
    | python3 -c "import sys,json; r=json.load(sys.stdin); c=r['choices'][0]['message']['content']; print(c); print('---'); print('finish_reason:', r['choices'][0].get('finish_reason')); print('usage:', r.get('usage'))" 2>&1

echo ''
echo '=== [TEST 3] /v1/models endpoint ==='
curl -s http://localhost:${LLM_PORT}/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print([m['id'] for m in d.get('data',[])])" 2>&1

echo ''
echo "[INFO] Serve test complete. Stopping vLLM server."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
