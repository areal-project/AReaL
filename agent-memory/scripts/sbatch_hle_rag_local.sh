#!/bin/bash
#SBATCH --job-name=yl-hle-rag
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=1200G
#SBATCH --gres=gpu:8
#SBATCH --output=logs/hle_rag_%j.log
#SBATCH --error=logs/hle_rag_%j.log

# HLE RAG baseline: raw trajectory storage + pure sim retrieval (no Q, no proceduralization)
# 10 sections, standard semantic retrieval

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SGLANG_IMG="/storage/openpsi/images/sglang-v0.5.10.sif"
RUNNER_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-v3.2"
LLM_PORT=30000

echo "=========================================="
echo "HLE RAG Baseline: deepseek-v3.2 via SGLang"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"
echo "=========================================="

fuser -k ${LLM_PORT}/tcp 2>/dev/null || true
sleep 5

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
VLLM_PID=$!

for i in $(seq 1 5400); do
    PROBE=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://localhost:${LLM_PORT}/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"deepseek/deepseek-v3.2","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)
    if [ "$PROBE" = "200" ]; then echo "[INFO] SGLang ready!"; break; fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[ERROR] SGLang died."; exit 1; fi
    if [ $i -eq 5400 ]; then echo "[ERROR] SGLang timeout."; kill $VLLM_PID 2>/dev/null; exit 1; fi
    sleep 1
done

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $RUNNER_IMG \
    bash -c "
cd ${MEMRL_DIR}
pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard --quiet 2>/dev/null || true
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MEMRL_EMBED_MIN_INTERVAL=0.5
python run/run_hle.py \
    --config configs/rl_hle_config.rag_local.yaml \
    --train data/hle/hle_test.parquet \
    --text_only \
    --judge_model gpt-4o-2024-11-20 \
    --judge_base_url https://matrixllm.alipay.com/v1/ \
    --judge_api_key sk-43dd5f664179406d92fec42a9364f8a5
"
EXIT_CODE=$?

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
echo "[INFO] Done. Exit: $EXIT_CODE | End: $(date)"
