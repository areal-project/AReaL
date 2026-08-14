#!/bin/bash
#SBATCH --job-name=yl-bcb-oracle-ceil
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:5
#SBATCH --output=logs/bcb_oracle_ceiling_5gpu_%j.log
#SBATCH --error=logs/bcb_oracle_ceiling_5gpu_%j.log

# 3-mode oracle ceiling experiment for System holdout transfer diagnostic.
# Reuses the memory snapshot from 927459 epoch 2 (the failed run that gave
# 34.1% with current retrieval). Runs no_mem / current / oracle on the same
# 82 System holdout tasks and same DeepSeek-R1-32B endpoint for fair comparison.

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
MODEL_PATH="/storage/openpsi/models/deepseek-ai__DeepSeek-R1-Distill-Qwen-32B"
EMBED_MODEL_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
LLM_PORT=$((8000 + ($SLURM_JOB_ID % 100) * 2))
EMBED_PORT=$((LLM_PORT + 1))

# Snapshot from 927459 epoch 2: contains 1490 trained memories
SNAPSHOT_ROOT="${MEMRL_DIR}/results/deepseek_holdout_system_5gpu/bigcodebench_eval/instruct_full/region/20260601_231704_deepseek-ai_DeepSeek-R1-Distill-Qwen-32B_region/epoch2/snapshot"
ORACLE_POOL_DIR="${SNAPSHOT_ROOT}/2"  # cube/textual_memory.json + local_cache

echo "=========================================="
echo "BCB Oracle Ceiling Experiment (3 modes)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Snapshot: $SNAPSHOT_ROOT"
echo "Start time: $(date)"
echo "=========================================="

singularity exec --nv --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
echo '[INFO] Starting on \$(hostname)'

git checkout region-dev --quiet 2>/dev/null || true
echo \"[INFO] Current branch: \$(git branch --show-current)\"

pip install -e . --quiet 2>/dev/null || true
pip install memoryos memos mem0ai 'chonkie==1.2.1' tensorboard hdbscan --quiet 2>/dev/null || true
pip install vllm --quiet 2>/dev/null || true

pip install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt --quiet 2>/dev/null || true
pip install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt --quiet 2>&1 | tail -5 || true
pip install faker statsmodels django openpyxl python-docx xlwt sendgrid scikit-image pyquery \
    geopandas geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays folium pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein natsort librosa Flask-WTF flask-restful --quiet 2>&1 | tail -3 || true

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface

TEMP_CONFIG=/tmp/oracle_config_\$\$.yaml
sed \"s|localhost:8000|localhost:${LLM_PORT}|g; s|localhost:8001|localhost:${EMBED_PORT}|g\" configs/rl_bcb_config.deepseek_old_region_bs8.yaml > \$TEMP_CONFIG

echo '[INFO] Starting embedding vLLM on GPU 4...'
CUDA_VISIBLE_DEVICES=4 python -m vllm.entrypoints.openai.api_server \
    --model ${EMBED_MODEL_PATH} \
    --served-model-name Qwen/Qwen3-Embedding-8B \
    --port ${EMBED_PORT} \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.30 \
    --trust-remote-code \
    --disable-log-requests \
    --seed 42 \
    &
EMBED_PID=\$!

for i in \$(seq 1 1200); do
    curl -s http://localhost:${EMBED_PORT}/health > /dev/null 2>&1 && { echo '[INFO] Embed vLLM ready'; break; }
    [ \$i -eq 1200 ] && { echo '[ERROR] Embed timeout'; kill \$EMBED_PID 2>/dev/null; exit 1; }
    sleep 1
done

echo '[INFO] Starting LLM vLLM on GPUs 0-3 (TP=4)...'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --served-model-name deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 4 \
    --port ${LLM_PORT} \
    --trust-remote-code \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.80 \
    --disable-log-requests \
    --seed 42 \
    --disable-frontend-multiprocessing \
    &
LLM_PID=\$!

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for i in \$(seq 1 900); do
    curl -s http://localhost:${LLM_PORT}/health > /dev/null 2>&1 && { echo '[INFO] LLM vLLM ready'; break; }
    [ \$i -eq 900 ] && { echo '[ERROR] LLM timeout'; kill \$EMBED_PID \$LLM_PID 2>/dev/null; exit 1; }
    sleep 1
done

# Run all 3 retrieval modes sequentially on the same vLLM
OUT_BASE=./results/deepseek_holdout_system_oracle_ceiling
COMMON_FLAGS=\"--config \$TEMP_CONFIG --split instruct --subset full --eval_timeout 240 --untrusted_hard_timeout 300 --strip_think --holdout_subtask bcb/System --task_cluster_k 0 --region_gating_mode additive --region_utility_mode beta --region_temperature 0.3 --shrinkage_top_n 1 --region_min_cluster_size 15 --epochs 1 --eval_only --resume_checkpoint_path ${SNAPSHOT_ROOT}\"

for MODE in no_mem current oracle; do
    echo ''
    echo '=========================================='
    echo \"[ORACLE-CEILING] Running mode=\${MODE}\"
    echo '=========================================='

    EXTRA_FLAGS=\"\"
    if [ \"\${MODE}\" = \"oracle\" ]; then
        EXTRA_FLAGS=\"--oracle_snapshot_dir ${ORACLE_POOL_DIR}\"
    fi

    python run/run_bcb_region.py \$COMMON_FLAGS \\
        --retrieval_mode \${MODE} \\
        --output_dir \${OUT_BASE}_\${MODE} \\
        \${EXTRA_FLAGS}

    EXIT_CODE=\$?
    echo \"[INFO] Mode=\${MODE} exited with \$EXIT_CODE\"
done

kill \$EMBED_PID \$LLM_PID 2>/dev/null
wait \$EMBED_PID 2>/dev/null
wait \$LLM_PID 2>/dev/null
echo '[INFO] vLLM stopped.'
"

echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $?"
echo "=========================================="
