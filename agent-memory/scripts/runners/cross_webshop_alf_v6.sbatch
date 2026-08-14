#!/bin/bash
#SBATCH --job-name=cross_webshop_alf_v6
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_webshop_alf_v6_%j.out
#SBATCH --error=/storage/openpsi/users/yl/agent-memory/MemRL/logs/cross_webshop_alf_v6_%j.err

set -ex

echo "[INFO] Starting job on $(hostname)"
cd /storage/openpsi/users/yl/agent-memory/MemRL

singularity exec --nv \
    --bind /storage:/storage \
    /storage/openpsi/images/areal-latest.sif \
    bash << 'INNER_SCRIPT'
set -ex
cd /storage/openpsi/users/yl/agent-memory/MemRL

export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0

# Install dependencies
pip install 'litellm[proxy]' -q 2>&1 | tail -3

# Start LiteLLM
LITELLM_PORT=$((10000 + RANDOM % 50000))
cp /storage/openpsi/users/yl/.claude/config.yaml /tmp/litellm_webshop.yaml

echo "[INFO] Starting LiteLLM on port ${LITELLM_PORT}..."
python3 -m litellm.proxy.proxy_cli --config /tmp/litellm_webshop.yaml --port ${LITELLM_PORT} --host 127.0.0.1 > /tmp/litellm_webshop.log 2>&1 &
LITELLM_PID=$!

# Wait for LiteLLM
LITELLM_URL="http://127.0.0.1:${LITELLM_PORT}"
for i in $(seq 1 60); do
    if curl -s ${LITELLM_URL}/health > /dev/null 2>&1; then
        echo "[INFO] LiteLLM ready at ${LITELLM_URL}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "[ERROR] LiteLLM timeout"
        cat /tmp/litellm_webshop.log
        exit 1
    fi
    sleep 2
done

# Run WebShop training directly with exec to avoid file read issues
echo "[INFO] Starting WebShop training..."
python3 -c "
import sys
sys.path.insert(0, '/storage/openpsi/users/yl/agent-memory/MemRL')

import os
import time
import tempfile
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from memrl.providers.llm import OpenAILLM
from memrl.providers.embedding import OpenAIEmbedder
from memrl.agent.memp_agent import MempAgent
from memrl.run.webshop_rl_runner import WebShopRunner

project_root = Path('/storage/openpsi/users/yl/agent-memory/MemRL')
out_dir = project_root / 'results'
run_id = time.strftime('%Y%m%d-%H%M%S')
log_dir = out_dir / 'webshop' / f'exp_cross_webshop_alf_v6_{run_id}' / 'local_cache'
log_dir.mkdir(parents=True, exist_ok=True)

# Initialize LLM
llm = OpenAILLM(
    api_key='sk-placeholder',
    base_url='${LITELLM_URL}',
    model='gpt-4o-2024-11-20',
    default_temperature=0.0,
    default_max_tokens=2048,
    token_log_dir=str(log_dir),
)

# Initialize Embedder
embedder = OpenAIEmbedder(
    api_key='sk-placeholder',
    base_url='${LITELLM_URL}',
    model='text-embedding-3-small',
    max_text_len=6000,
    token_log_dir=str(log_dir),
)

# Initialize Agent
agent = MempAgent(llm_provider=llm, few_shot_examples=[])

# Initialize Runner (without memory service for simple test)
runner = WebShopRunner(
    agent=agent,
    root=project_root,
    memory_service=None,
    exp_name='cross_webshop_alf_v6',
    num_sections=5,
    batch_size=5,
    max_steps=15,
    rl_config=None,
    ck_dir=out_dir / 'webshop' / f'exp_cross_webshop_alf_v6_{run_id}',
    retrieve_k=10,
    mode='train',
    dataset_ratio=1.0,
    random_seed=42,
)

# Run
runner.run()
logger.info('WebShop training completed')
"

echo "[INFO] Stopping LiteLLM..."
kill ${LITELLM_PID} 2>/dev/null || true

echo "[INFO] WebShop→ALF experiment completed"
INNER_SCRIPT
