#!/bin/bash
# 极简镜像探测: 确认某镜像里 sglang / vllm / torch / GPU 是否齐备.
# 用法(登录节点):
#   KM_IMAGE=acr-sh-...:dev-vllm-20260429 bash MemRL/scripts/probe_image.sh
# 结果写到 MemRL/logs/probe_image_<ts>.log, 登录节点可 tail -f.
set -uo pipefail

echo "=========================================="
echo "IMAGE PROBE | Start: $(date)"
echo "Image: ${KM_IMAGE:-<default>}"
echo "=========================================="

echo "[DEBUG] which python: $(which python 2>/dev/null || echo none)"
echo "[DEBUG] python version: $(python --version 2>&1 || echo none)"

echo "--- sglang ---"
python -c "import sglang; print('[OK] sglang', sglang.__version__)" 2>&1 | head -5
echo "--- vllm ---"
python -c "import vllm; print('[OK] vllm', vllm.__version__)" 2>&1 | head -5
echo "--- torch + cuda ---"
python -c "import torch; print('[OK] torch', torch.__version__, 'cuda', torch.version.cuda)" 2>&1 | head -5

echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=index,name,memory.total --format=csv 2>&1 | head -10 || echo "nvidia-smi failed"

echo "--- pip show (versions) ---"
/usr/local/bin/pip show sglang vllm 2>/dev/null | grep -E "^Name|^Version|^Location" || true

echo "[DONE] $(date)"
