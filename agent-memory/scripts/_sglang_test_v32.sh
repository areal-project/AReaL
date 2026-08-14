#!/bin/bash
IMG=/storage/openpsi/images/sglang-v0.5.10.sif
MODEL=/storage/openpsi/models/deepseek-v3.2
singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage $IMG bash -c "
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo '[INFO] launching sglang server (tp=8) for V3.2...'
python -m sglang.launch_server \
  --model-path $MODEL \
  --served-model-name deepseek-v3.2 \
  --tp 8 \
  --trust-remote-code \
  --host 127.0.0.1 --port 30000 \
  --context-length 65536 &
SGL=\$!
# 等就绪 (最多 25 min, 看 health)
for i in \$(seq 1 1500); do
  if curl -s http://127.0.0.1:30000/health >/dev/null 2>&1; then echo '[INFO] SGLANG_READY'; break; fi
  if ! kill -0 \$SGL 2>/dev/null; then echo '[ERROR] SGLANG_DIED'; break; fi
  sleep 1
done
# 真实请求测试
echo '[INFO] test chat completion:'
curl -s -X POST http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"deepseek-v3.2\",\"max_tokens\":200,\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? reason briefly.\"}]}' | head -c 600
kill \$SGL 2>/dev/null
"
