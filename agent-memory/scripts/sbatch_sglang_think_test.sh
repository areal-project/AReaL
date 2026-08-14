#!/bin/bash
#SBATCH --job-name=yl-sglang-think-test
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:8
#SBATCH --exclude=slurmd-24,slurmd-16
#SBATCH --output=logs/sglang_think_test_%j.log
#SBATCH --error=logs/sglang_think_test_%j.log

IMG=/storage/openpsi/images/sglang-v0.5.10.sif
MODEL=/storage/openpsi/models/deepseek-v3.2
echo "=== SGLang V3.2 thinking test | Job $SLURM_JOB_ID | $SLURMD_NODENAME | $(date) ==="

singularity exec --nv --no-home --writable-tmpfs --bind /storage:/storage $IMG bash -c "
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m sglang.launch_server --model-path $MODEL --served-model-name deepseek/deepseek-v3.2 \
  --tp 8 --trust-remote-code --host 127.0.0.1 --port 30000 --context-length 65536 \
  --reasoning-parser deepseek-v3 --enforce-disable-flashinfer-allreduce-fusion &
SGL=\$!
for i in \$(seq 1 5400); do
  curl -s http://127.0.0.1:30000/health >/dev/null 2>&1 && { echo SGLANG_READY; break; }
  kill -0 \$SGL 2>/dev/null || { echo SGLANG_DIED; exit 1; }
  sleep 1
done

echo '=== TEST A: chat_template_kwargs thinking=true ==='
curl -s -X POST http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"deepseek/deepseek-v3.2\",\"max_tokens\":500,\"chat_template_kwargs\":{\"thinking\":true},\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23?\"}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); u=d.get(\"usage\",{}); print(\"reasoning_tokens:\", u.get(\"completion_tokens_details\",{}).get(\"reasoning_tokens\")); m=d[\"choices\"][0][\"message\"]; print(\"has reasoning field:\", bool(m.get(\"reasoning_content\") or m.get(\"reasoning\"))); print(\"content head:\", (m.get(\"content\") or \"\")[:80])'

echo '=== TEST B: 不带 thinking (默认) ==='
curl -s -X POST http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"deepseek/deepseek-v3.2\",\"max_tokens\":500,\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23?\"}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); u=d.get(\"usage\",{}); print(\"reasoning_tokens:\", u.get(\"completion_tokens_details\",{}).get(\"reasoning_tokens\"))'

echo '=== tokenizer.chat_template 是否 None (决定 use_dpsk_v32_encoding) ==='
python3 -c \"from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('$MODEL',trust_remote_code=True); print('chat_template is None:', t.chat_template is None)\" 2>&1 | tail -1

kill \$SGL 2>/dev/null
echo TEST_DONE
"
echo "=== end $(date) ==="
