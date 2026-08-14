#!/bin/bash
#SBATCH --job-name=yl-sglang-v32-test
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:8
#SBATCH --exclude=slurmd-24,slurmd-16
#SBATCH --output=logs/sglang_v32_test_%j.log
#SBATCH --error=logs/sglang_v32_test_%j.log

IMG=/storage/openpsi/images/sglang-v0.5.10.sif
MODEL=/storage/openpsi/models/deepseek-v3.2
echo "=== SGLang V3.2 serve test | Job $SLURM_JOB_ID | Node $SLURMD_NODENAME | $(date) ==="

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
for i in \$(seq 1 2400); do
  if curl -s http://127.0.0.1:30000/health >/dev/null 2>&1; then echo '[INFO] SGLANG_READY'; break; fi
  if ! kill -0 \$SGL 2>/dev/null; then echo '[ERROR] SGLANG_DIED'; break; fi
  sleep 1
done
echo '[INFO] test chat completion (reason+answer):'
curl -s -X POST http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{\"model\":\"deepseek-v3.2\",\"max_tokens\":300,\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Reason step by step then give the answer.\"}]}'
echo ''
echo '[INFO] test done.'
kill \$SGL 2>/dev/null
"
echo "=== end $(date) ==="
