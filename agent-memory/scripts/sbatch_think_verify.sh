#!/bin/bash
#SBATCH --job-name=yl-think-verify
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --gres=gpu:8
#SBATCH --exclude=slurmd-24,slurmd-16
#SBATCH --output=logs/think_verify_%j.log
#SBATCH --error=logs/think_verify_%j.log

IMG=/storage/openpsi/images/sglang-v0.5.10.sif
MODEL=/storage/openpsi/models/deepseek-v3.2
echo "=== thinking verify (SDK extra_body vs curl) | $SLURM_JOB_ID | $SLURMD_NODENAME | $(date) ==="

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

pip install openai --quiet 2>/dev/null || true
python3 << 'PYEOF'
from openai import OpenAI
c=OpenAI(api_key='EMPTY', base_url='http://127.0.0.1:30000/v1/')
Q='Prove that there are infinitely many primes. Reason carefully.'

def run(tag, extra_body):
    r=c.chat.completions.create(model='deepseek/deepseek-v3.2', max_tokens=4000,
        messages=[{'role':'user','content':Q}], extra_body=extra_body)
    m=r.choices[0].message
    reasoning=getattr(m,'reasoning_content',None) or getattr(m,'reasoning',None)
    print(f'[{tag}] content_len={len(m.content or \"\")} reasoning_len={len(reasoning or \"\")} reasoning_present={bool(reasoning)}')

print('=== SDK extra_body chat_template_kwargs thinking=True (provider 方式) ===')
run('SDK_THINK_ON', {'chat_template_kwargs':{'thinking':True}})
print('=== SDK 不带 thinking ===')
run('SDK_DEFAULT', {})
PYEOF
echo VERIFY_DONE
kill \$SGL 2>/dev/null
"
echo "=== end $(date) ==="
