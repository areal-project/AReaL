#!/bin/bash
set -euo pipefail

workspace=$1
output=$2
job_id=$3
venv_dir=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["venv"])' "$workspace/research/muon/local_environment.json")
image=${AREAL_MUON_IMAGE:-/storage/openpsi/images/areal-dev-sglang-0821.sif}
mkdir -p "$output"
squeue -j "$job_id" > "$output/allocation.txt"

if [ "$#" -gt 3 ]; then
  optimizers=("${@:4}")
else
  optimizers=(adamw muon)
fi
for optimizer in "${optimizers[@]}"; do
  run_dir="$output/$optimizer"
  mkdir -p "$run_dir"
  srun --mpi=none --jobid="$job_id" --overlap -N1 -n1 \
    apptainer exec --nv --bind /storage:/storage "$image" \
    bash -s -- "$workspace" "$run_dir" "$venv_dir" "$optimizer" <<'INNER'
set -euo pipefail
cd "$1"
source "$3/bin/activate"
source research/muon/container_environment.sh
export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
ca_bundle=$(python -c 'import certifi; print(certifi.where())')
export SSL_CERT_FILE="$ca_bundle" REQUESTS_CA_BUNDLE="$ca_bundle" CURL_CA_BUNDLE="$ca_bundle"
export AREAL_MUON_ADMIN_KEY
AREAL_MUON_ADMIN_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
model=/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct
data=/storage/openpsi/data/gsm8k
args=(
  --config examples/math/gsm8k_grpo_megatron.yaml
  "experiment_name=gsm8k-grpo-megatron-$4-full3"
  scheduler.type=local
  total_train_epochs=3
  "cluster.fileroot=$2/results"
  "actor.path=$model"
  "train_dataset.path=$data"
  "valid_dataset.path=$data"
  '+rollout.agent.admin_api_key=${oc.env:AREAL_MUON_ADMIN_KEY}'
)
if [ "$4" = muon ]; then
  args+=(actor.optimizer.type=muon ++actor.megatron.ddp.use_distributed_optimizer=false)
fi
printf -v shell_command '%q ' python examples/math/gsm8k_rl.py "${args[@]}"
printf '%s\n' "${shell_command% }" > "$2/command.txt"
python -c 'import sys,importlib.metadata as m; print(sys.executable,sys.version); print({n:m.version(n) for n in ["torch","megatron-core","emerging-optimizers","sglang","transformers"]})' > "$2/environment.txt"
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 1 > "$2/gpu-memory.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
date -u +%Y-%m-%dT%H:%M:%SZ > "$2/start.utc"
set +e
python examples/math/gsm8k_rl.py "${args[@]}" > "$2/grpo.log" 2>&1
status=$?
set -e
date -u +%Y-%m-%dT%H:%M:%SZ > "$2/end.utc"
echo "$status" > "$2/exit-code.txt"
exit "$status"
INNER
done
