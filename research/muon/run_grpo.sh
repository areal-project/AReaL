#!/bin/bash
set -euo pipefail
umask 077
workspace=$1
output=$2
squeue -j 972320 > "$output/allocation.txt"
srun --mpi=none --jobid=972320 --overlap -N1 -n1 \
  apptainer exec --nv --bind /storage:/storage \
  /storage/openpsi/images/areal-dev-sglang-0821.sif \
  bash -s -- "$workspace" "$output" <<'INNER'
set -euo pipefail
cd "$1"
env_dir=$(python3 -c 'import json; print(json.load(open("research/muon/local_environment.json"))["venv"])')
source "$env_dir/bin/activate"
source research/muon/container_environment.sh
export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}"
# The image's loopback proxy is unavailable on the allocated GPU node.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
ca_bundle=$(python -c 'import certifi; print(certifi.where())')
export SSL_CERT_FILE="$ca_bundle" REQUESTS_CA_BUNDLE="$ca_bundle" CURL_CA_BUNDLE="$ca_bundle"
export AREAL_MUON_MODEL=/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct
export AREAL_MUON_DATA=/storage/openpsi/data/gsm8k
export AREAL_MUON_OUTPUT="$2/grpo"
export AREAL_MUON_ADMIN_KEY
AREAL_MUON_ADMIN_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')
python -c 'import sys, importlib.metadata as m; print(sys.executable, sys.version); print({n: m.version(n) for n in ["torch", "megatron-core", "emerging-optimizers", "sglang", "transformers"]})' | tee "$2/environment.log"
python examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo_megatron_muon.yaml 2>&1 | tee "$2/grpo.log"
python research/muon/summarize_grpo.py "$2" "$2/summary.json"
INNER
