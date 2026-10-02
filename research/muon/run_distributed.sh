#!/bin/bash
set -euo pipefail
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
export AREAL_MUON_TEST_MODEL=/storage/openpsi/models/Qwen__Qwen2.5-1.5B-Instruct
python -m pytest -q -s tests/test_megatron_muon_distributed.py --basetemp="$2/distributed" 2>&1 | tee "$2/distributed.log"
INNER
