#!/bin/bash
set -euo pipefail
workspace=$1
output=$2
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
python -m pytest -q tests/test_megatron_muon_config.py tests/test_megatron_optimizer_config.py tests/test_megatron_async_save.py 2>&1 | tee "$2/pytest.log"
INNER
