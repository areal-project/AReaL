#!/bin/bash
set -euo pipefail
workspace=$1
output=$2
mkdir -p "$output"
squeue -j 972320 > "$output/allocation.txt"
srun --mpi=none --jobid=972320 --overlap -N1 -n1 \
  apptainer exec --nv --bind /storage:/storage \
  /storage/openpsi/images/areal-dev-sglang-0821.sif \
  python3 - "$workspace" > "$output/environment.txt" 2>&1 <<'PY'
import importlib.metadata as m
import os
import sys
print('python', sys.executable, sys.version)
for name in ['torch', 'megatron-core', 'emerging-optimizers', 'sglang', 'transformers', 'mbridge', 'pre-commit']:
    try:
        print(name, m.version(name), m.distribution(name).locate_file(''))
    except m.PackageNotFoundError:
        print(name, 'MISSING')
print('workspace_visible', os.path.isdir(sys.argv[1]))
import torch
print('cuda', torch.cuda.is_available(), torch.cuda.device_count())
PY
cat "$output/environment.txt"
