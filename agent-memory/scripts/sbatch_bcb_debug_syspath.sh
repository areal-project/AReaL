#!/bin/bash
#SBATCH --job-name=yl-bcb-debug-syspath
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --exclude=slurmd-23,slurmd-24,slurmd-45
#SBATCH --output=logs/bcb_debug_syspath_%j.log
#SBATCH --error=logs/bcb_debug_syspath_%j.log

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
TARGET_DIR="/storage/openpsi/users/yl/agent-memory/.cache/bcb_eval_libs"

singularity exec --no-home --writable-tmpfs --bind /storage:/storage $SINGULARITY_IMG \
    bash -c "
cd ${MEMRL_DIR}
export BCB_EVAL_LIBS_DIR=${TARGET_DIR}
python <<'PY'
import sys, os
print('=== Initial sys.path (full) ===')
for i, p in enumerate(sys.path):
    print(f'  [{i}] {p}')
print()
print('=== After injection ===')
sys.path.insert(0, '${TARGET_DIR}')
for i, p in enumerate(sys.path):
    print(f'  [{i}] {p}')
print()
print('=== Try import sklearn ===')
try:
    import sklearn
    print(f'sklearn: {sklearn.__version__} from {sklearn.__file__}')
except Exception as e:
    print(f'FAIL: {type(e).__name__}: {e}')

print()
print('=== Try import sklearn.model_selection ===')
try:
    from sklearn.model_selection import train_test_split
    print('OK')
except Exception as e:
    print(f'FAIL: {type(e).__name__}: {e}')

print()
print('=== Check what is in /usr/local/lib/python3.12/dist-packages (filtered) ===')
import os
ip = '/usr/local/lib/python3.12/dist-packages'
contents = sorted(os.listdir(ip))
for d in contents:
    if any(k in d.lower() for k in ['sklearn','scikit','scipy','numpy','tensor','torch','pandas','vllm']):
        print(f'  {d}')
print(f'TOTAL entries: {len(contents)}')
PY
"
