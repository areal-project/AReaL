#!/bin/bash
#SBATCH --job-name=yl-bcb-verify-tf
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=slurmd-23,slurmd-24,slurmd-45
#SBATCH --output=logs/bcb_verify_tf_%j.log
#SBATCH --error=logs/bcb_verify_tf_%j.log

# Verify: 4 BCB tasks needing tf/keras
#   1. Can we import tensorflow/keras in eval subprocess after BCB_EVAL_LIBS_DIR injection?
#   2. Do the previous-generated solutions actually run (PASS or non-import error)?
#   3. Does setting PYTHONPATH not break BCB's own test code?

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
SINGULARITY_IMG="/storage/openpsi/images/areal-latest.sif"
TARGET_DIR="/storage/openpsi/users/yl/agent-memory/.cache/bcb_eval_libs"

echo "=========================================="
echo "BCB TF/Keras End-to-End Verify"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="

singularity exec --no-home --writable-tmpfs \
    --bind /storage:/storage \
    $SINGULARITY_IMG \
    bash -c "
set -e
cd ${MEMRL_DIR}
export BCB_EVAL_LIBS_DIR=${TARGET_DIR}
export TF_USE_LEGACY_KERAS=1

echo '[INFO] Python: '\$(python --version)
echo '[INFO] BCB_EVAL_LIBS_DIR: '\$BCB_EVAL_LIBS_DIR
echo ''

# Phase A: confirm subprocess injection logic works (no PYTHONPATH set globally)
echo '=== Phase A: simulate eval_utils._untrusted_check_worker injection ==='
python <<'PY'
import os, sys
# Reproduce the logic from memrl/bigcodebench_eval/eval_utils.py
libs_dir = os.environ.get('BCB_EVAL_LIBS_DIR')
print(f'libs_dir from env: {libs_dir}')
print(f'sys.path before injection (first 3): {sys.path[:3]}')
if libs_dir and os.path.isdir(libs_dir):
    if libs_dir not in sys.path:
        sys.path.insert(0, libs_dir)
    os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
print(f'sys.path after injection (first 3): {sys.path[:3]}')

# Verify numpy/scipy etc do NOT come from libs_dir (they should still be image's)
# unless explicitly loaded with libs_dir in sys.path
import numpy
print(f'numpy: {numpy.__version__} from {numpy.__file__}')
# Now import tensorflow — should pull from libs_dir
import tensorflow as tf
print(f'tensorflow: {tf.__version__} from {tf.__file__}')
import keras
print(f'keras: {keras.__version__} from {keras.__file__}')
PY

echo ''
echo '=== Phase B: run BCB official untrusted_check on previous solutions ==='
python <<'PY'
import os, sys, json
# Set up the injection as eval_utils does
libs_dir = os.environ.get('BCB_EVAL_LIBS_DIR')
if libs_dir:
    sys.path.insert(0, libs_dir)
    os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

# Load BCB framework
sys.path.insert(0, '${MEMRL_DIR}/3rdparty/bigcodebench-main')
sys.path.insert(0, '${MEMRL_DIR}')
from bigcodebench.eval import untrusted_check, PASS, FAIL, TIMEOUT
from memrl.bigcodebench_eval.task_wrappers import load_bcb_data

print('Loading BCB tasks...')
tasks = load_bcb_data(subset='full')
print(f'Loaded {len(tasks)} tasks')

# Saved solutions from previous run
with open('/storage/openpsi/users/yl/agent-memory/.cache/bcb_tf_keras_solutions.json') as f:
    solutions = json.load(f)

target_ids = ['BigCodeBench/289','BigCodeBench/417','BigCodeBench/418','BigCodeBench/419']
print()
for tid in target_ids:
    if tid not in tasks:
        print(f'  {tid}: NOT IN BCB DATASET')
        continue
    task = tasks[tid]
    sol = solutions.get(tid, '')
    if not sol:
        print(f'  {tid}: NO PREVIOUS SOLUTION')
        continue
    print(f'--- {tid} ---')
    print(f'  solution length: {len(sol)}')
    try:
        stat, details = untrusted_check(
            code=sol,
            test_code=task['test'],
            entry_point=task['entry_point'],
            max_as_limit=30 * 1024,
            max_data_limit=30 * 1024,
            max_stack_limit=10,
            min_time_limit=10,
            gt_time_limit=60,
        )
        print(f'  STATUS: {stat}')
        if details is not None:
            # details typically dict with 'ALL' key
            if isinstance(details, dict):
                for k, v in details.items():
                    txt = str(v)[:200]
                    print(f'    {k}: {txt}')
            else:
                print(f'    details: {str(details)[:200]}')
    except Exception as e:
        print(f'  EXCEPTION: {type(e).__name__}: {str(e)[:200]}')
    print()
PY

echo ''
echo '=== Phase C: confirm vLLM still works WITHOUT BCB_EVAL_LIBS_DIR injection ==='
# This is the main process behavior — no PYTHONPATH from libs_dir.
python <<'PY'
import sys
# Do NOT add libs_dir to sys.path here (simulates main process)
import numpy
print(f'main-process numpy: {numpy.__version__} from {numpy.__file__}')
# Should be image's numpy (likely 2.x), NOT libs_dir's 1.26
if 'bcb_eval_libs' in numpy.__file__:
    print('WARNING: main process picked up libs_dir numpy — isolation BROKEN!')
else:
    print('OK: main process uses image numpy, libs_dir is isolated.')
PY

echo ''
echo 'Verify complete.'
"

EXIT=$?
echo ''
echo "End: $(date)"
echo "Exit: $EXIT"
