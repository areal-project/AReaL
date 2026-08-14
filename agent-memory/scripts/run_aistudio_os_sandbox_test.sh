#!/bin/bash
# Regression test: run the sandboxed LocalOSContainer end-to-end inside aistudio.
# /storage-safe (creates its own canary, verifies survival; never touches other data).
set +e
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/aistudio_os_sandbox_test_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a $LOGFILE) 2>&1
echo "=== OS sandbox regression test | $(hostname) | $(date) ==="
export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/3rdparty/LifelongAgentBench:${LOCAL_SP}:${PYTHONPATH}
# NOTE: intentionally NOT setting MEMRL_OS_SANDBOX — test verifies it defaults ON.
export MEMRL_OS_BACKEND=local
python3 ${PROJECT_DIR}/scripts/test_os_sandbox_container.py
echo "=== DONE $(date) ==="
