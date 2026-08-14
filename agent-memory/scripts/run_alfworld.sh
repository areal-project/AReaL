#!/bin/bash
export PYTHONUSERBASE=/storage/openpsi/users/yl/agent-memory/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=/storage/openpsi/users/yl/agent-memory/MemRL:$PYTHONPATH
export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd /storage/openpsi/users/yl/agent-memory/MemRL
python memrl/run/alfworld_rl_runner.py --config configs/rl_alf_config.local.yaml 2>&1 | tee logs/alfworld_run_$(date +%Y%m%d_%H%M%S).log
