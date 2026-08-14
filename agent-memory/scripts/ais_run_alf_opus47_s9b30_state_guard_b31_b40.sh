#!/bin/bash
export MEMRL_ALFWORLD_STATE_GUARD_PROMPT=1
export MEMRL_ALFWORLD_PROGRAM_GUIDE=0
export MEMRL_ALFWORLD_DEFERRED_REPAIR=1
export MEMRL_ALFWORLD_STOP_AFTER_BATCH=40
exec bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/ais_run_alf_opus47_s9b20_sharpen3_cap128.sh
