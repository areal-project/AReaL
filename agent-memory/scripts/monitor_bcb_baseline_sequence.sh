#!/bin/bash
# Sequence: current MemRL completion -> current no-memory/pass@10 -> final Region+FS.
set -euo pipefail
MEMRL=/storage/openpsi/users/yl/agent-memory/MemRL
OUT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench
STATE=$MEMRL/scripts/.monitor_state/current_bcb_sequence_20260808
mkdir -p "$STATE"
log(){ echo "[$(date -Iseconds)] $*" | tee -a "$STATE/monitor.log"; }
MEMRL_ROOT=$OUT/deepseek_v3_local_memrl_current_e2_e10
while ! find "$MEMRL_ROOT" -type f -path '*/epoch10/val/metrics.json' -print -quit | grep -q .; do
  log 'waiting for MemRL E10 val'; sleep 300
done
log 'MemRL E10 val found; submitting no-memory E1 + Pass@10'
if [ ! -f "$STATE/passk_submitted" ]; then
  (cd "$MEMRL" && bash scripts/submit_bcb_current_nomem_passk10.sh) 2>&1 | tee -a "$STATE/passk_submit.log"
  date -Iseconds > "$STATE/passk_submitted"
fi
PASSK_STATE=$OUT/current_nomem_passk10_20260807_state
while [ ! -f "$PASSK_STATE/passk10.done" ]; do
  log 'waiting for Pass@10 completion'; sleep 300
done
log 'Pass@10 done; submitting final Region+FS'
if [ ! -f "$STATE/region_submitted" ]; then
  (cd "$MEMRL" && bash scripts/submit_bcb_region_fs_final_stable.sh) 2>&1 | tee -a "$STATE/region_submit.log"
  date -Iseconds > "$STATE/region_submitted"
fi
log 'sequence submission complete'
