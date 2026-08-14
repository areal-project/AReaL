#!/bin/bash
set -euo pipefail
ROOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld_holdout_qwen72b_picktwo_3arm_20260730/alfworld
STATE=/storage/openpsi/users/yl/agent-memory/MemRL/scripts/.monitor_state/picktwo_to_s10
mkdir -p "$STATE"
LOG="$STATE/monitor.log"
exec >>"$LOG" 2>&1
printf '[%s] monitor start\n' "$(date -Is)"
patterns=(
 'exp_alfworld_holdout_picktwo_qwen72b_memrl_traj_20260731_101105_*/local_cache/snapshot/4/local_cache/cum_state.json'
 'exp_alfworld_holdout_picktwo_qwen72b_selfrag_traj_20260731_101105_*/local_cache/snapshot/4/local_cache/cum_state.json'
 'exp_alfworld_holdout_picktwo_qwen72b_regionfs_traj_20260731_101105_*/local_cache/snapshot/4/local_cache/cum_state.json'
)
while true; do
  ok=1
  for pat in "${patterns[@]}"; do
    compgen -G "$ROOT/$pat" >/dev/null || ok=0
  done
  if ((ok)); then break; fi
  sleep 300
done
# Wait for the original shared job to stop writing, avoiding concurrent writes
# to the same destinations. Require 10 minutes with no size/mtime change.
DRIVER=/storage/openpsi/users/yl/agent-memory/MemRL/logs/alfworld_holdout_picktwo_qwen72b_3arm_20260731_101105/driver.log
prev=''
stable=0
while ((stable < 2)); do
  cur=$(stat -c '%Y:%s' "$DRIVER" 2>/dev/null || echo missing)
  if [[ "$cur" == "$prev" ]]; then stable=$((stable+1)); else stable=0; prev="$cur"; fi
  sleep 300
done
printf '[%s] all S4 checkpoints healthy and original log stable; submitting S5-S10\n' "$(date -Is)"
bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_qwen72b_holdout_picktwo_to_s10.sh >"$STATE/submit.out" 2>&1
touch "$STATE/submitted"
printf '[%s] submit done\n' "$(date -Is)"
