#!/usr/bin/env bash
set -u

ROOT=/storage/openpsi/users/yl/agent-memory/MemRL
MEMP_LOG="$ROOT/logs/llb_db_memp_gpt41mini_0166115_20260717_024527.log"
MEMRL_LOG="$ROOT/logs/llb_db_memrl_gpt41mini_v2reflect_0160122_20260717_024358.log"
SUBMIT="$ROOT/scripts/submit_llb_db_baselines_gpt41mini.sh"
STATE_DIR="$ROOT/scripts/.monitor_state/llb_db_rag_selfrag"
MONITOR_LOG="$STATE_DIR/monitor.log"
INTERVAL="${INTERVAL:-300}"
PERSIST_WAIT="${PERSIST_WAIT:-120}"
MEMP_RECORD_ID="${MEMP_RECORD_ID:-}"
MEMRL_RECORD_ID="${MEMRL_RECORD_ID:-}"

mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/monitor.lock"
flock -n 9 || exit 0
exec >>"$MONITOR_LOG" 2>&1

echo "[$(date -Is)] monitor started pid=$$ interval=${INTERVAL}s"

complete() {
  local log=$1
  [[ -f "$log" ]] && \
    grep -aq -- 'Section 10 complete. Total 361 trajectories collected.' "$log" && \
    grep -aq -- '--- Validation Evaluation Complete (after Section 10) ---' "$log" && \
    grep -aq -- "'checkpoint_id': 10" "$log"
}

submit_once() {
  local method=$1 marker="$STATE_DIR/${1}.submitted" out="$STATE_DIR/${1}.submit.log"
  [[ -f "$marker" ]] && return 0
  : >"$out"
  echo "[$(date -Is)] submitting $method"
  if bash "$SUBMIT" "$method" >>"$out" 2>&1; then
    local rid
    rid=$(grep -aoE 'record id=[0-9]+' "$out" | tail -1 | cut -d= -f2 || true)
    [[ -n "$rid" ]] || { echo "[$(date -Is)] $method submit exited 0 but record id missing; retrying later"; return 1; }
    printf '%s record_id=%s\n' "$(date -Is)" "$rid" >"$marker"
    echo "[$(date -Is)] $method submitted record_id=$rid"
    return 0
  fi
  echo "[$(date -Is)] $method submission failed; retrying later"
  return 1
}

cleanup_one() {
  local label=$1 rid=$2 marker="$STATE_DIR/${1}.cleanup"
  [[ -f "$marker" ]] && return 0
  if [[ -z "$rid" ]]; then
    echo "[$(date -Is)] $label record ID unavailable; cleanup pending (set ${label^^}_RECORD_ID)"
    return 1
  fi
  LABEL="$label" RID="$rid" python - <<'PY'
import os, time
from aistudio_common.rest import job
label=os.environ['LABEL']; rid=str(os.environ['RID'])
terminal={'success','succeeded','failed','failure','stopped','killed','cancelled','canceled','terminated'}
status=job.query_job_status(rid)
print(f'{label}_before record_id={rid} status={status}')
if status is None:
    raise SystemExit(2)
if str(status).lower() not in terminal:
    print(job.stop_workflow(rid))
    time.sleep(5)
    print(f'{label}_after record_id={rid} status={job.query_job_status(rid)}')
PY
  local rc=$?
  if [[ $rc -eq 0 ]]; then printf '%s record_id=%s cleanup_checked\n' "$(date -Is)" "$rid" >"$marker"; return 0; fi
  return "$rc"
}

while true; do
  if ! complete "$MEMP_LOG" || ! complete "$MEMRL_LOG"; then
    echo "[$(date -Is)] waiting: memp=$(complete "$MEMP_LOG" && echo complete || echo running) memrl=$(complete "$MEMRL_LOG" && echo complete || echo running)"
    sleep "$INTERVAL"
    continue
  fi

  echo "[$(date -Is)] both runs complete; waiting ${PERSIST_WAIT}s for final persistence"
  sleep "$PERSIST_WAIT"
  complete "$MEMP_LOG" && complete "$MEMRL_LOG" || continue

  # Submit first; cleanup only after both durable record IDs have been captured.
  submit_once rag || { sleep "$INTERVAL"; continue; }
  submit_once selfrag || { sleep "$INTERVAL"; continue; }

  a=0; b=0
  cleanup_one memp "$MEMP_RECORD_ID" || a=$?
  cleanup_one memrl "$MEMRL_RECORD_ID" || b=$?
  if [[ $a -eq 0 && $b -eq 0 ]]; then
    echo "[$(date -Is)] all requested actions complete; monitor exiting"
    exit 0
  fi
  sleep "$INTERVAL"
done
