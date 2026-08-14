#!/usr/bin/env bash
# Wait for a fully persisted s10_b60, stop only the verified Opus workflow,
# then submit the trajectory resume with FS=1 and LLM concurrency=16.
set -u

ROOT=/storage/openpsi/users/yl/agent-memory/MemRL
OLD_RECORD_ID=324432334
OLD_LOG="$ROOT/logs/aistudio_alf_opus47_region_20260718_230127.log"
EXP_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld/alfworld/exp_alfworld_region_opus47_20260623-100806
SNAPSHOT="$EXP_DIR/local_cache/snapshot/s10_b60"
SUBMIT="$ROOT/scripts/ais_submit_alf_opus47_region_traj_resume.sh"
STATE_DIR="$ROOT/scripts/.monitor_state/alf_opus47_b60_fs1_c16"
MONITOR_LOG="$STATE_DIR/monitor.log"
INTERVAL="${OPUS_B60_MONITOR_INTERVAL:-60}"
PERSIST_WAIT="${OPUS_B60_PERSIST_WAIT:-120}"
STOP_WAIT="${OPUS_B60_STOP_WAIT:-30}"

mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/monitor.lock"
if ! flock -n 9; then
  echo "another monitor already owns $STATE_DIR/monitor.lock" >&2
  exit 0
fi
exec >>"$MONITOR_LOG" 2>&1

echo "[$(date -Is)] monitor started pid=$$ old_record_id=$OLD_RECORD_ID snapshot=$SNAPSHOT"

snapshot_healthy() {
  SNAPSHOT="$SNAPSHOT" python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['SNAPSHOT'])
required = [
    p/'snapshot_meta.json',
    p/'cube'/'textual_memory.json',
    p/'local_cache'/'q_cache.json',
    p/'local_cache'/'region_manager.json',
]
if any(not x.is_file() or x.stat().st_size <= 0 for x in required):
    raise SystemExit(1)
try:
    meta = json.loads((p/'snapshot_meta.json').read_text())
except Exception:
    raise SystemExit(1)
if str(meta.get('checkpoint_id')) != 's10_b60':
    raise SystemExit(1)
q = p/'qdrant'
if not q.is_dir() or not any(x.is_file() and x.stat().st_size > 0 for x in q.rglob('*')):
    raise SystemExit(1)
PY
}

snapshot_fingerprint() {
  SNAPSHOT="$SNAPSHOT" python - <<'PY'
import os
from pathlib import Path
p=Path(os.environ['SNAPSHOT'])
files=[x for x in p.rglob('*') if x.is_file()]
print(len(files), sum(x.stat().st_size for x in files), max((x.stat().st_mtime_ns for x in files), default=0))
PY
}

stop_old_workflow() {
  [[ -f "$STATE_DIR/cancel.ok" ]] && return 0
  OLD_RECORD_ID="$OLD_RECORD_ID" python - <<'PY'
import os, sys, time
from aistudio_common.rest import job
rid=os.environ['OLD_RECORD_ID']
terminal={'success','succeeded','failed','failure','stopped','killed','cancelled','canceled','terminated'}
status=job.query_job_status(rid)
normalized=str(status or '').strip().lower()
print(f'before_stop record_id={rid} status={status}')
if not normalized:
    print('status unknown; refusing to stop')
    raise SystemExit(2)
if normalized not in terminal:
    response=job.stop_workflow(rid)
    print(f'stop_requested record_id={rid} response={response}')
for _ in range(20):
    time.sleep(5)
    status=job.query_job_status(rid)
    normalized=str(status or '').strip().lower()
    print(f'stop_poll record_id={rid} status={status}')
    if normalized in terminal:
        raise SystemExit(0)
print('workflow did not reach a terminal state in time')
raise SystemExit(3)
PY
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    printf '%s old_record_id=%s stop_confirmed\n' "$(date -Is)" "$OLD_RECORD_ID" > "$STATE_DIR/cancel.ok"
    return 0
  fi
  return "$rc"
}

submit_resume_once() {
  [[ -f "$STATE_DIR/submitted.ok" ]] && return 0
  local out="$STATE_DIR/submit.log" rid
  : > "$out"
  echo "[$(date -Is)] submitting trajectory resume with FS=1 concurrency=16"
  if bash "$SUBMIT" >>"$out" 2>&1; then
    rid=$(grep -aoE '(record id=|record_id[=: ]+)[0-9]+' "$out" | tail -1 | grep -aoE '[0-9]+' || true)
    if [[ ! "$rid" =~ ^[0-9]+$ ]]; then
      echo "[$(date -Is)] submit exited 0 but no record ID was captured; refusing duplicate auto-submit"
      printf '%s submit_succeeded_record_id_uncaptured\n' "$(date -Is)" > "$STATE_DIR/submitted.uncaptured"
      return 2
    fi
    printf '%s\n' "$rid" > "$STATE_DIR/new_record_id"
    printf '%s new_record_id=%s fs_slots=1 concurrency=16 resume_snapshot=s10_b60\n' "$(date -Is)" "$rid" > "$STATE_DIR/submitted.ok"
    echo "[$(date -Is)] resume submitted record_id=$rid"
    return 0
  fi
  echo "[$(date -Is)] resume submission failed; will retry"
  return 1
}

while true; do
  if ! snapshot_healthy; then
    progress=$(grep -aoE 'Processing mini-batch [0-9]+/112 in section 10|mini-batch [0-9]+/112: success=[0-9]+/32' "$OLD_LOG" 2>/dev/null | tail -1 || true)
    echo "[$(date -Is)] waiting for healthy s10_b60; ${progress:-no-progress-marker}"
    sleep "$INTERVAL"
    continue
  fi

  fp_before=$(snapshot_fingerprint) || { sleep "$INTERVAL"; continue; }
  echo "[$(date -Is)] healthy s10_b60 detected fingerprint=$fp_before; waiting ${PERSIST_WAIT}s"
  sleep "$PERSIST_WAIT"
  if ! snapshot_healthy; then
    echo "[$(date -Is)] health revalidation failed; returning to polling"
    sleep "$INTERVAL"
    continue
  fi
  fp_after=$(snapshot_fingerprint) || { sleep "$INTERVAL"; continue; }
  if [[ "$fp_before" != "$fp_after" ]]; then
    echo "[$(date -Is)] snapshot still changing before='$fp_before' after='$fp_after'; returning to polling"
    sleep "$INTERVAL"
    continue
  fi
  printf '%s snapshot=%s fingerprint=%s\n' "$(date -Is)" "$SNAPSHOT" "$fp_after" > "$STATE_DIR/snapshot.stable"

  if ! stop_old_workflow; then
    echo "[$(date -Is)] old workflow stop/status check failed; retrying in ${STOP_WAIT}s"
    sleep "$STOP_WAIT"
    continue
  fi
  if [[ -f "$STATE_DIR/submitted.uncaptured" && ! -f "$STATE_DIR/submitted.ok" ]]; then
    echo "[$(date -Is)] prior submit may have succeeded but record ID was not captured; manual check required; exiting"
    exit 2
  fi
  submit_resume_once || { rc=$?; [[ $rc -eq 2 ]] && exit 2; sleep "$INTERVAL"; continue; }
  echo "[$(date -Is)] b60 switch completed; monitor exiting"
  exit 0
done
