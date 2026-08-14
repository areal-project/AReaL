#!/usr/bin/env bash
# Wait for the LLB OS MemRL GPT-4.1-mini run to finish Section-10 validation,
# submit Pass@10 through AIStudio exactly once, then stop the completed MemRL
# workflow if AIStudio still reports it as non-terminal.
set -u

ROOT=/storage/openpsi/users/yl/agent-memory
MEMRL_LOG="$ROOT/MemRL/logs/llb_os_memrl_gpt41mini_20260716-230220.log"
SUBMIT="$ROOT/MemRL/scripts/submit_llb_os_passk_gpt41mini.sh"
STATE_DIR="$ROOT/MemRL/scripts/.monitor_state/llb_os_passk"
MONITOR_LOG="$STATE_DIR/monitor.log"
SUBMIT_LOG="$STATE_DIR/submit.log"
SUBMITTED_FILE="$STATE_DIR/submitted.ok"
PASSK_RECORD_FILE="$STATE_DIR/passk_record_id"
CANCEL_FILE="$STATE_DIR/memrl_cancel.ok"
LOCK_FILE="$STATE_DIR/monitor.lock"
INTERVAL="${LLB_OS_PASSK_MONITOR_INTERVAL:-600}"
PERSIST_WAIT="${LLB_OS_PASSK_PERSIST_WAIT:-300}"

# Verified from the original AIStudio submission/session history on 2026-07-17.
MEMRL_RECORD_ID=323780044
MEMRL_EXPECTED_JOB_NAME=yl-llbos-memrl-gpt41mini-resume-0716-0934

mkdir -p "$STATE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another monitor instance already owns $LOCK_FILE" >&2
  exit 0
fi
exec >>"$MONITOR_LOG" 2>&1

echo "[$(date -Is)] monitor started pid=$$ interval=${INTERVAL}s memrl_record_id=$MEMRL_RECORD_ID expected_job=$MEMRL_EXPECTED_JOB_NAME"

memrl_finished() {
  [[ -f "$MEMRL_LOG" ]] &&
    grep -aq -- '--- Validation Evaluation Complete (after Section 10) ---' "$MEMRL_LOG"
}

submit_passk_once() {
  if [[ -f "$SUBMITTED_FILE" ]]; then
    echo "[$(date -Is)] Pass@10 already submitted: $(cat "$SUBMITTED_FILE")"
    return 0
  fi

  echo "[$(date -Is)] submitting LLB OS Pass@10 through AIStudio"
  : > "$SUBMIT_LOG"
  local rc rid
  if bash "$SUBMIT" >>"$SUBMIT_LOG" 2>&1; then
    rid=$(grep -aoE 'record id=[0-9]+' "$SUBMIT_LOG" | tail -1 | cut -d= -f2 || true)
    if [[ ! "$rid" =~ ^[0-9]+$ ]]; then
      echo "[$(date -Is)] submit command exited 0 but no Pass@10 record ID was found; will retry"
      return 1
    fi
    printf '%s\n' "$rid" > "$PASSK_RECORD_FILE"
    printf '%s passk_record_id=%s\n' "$(date -Is)" "$rid" > "$SUBMITTED_FILE"
    echo "[$(date -Is)] Pass@10 submission succeeded record_id=$rid"
    return 0
  else
    rc=$?
    echo "[$(date -Is)] Pass@10 submission failed rc=$rc; will retry after ${INTERVAL}s (see $SUBMIT_LOG)"
    return "$rc"
  fi
}

cancel_memrl_if_needed() {
  if [[ -f "$CANCEL_FILE" ]]; then
    echo "[$(date -Is)] MemRL workflow cleanup already recorded: $(cat "$CANCEL_FILE")"
    return 0
  fi

  MEMRL_RECORD_ID="$MEMRL_RECORD_ID" python - <<'PY'
import os
import sys
import time
from aistudio_common.rest import job

rid = os.environ["MEMRL_RECORD_ID"]
terminal = {"success", "succeeded", "failed", "failure", "stopped", "killed", "cancelled", "canceled", "terminated"}
status = job.query_job_status(rid)
normalized = str(status or "").strip().lower()
print(f"memrl_before record_id={rid} status={status}")
if normalized in terminal:
    print(f"memrl_terminal record_id={rid} status={status}; no stop needed")
    sys.exit(0)
if not normalized:
    print(f"memrl_status_unknown record_id={rid}; refusing to stop until status can be verified")
    sys.exit(2)

response = job.stop_workflow(rid)
print(f"memrl_stop_requested record_id={rid} response={response}")
time.sleep(5)
after = job.query_job_status(rid)
print(f"memrl_after record_id={rid} status={after}")
sys.exit(0)
PY
  local rc=$?
  if [[ "$rc" == 0 ]]; then
    printf '%s memrl_record_id=%s cleanup_checked\n' "$(date -Is)" "$MEMRL_RECORD_ID" > "$CANCEL_FILE"
    echo "[$(date -Is)] MemRL workflow cleanup completed/confirmed"
    return 0
  fi
  echo "[$(date -Is)] MemRL workflow cleanup deferred rc=$rc; will retry"
  return "$rc"
}

while true; do
  if ! memrl_finished; then
    echo "[$(date -Is)] waiting for Section-10 validation completion; log_mtime=$(stat -c %y "$MEMRL_LOG" 2>/dev/null || echo missing)"
    sleep "$INTERVAL"
    continue
  fi

  if [[ ! -f "$SUBMITTED_FILE" ]]; then
    echo "[$(date -Is)] Section-10 validation completion detected; waiting ${PERSIST_WAIT}s for final persistence"
    sleep "$PERSIST_WAIT"
    if ! memrl_finished; then
      echo "[$(date -Is)] completion marker disappeared; returning to polling"
      sleep "$INTERVAL"
      continue
    fi
    submit_passk_once || { sleep "$INTERVAL"; continue; }
  fi

  # Submission is durable before cleanup, so a cancellation/API failure can
  # never cause duplicate Pass@10 submissions.
  if cancel_memrl_if_needed; then
    echo "[$(date -Is)] all requested actions complete; monitor exiting"
    exit 0
  fi
  sleep "$INTERVAL"
done
