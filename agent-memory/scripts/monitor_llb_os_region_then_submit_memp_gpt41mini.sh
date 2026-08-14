#!/usr/bin/env bash
# After Region+FS S10 validation is persisted, stop any residual Region workflow,
# then submit the LLB OS GPT-4.1-mini MemP experiment exactly once via AIStudio.
set -u

ROOT=/storage/openpsi/users/yl/agent-memory
REGION_LOG="$ROOT/MemRL/logs/llb_os_region_fs_gpt41mini_hardfix_resume_s6b49_0160122_20260717_090730.log"
SUBMIT="$ROOT/MemRL/scripts/submit_llb_os_memp_gpt41mini.sh"
STATE_DIR="$ROOT/MemRL/scripts/.monitor_state/llb_os_region_to_memp_gpt41mini"
MONITOR_LOG="$STATE_DIR/monitor.log"
SUBMIT_LOG="$STATE_DIR/submit.log"
REGION_STOP_FILE="$STATE_DIR/region_stop.ok"
SUBMITTED_FILE="$STATE_DIR/submitted.ok"
MEMP_RECORD_FILE="$STATE_DIR/memp_record_id"
LOCK_FILE="$STATE_DIR/monitor.lock"
INTERVAL="${LLB_OS_REGION_TO_MEMP_INTERVAL:-300}"
PERSIST_WAIT="${LLB_OS_REGION_TO_MEMP_PERSIST_WAIT:-300}"
REGION_RECORD_ID=324125969

mkdir -p "$STATE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another monitor instance already owns $LOCK_FILE" >&2
  exit 0
fi
exec >>"$MONITOR_LOG" 2>&1

echo "[$(date -Is)] monitor started pid=$$ region_record_id=$REGION_RECORD_ID interval=${INTERVAL}s"

region_finished() {
  [[ -f "$REGION_LOG" ]] &&
    grep -aq -- '--- Validation Evaluation Complete (after Section 10) ---' "$REGION_LOG"
}

region_status() {
  REGION_RECORD_ID="$REGION_RECORD_ID" python - <<'PY'
import os
from aistudio_common.rest import job
print(job.query_job_status(os.environ['REGION_RECORD_ID']) or '')
PY
}

stop_region_residual() {
  local status normalized response after
  status=$(region_status 2>&1) || {
    echo "[$(date -Is)] failed to query Region status: $status"
    return 1
  }
  normalized=$(printf '%s' "$status" | tr '[:upper:]' '[:lower:]' | xargs)
  echo "[$(date -Is)] Region before cleanup record_id=$REGION_RECORD_ID status=$status"
  case "$normalized" in
    success|succeeded|failed|failure|stopped|killed|cancelled|canceled|terminated)
      printf '%s region_record_id=%s terminal_status=%s\n' "$(date -Is)" "$REGION_RECORD_ID" "$status" > "$REGION_STOP_FILE"
      return 0
      ;;
    '')
      echo "[$(date -Is)] Region status unknown; refusing to stop or submit MemP"
      return 1
      ;;
  esac

  response=$(REGION_RECORD_ID="$REGION_RECORD_ID" python - <<'PY'
import os
from aistudio_common.rest import job
print(job.stop_workflow(os.environ['REGION_RECORD_ID']))
PY
  ) || {
    echo "[$(date -Is)] Region stop request failed: $response"
    return 1
  }
  echo "[$(date -Is)] Region stop requested response=$response"

  # Do not overlap MemP with a Region workflow that is still shutting down.
  for _ in $(seq 1 30); do
    sleep 10
    after=$(region_status 2>&1) || after=""
    normalized=$(printf '%s' "$after" | tr '[:upper:]' '[:lower:]' | xargs)
    echo "[$(date -Is)] Region cleanup poll status=$after"
    case "$normalized" in
      success|succeeded|failed|failure|stopped|killed|cancelled|canceled|terminated)
        printf '%s region_record_id=%s terminal_status=%s\n' "$(date -Is)" "$REGION_RECORD_ID" "$after" > "$REGION_STOP_FILE"
        return 0
        ;;
    esac
  done
  echo "[$(date -Is)] Region did not reach terminal state within 300s; will retry later and will not submit MemP yet"
  return 1
}

submit_memp_once() {
  if [[ -f "$SUBMITTED_FILE" ]]; then
    echo "[$(date -Is)] MemP already submitted: $(cat "$SUBMITTED_FILE")"
    return 0
  fi

  echo "[$(date -Is)] submitting LLB OS GPT-4.1-mini MemP through AIStudio"
  : > "$SUBMIT_LOG"
  local rc rid
  if sudo -n -E bash "$SUBMIT" >>"$SUBMIT_LOG" 2>&1; then
    rid=$(grep -aoE 'record id=[0-9]+' "$SUBMIT_LOG" | tail -1 | cut -d= -f2 || true)
    if [[ ! "$rid" =~ ^[0-9]+$ ]]; then
      echo "[$(date -Is)] submit exited 0 but no MemP record ID was found; will retry"
      return 1
    fi
    printf '%s\n' "$rid" > "$MEMP_RECORD_FILE"
    printf '%s memp_record_id=%s\n' "$(date -Is)" "$rid" > "$SUBMITTED_FILE"
    echo "[$(date -Is)] MemP submission succeeded record_id=$rid"
    return 0
  else
    rc=$?
    echo "[$(date -Is)] MemP submission failed rc=$rc; will retry after ${INTERVAL}s (see $SUBMIT_LOG)"
    return "$rc"
  fi
}

while true; do
  if [[ -f "$SUBMITTED_FILE" ]]; then
    echo "[$(date -Is)] chain complete: $(cat "$SUBMITTED_FILE")"
    exit 0
  fi

  if ! region_finished; then
    echo "[$(date -Is)] waiting for Region S10 validation; log_mtime=$(stat -c %y "$REGION_LOG" 2>/dev/null || echo missing)"
    sleep "$INTERVAL"
    continue
  fi

  if [[ ! -f "$REGION_STOP_FILE" ]]; then
    echo "[$(date -Is)] Region S10 validation detected; waiting ${PERSIST_WAIT}s for final persistence"
    sleep "$PERSIST_WAIT"
    if ! region_finished; then
      echo "[$(date -Is)] completion marker disappeared; returning to polling"
      sleep "$INTERVAL"
      continue
    fi
    if ! stop_region_residual; then
      sleep "$INTERVAL"
      continue
    fi
  fi

  if submit_memp_once; then
    echo "[$(date -Is)] Region->MemP chain completed"
    exit 0
  fi
  sleep "$INTERVAL"
done
