#!/usr/bin/env bash
set -u

ROOT=/storage/openpsi/users/yl/agent-memory/MemRL
SELF_RAG_RECORD_ID=324338591
SELF_RAG_LOG="$ROOT/logs/llb_db_selfrag_gpt41mini_20260718-013611.log"
SELF_RAG_SNAPSHOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_baselines/exp_llb_db_selfrag_gpt41mini_selfrag-db-gpt41mini-20260718/snapshot/10
SUBMIT="$ROOT/scripts/submit_llb_db_baselines_gpt41mini.sh"
STATE_DIR="$ROOT/scripts/.monitor_state/llb_db_selfrag_then_mem0"
MONITOR_LOG="$STATE_DIR/monitor.log"
INTERVAL="${INTERVAL:-120}"
PERSIST_WAIT="${PERSIST_WAIT:-120}"
mkdir -p "$STATE_DIR"
exec 9>"$STATE_DIR/monitor.lock"
flock -n 9 || exit 0
exec >>"$MONITOR_LOG" 2>&1

echo "[$(date -Is)] monitor started pid=$$ record_id=$SELF_RAG_RECORD_ID"

log_complete() {
  [[ -f "$SELF_RAG_LOG" ]] &&
    grep -aq -- 'Section 10 complete. Total 361 trajectories collected.' "$SELF_RAG_LOG" &&
    grep -aq -- 'Validation Evaluation Complete (after Section 10)' "$SELF_RAG_LOG"
}

snapshot_healthy() {
  SNAPSHOT="$SELF_RAG_SNAPSHOT" python - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ['SNAPSHOT'])
required=[p/'snapshot_meta.json', p/'cube'/'textual_memory.json', p/'local_cache'/'q_cache.json', p/'local_cache'/'dict_memory.json']
if any(not x.is_file() or x.stat().st_size <= 0 for x in required):
    raise SystemExit(1)
try:
    meta=json.loads((p/'snapshot_meta.json').read_text())
except Exception:
    raise SystemExit(1)
if int(meta.get('checkpoint_id', -1)) != 10:
    raise SystemExit(1)
q=p/'qdrant'
if not q.is_dir() or not any(x.is_file() and x.stat().st_size > 0 for x in q.rglob('*')):
    raise SystemExit(1)
PY
}

cancel_if_lingering() {
  [[ -f "$STATE_DIR/selfrag.cancelled" ]] && return 0
  RID="$SELF_RAG_RECORD_ID" PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-} python - <<'PY'
import os, time
from aistudio_common.rest import job
rid=os.environ['RID']
terminal={'success','succeeded','failed','failure','stopped','killed','cancelled','canceled','terminated'}
status=job.query_job_status(rid)
print(f'before_cancel record_id={rid} status={status}')
if status is None:
    raise SystemExit(2)
if str(status).lower() not in terminal:
    print(job.stop_workflow(rid))
    time.sleep(10)
    status=job.query_job_status(rid)
    print(f'after_cancel record_id={rid} status={status}')
PY
  rc=$?
  if [[ $rc -eq 0 ]]; then
    printf '%s record_id=%s cancel_checked\n' "$(date -Is)" "$SELF_RAG_RECORD_ID" > "$STATE_DIR/selfrag.cancelled"
    return 0
  fi
  return "$rc"
}

submit_mem0_once() {
  [[ -f "$STATE_DIR/mem0.submitted" ]] && return 0
  local out="$STATE_DIR/mem0.submit.log"
  : > "$out"
  echo "[$(date -Is)] submitting Mem0 with stable run id"
  if MEMRL_RUN_ID=mem0-db-gpt41mini-20260719 bash "$SUBMIT" mem0 >>"$out" 2>&1; then
    local rid
    rid=$(grep -aoE '(record id=|record_id[=: ]+)[0-9]+' "$out" | tail -1 | grep -aoE '[0-9]+' || true)
    if [[ -z "$rid" ]]; then
      echo "[$(date -Is)] submission returned success but record id was not captured; will retry"
      return 1
    fi
    printf '%s record_id=%s run_id=mem0-db-gpt41mini-20260719\n' "$(date -Is)" "$rid" > "$STATE_DIR/mem0.submitted"
    echo "[$(date -Is)] Mem0 submitted record_id=$rid"
    return 0
  fi
  echo "[$(date -Is)] Mem0 submission failed; will retry"
  return 1
}

while true; do
  if ! log_complete || ! snapshot_healthy; then
    progress=$(grep -aoE 'Processing mini-batch [0-9]+/73 in section 10' "$SELF_RAG_LOG" 2>/dev/null | tail -1 || true)
    echo "[$(date -Is)] waiting for Self-RAG final eval + healthy snapshot/10; ${progress:-no-progress-marker}"
    sleep "$INTERVAL"
    continue
  fi

  if [[ ! -f "$STATE_DIR/selfrag.persist_wait_done" ]]; then
    echo "[$(date -Is)] final eval and snapshot/10 detected; waiting ${PERSIST_WAIT}s for persistence"
    sleep "$PERSIST_WAIT"
    if ! log_complete || ! snapshot_healthy; then
      echo "[$(date -Is)] persistence revalidation failed; returning to polling"
      continue
    fi
    printf '%s healthy snapshot=%s\n' "$(date -Is)" "$SELF_RAG_SNAPSHOT" > "$STATE_DIR/selfrag.persist_wait_done"
  fi

  cancel_if_lingering || { echo "[$(date -Is)] cancel/status check failed; retrying"; sleep "$INTERVAL"; continue; }
  submit_mem0_once || { sleep "$INTERVAL"; continue; }
  echo "[$(date -Is)] requested chain completed; monitor exiting"
  exit 0
done
