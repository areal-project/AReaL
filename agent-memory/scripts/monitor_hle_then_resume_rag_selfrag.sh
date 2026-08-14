#!/usr/bin/env bash
set -u

ROOT=/storage/openpsi/users/yl/agent-memory
MEMRL_LOG="$ROOT/MemRL/logs/aistudio_hle_memrl_gemini35flash_20260707-115246.log"
MEMP_LOG="$ROOT/MemRL/logs/aistudio_hle_memp_gemini35flash_yl-hle-memp-g35f-20260708-143854.log"
SUBMIT_ONE="$ROOT/MemRL/scripts/submit_hle_one_followup.sh"
HEALTH_MONITOR="$ROOT/MemRL/scripts/monitor_hle_followup_health.sh"
STATE_DIR="$ROOT/MemRL/logs/hle_resume_monitor"
MONITOR_LOG="$STATE_DIR/monitor.log"
LOCK_DIR="$STATE_DIR/submission.lock"
DONE_FILE="$STATE_DIR/submitted.ok"
SUBMIT_LOG="$STATE_DIR/submit.log"
SUMMARY_FILE="$STATE_DIR/final_summary.txt"
MANIFEST="$STATE_DIR/followup_manifest.env"
INTERVAL="${HLE_MONITOR_INTERVAL:-600}"
STAGGER="${HLE_FOLLOWUP_STAGGER:-1800}"

mkdir -p "$STATE_DIR"
exec >>"$MONITOR_LOG" 2>&1

echo "[$(date -Is)] monitor started pid=$$ interval=${INTERVAL}s stagger=${STAGGER}s"

is_finished() {
  local log="$1"
  # Runner order is: section metric -> final snapshot/10 -> cumulative metric/state.
  [[ -f "$log" ]] || return 1
  grep -aq 'Section 10 Train Acc:' "$log" &&
    grep -aqE ' Saved ckpt:.*snapshot/10([/[:space:]},]|$)' "$log" &&
    grep -aq '\[Train\] Cumulative Acc after section 10:' "$log"
}

safe_sleep() {
  local seconds="$1"
  (( seconds <= 0 )) && return 0
  if ! sleep "$seconds"; then
    echo "[$(date -Is)] WARN sleep ${seconds}s failed; retrying after 5s"
    sleep 5 || true
  fi
}

stage_file() { printf '%s/%s.submitted' "$STATE_DIR" "$1"; }

record_job_id() {
  local mode="$1" rid="$2" before_lines="$3"
  local job_id stage
  stage=$(stage_file "$mode")
  job_id=$(tail -n "+$((before_lines + 1))" "$SUBMIT_LOG" 2>/dev/null | grep -aEo '(Job ID:[[:space:]]*|record id=)[0-9]+' | tail -n 1 | grep -Eo '[0-9]+' || true)
  printf 'MODE=%q\nRID=%q\nJOB_ID=%q\nSUBMITTED_EPOCH=%q\nSUBMITTED_AT=%q\n' \
    "$mode" "$rid" "$job_id" "$(date +%s)" "$(date -Is)" > "$stage"
  if [[ -n "$job_id" ]]; then
    printf -v "${mode^^}_JOB_ID" '%s' "$job_id"
    export "${mode^^}_JOB_ID"
    printf '%s_JOB_ID=%q\n' "${mode^^}" "$job_id" >> "$MANIFEST"
    echo "[$(date -Is)] recorded ${mode} job_id=$job_id rid=$rid"
  else
    echo "[$(date -Is)] WARN ${mode} submitted but no Job ID parsed; rid=$rid"
  fi
}

wait_after_stage() {
  local mode="$1" stage epoch elapsed remaining
  stage=$(stage_file "$mode")
  [[ -f "$stage" ]] || return 0
  epoch=$(awk -F= '$1=="SUBMITTED_EPOCH"{print $2}' "$stage" | tr -d "'\"")
  [[ "$epoch" =~ ^[0-9]+$ ]] || return 0
  elapsed=$(( $(date +%s) - epoch ))
  remaining=$(( STAGGER - elapsed ))
  if (( remaining > 0 )); then
    echo "[$(date -Is)] preserving stagger: waiting ${remaining}s after ${mode}"
    safe_sleep "$remaining"
  fi
}

submit_stage() {
  local mode="$1" rid="$2" before_lines rc
  if [[ -f "$(stage_file "$mode")" ]]; then
    echo "[$(date -Is)] ${mode} already submitted; not duplicating"
    return 0
  fi
  before_lines=$(wc -l < "$SUBMIT_LOG" 2>/dev/null || echo 0)
  echo "[$(date -Is)] submitting ${mode} rid=$rid"
  sudo -n -E bash "$SUBMIT_ONE" "$mode" "$rid" >>"$SUBMIT_LOG" 2>&1
  rc=$?
  if (( rc != 0 )); then
    echo "[$(date -Is)] ${mode} submission failed rc=$rc; see $SUBMIT_LOG"
    return "$rc"
  fi
  record_job_id "$mode" "$rid" "$before_lines"
}

release_lock() { rm -f "$LOCK_DIR/pid" 2>/dev/null || true; rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap release_lock EXIT
trap 'release_lock; jobs -pr | xargs -r kill 2>/dev/null || true; exit 0' INT TERM

while true; do
  if [[ -f "$DONE_FILE" ]]; then
    echo "[$(date -Is)] submission already completed: $(cat "$DONE_FILE")"
    exit 0
  fi

  memrl_done=0; memp_done=0
  is_finished "$MEMRL_LOG" && memrl_done=1
  is_finished "$MEMP_LOG" && memp_done=1
  echo "[$(date -Is)] poll memrl_done=$memrl_done memp_done=$memp_done memrl_mtime=$(stat -c %y "$MEMRL_LOG" 2>/dev/null || echo missing) memp_mtime=$(stat -c %y "$MEMP_LOG" 2>/dev/null || echo missing)"

  if [[ "$memrl_done" == 1 && "$memp_done" == 1 ]]; then
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
      # Remove only a demonstrably stale lock (no live monitor PID recorded in it).
      lock_pid=$(cat "$LOCK_DIR/pid" 2>/dev/null || true)
      if [[ "$lock_pid" =~ ^[0-9]+$ ]] && kill -0 "$lock_pid" 2>/dev/null; then
        echo "[$(date -Is)] another monitor pid=$lock_pid owns submission lock"
        safe_sleep "$INTERVAL"
        continue
      fi
      echo "[$(date -Is)] removing stale submission lock pid=${lock_pid:-unknown}"
      rm -f "$LOCK_DIR/pid" 2>/dev/null || true
      rmdir "$LOCK_DIR" 2>/dev/null || true
      mkdir "$LOCK_DIR" 2>/dev/null || { safe_sleep "$INTERVAL"; continue; }
    fi
    printf '%s\n' "$$" > "$LOCK_DIR/pid"

    echo "[$(date -Is)] both final Section 10 snapshots/results detected; waiting 600s for persistence"
    safe_sleep 600
    if ! is_finished "$MEMRL_LOG" || ! is_finished "$MEMP_LOG"; then
      echo "[$(date -Is)] completion condition no longer satisfied; releasing lock"
      release_lock
      safe_sleep "$INTERVAL"
      continue
    fi

    if [[ ! -f "$MANIFEST" ]]; then
      TS=$(date +%Y%m%d-%H%M%S)
      RAG_RID="yl-hle-rag-g35f-resume-${TS}"
      SELFRAG_RID="yl-hle-selfrag-g35f-resume-${TS}"
      MEM0_RID="yl-hle-mem0-g35f-${TS}"
      printf 'RAG_RID=%q\nSELFRAG_RID=%q\nMEM0_RID=%q\n' "$RAG_RID" "$SELFRAG_RID" "$MEM0_RID" > "$MANIFEST"
    fi
    # shellcheck disable=SC1090
    source "$MANIFEST"

    submit_stage rag "$RAG_RID" || { release_lock; safe_sleep "$INTERVAL"; continue; }
    wait_after_stage rag
    submit_stage selfrag "$SELFRAG_RID" || { release_lock; safe_sleep "$INTERVAL"; continue; }
    wait_after_stage selfrag
    submit_stage mem0 "$MEM0_RID" || { release_lock; safe_sleep "$INTERVAL"; continue; }

    # Reload persisted IDs so restart/resume of this monitor still produces a complete summary.
    # shellcheck disable=SC1090
    source "$MANIFEST"
    printf '%s\n' "$(date -Is) rag=$RAG_RID selfrag=$SELFRAG_RID mem0=$MEM0_RID" > "$DONE_FILE"
    {
      echo "submitted_at=$(date -Is)"
      echo "rag_run_id=$RAG_RID"
      echo "rag_job_id=${RAG_JOB_ID:-unknown}"
      echo "rag_config=$ROOT/MemRL/configs/rl_hle_config.rag_gemini35flash_resume_1b9.yaml"
      echo "rag_resume=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_rag_gemini35flash_yl-hle-rag-g35f-20260717-103921/snapshot/1_b9"
      echo "rag_log=$ROOT/MemRL/logs/aistudio_hle_rag_gemini35flash_${RAG_RID}.log"
      echo "selfrag_run_id=$SELFRAG_RID"
      echo "selfrag_job_id=${SELFRAG_JOB_ID:-unknown}"
      echo "selfrag_config=$ROOT/MemRL/configs/rl_hle_config.selfrag_gemini35flash_resume_1b9.yaml"
      echo "selfrag_resume=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_selfrag_gemini35flash_yl-hle-selfrag-g35f-20260717-103921/snapshot/1_b9"
      echo "selfrag_log=$ROOT/MemRL/logs/aistudio_hle_selfrag_gemini35flash_${SELFRAG_RID}.log"
      echo "mem0_run_id=$MEM0_RID"
      echo "mem0_job_id=${MEM0_JOB_ID:-unknown}"
      echo "mem0_config=generated:/tmp/hle_mem0_${MEM0_RID}.yaml"
      echo "mem0_resume=fresh_start; future resume requires mem0_qdrant+mem0_id_metadata+aligned collection/runner state"
      echo "mem0_collection=hle_mem0_${MEM0_RID//-/_}"
      echo "mem0_log=$ROOT/MemRL/logs/aistudio_hle_mem0_gemini35flash_${MEM0_RID}.log"
    } > "$SUMMARY_FILE"
    nohup bash "$HEALTH_MONITOR" "$MANIFEST" >/dev/null 2>&1 &
    echo "[$(date -Is)] all three submissions succeeded; health monitor pid=$!"
    exit 0
  fi
  safe_sleep "$INTERVAL"
done
