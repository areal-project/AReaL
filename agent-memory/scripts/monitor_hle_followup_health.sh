#!/usr/bin/env bash
set -u
ROOT=/storage/openpsi/users/yl/agent-memory
STATE_DIR="$ROOT/MemRL/logs/hle_resume_monitor"
MANIFEST="${1:-$STATE_DIR/followup_manifest.env}"
HEALTH_LOG="$STATE_DIR/health.log"
INTERVAL="${HLE_HEALTH_INTERVAL:-300}"
MAX_POLLS="${HLE_HEALTH_MAX_POLLS:-72}"
mkdir -p "$STATE_DIR"
exec >>"$HEALTH_LOG" 2>&1
[[ -f "$MANIFEST" ]] || { echo "[$(date -Is)] FATAL manifest missing: $MANIFEST"; exit 1; }
# shellcheck disable=SC1090
source "$MANIFEST"
echo "[$(date -Is)] health monitor started manifest=$MANIFEST"
log_for() { printf '%s/MemRL/logs/aistudio_hle_%s_gemini35flash_%s.log' "$ROOT" "$1" "$2"; }
has_fatal() { [[ -f "$1" ]] && grep -aqE 'ModuleNotFoundError|ImportError:|No module named|HLE run failed:|Traceback \(most recent call last\)' "$1"; }
meaningful_progress() { [[ -f "$1" ]] && grep -aqE '\[train sec [0-9]+\][[:space:]]+[0-9]+/[0-9]+[[:space:]]+\| Acc so far:|Section [0-9]+ Train Acc:' "$1"; }
rag_ok=0; selfrag_ok=0; mem0_ok=0
for ((poll=1; poll<=MAX_POLLS; poll++)); do
  RAG_LOG=$(log_for rag "$RAG_RID")
  SELFRAG_LOG=$(log_for selfrag "$SELFRAG_RID")
  MEM0_LOG=$(log_for mem0 "$MEM0_RID")
  if [[ "$rag_ok" == 0 && -f "$RAG_LOG" ]]; then
    if grep -aFq '[RAG] PRECHECK_OK resume=1_b9' "$RAG_LOG" && meaningful_progress "$RAG_LOG"; then rag_ok=1; echo "[$(date -Is)] OK RAG resumed and progressing log=$RAG_LOG"
    elif has_fatal "$RAG_LOG"; then echo "[$(date -Is)] ALERT RAG fatal/startup-or-runtime error before verified progress log=$RAG_LOG"; fi
  fi
  if [[ "$selfrag_ok" == 0 && -f "$SELFRAG_LOG" ]]; then
    if grep -aFq '[Self-RAG] PRECHECK_OK enabled=true' "$SELFRAG_LOG" && grep -aFq '[Self-RAG] Critique' "$SELFRAG_LOG" && meaningful_progress "$SELFRAG_LOG"; then selfrag_ok=1; echo "[$(date -Is)] OK Self-RAG critique active and progressing log=$SELFRAG_LOG"
    elif has_fatal "$SELFRAG_LOG"; then echo "[$(date -Is)] ALERT Self-RAG fatal/startup-or-runtime error before verified progress log=$SELFRAG_LOG"
    elif grep -aFq '[Self-RAG] PRECHECK_OK enabled=true' "$SELFRAG_LOG"; then echo "[$(date -Is)] Self-RAG precheck passed; waiting for critique"; fi
  fi
  if [[ "$mem0_ok" == 0 && -f "$MEM0_LOG" ]]; then
    if grep -aFq '[Mem0] PRECHECK_OK' "$MEM0_LOG" && grep -aFq 'Using Mem0MemoryService (infer=true' "$MEM0_LOG" && grep -aFq '[Mem0MemoryService] add_memories:' "$MEM0_LOG" && grep -aFq 'fresh_start=true' "$MEM0_LOG" && meaningful_progress "$MEM0_LOG"; then mem0_ok=1; echo "[$(date -Is)] OK Mem0 active with infer=true log=$MEM0_LOG"
    elif has_fatal "$MEM0_LOG"; then echo "[$(date -Is)] ALERT Mem0 dependency/runtime failure before verified progress log=$MEM0_LOG"
    elif grep -aFq '[Mem0] PRECHECK_OK' "$MEM0_LOG"; then echo "[$(date -Is)] Mem0 imports passed; waiting for service activity"; fi
  fi
  echo "[$(date -Is)] poll=$poll rag_ok=$rag_ok selfrag_ok=$selfrag_ok mem0_ok=$mem0_ok"
  if [[ "$rag_ok" == 1 && "$selfrag_ok" == 1 && "$mem0_ok" == 1 ]]; then
    printf '%s\n' "$(date -Is) rag_ok=1 selfrag_ok=1 mem0_ok=1" > "$STATE_DIR/health.ok"
    exit 0
  fi
  sleep "$INTERVAL"
done
echo "[$(date -Is)] ALERT health verification timed out rag_ok=$rag_ok selfrag_ok=$selfrag_ok mem0_ok=$mem0_ok"
exit 1
