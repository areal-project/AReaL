#!/bin/bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
STATE="$ROOT/MemRL/logs/hle_resume_monitor"
LOG="$STATE/mem0_bm25_restart.log"
OLD_JOB=324760052
RID=yl-hle-mem0-g35f-20260719-185247
SNAP_ROOT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_mem0_gemini35flash_${RID}/snapshot
BASE=1_b9
export PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-}
exec >>"$LOG" 2>&1
trap 'rc=$?; echo "[$(date -Is)] monitor exit rc=$rc line=${LINENO}"' EXIT
echo "[$(date -Is)] monitor start old_job=$OLD_JOB baseline=$BASE"
while true; do
  newest=$(find "$SNAP_ROOT" -mindepth 1 -maxdepth 1 -type d -name '*_b*' -printf '%f\n' 2>/dev/null | sort -t b -k2,2n | tail -1 || true)
  if [[ -n "$newest" && "$newest" != "$BASE" && -f "$SNAP_ROOT/$newest/snapshot_meta.json" && -d "$SNAP_ROOT/$newest/mem0_qdrant" ]]; then
    echo "[$(date -Is)] new valid checkpoint detected: $newest"
    break
  fi
  if status=$(python - <<PY
from aistudio_common.rest import job
print(job.query_job_status('$OLD_JOB'))
PY
  ); then
    :
  else
    status=unknown
    echo "[$(date -Is)] job status query failed; continuing to wait for a complete checkpoint"
  fi
  echo "[$(date -Is)] waiting newest=${newest:-none} old_status=$status"
  case "${status,,}" in success|succeeded|failed|failure|stopped|killed|cancelled|canceled|terminated)
    echo "[$(date -Is)] old job became terminal before a newer checkpoint; not auto-submitting"
    exit 2;;
  esac
  sleep 60
done
python - <<PY
import time
from aistudio_common.rest import job
rid='$OLD_JOB'
print('stop_before', job.query_job_status(rid))
print('stop_result', job.stop_workflow(rid))
for _ in range(30):
    time.sleep(5)
    s=job.query_job_status(rid)
    print('stop_wait', s, flush=True)
    if str(s).lower() not in {'running','prepare','preparing','pending'}:
        break
PY
sleep 10
cd /tmp
export PYTHONPATH=/tmp/yl_pypai:${PYTHONPATH:-}
out="$STATE/mem0_bm25_resubmit.out"
bash "$ROOT/MemRL/scripts/submit_hle_one_followup.sh" mem0 "$RID" >"$out" 2>&1
new_job=$(grep -aoE 'record id=[0-9]+' "$out" | tail -1 | cut -d= -f2)
[[ -n "$new_job" ]]
printf '%s\n' "$new_job" >"$STATE/mem0.bm25_job_id"
python - <<PY
from pathlib import Path
p=Path('$STATE/followup_manifest.env')
s=p.read_text()
lines=[]
for line in s.splitlines():
    if line.startswith('MEM0_JOB_ID='):
        line='MEM0_JOB_ID=$new_job'
    lines.append(line)
p.write_text('\n'.join(lines)+'\n')
PY
echo "[$(date -Is)] BM25 Mem0 resumed new_job=$new_job checkpoint=$newest"
