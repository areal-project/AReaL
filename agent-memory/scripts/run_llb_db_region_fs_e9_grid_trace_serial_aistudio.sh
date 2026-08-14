#!/usr/bin/env bash
set -euo pipefail
SNAP=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect/exp_llb_db_region_fs_gpt41mini_splitpriorfix_region-fs-db-gpt41mini-splitpriorfix-20260722/snapshot/9
RUNNER=/storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_db_region_fs_readonly_grid_trace_aistudio.sh
OUT=/storage/openpsi/users/yl/agent-memory/MemRL/analysis_outputs/llb_db_region_e9_grid_20260727
mkdir -p "$OUT"
eval_cell() {
  local label=$1 ws=$2 wq=$3 fs=$4
  echo "=== TRACE GRID $label sim=$ws q=$wq fs=$fs ==="
  SOURCE_SNAPSHOT="$SNAP" EVAL_SECTION=9 EVAL_LABEL="$label" EVAL_WEIGHT_SIM="$ws" EVAL_WEIGHT_Q="$wq" EVAL_FAILURE_SLOTS="$fs" TRACE_OUTPUT="$OUT/${label}.jsonl" bash "$RUNNER"
}
eval_cell region_e9_control_05q05_fs2 0.5 0.5 2
eval_cell region_e9_A_07q03_fs2 0.7 0.3 2
eval_cell region_e9_D_07q03_fs1 0.7 0.3 1
echo '=== TRACE GRID COMPLETE ==='
