#!/usr/bin/env bash
# Serial read-only E9 corrected Region validation grid: A(fs2) then D(fs1).
set -euo pipefail
SNAP=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect/exp_llb_db_region_fs_gpt41mini_splitpriorfix_region-fs-db-gpt41mini-splitpriorfix-20260722/snapshot/9
RUNNER=/storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_db_region_fs_readonly_grid_aistudio.sh
printf '%s\n' '=== SERIAL REGION GRID: A sim07/q03 fs2 ==='
SOURCE_SNAPSHOT="$SNAP" EVAL_SECTION=9 EVAL_LABEL=region_e9_sim07q03_fs2_serial EVAL_WEIGHT_SIM=0.7 EVAL_WEIGHT_Q=0.3 EVAL_FAILURE_SLOTS=2 bash "$RUNNER"
printf '%s\n' '=== SERIAL REGION GRID: D sim07/q03 fs1 ==='
SOURCE_SNAPSHOT="$SNAP" EVAL_SECTION=9 EVAL_LABEL=region_e9_sim07q03_fs1_serial EVAL_WEIGHT_SIM=0.7 EVAL_WEIGHT_Q=0.3 EVAL_FAILURE_SLOTS=1 bash "$RUNNER"
printf '%s\n' '=== SERIAL REGION GRID COMPLETE ==='
