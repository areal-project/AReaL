#!/usr/bin/env bash
set -euo pipefail
SNAP=${SOURCE_SNAPSHOT:-/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect/exp_llb_db_region_fs_gpt41mini_splitpriorfix_region-fs-db-gpt41mini-splitpriorfix-20260722/snapshot/9}
RUNNER=/storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_db_region_fs_v2_readonly_aistudio.sh
OUT=/storage/openpsi/users/yl/agent-memory/MemRL/analysis_outputs/llb_db_regionfs_v2_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
cell() { local label=$1 fs=$2 fsv2=$3 lambda=$4; SOURCE_SNAPSHOT="$SNAP" EVAL_LABEL="$label" EVAL_FAILURE_SLOTS="$fs" EVAL_FS_V2="$fsv2" EVAL_REGION_LAMBDA="$lambda" TRACE_OUTPUT="$OUT/$label.jsonl" bash "$RUNNER"; }
cell A_dedup_fs0 0 0 0
cell B_dedup_fs1v2 1 1 0
cell C_dedup_region015_fs0 0 0 0.15
cell D_dedup_region015_fs1v2 1 1 0.15
echo "REGION_FS_V2_GRID_COMPLETE output=$OUT"
