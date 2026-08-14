#!/usr/bin/env bash
set -euo pipefail
SOURCE_SNAPSHOT=${SOURCE_SNAPSHOT:-/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect/exp_llb_db_region_fs_gpt41mini_splitpriorfix_region-fs-db-gpt41mini-splitpriorfix-20260722/snapshot/9}
OUT=/storage/openpsi/users/yl/agent-memory/MemRL/analysis_outputs/llb_db_regionfs_structured_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
SOURCE_SNAPSHOT="$SOURCE_SNAPSHOT" \
EVAL_LABEL=structured_regionfs_single \
EVAL_WEIGHT_SIM=0.7 EVAL_WEIGHT_Q=0.3 \
EVAL_FAILURE_SLOTS=1 EVAL_FS_V2=1 EVAL_DB_STRUCTURED_FS=1 \
EVAL_REGION_LAMBDA=0 EVAL_DEDUP=1 EVAL_FS_MIN_SIM=0.50 \
TRACE_OUTPUT="$OUT/structured_regionfs_single.jsonl" \
bash /storage/openpsi/users/yl/agent-memory/MemRL/scripts/run_llb_db_region_fs_v2_readonly_aistudio.sh
echo "STRUCTURED_REGION_FS_SINGLE_COMPLETE output=$OUT"
