#!/bin/bash
set -euo pipefail
BASE=/storage/openpsi/users/yl/agent-memory/MemRL/scripts
ROOT_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
for R in 1 2 3; do
  echo "================ DIAGNOSTIC RUN $R/3 ================"
  RUN_TAG="${ROOT_TAG}-r${R}" bash "$BASE/ais_run_alf_opus47_s8_diag3_worker.sh"
done
