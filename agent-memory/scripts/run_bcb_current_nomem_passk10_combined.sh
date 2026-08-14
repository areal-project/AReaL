#!/bin/bash
# Current-serving baseline rebuild: no-memory E1 train+val, then train pass@10.
set -euo pipefail
MEMRL_DIR=${MEMRL_DIR:-/storage/openpsi/users/yl/agent-memory/MemRL}
OUT=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench
STATE=$OUT/current_nomem_passk10_20260807_state
LLM_BASE_URL=${BCB_LLM_BASE_URL:?parent strict runner must export BCB_LLM_BASE_URL}
mkdir -p "$STATE/configs" "$STATE/logs"
cd "$MEMRL_DIR"

make_cfg() {
  local dst=$1 baseline=$2
  python - "$dst" "$baseline" "$LLM_BASE_URL" <<'PY'
import sys,yaml
from pathlib import Path
dst,baseline,url=sys.argv[1:]
cfg=yaml.safe_load(Path('configs/rl_bcb_config.passk10_local.yaml').read_text())
cfg['llm']['base_url']=url
cfg['experiment']['experiment_name']='bcb_current_'+baseline
if baseline=='nomem_e1':
    cfg['experiment']['baseline_mode']=None
    cfg['experiment']['baseline_k']=10
    cfg['experiment']['bcb_run_validation']=True
else:
    cfg['experiment']['baseline_mode']='passk'
    cfg['experiment']['baseline_k']=10
    cfg['experiment']['bcb_run_validation']=False
Path(dst).write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
}
run_stage() {
  local name=$1; shift
  if [ -f "$STATE/$name.done" ]; then echo "[BASELINE] SKIP $name"; return; fi
  echo "[BASELINE] START $name: $*"
  "$@" 2>&1 | tee -a "$STATE/logs/$name.log"
  test ${PIPESTATUS[0]} -eq 0
  date -Iseconds > "$STATE/$name.done"
}
NOMEM_CFG=$STATE/configs/nomem_e1.yaml
PASSK_CFG=$STATE/configs/passk10.yaml
make_cfg "$NOMEM_CFG" nomem_e1
make_cfg "$PASSK_CFG" passk10
COMMON=(--split instruct --subset full --checkpoint_interval 100 --max_checkpoints 3 --eval_timeout 240 --untrusted_hard_timeout 300)
run_stage nomem_e1 python run/run_bcb.py --config "$NOMEM_CFG" --epochs 1 \
  --output_dir "$OUT/deepseek_v3_current_nomem_e1" --n_eval_runs 1 "${COMMON[@]}"
run_stage passk10 python run/run_bcb.py --config "$PASSK_CFG" --epochs 10 \
  --output_dir "$OUT/deepseek_v3_current_passk10" --baseline_mode passk --baseline_k 10 "${COMMON[@]}"
echo '[BASELINE] ALL DONE'
