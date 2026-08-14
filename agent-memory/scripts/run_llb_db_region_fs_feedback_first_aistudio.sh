#!/usr/bin/env bash
# Fresh 10-epoch LLB-DB corrected Region training with SQL-structured FS.
set -euo pipefail
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | sed 's/^gpulingjun//; s/\..*$//')
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$PROJECT_DIR/logs/llb_db_region_fs_feedback_first_gpt41mini_${HOST_SHORT}_${TS}.log"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$PROJECT_DIR"
echo '=========================================='
echo 'LLB DB feedback-first Region+Structured-FS 10-epoch training (gpt-4.1-mini)'
echo "Host: $(hostname)"; echo "Start time: $(date)"; echo "Log: $LOGFILE"
echo '=========================================='

export PYTHONPATH="$PROJECT_DIR:$LOCAL_SP:${PYTHONPATH:-}"
export MEMRL_OS_BACKEND=local MEMRL_DB_BACKEND=auto MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=3.0 MEMRL_EMBED_GLOBAL_MIN_INTERVAL=3.0 MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_LLB_REFLECTION_PROMPT=v2 MEMRL_LLB_SCRIPT_DETAIL=db_pattern
export MEMRL_LLM_MODEL=gpt-4.1-mini-2025-04-14
# Stable identity: AIS ON_EVICTION retries resume the same checkpoint chain.
export MEMRL_RUN_ID=region-fs-db-gpt41mini-feedback-first-20260805
# Keep the validated corrected Region topology/Q behavior. Only FS abstraction changes.
unset MEMRL_DB_MULTI_AXIS MEMRL_REGION_RETRIEVE_MODE MEMRL_WEIGHTED_REGION_QUOTA || true
unset MEMRL_EXPLICIT_REGION_LAMBDA MEMRL_EXPLICIT_REGION_MIN_RANGE || true
unset MEMRL_REGION_SPLIT_RANGE_FRACTION MEMRL_REGION_SPLIT_ON_RESUME || true
unset MEMRL_FINAL_MEMORY_DEDUP || true
unset MEMRL_REGION_TOPOLOGY_COOLDOWN_SECTIONS || true
export MEMRL_REGION_REGISTER_ON_CREATE=0
export MEMRL_REGION_BACKFILL_ON_RESTORE=0
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
printf '%s\n' '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || { echo 'ERROR: MariaDB install failed' >&2; exit 3; }
printf '%s\n' '[INFO] Installing runtime deps...'
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm concurrent-log-handler mysql-connector-python hdbscan --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/
python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"

CONFIG=/tmp/rl_llb_db_region_fs_structured_full.yaml
python3 - "$CONFIG" <<'PYCFG'
import sys,yaml
from pathlib import Path
src=Path('configs/rl_llb_db_region_fs_hardfix_s3.yaml'); dst=Path(sys.argv[1])
cfg=yaml.safe_load(src.read_text())
cfg['llm']['api_key']='runtime-injected'; cfg['embedding']['api_key']='runtime-injected'
# Fresh chain only. The runner's stable-run auto-resume handles AIS retries.
cfg['memory']['load_from_checkpoint']=False; cfg['memory']['checkpoint_path']=''
cfg['memory']['user_id']='llb_db_region_fs_feedback_first_user'
cfg['experiment']['experiment_name']='llb_db_region_fs_gpt41mini_feedback_first'
cfg['experiment']['num_sections']=10
cfg['experiment']['ckpt_save_every_n_batches']=10; cfg['experiment']['ckpt_max_keep']=3
cfg['experiment']['eval_runs']=1; cfg['experiment']['eval_temperature']=0.0
cfg['experiment']['llb_dedup_by_task_id']=False
cfg['rl_config']['weight_sim']=0.5; cfg['rl_config']['weight_q']=0.5
dst.write_text(yaml.safe_dump(cfg,sort_keys=False))
PYCFG

printf '%s\n' '[INFO] Fresh feedback-first Region: cold-start inline Structured FS, utility-before-membership.'
printf '%s\n' '[INFO] FS contract: independent pool, 4 success minimum, 1 FS, sim>=0.50, abstain without compatible evidence.'
printf '%s\n' "[INFO] MEMRL_RUN_ID=$MEMRL_RUN_ID"
python3 scripts/run_llb_with_rotated_matrix_credentials.py \
  --config "$CONFIG" \
  --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect \
  --region --region_gating_mode additive \
  --propagation_eta 0.12 --shrinkage_confidence_k 3.0 --no_z_norm \
  --explore_schedule '0,4,3,2,2,1,1,1,1,0' \
  --failure_summary_n_slots 1 \
  --failure_summary_independent_pool \
  --failure_summary_min_success 4 \
  --failure_summary_min_similarity 0.50 \
  --failure_summary_min_evidence 1 \
  --failure_summary_db_structured \
  --resume_eval_section -1

echo '=========================================='
echo "End time: $(date)"
echo '=========================================='
