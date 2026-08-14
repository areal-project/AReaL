#!/usr/bin/env bash
# Fresh corrected proceduralization DB early-progressive-topology screening run.
# Only topology policy changes: best split x1, evidence/child gates, merge x1.
set -euo pipefail

PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | sed 's/^gpulingjun//; s/\..*$//')
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$PROJECT_DIR/logs/llb_db_region_fs_early_progressive_gpt41mini_${HOST_SHORT}_${TS}.log"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$PROJECT_DIR"

echo '=========================================='
echo 'LLB DB corrected Region early-progressive-topology E1-E3 screening'
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Log: $LOGFILE"
echo '=========================================='

export PYTHONPATH="$PROJECT_DIR:$LOCAL_SP:${PYTHONPATH:-}"
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=auto
# DB-specific progressive topology. Everything else matches corrected Region.
export MEMRL_REGION_CLUSTER_INIT_STEP=180
export MEMRL_REGION_TOPOLOGY_COOLDOWN_SECTIONS=0
export MEMRL_REGION_SPLIT_RANGE_FRACTION=0.12
export MEMRL_REGION_MAX_VARIANCE_SPLITS_PER_EPOCH=1
export MEMRL_REGION_SPLIT_MIN_EFFECTIVE_EVIDENCE=120
export MEMRL_REGION_PROGRESSIVE_BEST_SPLIT=1
export MEMRL_REGION_MAX_MERGES_PER_EPOCH=1
export MEMRL_REGION_SPLIT_MIN_CHILD_SIZE=25
export MEMRL_REGION_PROTECT_NEW_SPLIT_CHILDREN=1
unset MEMRL_DB_MULTI_AXIS || true
unset MEMRL_REGION_SPLIT_ON_RESUME || true
unset MEMRL_REGION_RETRIEVE_MODE || true
unset MEMRL_WEIGHTED_REGION_QUOTA || true
unset MEMRL_RETRIEVAL_AUDIT_PATH || true
unset MEMRL_FINAL_MEMORY_DEDUP || true
export MEMRL_UPDATE_MAX_WORKERS=1
# Next fresh run: enforce a shared cross-process 3s interval for every
# embedding-backed operation (retrieval plus text_mem.add), not merely a local
# fallback throttle. Current running job is intentionally unaffected.
export MEMRL_EMBED_THROTTLE=5.0
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=5.0
export MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_LLB_REFLECTION_PROMPT=v2
export MEMRL_LLB_SCRIPT_DETAIL=db_pattern
export MEMRL_LLM_MODEL=gpt-4.1-mini-2025-04-14
export MEMRL_RUN_ID="region-fs-db-gpt41mini-early-progressive-topology-20260730"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
printf '%s\n' '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed'
printf '%s\n' '[INFO] Installing runtime deps...'
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm concurrent-log-handler mysql-connector-python hdbscan --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/
python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"

# Generated config is node-local: no credentials and no output are written to
# the permanent checkpoint tree except the new experiment's own snapshots.
CONFIG=/tmp/rl_llb_db_region_fs_early_progressive.yaml
python3 - "$CONFIG" <<'PYCFG'
import sys
from pathlib import Path
import yaml
src=Path('configs/rl_llb_db_region_fs_hardfix_s3.yaml')
dst=Path(sys.argv[1])
cfg=yaml.safe_load(src.read_text())
# Credentials are injected into the in-memory MempConfig by the wrapper.
cfg['llm']['api_key']='runtime-injected'
cfg['embedding']['api_key']='runtime-injected'
# Critical: no old Region checkpoint/polluted split posterior can enter this run.
cfg['memory']['load_from_checkpoint']=False
cfg['memory']['checkpoint_path']=''
cfg['memory']['user_id']='llb_db_region_fs_early_progressive_user'
cfg['experiment']['experiment_name']='llb_db_region_fs_gpt41mini_early_progressive'
cfg['experiment']['ckpt_save_every_n_batches']=10
cfg['experiment']['ckpt_max_keep']=3
cfg['experiment']['eval_runs']=1
cfg['experiment']['eval_temperature']=0.0
cfg['experiment']['num_sections']=3
dst.write_text(yaml.safe_dump(cfg,sort_keys=False))
PYCFG

printf '%s\n' '[INFO] Fresh early progressive topology: init_step=180, split_fraction=.12, max_split=1, max_merge=1, evidence>=120, child>=25.'
printf '%s\n' "[INFO] MEMRL_RUN_ID=$MEMRL_RUN_ID"
python3 scripts/run_llb_with_rotated_matrix_credentials.py \
  --config "$CONFIG" \
  --output_dir /storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect \
  --region --region_gating_mode additive \
  --propagation_eta 0.12 \
  --shrinkage_confidence_k 3.0 \
  --no_z_norm \
  --explore_schedule '0,4,3,2,2,1,1,1,1,0' \
  --failure_summary_n_slots 2 \
  --resume_eval_section -1

echo '=========================================='
echo "End time: $(date)"
echo '=========================================='
