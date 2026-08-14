#!/usr/bin/env bash
# Resume LLB-DB trajectory + multi-axis with adaptive variance split from the existing stable run.
set -euo pipefail
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | sed 's/^gpulingjun//; s/\..*$//')
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$PROJECT_DIR/logs/llb_db_region_fs_trajectory_multiaxis_gpt41mini_${HOST_SHORT}_${TS}.log"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$PROJECT_DIR"

echo '=========================================='
echo 'LLB DB Region+FS trajectory + multi-axis Q + adaptive split=0.10 (resume)'
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Log: $LOGFILE"
echo '=========================================='

export PYTHONPATH="$PROJECT_DIR:$LOCAL_SP:${PYTHONPATH:-}"
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=auto
export MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=3.0
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=3.0
export MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_LLB_REFLECTION_PROMPT=v2
export MEMRL_LLB_SCRIPT_DETAIL=db_pattern
export MEMRL_LLM_MODEL=gpt-4.1-mini-2025-04-14
export MEMRL_DB_MULTI_AXIS=1
export MEMRL_REGION_SPLIT_RANGE_FRACTION=0.10
export MEMRL_REGION_SPLIT_ON_RESUME=1
export MEMRL_REGION_RETRIEVE_MODE=weighted_quota
export MEMRL_WEIGHTED_REGION_QUOTA=1
export MEMRL_WEIGHTED_REGION_MIN_SIM=0.55
export MEMRL_WEIGHTED_REGION_UTILITY_MARGIN=0.05
export MEMRL_WEIGHTED_REGION_MIN_COUNT=30
export MEMRL_RUN_ID=region-fs-db-gpt41mini-trajectory-multiaxis-20260724
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || { echo 'ERROR: MariaDB install failed' >&2; exit 3; }
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm concurrent-log-handler mysql-connector-python hdbscan --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/
python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"

CONFIG=/tmp/rl_llb_db_region_fs_trajectory_multiaxis.yaml
python3 - "$CONFIG" <<'PYCFG'
import sys
from pathlib import Path
import yaml
src=Path('configs/rl_llb_db_region_fs_hardfix_s3.yaml')
dst=Path(sys.argv[1])
cfg=yaml.safe_load(src.read_text())
cfg['llm']['api_key']='runtime-injected'
cfg['embedding']['api_key']='runtime-injected'
cfg['memory']['build_strategy']='trajectory'
cfg['memory']['load_from_checkpoint']=False
cfg['memory']['checkpoint_path']=''
cfg['memory']['user_id']='llb_db_region_fs_trajectory_multiaxis_user'
cfg['experiment']['experiment_name']='llb_db_region_fs_gpt41mini_trajectory_multiaxis'
cfg['experiment']['ckpt_save_every_n_batches']=10
cfg['experiment']['ckpt_max_keep']=3
cfg['experiment']['eval_runs']=1
cfg['experiment']['eval_temperature']=0.0
dst.write_text(yaml.safe_dump(cfg,sort_keys=False))
PYCFG

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
