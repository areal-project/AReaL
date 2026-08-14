#!/usr/bin/env bash
# Strict read-only LLB-DB checkpoint validation. Permanent checkpoints are only
# copied to node-local /tmp and never used as an output location.
set -euo pipefail
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
SOURCE_SNAPSHOT=${SOURCE_SNAPSHOT:?SOURCE_SNAPSHOT is required}
SOURCE_CONFIG=${SOURCE_CONFIG:?SOURCE_CONFIG is required}
EVAL_SECTION=${EVAL_SECTION:?EVAL_SECTION is required}
EVAL_LABEL=${EVAL_LABEL:?EVAL_LABEL is required}
HOST_SHORT=$(hostname | sed 's/^gpulingjun//; s/\..*$//')
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$PROJECT_DIR/logs/llb_db_${EVAL_LABEL}_readonly_valeval_${HOST_SHORT}_${TS}.log"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$PROJECT_DIR"
echo "READONLY_CHECKPOINT_EVAL label=$EVAL_LABEL source=$SOURCE_SNAPSHOT section=$EVAL_SECTION"
[[ -f "$SOURCE_SNAPSHOT/snapshot_meta.json" && -d "$SOURCE_SNAPSHOT/cube" && -d "$SOURCE_SNAPSHOT/qdrant" ]] || { echo 'ERROR: unhealthy source snapshot' >&2; exit 2; }
export PYTHONPATH="$PROJECT_DIR:$LOCAL_SP:${PYTHONPATH:-}"
export MEMRL_OS_BACKEND=local MEMRL_DB_BACKEND=auto MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_EMBED_THROTTLE=0.8 MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_LLM_MODEL=gpt-4.1-mini-2025-04-14
export MEMRL_LLB_REFLECTION_PROMPT=v2 MEMRL_LLB_SCRIPT_DETAIL=db_pattern
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/tmp/huggingface
export MEMRL_EVAL_ONLY_SECTION="$EVAL_SECTION"
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || { echo 'ERROR: mariadb install failed' >&2; exit 3; }
command -v mysqld >/dev/null || { echo 'ERROR: mysqld unavailable after install' >&2; exit 3; }
VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
pip install mem0ai 'chonkie==1.2.1' tensorboard pandas tqdm concurrent-log-handler mysql-connector-python --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/
python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"
WORKDIR=$(mktemp -d /tmp/llb_db_readonly_eval.XXXXXX)
trap 'rm -rf "$WORKDIR"' EXIT
LOCAL_SNAPSHOT="$WORKDIR/snapshot"
LOCAL_CONFIG="$WORKDIR/config.yaml"
LOCAL_OUTPUT="$WORKDIR/out"
cp -a "$SOURCE_SNAPSHOT" "$LOCAL_SNAPSHOT"
python3 - "$SOURCE_CONFIG" "$LOCAL_CONFIG" "$LOCAL_SNAPSHOT" "$LOCAL_OUTPUT" <<'PYCFG'
import sys
from pathlib import Path
import yaml
src, dst, snapshot, output=map(Path,sys.argv[1:])
cfg=yaml.safe_load(src.read_text())
cfg['llm']['api_key']='runtime-injected'
cfg['embedding']['api_key']='runtime-injected'
cfg['llm']['model']='gpt-4.1-mini-2025-04-14'
cfg['memory']['load_from_checkpoint']=True
cfg['memory']['checkpoint_path']=str(snapshot)
cfg['memory']['user_id']='readonly_checkpoint_eval'
cfg['experiment']['output_dir']=str(output)
cfg['experiment']['save_trajectories']=False
cfg['experiment']['save_memories']=False
cfg['experiment']['ckpt_save_every_n_batches']=0
cfg['experiment']['ckpt_max_keep']=1
cfg['experiment']['eval_runs']=1
cfg['experiment']['eval_temperature']=0.0
dst.write_text(yaml.safe_dump(cfg,sort_keys=False))
PYCFG
python3 scripts/run_llb_with_rotated_matrix_credentials.py --config "$LOCAL_CONFIG" --output_dir "$LOCAL_OUTPUT"
echo "READONLY_CHECKPOINT_EVAL_COMPLETE label=$EVAL_LABEL"
