#!/usr/bin/env bash
# Clean OS Region+FS replacement: proceduralization, fresh state, correct split evidence,
# and conservative progressive topology after initial E1 clustering.
set -euo pipefail
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | sed 's/^gpulingjun//;s/\..*$//')
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$PROJECT_DIR/logs/llb_os_region_fs_splitprior_topology_clean_gpt41mini_${HOST_SHORT}_${TS}.log"
mkdir -p "$PROJECT_DIR/logs"
exec > >(tee -a "$LOGFILE") 2>&1
cd "$PROJECT_DIR"
# hdbscan is imported lazily at the first utility clustering step (E1 batch 60
# in this schedule). Install it in an isolated job-local target and make the
# target precede source imports; do not depend on a transient shared venv.
REGION_DEPS=/tmp/llb_os_regionfs_clean_deps
mkdir -p "$REGION_DEPS"
command -v pip >/dev/null
pip --version
pip install --disable-pip-version-check --target "$REGION_DEPS" \
  'hdbscan==0.8.40' -i https://pypi.antfin-inc.com/simple/
export PYTHONPATH="$REGION_DEPS:$PROJECT_DIR:$PROJECT_DIR/3rdparty/LifelongAgentBench:$LOCAL_SP:${PYTHONPATH:-}"
python3 - <<'PYHDBSCAN'
import hdbscan
print('[PRECHECK] hdbscan import OK:', getattr(hdbscan, '__version__', 'installed'))
PYHDBSCAN
export MEMRL_OS_BACKEND=local MEMRL_OS_SANDBOX=1 MEMRL_UPDATE_MAX_WORKERS=1
export MEMRL_LLB_REQUEST_INTERVAL=2.0 MEMRL_LLM_MIN_INTERVAL=1.0
export MEMRL_EMBED_THROTTLE=2.0 MEMRL_EMBED_GLOBAL_MIN_INTERVAL=2.0 MEMRL_EMBED_MAX_RETRIES=8
export MEMRL_EMBED_429_BASE_DELAY=10 MEMRL_EMBED_429_MAX_DELAY=120 MEMRL_EMBED_RETRY_JITTER=2
export MEMRL_EMBED_RATE_LIMIT_DIR=/storage/openpsi/users/yl/agent-memory/.cache/embedding_rate_limits
export MEMRL_EMBED_RATE_LIMIT_KEY=llb-os-regionfs-clean-gpt41mini-text-embedding-3-large
export MEMRL_MATRIX_CREDENTIAL_CONFIG=/storage/openpsi/users/yl/cfuse/rq3_v11_clean_20260719/config_multisurface_isolated.yaml
export HF_ENDPOINT=https://hf-mirror.com HF_HOME=/tmp/huggingface
# Source-conserving split + stability controls. Initial cluster occurs at the end
# of E1 (global step 300); one full cooldown section prevents immediate churn.
# Afterwards only the single best-supported split and at most one merge may occur.
export MEMRL_REGION_SPLIT_EVIDENCE_MIGRATION_MODE=soft_source_conserving
export MEMRL_REGION_TOPOLOGY_UPDATES_ENABLED=1
export MEMRL_REGION_CLUSTER_INIT_STEP=300
export MEMRL_REGION_TOPOLOGY_COOLDOWN_SECTIONS=1
export MEMRL_REGION_EVIDENCE_SHARPEN_ALPHA=2.0 MEMRL_REGION_SPLIT_RANGE_FRACTION=0.15
export MEMRL_REGION_MAX_VARIANCE_SPLITS_PER_EPOCH=1 MEMRL_REGION_SPLIT_MIN_EFFECTIVE_EVIDENCE=80
export MEMRL_REGION_PROGRESSIVE_BEST_SPLIT=1 MEMRL_REGION_MAX_MERGES_PER_EPOCH=1
export MEMRL_REGION_SPLIT_MIN_CHILD_SIZE=20 MEMRL_REGION_PROTECT_NEW_SPLIT_CHILDREN=1
: "${MEMRL_RUN_ID:?MEMRL_RUN_ID must be supplied by the job launcher}"
source "$PROJECT_DIR/scripts/llb_os_auto_resume.sh" llb_os_region_fs_splitprior_topology_clean_gpt41mini
python3 scripts/test_region_split_prior.py
python3 scripts/test_region_member_fs_argmax.py
CONFIG=/tmp/rl_llb_os_region_fs_splitprior_topology_clean.yaml
python3 - "$CONFIG" <<'PYCFG'
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('configs/rl_llb_os_region_gpt41mini.yaml').read_text())
cfg['llm']['api_key'] = 'runtime-injected'
cfg['embedding']['api_key'] = 'runtime-injected'
cfg['memory'].update(build_strategy='proceduralization', load_from_checkpoint=False, checkpoint_path='', user_id='llb_os_region_fs_splitprior_topology_clean_user')
cfg['experiment'].update(experiment_name='llb_os_region_fs_splitprior_topology_clean_gpt41mini', region_cluster_init_step=300, ckpt_save_every_n_batches=10, ckpt_max_keep=3, eval_runs=1, eval_temperature=0.0)
cfg['rl_config'].update(tau=0.60, weight_sim=0.45, weight_q=0.55)
Path(sys.argv[1]).write_text(yaml.safe_dump(cfg, sort_keys=False))
PYCFG
python3 - "$CONFIG" <<'PYCHECK'
import os, sys, yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text())
assert cfg['experiment']['task'] == 'os'
assert cfg['memory']['build_strategy'] == 'proceduralization'
assert cfg['memory']['load_from_checkpoint'] is False
assert cfg['experiment']['region_cluster_init_step'] == 300
assert os.environ['MEMRL_REGION_SPLIT_EVIDENCE_MIGRATION_MODE'] == 'soft_source_conserving'
assert os.environ['MEMRL_REGION_TOPOLOGY_UPDATES_ENABLED'] == '1'
assert os.environ['MEMRL_REGION_TOPOLOGY_COOLDOWN_SECTIONS'] == '1'
assert os.environ['MEMRL_REGION_PROGRESSIVE_BEST_SPLIT'] == '1'
assert cfg['rl_config']['tau'] == 0.60
assert cfg['rl_config']['weight_sim'] == 0.45
assert cfg['rl_config']['weight_q'] == 0.55
print('[PRECHECK] clean OS Region+FS proceduralization; no legacy checkpoint; source-conserving split; E1-late progressive topology; selective FS')
PYCHECK
python3 scripts/run_llb_region_clean_with_credentials.py --config "$CONFIG" --region --region_k 8 --region_gating_mode additive --failure_summary_n_slots 1 --failure_summary_k 10 --failure_summary_independent_pool --failure_summary_min_success 4 --failure_summary_min_similarity 0.50 --propagation_eta 0.03 --shrinkage_confidence_k 2.5 --no_z_norm --explore_schedule '0,1,1,1,0,0,0,0,0,0' --resume_eval_section -1
