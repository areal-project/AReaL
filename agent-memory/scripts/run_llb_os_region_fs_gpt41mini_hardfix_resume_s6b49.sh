#!/bin/bash
# Resume the legacy LLB OS Region+FS run from the isolated copy of snapshot/6_b49.
# The current code canonicalizes hard memberships from soft argmax on restore,
# rebuilds region summaries, and routes LLB region FS by soft argmax.
set +e
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | cut -d. -f1 | tail -c 8)
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE=${PROJECT_DIR}/logs/llb_os_region_fs_gpt41mini_hardfix_resume_s6b49_${HOST_SHORT}_${TS}.log
mkdir -p ${PROJECT_DIR}/logs
exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "RESUME LLB OS Region+FS hard-membership fix from isolated snapshot/6_b49"
echo "Host: $(hostname)  Start: $(date)  Log: $LOGFILE"
echo "=========================================="

export PYTHONPATH=${PROJECT_DIR}:${PROJECT_DIR}/3rdparty/LifelongAgentBench:${LOCAL_SP}:${PYTHONPATH}
export MEMRL_OS_BACKEND=local
export MEMRL_OS_SANDBOX=1
export MEMRL_UPDATE_MAX_WORKERS=1
# Shared embedding throttle/backoff for all LLB OS gpt-4.1-mini AIS jobs.
# The shared /storage state + common key coordinates separate AIS containers.
export MEMRL_EMBED_THROTTLE=1.0
export MEMRL_EMBED_GLOBAL_MIN_INTERVAL=1.0
export MEMRL_LLB_REQUEST_INTERVAL=1.5
export MEMRL_EMBED_MAX_RETRIES=8
export MEMRL_EMBED_429_BASE_DELAY=10
export MEMRL_EMBED_429_MAX_DELAY=120
export MEMRL_EMBED_RETRY_JITTER=2
export MEMRL_EMBED_RATE_LIMIT_DIR=/storage/openpsi/users/yl/agent-memory/.cache/embedding_rate_limits
export MEMRL_EMBED_RATE_LIMIT_KEY=llb-os-text-embedding-3-large
export MEMRL_LLM_MIN_INTERVAL=3.0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface
# Dedicated run directory cloned from legacy snapshot/6_b49. Never shares writes
# with the still-running legacy AIS task.
export MEMRL_RUN_ID=20260717-regionfs-hardfix-s6b49

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
cd ${PROJECT_DIR}

# Prepare an isolated checkpoint tree inside AIS (the local login node cannot write
# the experiment checkpoint root). Pin exactly the last complete legacy batch snapshot.
CKPT_BASE=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_v2reflect
SRC_RUN=${CKPT_BASE}/exp_llb_os_region_fs_gpt41mini_20260715-114747
DST_RUN=${CKPT_BASE}/exp_llb_os_region_fs_gpt41mini_${MEMRL_RUN_ID}
if [ ! -f "${DST_RUN}/snapshot/6_b49/snapshot_meta.json" ]; then
  echo "[INFO] Cloning legacy snapshot/6_b49 into isolated hard-fix run..."
  mkdir -p "${DST_RUN}/snapshot"
  cp -a "${SRC_RUN}/snapshot/6_b49" "${DST_RUN}/snapshot/6_b49" || exit 21
  cp -a "${SRC_RUN}/llb_batch_progress.json" "${DST_RUN}/llb_batch_progress.json" || true
  sed -i "s|${SRC_RUN}/snapshot/6_b49|${DST_RUN}/snapshot/6_b49|g" \
    "${DST_RUN}/snapshot/6_b49/snapshot_meta.json"
else
  echo "[INFO] Isolated snapshot already exists; preserving it for AIS retry/resume."
fi
echo "[INFO] Isolated snapshot metadata:"
cat "${DST_RUN}/snapshot/6_b49/snapshot_meta.json"

echo "  nsenter=$(command -v nsenter) unshare=$(command -v unshare) MEMRL_RUN_ID=$MEMRL_RUN_ID"
echo '[INFO] Installing runtime deps...'
pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler hdbscan --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/ || echo 'Warning: pip deps failed'
python3 -c "import memos, memrl; print('imports OK')"

echo '[INFO] Resuming fixed Region+FS from isolated snapshot/6_b49...'
python3 run/run_llb.py --config configs/rl_llb_os_region_gpt41mini.yaml \
  --region --region_k 8 --region_gating_mode additive \
  --failure_summary_n_slots 2 --failure_summary_k 10 \
  --explore_schedule 0,2,2,1,1,1,1,0,0,0

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
