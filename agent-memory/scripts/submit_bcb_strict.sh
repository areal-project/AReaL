#!/bin/bash
# ============================================================================
# BCB 实验 aistudio 提交脚本
# 用法: bash scripts/submit_bcb.sh [experiment_name]
#   experiment_name: memp | passk10 | rag | selfrag | region_leaf
# ============================================================================
set -euo pipefail

EXPERIMENT="${1:-memp}"
RUNNER_SCRIPT="/storage/openpsi/users/yl/agent-memory/MemRL/scripts/aistudio_bcb_runner_strict.sh"
SUBMIT_SCRIPT="/storage/openpsi/users/yl/agent-memory/submit.sh"
OUTPUT_BASE="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/bigcodebench"

case "$EXPERIMENT" in
    memp)
        export JOB_NAME="yl-bcb-memp"
        export BCB_CONFIG="configs/rl_bcb_config.memp_local.yaml"
        export BCB_OUTPUT_DIR="${OUTPUT_BASE}/deepseek_v3_memp"
        export BCB_EXTRA_ARGS="--n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last"
        ;;
    passk10)
        export JOB_NAME="yl-bcb-passk10"
        export BCB_CONFIG="configs/rl_bcb_config.passk10_local.yaml"
        export BCB_OUTPUT_DIR="${OUTPUT_BASE}/deepseek_v3_passk10"
        export BCB_EXTRA_ARGS=""
        # pass@k has its own resume (baseline_passk/results.jsonl); must NOT use
        # the epoch-snapshot auto-resume (would read stale/wrong run snapshots).
        export BCB_NO_AUTORESUME=1
        ;;
    rag)
        export JOB_NAME="yl-bcb-rag"
        export BCB_CONFIG="configs/rl_bcb_config.rag_local.yaml"
        export BCB_OUTPUT_DIR="${OUTPUT_BASE}/deepseek_v3_rag"
        export BCB_EXTRA_ARGS="--n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last"
        ;;
    selfrag)
        export JOB_NAME="yl-bcb-selfrag"
        export BCB_CONFIG="configs/rl_bcb_config.selfrag_local.yaml"
        export BCB_OUTPUT_DIR="${OUTPUT_BASE}/deepseek_v3_selfrag"
        export BCB_EXTRA_ARGS="--n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last"
        ;;
    region_leaf)
        export JOB_NAME="yl-bcb-reg-leaf"
        export BCB_CONFIG="configs/rl_bcb_config.region_leaf_local.yaml"
        export BCB_OUTPUT_DIR="${OUTPUT_BASE}/deepseek_v3_region_leaf"
        export BCB_EXTRA_ARGS="--n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last"
        ;;
    mem0)
        export JOB_NAME="yl-bcb-mem0"
        export BCB_CONFIG="configs/rl_bcb_config.mem0_local.yaml"
        export BCB_OUTPUT_DIR="${OUTPUT_BASE}/deepseek_v3_mem0"
        # --mem0 uses Mem0MemoryService; mem0's LLM reads cfg.llm.base_url, which the
        # runner rewrites to the worker's main LLM in dual-node. DeepSeek-V3 is not a
        # thinking model (no reasoning-parser), so mem0 extraction can safely share the
        # action LLM server (unlike qwen3.6 which needs a separate extractor server).
        export BCB_EXTRA_ARGS="--mem0 --mem0_infer true --n_eval_runs 3 --eval_temperature 0.2 --multi_eval_epochs last"
        # SGLang/Qwen3-Embedding rejects OpenAI's `dimensions` field as an
        # unsupported Matryoshka resize. Request the model-native output.
        export MEMRL_MEM0_OMIT_EMBED_DIMENSIONS="${MEMRL_MEM0_OMIT_EMBED_DIMENSIONS:-1}"
        # mem0 metadata semantics changed; never auto-resume an older/incompatible
        # memory snapshot unless the caller explicitly supplies BCB_RESUME_FROM.
        export BCB_NO_AUTORESUME="${BCB_NO_AUTORESUME:-1}"
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Available: memp | passk10 | rag | selfrag | region_leaf | mem0"
        exit 1
        ;;
esac

# Optional resume from a checkpoint snapshot (survives low-priority preemption).
# Completed epoch: BCB_RESUME_FROM=.../epoch5/snapshot/5 BCB_RESUME_EPOCH=5
# resumes at E6. Mid-epoch: additionally set BCB_RESUME_STEP=<sample index>;
# this resumes that epoch and reloads samples_partial.jsonl instead of restarting.
if [ -n "${BCB_RESUME_FROM:-}" ]; then
    BCB_EXTRA_ARGS="${BCB_EXTRA_ARGS} --resume_from ${BCB_RESUME_FROM}"
    [ -n "${BCB_RESUME_EPOCH:-}" ] && BCB_EXTRA_ARGS="${BCB_EXTRA_ARGS} --resume_epoch ${BCB_RESUME_EPOCH}"
    [ -n "${BCB_RESUME_STEP:-}" ] && BCB_EXTRA_ARGS="${BCB_EXTRA_ARGS} --resume_step ${BCB_RESUME_STEP}"
    export BCB_EXTRA_ARGS
    echo "[submit] RESUME enabled: from=${BCB_RESUME_FROM} epoch=${BCB_RESUME_EPOCH:-auto} step=${BCB_RESUME_STEP:-next-epoch}"
fi

export BCB_EPOCHS="${BCB_EPOCHS:-10}"
# Dual-node by default: worker (8 GPU) serves DeepSeek-V3 TP=8, master serves the
# embedder standalone (GPU0) + runs run_bcb. This avoids the GPU0 memory
# contention that made the single-node layout OOM on the embedder.
# Set WORKER_NUM=0 to force the old single-node layout (embedder shares GPU0).
export WORKER_NUM="${WORKER_NUM:-1}"
# Use our own submit template so master can request just 1 GPU (embedding only)
# instead of 8, saving 7 idle cards. Falls back to public template only if unset.
export SUBMIT_TEMPLATE="${SUBMIT_TEMPLATE:-/storage/openpsi/users/yl/agent-memory/MemRL/scripts/submit_template_bcb_strict.py}"
# Dual-node master needs only 1 card (embedding); single-node needs all 8.
if [ "${WORKER_NUM}" -ge 1 ]; then
    export MASTER_GPU_NUM="${MASTER_GPU_NUM:-1}"
else
    export MASTER_GPU_NUM="${MASTER_GPU_NUM:-8}"
fi

LOG_DIR="/storage/openpsi/users/yl/agent-memory/MemRL/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# Per-node log: include POD_NAME so master and worker write SEPARATE files. This
# is essential for dual-node debugging — a shared file interleaves both nodes and
# hides which node failed first. $POD_NAME is resolved inside the container.
LOG_FILE_BASE="${LOG_DIR}/aistudio_${EXPERIMENT}_${TIMESTAMP}"
export JOB_NAME="${JOB_NAME}-${TIMESTAMP}"
export JOB_COMMAND="bash -c 'exec > >(tee -a ${LOG_FILE_BASE}_\${POD_NAME:-node}.log) 2>&1; BCB_CONFIG=\"${BCB_CONFIG}\" BCB_EPOCHS=\"${BCB_EPOCHS}\" BCB_OUTPUT_DIR=\"${BCB_OUTPUT_DIR}\" BCB_EXTRA_ARGS=\"${BCB_EXTRA_ARGS}\" BCB_NO_AUTORESUME=\"${BCB_NO_AUTORESUME:-0}\" MEMRL_MEM0_OMIT_EMBED_DIMENSIONS=\"${MEMRL_MEM0_OMIT_EMBED_DIMENSIONS:-0}\" bash ${RUNNER_SCRIPT}'"
LOG_FILE="${LOG_FILE_BASE}"  # base path; actual files get _<pod>.log suffix

echo "=========================================="
echo "Submitting: $EXPERIMENT"
echo "  Job name: $JOB_NAME"
echo "  Config:   $BCB_CONFIG"
echo "  Output:   $BCB_OUTPUT_DIR"
echo "  Extra:    $BCB_EXTRA_ARGS"
echo "  Log:      $LOG_FILE"
echo "=========================================="

cd /storage/openpsi/users/yl/agent-memory
bash submit.sh
