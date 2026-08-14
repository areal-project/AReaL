#!/bin/bash
# ============================================================================
# 通用 BCB 实验 runner — 用于 aistudio 容器内直接执行
# 通过环境变量传入参数（由 submit_bcb.sh 设置）
#
# 容器环境: areal-runtime:dev-sglang-20260401
#   - python3 = /AReaL/.venv/bin/python (3.12.3)
#   - pip = /usr/local/bin/pip (装到 system dist-packages, venv 看不到)
#   - 解法: pip install --target $VENV_SP
#   - 外网不通, 必须用 -i https://pypi.antfin-inc.com/simple/
# ============================================================================
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
BCB_CONFIG="${BCB_CONFIG:-configs/rl_bcb_config.memp_local.yaml}"
BCB_EPOCHS="${BCB_EPOCHS:-10}"
BCB_EXTRA_ARGS="${BCB_EXTRA_ARGS:-}"
BCB_OUTPUT_DIR="${BCB_OUTPUT_DIR:?BCB_OUTPUT_DIR must be set}"
BCB_MODEL_PATH="${BCB_MODEL_PATH:-/storage/openpsi/users/yl/models/DeepSeek-V3-mtp1}"
BCB_EMBED_MODEL_PATH="${BCB_EMBED_MODEL_PATH:-/storage/openpsi/models/Qwen3-Embedding-8B}"
BCB_LLM_PORT="${BCB_LLM_PORT:-8000}"
BCB_EMBED_PORT="${BCB_EMBED_PORT:-8001}"

VENV_SP="/AReaL/.venv/lib/python3.12/site-packages"
PIP="/usr/local/bin/pip"

VLLM_PID=""
EMBED_PID=""

cleanup() {
    # Preserve the real exit code: the trap fires on EXIT and the commands below
    # (echo/kill/wait) would otherwise overwrite $? and report success on failure.
    local rc=$?
    echo "[INFO] Cleaning up... (exit code=$rc)"
    # Signal the dual-node worker (if any) to stop serving and exit, so the whole
    # job terminates instead of the worker hanging until platform timeout.
    [ -n "${DONE_FLAG:-}" ] && touch "$DONE_FLAG" 2>/dev/null || true
    [ -n "$EMBED_PID" ] && kill $EMBED_PID 2>/dev/null && wait $EMBED_PID 2>/dev/null
    [ -n "$VLLM_PID" ] && kill $VLLM_PID 2>/dev/null && wait $VLLM_PID 2>/dev/null
    echo "[INFO] Done. End: $(date)"
    exit $rc
}
trap cleanup EXIT

echo "=========================================="
echo "BCB Runner (aistudio) | Start: $(date)"
echo "Config: $BCB_CONFIG"
echo "Epochs: $BCB_EPOCHS"
echo "Output: $BCB_OUTPUT_DIR"
echo "Extra:  $BCB_EXTRA_ARGS"
echo "=========================================="

# ---------------------------------------------------------------------------
# 1. Install dependencies
# ---------------------------------------------------------------------------
cd "$MEMRL_DIR"

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
# PYTHONPATH order matters:
#   1. $MEMRL_DIR first  -> `import memrl` always resolves to the live source tree
#      (pip `-e . --target` does NOT create a real editable link, so we cannot rely on it)
#   2. local site-packages -> memos 2.0.13 (short dep chain, importable as-is)
export PYTHONPATH=${MEMRL_DIR}:/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages:${PYTHONPATH:-}

find . -name '*.pyc' -delete 2>/dev/null || true
find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "[DEBUG] which python: $(which python)"
echo "[DEBUG] python version: $(python --version)"

# memrl itself is imported from $MEMRL_DIR via PYTHONPATH (see above); we only need
# to install its metadata/entry points and pull in third-party deps. Install deps
# (no memrl editable-target trick, which silently produces a non-editable copy).
#
# WORKER SKIPS pip entirely: it only runs the SGLang main LLM (sglang ships in the
# image), needs none of memrl/mem0ai/BCB deps, and two nodes writing the same
# CPFS `--target` concurrently corrupts site-packages. Detect role early via
# POD_NAME/RANK (before the later NODE_RANK computation) so only master installs.
IS_WORKER_EARLY=0
if [ -n "${RANK:-}" ]; then
    [ "${RANK}" != "0" ] && IS_WORKER_EARLY=1
elif [[ "${POD_NAME:-}" =~ [Ww]orker ]]; then
    IS_WORKER_EARLY=1
fi

if [ "$IS_WORKER_EARLY" = "0" ]; then
    $PIP install mem0ai 'chonkie==1.2.1' tensorboard hdbscan pandas tqdm \
        concurrent-log-handler ollama \
        --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ --quiet 2>&1 | tail -3 || true
    # pip failures above are non-fatal here: the import checks below are the real gate.
    python -c 'import memrl; print("[OK] memrl:", memrl.__file__)' || { echo '[FATAL] memrl import failed'; exit 1; }
    python -c 'from memos.mem_os.main import MOS; print("[OK] memos")' || { echo '[FATAL] memos import failed'; exit 1; }
else
    echo "[WORKER] skipping pip install and memrl/memos import (worker only serves SGLang LLM)."
fi

# ---------------------------------------------------------------------------
# 2. Detect GPU and configure multi-node
# ---------------------------------------------------------------------------
fuser -k ${BCB_LLM_PORT}/tcp 2>/dev/null || true
fuser -k ${BCB_EMBED_PORT}/tcp 2>/dev/null || true
sleep 2

python -c "import sglang; print('[OK] sglang:', sglang.__version__)" || { echo '[FATAL] sglang import failed'; exit 1; }

echo "[DEBUG] === reached GPU detection ==="
# Detect GPU. Every command substitution must tolerate non-zero exit under
# `set -euo pipefail` (torch CUDA init / nvidia-smi can fail on some hosts and
# would otherwise silently kill the whole script before any diagnostics print).
# Print raw nvidia-smi first (keep stderr) so we can see the real cause if the
# GPU is not visible instead of losing it to 2>/dev/null.
echo "[DEBUG] === nvidia-smi (raw, stderr kept) ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv || echo "  [WARN] nvidia-smi failed rc=$?"
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
GPU_MEM_GB=$( { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true; } | head -n1 | awk '{print int($1/1024)}' || true)
GPU_MEM_GB=${GPU_MEM_GB:-0}
echo "[DEBUG] GPU: $GPU_NAME x $GPU_COUNT (${GPU_MEM_GB}GB/card)"
echo "[DEBUG] HOSTNAME=$(hostname 2>/dev/null || echo unknown)"
echo "[DEBUG] === network interfaces (ipv4) ==="
ip -4 -o addr show 2>/dev/null | awk '{print "  "$2" "$4}' || true
echo "[DEBUG] === IB devices ==="
ibv_devices 2>/dev/null || echo "  ibv_devices unavailable"
ibdev2netdev 2>/dev/null || echo "  ibdev2netdev unavailable"
echo "[DEBUG] All rank/node env:"
python -c "import os; [print(f'  {k}={v}') for k,v in sorted(os.environ.items()) if any(x in k.upper() for x in ['RANK','NODE','WORKER','MASTER','WORLD','POD','INDEX'])]"

# Determine node rank. Prefer the platform-provided RANK; otherwise derive from
# POD_NAME. For workers, the trailing ordinal disambiguates 3+ nodes
# (worker-0 -> rank 1, worker-1 -> rank 2, ...); master is always rank 0.
if [ -n "${RANK:-}" ]; then
    NODE_RANK="$RANK"
elif [[ "${POD_NAME:-}" =~ [Mm]aster ]]; then
    NODE_RANK=0
elif [[ "${POD_NAME:-}" =~ [Ww]orker ]]; then
    WORKER_ORD=$(printf '%s' "${POD_NAME}" | grep -oE '[0-9]+$' || echo 0)
    NODE_RANK=$(( WORKER_ORD + 1 ))
else
    NODE_RANK=0
fi
echo "[DEBUG] Final NODE_RANK=$NODE_RANK (POD_NAME=${POD_NAME:-unset}, RANK=${RANK:-unset})"

# Decide topology PRIMARILY from the number of nodes the platform actually
# allocated (WORLD_SIZE, set by AIStudio = 1 + WORKER_NUM), NOT from a VRAM
# probe that can silently fail. VRAM is only a secondary hint for logging.
# Rationale: submit with WORKER_NUM=0 -> WORLD_SIZE=1 -> single node TP=8, which
# fits DeepSeek-V3 FP8 on 8x140GB L20X/H200 and avoids cross-node NCCL entirely.
ALLOC_NODES="${WORLD_SIZE:-1}"
case "$ALLOC_NODES" in ''|*[!0-9]*) ALLOC_NODES=1 ;; esac

# Shared dir for cross-node coordination (worker publishes its IP here, master
# writes a DONE flag here). Keyed by job id (from POD_NAME) so concurrent jobs
# don't collide. DNS between nodes does NOT work on this cluster (verified), so
# nodes rendezvous via POD_IP written to this CPFS-shared file.
JOBID=$(printf '%s' "${POD_NAME:-unknown}" | grep -oE 'aistudio-[a-z0-9]+' | head -n1 || echo "job")
JOBID=${JOBID:-job}
SHARE_DIR="/storage/openpsi/users/yl/agent-memory/MemRL/logs/dualnode_share/${JOBID}"
WORKER_IP_FILE="$SHARE_DIR/worker_ip.txt"
DONE_FLAG="$SHARE_DIR/DONE"
mkdir -p "$SHARE_DIR" 2>/dev/null || true

TP_SIZE=8
if [ "$ALLOC_NODES" -le 1 ]; then
    # Single-node fallback: main LLM + embedding share the 8 cards; embedding
    # squeezes onto GPU0 (memory-tight, see notes below).
    NNODES=1
    MEM_FRACTION="${BCB_MEM_FRACTION:-0.65}"
    echo "[INFO] Single node (WORLD_SIZE=$ALLOC_NODES, GPU=$GPU_NAME ${GPU_MEM_GB}GB). TP=$TP_SIZE, mem-fraction=$MEM_FRACTION."
else
    # Dual-node: worker runs the main LLM TP=8 alone (full 8 cards, no embedding
    # contention); master runs the embedder standalone + the experiment. Nodes
    # talk over HTTP (not NCCL), so no cross-node NCCL config is needed.
    NNODES="$ALLOC_NODES"
    MEM_FRACTION="${BCB_MEM_FRACTION:-0.85}"
    echo "[INFO] Dual-node (WORLD_SIZE=$ALLOC_NODES, GPU=$GPU_NAME ${GPU_MEM_GB}GB). Worker=main LLM TP=$TP_SIZE, master=embedding+run_bcb. SHARE_DIR=$SHARE_DIR"
fi

# ---------------------------------------------------------------------------
# 3. WORKER node (dual-node only): serve the main LLM standalone (TP=8) over the
#    network, publish own IP so master can find it (DNS doesn't work here), then
#    stay alive until master signals DONE.
# ---------------------------------------------------------------------------
if [ "$NNODES" -gt 1 ] && [ "$NODE_RANK" != "0" ]; then
    SELF_IP="${POD_IP:-${NODE_IP:-}}"
    echo "[WORKER] rank=$NODE_RANK IP=$SELF_IP. Starting main LLM (TP=$TP_SIZE) on 0.0.0.0:${BCB_LLM_PORT}."
    python -u -m sglang.launch_server \
        --model-path "${BCB_MODEL_PATH}" \
        --served-model-name deepseek-ai/DeepSeek-V3 \
        --tp $TP_SIZE \
        --trust-remote-code \
        --host 0.0.0.0 --port ${BCB_LLM_PORT} \
        --context-length 32768 \
        --mem-fraction-static ${MEM_FRACTION} \
        --dist-timeout 1800 &
    VLLM_PID=$!

    # Wait until the LLM answers locally, then publish IP for the master.
    echo "[WORKER] waiting for local LLM to become ready before publishing IP..."
    for i in $(seq 1 5400); do
        PROBE=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://localhost:${BCB_LLM_PORT}/v1/chat/completions \
            -H 'Content-Type: application/json' \
            -d '{"model":"deepseek-ai/DeepSeek-V3","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null || true)
        if [ "$PROBE" = "200" ]; then
            echo "$SELF_IP" > "$WORKER_IP_FILE"
            echo "[WORKER] LLM ready. Published IP=$SELF_IP -> $WORKER_IP_FILE"
            break
        fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[WORKER] LLM process died before ready."; exit 1; fi
        if [ $i -eq 5400 ]; then echo "[WORKER] LLM ready timeout (90min)"; exit 1; fi
        sleep 1
    done

    # Stay alive serving requests until master finishes and writes DONE.
    echo "[WORKER] serving; polling for master DONE flag ($DONE_FLAG)."
    while true; do
        if [ -f "$DONE_FLAG" ]; then echo "[WORKER] master DONE seen. Shutting down LLM and exiting 0."; kill $VLLM_PID 2>/dev/null || true; exit 0; fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[WORKER] LLM process died while serving."; exit 1; fi
        sleep 15
    done
fi

# ---------------------------------------------------------------------------
# 4. Main LLM startup.
#    - Single-node: master starts the LLM locally (below), then embedding on GPU0.
#    - Dual-node: the LLM runs on the WORKER; master instead waits for the worker's
#      published IP and points BCB_LLM_BASE at it. No local LLM on master.
# ---------------------------------------------------------------------------
if [ "$NNODES" -gt 1 ]; then
    # DUAL-NODE MASTER: discover worker IP (published to CPFS; DNS unavailable).
    echo "[MASTER] dual-node. Waiting for worker to publish its IP -> $WORKER_IP_FILE"
    WORKER_IP=""
    for i in $(seq 1 5400); do   # up to 90 min (worker cold start + weight load)
        if [ -f "$WORKER_IP_FILE" ]; then
            WORKER_IP=$(cat "$WORKER_IP_FILE" 2>/dev/null | tr -d '[:space:]' || true)
            [ -n "$WORKER_IP" ] && { echo "[MASTER] worker LLM IP=$WORKER_IP (after ~${i}s)"; break; }
        fi
        if [ $((i % 60)) -eq 0 ]; then echo "[MASTER] still waiting for worker IP... ~$((i/60)) min"; fi
        sleep 1
    done
    if [ -z "$WORKER_IP" ]; then echo "[FATAL] worker never published IP within 90min."; exit 1; fi
    BCB_LLM_HOST="$WORKER_IP"
else
    # SINGLE-NODE MASTER: start the LLM locally.
    BCB_LLM_HOST="localhost"
    python -u -m sglang.launch_server \
        --model-path "${BCB_MODEL_PATH}" \
        --served-model-name deepseek-ai/DeepSeek-V3 \
        --tp $TP_SIZE \
        --trust-remote-code \
        --host 0.0.0.0 --port ${BCB_LLM_PORT} \
        --context-length 32768 \
        --mem-fraction-static ${MEM_FRACTION} \
        --dist-timeout 1800 &
    VLLM_PID=$!

    echo "[INFO] Waiting for SGLang (pid=$VLLM_PID) on port $BCB_LLM_PORT..."
    for i in $(seq 1 5400); do
        # `|| true`: while the server is still starting, curl exits 7 (couldn't
        # connect). Under `set -e` that non-zero would propagate out of $(...) and
        # kill the whole script on the very first loop iteration.
        PROBE=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://localhost:${BCB_LLM_PORT}/v1/chat/completions \
            -H 'Content-Type: application/json' \
            -d '{"model":"deepseek-ai/DeepSeek-V3","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null || true)
        if [ "$PROBE" = "200" ]; then echo "[INFO] SGLang ready!"; break; fi
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "[ERROR] SGLang died."
            sleep 2
            exit 1
        fi
        if [ $i -eq 5400 ]; then echo "[ERROR] SGLang timeout (90min)"; exit 1; fi
        sleep 1
    done
fi

# ---------------------------------------------------------------------------
# 5. Verify the (remote or local) main LLM answers, then start the embedding
#    server. In dual-node the LLM is on the worker (BCB_LLM_HOST=worker IP);
#    the embedder gets its own GPU0 with no contention, so it can use a large
#    mem-fraction. In single-node it shares GPU0 (tighter fraction, --nccl-port
#    avoids the EADDRINUSE clash with the main server's torch.distributed port).
# ---------------------------------------------------------------------------
if [ "$NNODES" -gt 1 ]; then
    # Confirm the remote worker LLM is reachable from master before proceeding.
    echo "[MASTER] verifying remote LLM at http://${BCB_LLM_HOST}:${BCB_LLM_PORT} ..."
    for i in $(seq 1 600); do
        PROBE=$(curl -s -o /dev/null -w '%{http_code}' -X POST http://${BCB_LLM_HOST}:${BCB_LLM_PORT}/v1/chat/completions \
            -H 'Content-Type: application/json' \
            -d '{"model":"deepseek-ai/DeepSeek-V3","max_tokens":4,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null || true)
        if [ "$PROBE" = "200" ]; then echo "[MASTER] remote LLM reachable."; break; fi
        if [ $i -eq 600 ]; then echo "[FATAL] remote worker LLM not reachable from master (10min)."; exit 1; fi
        sleep 1
    done
    EMBED_MEM_FRACTION="${BCB_EMBED_MEM_FRACTION:-0.85}"
    EMBED_NCCL_ARG=""
else
    EMBED_MEM_FRACTION="${BCB_EMBED_MEM_FRACTION:-0.30}"
    EMBED_NCCL_ARG="--nccl-port 21000"
fi

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
    --model-path "${BCB_EMBED_MODEL_PATH}" \
    --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port ${BCB_EMBED_PORT} \
    ${EMBED_NCCL_ARG} \
    --context-length 8192 \
    --mem-fraction-static ${EMBED_MEM_FRACTION} \
    --trust-remote-code \
    --is-embedding &
EMBED_PID=$!

echo "[INFO] Waiting for embedding (pid=$EMBED_PID) on port $BCB_EMBED_PORT (mem-fraction=$EMBED_MEM_FRACTION)..."
for i in $(seq 1 600); do
    if curl -s "http://localhost:${BCB_EMBED_PORT}/v1/models" 2>/dev/null | grep -q model; then echo "[INFO] Embedding ready!"; break; fi
    if ! kill -0 $EMBED_PID 2>/dev/null; then echo "[ERROR] Embedding died"; exit 1; fi
    if [ $i -eq 600 ]; then echo "[ERROR] Embedding timeout (10min)"; exit 1; fi
    sleep 1
done

# ---------------------------------------------------------------------------
# 5b. Dual-node: point the BCB config's llm.base_url at the remote worker LLM.
#     run_bcb.py reads base_url only from the yaml (no CLI/env override). To avoid
#     polluting the shared source config with a per-job worker IP, we copy it to a
#     temp file, rewrite llm.base_url there, and run from the copy.
# ---------------------------------------------------------------------------
CFG_SRC="$BCB_CONFIG"
case "$CFG_SRC" in /*) : ;; *) CFG_SRC="$MEMRL_DIR/$CFG_SRC" ;; esac
RUN_CONFIG="$CFG_SRC"
if [ "$NNODES" -gt 1 ]; then
    RUN_CONFIG="$SHARE_DIR/run_config.yaml"
    cp "$CFG_SRC" "$RUN_CONFIG"
    NEW_LLM_URL="http://${BCB_LLM_HOST}:${BCB_LLM_PORT}/v1/"
    # Narrow replace: only the llm base_url (localhost:8000). Embedding stays local.
    sed -i "s#http://localhost:${BCB_LLM_PORT}/v1/#${NEW_LLM_URL}#g" "$RUN_CONFIG"
    if grep -q "$NEW_LLM_URL" "$RUN_CONFIG"; then
        echo "[MASTER] run config $RUN_CONFIG llm.base_url -> $NEW_LLM_URL"
    else
        echo "[FATAL] failed to rewrite llm.base_url in $RUN_CONFIG to $NEW_LLM_URL"; grep -n "base_url" "$RUN_CONFIG" || true; exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 5c. AUTO-RESUME (preemption tolerance). low-priority jobs get preempted often;
#     on restart, auto-scan $BCB_OUTPUT_DIR for the highest COMPLETED epoch
#     snapshot (epoch<N>/snapshot/<N> containing snapshot_meta.json) and resume
#     from it. Explicit --resume_from in BCB_EXTRA_ARGS (if set) takes priority.
#     BCB_NO_AUTORESUME=1 disables it (baseline modes like pass@k use their own
#     resume via results.jsonl and must NOT read epoch snapshots).
# ---------------------------------------------------------------------------
if [ "${BCB_NO_AUTORESUME:-0}" = "1" ]; then
    echo "[MASTER] BCB_NO_AUTORESUME=1; skipping epoch-snapshot auto-resume (baseline mode uses its own resume)."
else
case " $BCB_EXTRA_ARGS " in
    *" --resume_from "*)
        echo "[MASTER] explicit --resume_from present; skipping auto-resume scan." ;;
    *)
        AUTO_RESUME=$(python3 - "$BCB_OUTPUT_DIR" <<'PY'
import glob, os, re, sys
base = sys.argv[1]
best_n, best_dir = 0, None
for meta in glob.glob(os.path.join(base, "**", "epoch*/snapshot/*/snapshot_meta.json"), recursive=True):
    d = os.path.dirname(meta)
    m = re.search(r"/epoch(\d+)/snapshot/(\d+)$", d)
    if not m:
        continue
    ep, sn = int(m.group(1)), int(m.group(2))
    if ep == sn and ep > best_n:   # completed full-epoch snapshot
        best_n, best_dir = ep, d
if best_dir:
    print(f"{best_n}\t{best_dir}")
PY
)
        if [ -n "$AUTO_RESUME" ]; then
            RESUME_EP=$(printf '%s' "$AUTO_RESUME" | cut -f1)
            RESUME_DIR=$(printf '%s' "$AUTO_RESUME" | cut -f2)
            BCB_EXTRA_ARGS="${BCB_EXTRA_ARGS} --resume_from ${RESUME_DIR} --resume_epoch ${RESUME_EP}"
            echo "[MASTER] AUTO-RESUME from completed epoch ${RESUME_EP}: ${RESUME_DIR} (will start at E$((RESUME_EP+1)))"
        else
            echo "[MASTER] no completed epoch snapshot found; starting fresh from E1."
        fi ;;
esac
fi

# ---------------------------------------------------------------------------
# 4. Install BCB eval dependencies (after servers are up, failures non-fatal)
# ---------------------------------------------------------------------------
$PIP install -r 3rdparty/bigcodebench-main/Requirements/requirements.txt \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ --quiet 2>&1 | tail -3 || true
$PIP install -r 3rdparty/bigcodebench-main/Requirements/requirements-eval.txt \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ --quiet 2>&1 | tail -3 || true
$PIP install faker statsmodels xlwt python-docx sendgrid openpyxl xlrd seaborn pyarrow shapely geopandas folium \
    django scikit-image pyquery geopy xmltodict Flask-Mail flask_login pyfakefs texttable textblob gensim \
    pytesseract holidays pycryptodome mechanize wikipedia wordcloud wordninja requests_mock python-Levenshtein natsort librosa Flask-WTF flask-restful \
    --target $VENV_SP -i https://pypi.antfin-inc.com/simple/ --quiet 2>&1 | tail -3 || true

# ---------------------------------------------------------------------------
# 5. Run experiment
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "[RUN] python run/run_bcb.py"
echo "  --config $RUN_CONFIG"
echo "  --split instruct --subset full --epochs $BCB_EPOCHS"
echo "  --output_dir $BCB_OUTPUT_DIR"
echo "  $BCB_EXTRA_ARGS"
echo "=========================================="

# Capture the real exit code even under `set -e` (bare non-zero would exit before
# the reporting line). The EXIT trap re-exits with this preserved code.
EXIT_CODE=0
python run/run_bcb.py \
    --config "$RUN_CONFIG" \
    --split instruct --subset full --epochs "$BCB_EPOCHS" \
    --checkpoint_interval 100 --max_checkpoints 3 \
    --eval_timeout 240 --untrusted_hard_timeout 300 \
    --output_dir "$BCB_OUTPUT_DIR" \
    $BCB_EXTRA_ARGS || EXIT_CODE=$?
echo "[INFO] Experiment finished. Exit code: $EXIT_CODE"
exit $EXIT_CODE
