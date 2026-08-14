#!/bin/bash
# AIStudio: Qwen2.5-72B ALFWorld Mem0 baseline (SGLang, 3x H200).
# GPU 0: Qwen3-Embedding-8B; GPU 1-2: Qwen2.5-72B TP=2.
# IMPORTANT: Mem0 resume is destination-auto-resume only. Keep MEMRL_RUN_ID
# stable across retries; do not point generic ckpt_resume_path at a Mem0 snapshot.
set -euo pipefail

MEMRL_DIR="/storage/openpsi/users/yl/agent-memory/MemRL"
QWEN72B_PATH="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct"
EMBED_PATH="/storage/openpsi/models/Qwen3-Embedding-8B"
RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
LOGFILE="${MEMRL_DIR}/logs/aistudio_qwen72b_mem0_vllm_${RUN_TAG}.log"
EMBED_PORT="${EMBED_PORT:-19090}"
LLM_PORT="${LLM_PORT:-19290}"
NCCL_PORT="${NCCL_PORT:-29690}"
export MEMRL_RUN_ID="${MEMRL_RUN_ID:-qwen72b_mem0_v1}"
MEM0_COLLECTION="${MEM0_COLLECTION:-memrl_mem0_alf_qwen72b_v1}"
# Optional explicit external batch snapshot for an isolated continuation.
# Do not silently use an ambiguous generic checkpoint: a batch continuation
# must name the exact source snapshot (e.g. .../snapshot/s9_b10).
MEM0_RESUME_SOURCE="${MEM0_RESUME_SOURCE:-}"
MEM0_EXPECTED_RESUME="${MEM0_EXPECTED_RESUME:-}"
EXPERIMENT_NAME="alfworld_mem0_qwen72b"
OUTPUT_DIR="/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/alfworld"
DEST_DIR="${OUTPUT_DIR}/alfworld/exp_${EXPERIMENT_NAME}_${MEMRL_RUN_ID}"
RUNTIME_CONFIG="/tmp/alf_mem0_qwen72b_${$}.yaml"

mkdir -p "$(dirname "$LOGFILE")"
exec > >(tee -a "$LOGFILE") 2>&1
SERVER_PIDS=()
cleanup() {
    local rc=$?
    rm -f "$RUNTIME_CONFIG"
    if ((${#SERVER_PIDS[@]})); then kill "${SERVER_PIDS[@]}" 2>/dev/null || true; fi
    echo "[INFO] Cleanup complete (exit=${rc}) at $(date)"
}
trap cleanup EXIT INT TERM

echo "=========================================="
echo "AIStudio: Qwen2.5-72B Mem0 (SGLang, 3 GPU)"
echo "Start: $(date)"
echo "Run ID: ${MEMRL_RUN_ID}"
echo "Destination: ${DEST_DIR}"
echo "Mem0 collection: ${MEM0_COLLECTION}"
echo "Explicit resume source: ${MEM0_RESUME_SOURCE:-<destination auto-resume>}"
echo "Expected resume: ${MEM0_EXPECTED_RESUME:-<auto>}"
echo "Ports: embedding=${EMBED_PORT}, llm=${LLM_PORT}, nccl=${NCCL_PORT}"
echo "=========================================="

export HF_HOME=/storage/openpsi/users/yl/agent-memory/.cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export FASTEMBED_CACHE_PATH="${MEMRL_DIR}/scripts/fastembed_cache"
export MEM0_TELEMETRY=false MEM0_TELEMETRY_SAMPLE_RATE=0
export MEMRL_MEM0_OMIT_EMBED_DIMENSIONS=1
export ANONYMIZED_TELEMETRY=False POSTHOG_DISABLED=1
export MEMRL_MEM0_LLM_BASE_URL="http://localhost:${LLM_PORT}/v1/"
export MEMRL_MEM0_LLM_MODEL="Qwen2.5-72B-Instruct"
export MEMRL_UPDATE_MAX_WORKERS="${MEMRL_UPDATE_MAX_WORKERS:-1}"
export MEMRL_MEM0_MIN_INTERVAL="${MEMRL_MEM0_MIN_INTERVAL:-0}"
DEPS_DIR="/tmp/qwen72b_mem0_deps"
SHARED_SP="/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages"
rm -rf "$DEPS_DIR"
mkdir -p "$DEPS_DIR"
# Do not put the shared ~/.local tree on PYTHONPATH: it currently contains
# other Mem0 installs and would silently override the pinned experiment environment.
export PYTHONPATH="${MEMRL_DIR}:${DEPS_DIR}:${SHARED_SP}"

cd "$MEMRL_DIR"
PIP_INDEX="https://pypi.antfin-inc.com/simple/"
# Install into an isolated per-job target so Mem0/OpenAI dependencies cannot
# mutate the SGLang runtime environment in /AReaL/.venv.
pip install "mem0ai==2.0.12" "qdrant-client[fastembed]>=1.17,<1.18" "chonkie==1.2.1" \
    "tokenizers>=0.22,<=0.23.0" "huggingface-hub>=0.34,<1.0" \
    tensorboard hdbscan pandas tqdm concurrent-log-handler textworld alfworld \
    --target "$DEPS_DIR" -i "$PIP_INDEX" 2>&1 | tail -15

PYTHONPATH="${MEMRL_DIR}:${DEPS_DIR}:${SHARED_SP}" python3 - <<'PYDEPS'
from importlib import metadata
import mem0, qdrant_client, fastembed, memrl, memos
from memrl.service.mem0_memory_service import Mem0MemoryService
print("[OK] Mem0 dependency imports passed")
for dist in ("mem0ai", "qdrant-client", "fastembed"):
    print(f"[OK] {dist}={metadata.version(dist)}")
print(f"[OK] mem0 module={getattr(mem0, '__file__', None)}")
print(f"[OK] qdrant_client module={getattr(qdrant_client, '__file__', None)}")
print(f"[OK] Mem0MemoryService={Mem0MemoryService.__module__}")
assert metadata.version("mem0ai") == "2.0.12", metadata.version("mem0ai")
assert "/tmp/qwen72b_mem0_deps/" in str(mem0.__file__), mem0.__file__
PYDEPS

# Validate complete Mem0 snapshots before launch; the runner then auto-loads the
# latest complete one from the same stable destination.
DEST_DIR="$DEST_DIR" python3 - <<'PYSNAP'
import json, os, re
from pathlib import Path
root = Path(os.environ["DEST_DIR"]) / "local_cache" / "snapshot"
def healthy(path):
    marker, metadata, qdrant = path/"snapshot_meta.json", path/"mem0_id_metadata.json", path/"mem0_qdrant"
    if not marker.is_file() or not metadata.is_file() or not qdrant.is_dir(): return False
    try:
        marker_data = json.loads(marker.read_text()); json.loads(metadata.read_text())
    except Exception: return False
    return marker_data.get("backend") == "mem0" and any(p.is_file() and p.stat().st_size > 0 for p in qdrant.rglob("*"))
candidates = []
if root.is_dir():
    for child in root.iterdir():
        if not child.is_dir() or not healthy(child): continue
        if child.name.isdigit(): candidates.append(((int(child.name), 10**9), child))
        else:
            match = re.fullmatch(r"s(\d+)_b(\d+)", child.name)
            if match: candidates.append(((int(match.group(1)), int(match.group(2))), child))
if candidates: print(f"[INFO] Healthy Mem0 auto-resume candidate: {max(candidates)[1]}")
else: print(f"[INFO] Mem0 fresh run: no healthy snapshot under {root}")
PYSNAP

# An isolated continuation may load a batch snapshot from another destination.
# Validate it before expensive server startup and make the intended position
# auditable in the business log.
if [[ -n "$MEM0_RESUME_SOURCE" ]]; then
    if [[ ! -f "$MEM0_RESUME_SOURCE/snapshot_meta.json" || ! -f "$MEM0_RESUME_SOURCE/mem0_id_metadata.json" || ! -d "$MEM0_RESUME_SOURCE/mem0_qdrant" || ! -f "$MEM0_RESUME_SOURCE/local_cache/cum_state.json" ]]; then
        echo "[ERROR] Explicit Mem0 resume snapshot is incomplete: $MEM0_RESUME_SOURCE" >&2
        exit 2
    fi
    if [[ -n "$MEM0_EXPECTED_RESUME" && ! "$MEM0_EXPECTED_RESUME" =~ ^section=[0-9]+,batch=[0-9]+$ ]]; then
        echo "[ERROR] Refusing malformed expected resume marker: $MEM0_EXPECTED_RESUME" >&2
        exit 2
    fi
    echo "[RESUME-PREFLIGHT] source=$MEM0_RESUME_SOURCE expected=${MEM0_EXPECTED_RESUME:-unspecified}"
fi

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$MEMRL_DIR" python3 -m sglang.launch_server \
    --model-path "$EMBED_PATH" --served-model-name Qwen/Qwen3-Embedding-8B \
    --host 127.0.0.1 --port "$EMBED_PORT" --context-length 8192 \
    --trust-remote-code --is-embedding &
SERVER_PIDS+=("$!")
CUDA_VISIBLE_DEVICES=1,2 PYTHONPATH="$MEMRL_DIR" python3 -m sglang.launch_server \
    --model-path "$QWEN72B_PATH" --served-model-name Qwen2.5-72B-Instruct \
    --tp 2 --host 127.0.0.1 --port "$LLM_PORT" --context-length 32768 \
    --trust-remote-code --nccl-port "$NCCL_PORT" &
SERVER_PIDS+=("$!")

wait_for_server() {
    local name="$1" port="$2" attempts="$3"
    echo "[INFO] Waiting for ${name} on port ${port}..."
    for i in $(seq 1 "$attempts"); do
        if curl -fsS "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q model; then
            echo "[INFO] ${name} ready"; return 0
        fi
        sleep 1
    done
    echo "[ERROR] ${name} timeout after ${attempts}s" >&2; return 1
}
wait_for_server Embed "$EMBED_PORT" 1200
wait_for_server LLM "$LLM_PORT" 1800

# Fail fast before spending hours on the full run. This exercises the exact
# pinned Mem0 add/search/checkpoint APIs against both live model servers.
MEM0_PREFLIGHT_DIR="/tmp/mem0_preflight_${$}"
rm -rf "$MEM0_PREFLIGHT_DIR"
MEM0_PREFLIGHT_DIR="$MEM0_PREFLIGHT_DIR" \
PYTHONPATH="${MEMRL_DIR}:${DEPS_DIR}:${SHARED_SP}" python3 - <<'PYMEM0'
import json, os
from pathlib import Path
from memrl.service.mem0_memory_service import Mem0MemoryService
root = Path(os.environ["MEM0_PREFLIGHT_DIR"])
svc = Mem0MemoryService(
    llm_base_url=os.environ["MEMRL_MEM0_LLM_BASE_URL"],
    llm_model=os.environ["MEMRL_MEM0_LLM_MODEL"],
    embed_base_url=f"http://localhost:{os.environ['EMBED_PORT']}/v1/",
    embed_model="Qwen/Qwen3-Embedding-8B",
    embedding_dims=4096,
    qdrant_path=str(root / "qdrant"),
    collection_name="memrl_mem0_qwen72b_preflight",
    user_id="preflight_agent",
    infer=True,
)
mid = svc.add_memory(
    "Put a clean apple in the refrigerator.",
    [{"role": "assistant", "content": "go to kitchen; take apple; open refrigerator; put apple in refrigerator"}],
    True,
)
assert mid, "Mem0 preflight add produced no memory ID"
result, _ = svc.retrieve_query("How should I put an apple in the refrigerator?", k=1)
assert result["selected"], "Mem0 preflight search returned no memory"
saved = Path(svc.save_checkpoint_snapshot(str(root / "checkpoint"), 1))
assert saved.is_dir() and any(p.is_file() and p.stat().st_size > 0 for p in saved.rglob("*"))
marker = json.loads((saved.parent / "snapshot_meta.json").read_text())
assert marker == {"ckpt_id": 1, "backend": "mem0"}, marker
print(f"[OK] Mem0 live preflight passed: memory_id={mid}, snapshot={saved}")
PYMEM0
rm -rf "$MEM0_PREFLIGHT_DIR"

cat > "$RUNTIME_CONFIG" <<CFGEOF
llm:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${LLM_PORT}/v1/
  model: Qwen2.5-72B-Instruct
  temperature: 0
  max_tokens: 4096
embedding:
  provider: openai
  api_key: EMPTY
  base_url: http://localhost:${EMBED_PORT}/v1/
  model: Qwen/Qwen3-Embedding-8B
  max_text_len: 8196
  dimension: 4096
memory:
  build_strategy: proceduralization
  retrieve_strategy: query
  update_strategy: adjustment
  k_retrieve: 3
  max_keywords: 5
  add_similarity_threshold: 0.9
  memory_budget_tokens: 0
  sim_norm_mean: 0.5187
  sim_norm_std: 0.1203
  load_from_checkpoint: false
  checkpoint_path: null
environment:
  alfworld_config_path: configs/envs/alfworld.yaml
  alfworld_env_type: AlfredTWEnv
experiment:
  random_seed: 42
  enable_value_driven: false
  experiment_name: ${EXPERIMENT_NAME}
  mode: train
  num_sections: 10
  batch_size: 128
  dataset_ratio: 1.0
  few_shot_path: data/alfworld/alfworld_examples.json
  baseline_mode: null
  baseline_k: 10
  output_dir: ${OUTPUT_DIR}
  max_steps: 30
  save_trajectories: true
  save_memories: true
  ckpt_resume_enabled: ${MEM0_CKPT_RESUME_ENABLED:-false}
  ckpt_resume_path: "${MEM0_RESUME_SOURCE}"
  ckpt_resume_epoch: null
  ckpt_save_every_n_batches: 10
  ckpt_max_keep: 1000
  n_eval_runs: 4
  eval_temperature: 0.2
rl_config:
  epsilon: 0
  tau: 0.0
  alpha: 0.3
  gamma: 0.0
  q_init_pos: 0
  q_init_neg: 0
  success_reward: 1.0
  failure_reward: -1.0
  topk: 3
  novelty_threshold: 0.85
  recency_boost: 0.0
  reward_merge_gain: 0.1
  q_min_threshold: -10
  weight_sim: 1.0
  weight_q: 0.0
CFGEOF
chmod 600 "$RUNTIME_CONFIG"
echo "[EXP] 72B Mem0 starting at $(date)"
PYTHONPATH="${MEMRL_DIR}:${DEPS_DIR}:${SHARED_SP}" python3 scripts/run_alfworld_mem0_bm25.py --config "$RUNTIME_CONFIG" \
    --mem0 --mem0_infer true --mem0_collection "$MEM0_COLLECTION" --skip_initial_eval
echo "[EXP] 72B Mem0 completed at $(date)"
