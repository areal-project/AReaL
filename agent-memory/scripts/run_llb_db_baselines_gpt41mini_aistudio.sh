#!/usr/bin/env bash
set -euo pipefail

BASELINE="${BASELINE:-rag}"
PROJECT_DIR=/storage/openpsi/users/yl/agent-memory/MemRL
LOCAL_SP=/storage/openpsi/users/yl/agent-memory/.local/lib/python3.12/site-packages
HOST_SHORT=$(hostname | sed 's/^gpulingjun//; s/\..*$//')
TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="${PROJECT_DIR}/logs/llb_db_${BASELINE}_gpt41mini_${HOST_SHORT}_${TS}.log"

exec > >(tee -a "$LOGFILE") 2>&1

cd "$PROJECT_DIR"
echo "=========================================="
echo "LLB DB baseline: ${BASELINE} (gpt-4.1-mini)"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Log: $LOGFILE"
echo "=========================================="

RUNTIME_OVERRIDE="${PROJECT_DIR}/scripts/runtime_overrides"
export PYTHONPATH="${RUNTIME_OVERRIDE}:${PROJECT_DIR}:${LOCAL_SP}:${PYTHONPATH:-}"
export MEMRL_LLM_MODEL=gpt-4.1-mini-2025-04-14
export MEMRL_RUN_ID="${MEMRL_RUN_ID:-${BASELINE}-db-gpt41mini-$(date +%Y%m%d)}"
export MEMRL_OS_BACKEND=local
export MEMRL_DB_BACKEND=auto
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/tmp/huggingface
export MEMRL_LLB_REFLECTION_PROMPT=v2
export MEMRL_LLB_SCRIPT_DETAIL=db_pattern

VENV_SP=/AReaL/.venv/lib/python3.12/site-packages
export PYTHONPATH="${RUNTIME_OVERRIDE}:${PROJECT_DIR}:${LOCAL_SP}:${VENV_SP}:${PYTHONPATH}"

echo '[INFO] Installing MariaDB server...'
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq mariadb-server >/dev/null 2>&1 || echo 'Warning: apt-get install mariadb-server failed'

echo '[INFO] Installing runtime deps...'
if [[ "$BASELINE" == "mem0" ]]; then
  # Must be set before the first mem0 import. Otherwise mem0 opens the global
  # /root/.mem0/migrations_qdrant telemetry store, and Memory re-init during
  # checkpoint restore collides with its own QdrantLocal lock.
  export MEM0_TELEMETRY=False
  export ANONYMIZED_TELEMETRY=False
  export POSTHOG_DISABLED=1
  # Keep Mem0/FastEmbed out of the shared AReaL environment. Installing the
  # latest transitive dependencies directly into VENV_SP previously upgraded
  # huggingface-hub to 1.x, incompatible with transformers 4.57.1.
  MEM0_DEPS=/tmp/llb_db_mem0_bm25_site
  rm -rf "$MEM0_DEPS"
  mkdir -p "$MEM0_DEPS"
  pip install \
    "mem0ai==2.0.12" \
    "qdrant-client[fastembed]>=1.17,<1.18" \
    "huggingface-hub>=0.34,<1.0" \
    "tokenizers>=0.22,<0.23" \
    --target "$MEM0_DEPS" \
    -i https://pypi.antfin-inc.com/simple/
  export PYTHONPATH="${RUNTIME_OVERRIDE}:${PROJECT_DIR}:${MEM0_DEPS}:${LOCAL_SP}:${VENV_SP}:${PYTHONPATH:-}"
else
  python3 -m pip install mem0ai "chonkie==1.2.1" tensorboard pandas tqdm concurrent-log-handler mysql-connector-python \
    --target "$VENV_SP" -i https://pypi.antfin-inc.com/simple/
fi

python3 -c "import memos, memrl; print('imports OK; memrl from:', memrl.__file__)"
if [[ "$BASELINE" == "mem0" ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export FASTEMBED_CACHE_PATH="${PROJECT_DIR}/scripts/fastembed_cache"
  python3 - <<'PYBM25'
import inspect
from importlib.metadata import version
from packaging.version import Version

hub = Version(version("huggingface-hub"))
assert Version("0.34") <= hub < Version("1.0"), hub
import transformers
import mem0
import qdrant_client
import fastembed
from fastembed import SparseTextEmbedding
from memrl.service.mem0_memory_service import Mem0MemoryService

encoder = SparseTextEmbedding(
    "Qdrant/bm25",
    cache_dir="/storage/openpsi/users/yl/agent-memory/MemRL/scripts/fastembed_cache",
    local_files_only=True,
)
probe = list(encoder.embed(["llb bm25 bootstrap"]))
assert probe and len(probe[0].indices) > 0
print(
    "mem0 BM25 dependency/cache imports OK; "
    f"huggingface-hub={hub}; transformers={transformers.__version__}; "
    f"service override={inspect.getsourcefile(Mem0MemoryService)}"
)
PYBM25
fi
echo "[INFO] BASELINE=$BASELINE"
echo "[INFO] MEMRL_LLM_MODEL=$MEMRL_LLM_MODEL"
echo "[INFO] MEMRL_RUN_ID=$MEMRL_RUN_ID"

# Keep large MariaDB data files on a unique shared scratch directory, while
# keeping Unix socket/pid files on node-local /tmp. Shared storage is permanent:
# this script never removes or overwrites any /storage path.
RUN_TAG=$(printf '%s' "$MEMRL_RUN_ID" | sha256sum | cut -c1-12)
RUNTIME_TMP="/storage/openpsi/users/yl/agent-memory/.t/${RUN_TAG}"
mkdir -p "$RUNTIME_TMP"
if [[ "$BASELINE" == "mem0" ]]; then
  # Mem0 live Qdrant and Python temporary files are node-local. Only explicit
  # snapshots are copied to permanent storage by the checkpoint API.
  export TMPDIR="/tmp"
else
  export TMPDIR="$RUNTIME_TMP"
fi
export MEMRL_DB_DATADIR_ROOT="$RUNTIME_TMP"
export MEMRL_DB_SOCKET_ROOT="/tmp"
echo "[INFO] TMPDIR=$TMPDIR"
echo "[INFO] MEMRL_DB_DATADIR_ROOT=$MEMRL_DB_DATADIR_ROOT"
echo "[INFO] MEMRL_DB_SOCKET_ROOT=$MEMRL_DB_SOCKET_ROOT"

OUTPUT_DIR=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/llb_baselines
case "$BASELINE" in
  rag)
    RAG_CONFIG="configs/rl_llb_db_rag.yaml"
    if [[ -n "${RAG_RESUME_CHECKPOINT:-}" ]]; then
      RAG_RESUME_SECTION="${RAG_RESUME_SECTION:-4}"
      RAG_EXPERIMENT_NAME="${RAG_EXPERIMENT_NAME:-llb_db_rag_gpt41mini_s4resume}"
      RAG_CONFIG="${RUNTIME_TMP}/rl_llb_db_rag_resume.yaml"
      RAG_DEST_DIR="${OUTPUT_DIR}/exp_${RAG_EXPERIMENT_NAME}_${MEMRL_RUN_ID}"
      export RAG_RESUME_CHECKPOINT RAG_RESUME_SECTION RAG_EXPERIMENT_NAME RAG_CONFIG RAG_DEST_DIR
      python3 - <<'PYCFG'
import json
import os
import re
from pathlib import Path
import yaml

src = Path("configs/rl_llb_db_rag.yaml")
dst = Path(os.environ["RAG_CONFIG"])
source_checkpoint = Path(os.environ["RAG_RESUME_CHECKPOINT"])
expected_section = int(os.environ["RAG_RESUME_SECTION"])
destination = Path(os.environ["RAG_DEST_DIR"])


def validate_snapshot(path: Path, expected_id=None):
    required_files = (
        path / "snapshot_meta.json",
        path / "cube" / "textual_memory.json",
        path / "local_cache" / "q_cache.json",
    )
    for item in required_files:
        if not item.is_file() or item.stat().st_size <= 0:
            return False, f"missing/empty {item.relative_to(path)}"
    qdrant = path / "qdrant"
    if not qdrant.is_dir() or not any(p.is_file() and p.stat().st_size > 0 for p in qdrant.rglob("*")):
        return False, "missing/empty qdrant"
    try:
        meta = json.loads((path / "snapshot_meta.json").read_text())
    except Exception as exc:
        return False, f"invalid snapshot_meta.json: {exc}"
    if expected_id is not None and int(meta.get("checkpoint_id", -1)) != expected_id:
        return False, f"checkpoint_id={meta.get('checkpoint_id')} (expected {expected_id})"
    return True, "ok"


def destination_has_healthy_snapshot(root: Path):
    snap_root = root / "snapshot"
    if not snap_root.is_dir():
        return None
    candidates = []
    for child in snap_root.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"(\d+)(?:_b(\d+))?", child.name)
        if not match:
            continue
        ok, _ = validate_snapshot(child)
        if ok:
            section = int(match.group(1))
            batch = int(match.group(2)) if match.group(2) is not None else 10**9
            candidates.append(((section, batch), child))
    return max(candidates, default=(None, None))[1]

config = yaml.safe_load(src.read_text())
config.setdefault("experiment", {})["experiment_name"] = os.environ["RAG_EXPERIMENT_NAME"]
config["experiment"]["ckpt_save_every_n_batches"] = 10
config["experiment"]["ckpt_max_keep"] = 3
memory = config.setdefault("memory", {})
healthy_destination = destination_has_healthy_snapshot(destination)
if healthy_destination is not None:
    # Platform retry: let run_llb auto-resume the newest destination snapshot,
    # rather than repeatedly loading the original source checkpoint.
    memory["load_from_checkpoint"] = False
    memory["checkpoint_path"] = None
    print(f"[INFO] Healthy destination snapshot found; auto-resume will use: {healthy_destination}")
else:
    ok, reason = validate_snapshot(source_checkpoint, expected_section)
    if not ok:
        raise SystemExit(f"refusing to resume from unhealthy checkpoint {source_checkpoint}: {reason}")
    memory["load_from_checkpoint"] = True
    memory["checkpoint_path"] = str(source_checkpoint)
    print(f"[INFO] First attempt will resume from validated checkpoint: {source_checkpoint}")

dst.write_text(yaml.safe_dump(config, sort_keys=False))
dst.chmod(0o600)
print(f"[INFO] Generated private resume config: {dst}")
PYCFG
    fi
    python3 run/run_llb.py \
      --config "$RAG_CONFIG" \
      --output_dir "$OUTPUT_DIR"
    ;;
  selfrag)
    python3 run/run_llb.py \
      --config configs/rl_llb_db_selfrag.yaml \
      --output_dir "$OUTPUT_DIR" \
      --self_rag
    ;;
  mem0)
    export MEMRL_UPDATE_MAX_WORKERS=1
    export MEMRL_MEM0_MIN_INTERVAL="${MEMRL_MEM0_MIN_INTERVAL:-3.7}"
    export MEM0_TELEMETRY=False
    export ANONYMIZED_TELEMETRY=False
    export POSTHOG_DISABLED=1
    MEM0_EXPERIMENT_NAME="${MEM0_EXPERIMENT_NAME:-llb_db_mem0_gpt41mini}"
    MEM0_CONFIG="${RUNTIME_TMP}/rl_llb_db_mem0_runtime.yaml"
    MEM0_DEST_DIR="${OUTPUT_DIR}/exp_${MEM0_EXPERIMENT_NAME}_${MEMRL_RUN_ID}"
    MEM0_COLLECTION="${MEM0_COLLECTION:-llb_db_mem0_${MEMRL_RUN_ID//-/_}}"
    export MEM0_EXPERIMENT_NAME MEM0_CONFIG MEM0_DEST_DIR MEM0_COLLECTION
    python3 - <<'PYCFG'
import json
import os
import re
from pathlib import Path
import yaml

src = Path("configs/rl_llb_db_mem0.yaml")
dst = Path(os.environ["MEM0_CONFIG"])
destination = Path(os.environ["MEM0_DEST_DIR"])


def valid_mem0_snapshot(path: Path) -> bool:
    marker = path / "snapshot_meta.json"
    metadata = path / "mem0_id_metadata.json"
    qdrant = path / "mem0_qdrant"
    if not marker.is_file() or not metadata.is_file():
        return False
    if marker.stat().st_size <= 0 or metadata.stat().st_size <= 0 or not qdrant.is_dir():
        return False
    try:
        payload = json.loads(marker.read_text())
        json.loads(metadata.read_text())
    except Exception:
        return False
    if payload.get("backend") != "mem0":
        return False
    return any(item.is_file() and item.stat().st_size > 0 for item in qdrant.rglob("*"))

healthy = []
snap_root = destination / "snapshot"
if snap_root.is_dir():
    for child in snap_root.iterdir():
        match = re.fullmatch(r"(\d+)(?:_b(\d+))?", child.name)
        if match and valid_mem0_snapshot(child):
            section = int(match.group(1))
            batch = int(match.group(2)) if match.group(2) is not None else 10**9
            healthy.append(((section, batch), child))

config = yaml.safe_load(src.read_text())
config.setdefault("experiment", {})["experiment_name"] = os.environ["MEM0_EXPERIMENT_NAME"]
config["experiment"]["ckpt_save_every_n_batches"] = 10
# Section-only Mem0 checkpoints: the generic batch-checkpoint cleanup uses recursive
# deletion under the permanent output tree. Keep permanent storage append-only.
config["experiment"]["ckpt_max_keep"] = 1000
# Mem0 resume is destination-auto-resume only. Do not point generic explicit
# checkpoint loading at a Mem0 snapshot: its layout and restore semantics differ.
config.setdefault("memory", {})["load_from_checkpoint"] = False
config["memory"]["checkpoint_path"] = None
if healthy:
    print(f"[INFO] Mem0 retry will auto-resume from: {max(healthy)[1]}")
else:
    print("[INFO] Mem0 fresh run: no healthy destination snapshot found")
dst.write_text(yaml.safe_dump(config, sort_keys=False))
dst.chmod(0o600)
print(f"[INFO] Generated private Mem0 config: {dst}")
PYCFG
    python3 scripts/run_llb_mem0_bm25.py \
      --config "$MEM0_CONFIG" \
      --output_dir "$OUTPUT_DIR" \
      --mem0 --mem0_infer true --mem0_collection "$MEM0_COLLECTION"
    ;;
  *)
    echo "ERROR: unsupported BASELINE=$BASELINE (expected rag, selfrag, or mem0)" >&2
    exit 2
    ;;
esac

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
