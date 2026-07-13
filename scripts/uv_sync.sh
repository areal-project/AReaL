#!/usr/bin/env bash
# Install project dependencies via `uv sync` for a given inference backend.
#
# Usage:
#   bash scripts/uv_sync.sh sglang          # sync with default pyproject.toml
#   bash scripts/uv_sync.sh vllm            # swap in vllm manifests, sync, revert
#   bash scripts/uv_sync.sh sglang --extra sandbox --group dev   # extra flags
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYPROJECT="${REPO_ROOT}/pyproject.toml"
VLLM_PYPROJECT="${REPO_ROOT}/pyproject.vllm.toml"
DEFAULT_LOCK="${REPO_ROOT}/uv.lock"
VLLM_LOCK="${REPO_ROOT}/uv.vllm.lock"

VARIANT="${1:-sglang}"
shift 2>/dev/null || true

case "$VARIANT" in
  sglang|vllm) ;;
  *)
    echo "[AReaL] Error: Invalid variant '${VARIANT}' (expected: sglang|vllm)" >&2
    exit 1
    ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "[AReaL] Error: 'uv' is not installed or not in PATH." >&2
  exit 1
fi

if [[ ! -f "${DEFAULT_PYPROJECT}" ]]; then
  echo "[AReaL] Error: pyproject.toml not found at ${DEFAULT_PYPROJECT}" >&2
  exit 1
fi

# Flags matching the Dockerfile's uv sync invocation (Stage 3), minus
# --active / --project which are Docker-specific.
UV_SYNC_ARGS=(
  --active
  --inexact
  --no-build-isolation
  --extra cuda
  "$@"
)

cd "${REPO_ROOT}"

if [[ "${VARIANT}" == "sglang" ]]; then
  echo "[AReaL] Running uv sync for sglang..."
  uv sync "${UV_SYNC_ARGS[@]}"
else
  # vllm: temporarily replace pyproject.toml / uv.lock with vllm variants,
  # run uv sync, then restore the originals (same swap pattern as uv_lock.sh).
  if [[ ! -f "${VLLM_PYPROJECT}" ]]; then
    echo "[AReaL] Error: pyproject.vllm.toml not found at ${VLLM_PYPROJECT}" >&2
    exit 1
  fi

  TMP_DIR="$(mktemp -d)"
  cleanup() {
    if [[ -f "${TMP_DIR}/pyproject.toml.bak" ]]; then
      cp "${TMP_DIR}/pyproject.toml.bak" "${DEFAULT_PYPROJECT}"
    fi
    if [[ -f "${TMP_DIR}/uv.lock.bak" ]]; then
      cp "${TMP_DIR}/uv.lock.bak" "${DEFAULT_LOCK}"
    fi
    rm -rf "${TMP_DIR}"
  }
  trap cleanup EXIT

  cp "${DEFAULT_PYPROJECT}" "${TMP_DIR}/pyproject.toml.bak"
  cp "${DEFAULT_LOCK}" "${TMP_DIR}/uv.lock.bak"

  cp "${VLLM_PYPROJECT}" "${DEFAULT_PYPROJECT}"
  cp "${VLLM_LOCK}" "${DEFAULT_LOCK}"

  echo "[AReaL] Running uv sync for vllm..."
  uv sync "${UV_SYNC_ARGS[@]}"
fi

echo "[AReaL] Done."
