#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYPROJECT="${REPO_ROOT}/pyproject.toml"
VLLM_PYPROJECT="${REPO_ROOT}/pyproject.vllm.toml"
DEFAULT_LOCK="${REPO_ROOT}/uv.lock"
VLLM_LOCK="${REPO_ROOT}/uv.vllm.lock"

if ! command -v uv >/dev/null 2>&1; then
  echo "[AReaL] Error: 'uv' is not installed or not in PATH." >&2
  exit 1
fi

if [[ ! -f "${DEFAULT_PYPROJECT}" || ! -f "${VLLM_PYPROJECT}" ]]; then
  echo "[AReaL] Error: pyproject.toml and pyproject.vllm.toml must both exist." >&2
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

cd "${REPO_ROOT}"

echo "[AReaL] Generating default lockfile from pyproject.toml..."
uv lock -v

cp "${DEFAULT_PYPROJECT}" "${TMP_DIR}/pyproject.toml.bak"
cp "${DEFAULT_LOCK}" "${TMP_DIR}/uv.lock.bak"

echo "[AReaL] Generating vLLM lockfile from pyproject.vllm.toml..."
cp "${VLLM_PYPROJECT}" "${DEFAULT_PYPROJECT}"
cp "${VLLM_LOCK}" "${DEFAULT_LOCK}"
uv lock -v
cp "${DEFAULT_LOCK}" "${VLLM_LOCK}"

echo "[AReaL] Done. Updated files:"
echo "  - uv.lock"
echo "  - uv.vllm.lock"
