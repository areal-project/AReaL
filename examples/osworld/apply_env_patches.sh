#!/usr/bin/env bash
# Apply runtime patches that the conda env needs but `uv sync` would overwrite.
#
# Run once after creating the env (or any time after `uv sync --extra cuda`).
#
#   bash examples/osworld/apply_env_patches.sh
#
# Optional env vars:
#   AREAL_ENV_PREFIX  - conda env prefix (default: ../../../env)
#
# Patches applied (each one is idempotent):
#   1. (Conditional) SGLang JIT kernels: c++20 -> c++17 only when the host's
#      nvcc is too old (e.g. CUDA 12.2) to accept `-std=c++20`. With CUDA
#      12.9+ this patch is HARMFUL — SGLang's templates use `std::integral`
#      and `std::ranges` which require C++20. The script auto-detects.
#   2. pydrive -> pydrive2 shim, because OSWorld imports the unmaintained
#      `pydrive` package which is broken against modern oauth2client.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AREAL_ENV_PREFIX="${AREAL_ENV_PREFIX:-$(cd "${SCRIPT_DIR}/../../../env" && pwd)}"

if [[ ! -d "${AREAL_ENV_PREFIX}" ]]; then
    echo "[apply_env_patches.sh] env not found at: ${AREAL_ENV_PREFIX}" >&2
    exit 1
fi

SP="${AREAL_ENV_PREFIX}/lib/python3.12/site-packages"

# -------- Patch 1: SGLang c++20 -> c++17 (only when nvcc rejects c++20) --------
SGL_UTILS="${SP}/sglang/jit_kernel/utils.py"
if [[ ! -f "${SGL_UTILS}" ]]; then
    echo "[apply_env_patches.sh] sglang utils.py missing: ${SGL_UTILS}" >&2
    exit 2
fi

# Probe: does nvcc accept -std=c++20?
NVCC_BIN="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
NVCC_C20_OK=0
if "${NVCC_BIN}" --help 2>/dev/null | grep -q 'c++20'; then
    NVCC_C20_OK=1
fi

if [[ "${NVCC_C20_OK}" == "1" ]]; then
    # nvcc supports c++20: revert any prior c++17 downgrade we (or older
    # versions of this script) introduced. SGLang templates require C++20.
    if grep -q '"-std=c++17"' "${SGL_UTILS}" && [[ -f "${SGL_UTILS}.bak.cuda122" ]]; then
        cp "${SGL_UTILS}.bak.cuda122" "${SGL_UTILS}"
        echo "[patch 1/2] SGLang: restored c++20 from backup (nvcc supports c++20)"
    elif grep -q '"-std=c++17"' "${SGL_UTILS}"; then
        sed -i 's/"-std=c++17"/"-std=c++20"/g' "${SGL_UTILS}"
        echo "[patch 1/2] SGLang: rewrote c++17 -> c++20 in place"
    else
        echo "[patch 1/2] SGLang: already at c++20 — skipping"
    fi
else
    if grep -q '"-std=c++20"' "${SGL_UTILS}"; then
        cp "${SGL_UTILS}" "${SGL_UTILS}.bak.cuda122"
        sed -i 's/"-std=c++20"/"-std=c++17"/g' "${SGL_UTILS}"
        echo "[patch 1/2] SGLang: c++20 -> c++17 ($(grep -c '"-std=c++17"' "${SGL_UTILS}") sites; old nvcc)"
    else
        echo "[patch 1/2] SGLang already at c++17 — skipping"
    fi
fi

# -------- Patch 2: pydrive -> pydrive2 shim --------
PYDRIVE_DIR="${SP}/pydrive"
PYDRIVE2_DIR="${SP}/pydrive2"
if [[ ! -d "${PYDRIVE2_DIR}" ]]; then
    echo "[patch 2/2] pydrive2 not installed — installing"
    "${AREAL_ENV_PREFIX}/bin/pip" install --no-cache-dir --progress-bar off \
        pydrive2 "oauth2client<4.1.4"
fi

# Remove the unmaintained PyDrive (if it sneaked back in) before writing shim.
if "${AREAL_ENV_PREFIX}/bin/pip" show -q pydrive 2>/dev/null; then
    "${AREAL_ENV_PREFIX}/bin/pip" uninstall -y pydrive >/dev/null
fi

mkdir -p "${PYDRIVE_DIR}"
cat > "${PYDRIVE_DIR}/__init__.py" <<'PY'
"""Compatibility shim: redirect pydrive imports to pydrive2."""
import sys
from pydrive2 import auth as _auth
from pydrive2 import drive as _drive
sys.modules.setdefault("pydrive.auth", _auth)
sys.modules.setdefault("pydrive.drive", _drive)
PY
echo "[patch 2/2] pydrive shim written to ${PYDRIVE_DIR}/__init__.py"

# -------- Cleanup any stale JIT cache from prior failed builds --------
TVM_FFI_CACHE="${HOME}/.cache/tvm-ffi"
if [[ -d "${TVM_FFI_CACHE}" ]]; then
    rm -rf "${TVM_FFI_CACHE}/sgl_kernel_jit_"*
    echo "[cleanup] purged stale sgl_kernel_jit_* under ${TVM_FFI_CACHE}"
fi

echo "[apply_env_patches.sh] done"
