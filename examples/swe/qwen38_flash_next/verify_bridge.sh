#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$recipe_dir/runtime.env"
: "${MCORE_BRIDGE_ROOT:?Set MCORE_BRIDGE_ROOT to the pinned clean checkout}"
actual=$(git -C "$MCORE_BRIDGE_ROOT" rev-parse HEAD)
if [[ "$actual" != "$QWEN_BRIDGE_COMMIT" ]]; then
  echo "Bridge revision mismatch: expected $QWEN_BRIDGE_COMMIT, got $actual" >&2
  exit 1
fi
if [[ -n $(git -C "$MCORE_BRIDGE_ROOT" status --porcelain --untracked-files=normal) ]]; then
  echo "Bridge checkout is dirty; use a clean pinned checkout" >&2
  exit 1
fi
echo "Bridge verified: $actual"
