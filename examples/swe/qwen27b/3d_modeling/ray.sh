#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
recipe=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  echo 'Usage: QWEN_ENV_FILE=<shared-profile> bash ray.sh [--config <train.yaml>] [--prepare-only] [--resume] [--set KEY=VALUE ...]'
  exit 0
fi
if [[ ${1:-} == --config ]]; then
  exec bash "$recipe/../ray_swemm.sh" "$@"
fi
exec bash "$recipe/../ray_swemm.sh" --config "$recipe/train.yaml" "$@"
