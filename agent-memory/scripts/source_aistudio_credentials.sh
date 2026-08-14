#!/usr/bin/env bash
# Source this file to populate AIS submit credentials from the legacy local
# submit wrapper. It never prints or persists credential values.
# Usage: source scripts/source_aistudio_credentials.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Please source this file: source scripts/source_aistudio_credentials.sh" >&2
  exit 2
fi

_legacy="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_llb_os_mem0_gpt41mini.sh"
[[ -r "$_legacy" ]] || { echo "AIS credential source is unavailable: $_legacy" >&2; return 1; }

_extract_legacy_export() {
  local name="$1"
  sed -n "s/^export ${name}=['\"]\([^'\"]*\)['\"]$/\1/p" "$_legacy" | head -n 1
}

export AISTUDIO_LOGIN_NAME="${AISTUDIO_LOGIN_NAME:-$(_extract_legacy_export AISTUDIO_LOGIN_NAME)}"
export AISTUDIO_USERNUMBER="${AISTUDIO_USERNUMBER:-$(_extract_legacy_export AISTUDIO_USERNUMBER)}"
export AISTUDIO_TOKEN="${AISTUDIO_TOKEN:-$(_extract_legacy_export AISTUDIO_TOKEN)}"
unset -f _extract_legacy_export
unset _legacy

[[ -n "${AISTUDIO_LOGIN_NAME:-}" && -n "${AISTUDIO_USERNUMBER:-}" && -n "${AISTUDIO_TOKEN:-}" ]] || {
  echo "Failed to load one or more AIS credential environment variables" >&2
  return 1
}
echo "AIS credential environment loaded (values redacted)."
