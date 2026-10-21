#!/usr/bin/env bash
# Usage: bash submit.sh [--sync-only]; defaults to the adjacent site.env.
set -euo pipefail
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "${1:-}" in
  ''|--sync-only|--from-snapshot) ;;
  *) echo 'Usage: submit.sh [--sync-only]' >&2; exit 2 ;;
esac

if [[ ${1:-} != --from-snapshot ]]; then
  export QWEN_ENV_FILE
  QWEN_ENV_FILE=$(realpath "${QWEN_ENV_FILE:-$recipe_dir/site.env}")
  export QWEN_RUN_USER=${QWEN_RUN_USER:-${SUDO_USER:-$(id -un)}}
  set -a
  source "$QWEN_ENV_FILE"
  set +a
  local_repo=${QWEN_REPO:-$(cd "$recipe_dir/../../../.." && pwd)}
  : "${QWEN_RUN_DIR:?Set the run directory in site.env}"
  : "${QWEN_OUTPUT_ROOT:?Set the output root in site.env}"
  snapshot=$(realpath -m "${QWEN_SYNC_DIR:-$QWEN_RUN_DIR/source}")
  privilege=()
  if (( EUID != 0 )); then
    privilege=(sudo)
  fi
  source_revision=$("${privilege[@]}" python3 "$recipe_dir/source_snapshot.py" stage \
    --source "$local_repo" --profile "$QWEN_ENV_FILE" --destination "$snapshot")
  echo "Shared source snapshot: $snapshot/code"
  if [[ ${1:-} == --sync-only ]]; then
    exit 0
  fi
  # Carry only routing/provenance here. Credentials are read from the staged profile.
  exec "${privilege[@]}" env \
    QWEN_ENV_FILE="$snapshot/code/examples/swe/rl/qwen38_27b/site.env" \
    QWEN_SUBMITTED_ENV_FILE="$snapshot/code/examples/swe/rl/qwen38_27b/site.env" \
    QWEN_SUBMITTED_REPO="$snapshot/code" \
    QWEN_SUBMITTED_TRIAL="$QWEN_ARENA_TRIAL" \
    QWEN_SUBMITTED_RUN_DIR="$QWEN_RUN_DIR" \
    QWEN_SUBMITTED_OUTPUT_ROOT="$QWEN_OUTPUT_ROOT" \
    QWEN_SUBMITTED_SNAPSHOT="$snapshot" \
    QWEN_SOURCE_REVISION="$source_revision" QWEN_RUN_USER="$QWEN_RUN_USER" \
    bash "$snapshot/code/examples/swe/rl/qwen38_27b/submit.sh" --from-snapshot
fi

: "${QWEN_SUBMITTED_SNAPSHOT:?Submit from the local checkout}"
python3 "$recipe_dir/source_snapshot.py" verify \
  --snapshot "$QWEN_SUBMITTED_SNAPSHOT" --revision "$QWEN_SOURCE_REVISION"
source "$recipe_dir/runtime.sh"
: "${QWEN_CONTROLLER_NODE:?Set the controller node in the site profile}"
checkout_root=$(realpath "$QWEN_REPO")
for output_dir in "$EXPERIMENT_ROOT" "$QWEN_OUTPUT_ROOT"; do
  case "$(realpath -m "$output_dir")/" in
    "$checkout_root/"*) echo 'Keep run files and output outside the source snapshot' >&2; exit 1 ;;
  esac
done
log_dir="$QWEN_OUTPUT_ROOT/logs/$QWEN_RUN_USER/$QWEN_EXPERIMENT_NAME/$QWEN_ARENA_TRIAL"
mkdir -p "$EXPERIMENT_ROOT" "$log_dir"
# The staged profile selects a recipe-local YAML; the runtime path stays train.yaml.
config_name=${QWEN_TRAIN_CONFIG:-train.yaml}
if [[ $config_name != "$(basename -- "$config_name")" || $config_name != *.yaml ]]; then
  echo 'QWEN_TRAIN_CONFIG must name a YAML file in the recipe directory' >&2
  exit 1
fi
config_source="$recipe_dir/$config_name"
test -f "$config_source"
if [[ -f $QWEN_CONFIG ]]; then
  cmp "$config_source" "$QWEN_CONFIG"
  test "$(cat "$EXPERIMENT_ROOT/source-revision")" = "$QWEN_SOURCE_REVISION"
else
  cp "$config_source" "$QWEN_CONFIG"
  printf '%s\n' "$QWEN_SOURCE_REVISION" > "$EXPERIMENT_ROOT/source-revision"
fi
export QWEN_SUBMITTED_RESERVATION=$QWEN_RESERVATION QWEN_SUBMITTED_NODELIST=$QWEN_NODELIST
args=(--parsable --partition="$QWEN_PARTITION" --nodelist="$QWEN_CONTROLLER_NODE"
  --chdir="$QWEN_REPO" --job-name="$QWEN_EXPERIMENT_NAME-$QWEN_ARENA_TRIAL"
  --output="$log_dir/controller-%j.out"
  --error="$log_dir/controller-%j.err"
  --export=ALL)
if [[ -n $QWEN_RESERVATION ]]; then
  args+=(--reservation="$QWEN_RESERVATION")
fi
exec sbatch "${args[@]}" "$recipe_dir/train.sbatch"
