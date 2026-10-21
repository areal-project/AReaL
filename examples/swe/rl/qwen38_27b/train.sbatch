#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:0
#SBATCH --time=UNLIMITED
#SBATCH --no-requeue
set -euo pipefail
umask 077

: "${QWEN_RUNTIME_SCRIPT:?Submit through submit.sh}"
source "$QWEN_RUNTIME_SCRIPT"
test -f "$EXPERIMENT_ROOT/train.yaml"
QWEN_SUBMITTED_RESERVATION=${QWEN_RESERVATION:-}
QWEN_SUBMITTED_NODELIST=${QWEN_NODELIST:-}
export QWEN_SUBMITTED_RESERVATION QWEN_SUBMITTED_NODELIST
export SBATCH_TIMELIMIT=168:00:00
export SBATCH_PARTITION="$QWEN_PARTITION" AREAL_APPTAINER_STAGGER_SECONDS=2
if [[ -n $QWEN_RESERVATION ]]; then
  export SBATCH_RESERVATION="$QWEN_RESERVATION"
else
  unset SBATCH_RESERVATION
fi
: "${WANDB_API_KEY:?WANDB_API_KEY missing from shared env}"
: "${ARENA_LLM_API_KEY:?ARENA_LLM_API_KEY missing from private env}"
: "${ARENA_OPENAPI_TOKEN:?ARENA_OPENAPI_TOKEN missing from private env}"
mkdir -p "$QWEN_OUTPUT_ROOT"

# Fail before allocating GPU workers if shared storage cannot durably write.
python3 - <<'VERIFY_STORAGE'
import os
import tempfile
from pathlib import Path

root = Path(os.environ["QWEN_ARENA_OUTPUT"])
fd, name = tempfile.mkstemp(prefix=".storage-preflight-", dir=root)
try:
    with os.fdopen(fd, "wb") as stream:
        stream.write(os.urandom(16 * 1024 * 1024))
        stream.flush()
        os.fsync(stream.fileno())
    if Path(name).stat().st_size != 16 * 1024 * 1024:
        raise OSError("Shared storage returned a short preflight write")
finally:
    if os.path.exists(name):
        os.unlink(name)
print("Shared storage preflight: 16 MiB fsync succeeded", flush=True)
VERIFY_STORAGE

# Validate the exact copied files, including uncommitted source and site profile.
# Compute hosts may have an old system Python; use the training image for scripts.
preflight_python() {
  singularity exec --no-eval --no-home --writable-tmpfs \
    --bind "$QWEN_CONTROLLER_MOUNTS" "$QWEN_ACTOR_IMAGE" python3 "$@"
}
preflight_python "$QWEN_RECIPE_DIR/source_snapshot.py" verify \
  --snapshot "$QWEN_SUBMITTED_SNAPSHOT" --revision "$QWEN_SOURCE_REVISION"
export SWE_RL_ADMIN_API_KEY
SWE_RL_ADMIN_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
cancel_arena_children() {
  local log_path="$QWEN_ARENA_OUTPUT/logs/$QWEN_RUN_USER/$QWEN_EXPERIMENT_NAME/$QWEN_ARENA_TRIAL/rollout.log"
  [[ -n ${QWEN_BATCH_REPLAY_PATH:-} ]] && return 0
  [[ -s $log_path ]] || return 0
  ARENA_CANCEL_ROLLOUT_LOG="$log_path" python3 - <<'PY'
import concurrent.futures
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

log_path = Path(os.environ["ARENA_CANCEL_ROLLOUT_LOG"])
config_path = Path(os.environ["QWEN_CONFIG"])
model_names = set()
model_pattern = re.compile(rb"model_name=(stream-areal-[a-z0-9-]+)")
with log_path.open("rb") as log_file:
    for line in log_file:
        model_names.update(model_pattern.findall(line))
stream_ids = re.findall(r"^    stream_id: ([^\s'\"]+)", config_path.read_text(), re.MULTILINE)
if not model_names:
    print("Arena child cleanup: no registered Arena models in this trial", flush=True)
    raise SystemExit(0)
if len(stream_ids) != 1:
    raise RuntimeError("Cannot identify this trial's Arena Stream")

base = os.environ["ARENA_OPENAPI_BASE"].rstrip("/")
headers = {"Authorization": "Bearer " + os.environ["ARENA_OPENAPI_TOKEN"]}
stream_id = urllib.parse.quote(stream_ids[0], safe="")


def request(method, url):
    req = urllib.request.Request(url, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=15) as response:
        return response.status, response.read()


def active_job_ids():
    def list_status(status):
        found = set()
        offset = 0
        while True:
            query = urllib.parse.urlencode(
                {"status": status, "all_experiments": "true", "limit": 1000, "offset": offset}
            )
            url = f"{base}/openapi/v1/streams/{stream_id}/tasks?{query}"
            payload = json.loads(request("GET", url)[1])
            items = payload.get("items", [])
            for item in items:
                job = item.get("job") or {}
                name = (job.get("model") or {}).get("model_name", "")
                if name.encode() in model_names:
                    found.add(job["job_id"])
            offset += len(items)
            if not items or offset >= payload.get("total", 0):
                break
        return found

    # scancel sends SIGKILL after KillWait (30 seconds on this cluster).
    # Query independent states together so the TERM trap can finish in time.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parts = pool.map(
            list_status, ("PENDING", "RUNNING", "AGGREGATING", "STOPPING")
        )
        return set().union(*parts)


def cancel(job_id):
    url = f"{base}/openapi/v1/jobs/{urllib.parse.quote(job_id, safe='')}/cancel"
    try:
        code, _ = request("POST", url)
        return {"job_id": job_id, "status": code}
    except urllib.error.HTTPError as exc:
        return {"job_id": job_id, "status": exc.code}
    except Exception as exc:
        return {"job_id": job_id, "error": type(exc).__name__}


job_ids = active_job_ids()
dry_run = os.environ.get("ARENA_CANCEL_DRY_RUN") == "1"
audit = Path(os.environ["EXPERIMENT_ROOT"]) / f"arena-cancel-{os.environ['SLURM_JOB_ID']}.json"
if dry_run:
    results = [{"job_id": job_id, "dry_run": True} for job_id in sorted(job_ids)]
    remaining = job_ids
else:
    # Preserve the exact target set even if Slurm ends the cleanup mid-flight.
    audit.write_text(json.dumps({"phase": "selected", "job_ids": sorted(job_ids)}) + "\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(cancel, sorted(job_ids)))
    # A second full Stream scan can outlive the Slurm TERM grace period.
    # Verify remaining jobs after scancel from outside the batch job.
    remaining = None
audit.write_text(
    json.dumps(
        {
            "selected": len(job_ids),
            "remaining": len(remaining) if remaining is not None else None,
            "results": results,
        },
        indent=2,
    ) + "\n"
)
print(
    f"Arena child cleanup: selected={len(job_ids)} "
    f"remaining={len(remaining) if remaining is not None else 'unverified'} audit={audit}",
    flush=True,
)
PY
}
cancel_slurm_children() {
  local prefix="${QWEN_EXPERIMENT_NAME}_${QWEN_ARENA_TRIAL}"
  local child_id
  while read -r child_id; do
    if [[ $child_id =~ ^[0-9]+$ ]]; then
      scancel "$child_id"
    fi
  done < <(squeue -h -n "${prefix}:actor,${prefix}:rollout" -o '%i')
}

if [[ ${QWEN_CLEANUP_ONLY:-0} == 1 ]]; then
  cancel_arena_children
  cancel_slurm_children
  exit 0
fi

# Check both credentials before GPU allocation; cleanup requires only OpenAPI.
preflight_python "$QWEN_RECIPE_DIR/validate_arena_auth.py" \
  --report "$EXPERIMENT_ROOT/monitoring/auth-preflight-${SLURM_JOB_ID}.json"

# Import the actual proxy entry and Arena workflow before allocating GPU workers.
singularity exec --no-home --writable-tmpfs --nv --bind "$QWEN_MOUNTS" \
  --env "PYTHONPATH=$QWEN_ROLLOUT_PYTHONPATH" \
  --env LD_LIBRARY_PATH= --env CUDNN_PATH= \
  "$QWEN_ROLLOUT_IMAGE" python3 "$QWEN_RECIPE_DIR/probe-rollout-imports.py"

# Slurm tools are available on the host, not in the actor SIF.
# Require a same-trial checkpoint for an explicitly requested recovery.
if [[ ${QWEN_REQUIRE_RECOVER:-0} == 1 ]]; then
  : "${QWEN_PREVIOUS_SUBMISSION:?Previous attempt receipt is required}"
  export QWEN_RECOVERY_HOST_REPORT="$EXPERIMENT_ROOT/monitoring/recovery-host-${SLURM_JOB_ID}.json"
  preflight_python "$QWEN_RECIPE_DIR/validate_recover_preflight.py" \
    --submission "$QWEN_PREVIOUS_SUBMISSION" --previous-jobs-only \
    --report "$QWEN_RECOVERY_HOST_REPORT"
fi

# An afterany job survives scancel and can finish Arena's slow Stream scan.
# Wait for a previous controller's cleanup before reusing this trial's model log.
prefix="${QWEN_EXPERIMENT_NAME}_${QWEN_ARENA_TRIAL}"
for attempt in {1..120}; do
  if [[ -z $(squeue -h -n "${prefix}:cleanup" -o '%i') ]]; then
    break
  fi
  if [[ $attempt == 120 ]]; then
    echo "Previous Arena cleanup is still active for $prefix" >&2
    exit 1
  fi
  sleep 5
done
cleanup_job_id=$(sbatch --parsable \
  --job-name="${prefix}:cleanup" \
  --dependency="afterany:${SLURM_JOB_ID}" \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --gres=gpu:0 \
  --time=00:20:00 --partition="$QWEN_PARTITION" \
  --output="$QWEN_OUTPUT_ROOT/logs/${prefix}-cleanup-%j.out" \
  --error="$QWEN_OUTPUT_ROOT/logs/${prefix}-cleanup-%j.err" \
  --export=ALL,QWEN_CLEANUP_ONLY=1 \
  "$QWEN_RECIPE_DIR/train.sbatch")
echo "Registered afterany Arena cleanup job $cleanup_job_id" >&2

cleanup_slurm_children() {
  local exit_code=$1
  trap - EXIT TERM INT
  set +e
  # The TERM trap is best effort; the afterany job completes slow cleanup.
  cancel_arena_children
  cancel_slurm_children
  exit "$exit_code"
}
trap 'cleanup_slurm_children $?' EXIT
trap 'cleanup_slurm_children 143' TERM
trap 'cleanup_slurm_children 130' INT

srun --mpi=none --ntasks=1 --cpus-per-task=4 --mem=64G \
  singularity exec --no-eval --pid --writable-tmpfs \
  --bind "$QWEN_CONTROLLER_MOUNTS" \
  --env "PYTHONPATH=$QWEN_CONTROLLER_PYTHONPATH" \
  "$QWEN_ACTOR_IMAGE" bash -c '
    set -euo pipefail
    source "$QWEN_RUNTIME_SCRIPT"
    # These describe the controller container, not paths on worker hosts.
    unset SINGULARITY_BIND SINGULARITY_BINDPATH APPTAINER_BIND APPTAINER_BINDPATH
    unset QWEN_QSA_STABLE_TOPK QWEN_ISOLATE_REQUEST_CACHE NCCL_ALGO NCCL_PROTO CUBLAS_WORKSPACE_CONFIG
    unset QWEN_BATCH_REPLAY_PATH QWEN_BATCH_REPLAY_PATHS QWEN_BATCH_SNAPSHOT_DIR
    cd "$QWEN_REPO"
    python3 "$QWEN_RECIPE_DIR/validate_config.py"
    python3 "$QWEN_RECIPE_DIR/validate_wandb.py"
    python3 "$QWEN_RECIPE_DIR/probe_runtime.py"
    if [[ ${QWEN_REQUIRE_RECOVER:-0} == 1 ]]; then
      python3 "$QWEN_RECIPE_DIR/validate_recover_preflight.py" \
        --submission "$QWEN_PREVIOUS_SUBMISSION" \
        --previous-jobs-report "$QWEN_RECOVERY_HOST_REPORT" \
        --config "$QWEN_CONFIG" \
        --report "$EXPERIMENT_ROOT/monitoring/recovery-full-${SLURM_JOB_ID}.json"
    fi
    exec python3 -m qwen27_train_rl swe --config "$QWEN_CONFIG"
  ' &
train_step_pid=$!
wait "$train_step_pid"
