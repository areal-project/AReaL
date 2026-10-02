#!/bin/bash
set -euo pipefail
workspace=$1
output=$2
mkdir -p "$output/checked"
tar -C "$workspace" --exclude=.git -cf - . | tar -C "$output/checked" -xf -
srun --mpi=none --jobid=972320 --overlap -N1 -n1 \
  apptainer exec --nv --bind /storage:/storage \
  /storage/openpsi/images/areal-dev-sglang-0821.sif \
  bash -s -- "$output/checked" "$output" <<'INNER'
set -euo pipefail
cd "$1"
env_dir=$(python3 -c 'import json; print(json.load(open("research/muon/local_environment.json"))["venv"])')
source "$env_dir/bin/activate"
source research/muon/container_environment.sh
export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$2/hardware.csv"
python -c 'import torch, importlib.metadata as m; print({n: m.version(n) for n in ["torch", "megatron-core", "emerging-optimizers", "transformer-engine", "nvidia-cudnn-cu12", "sglang", "mbridge"]}); print("CUDA", torch.version.cuda, "cuDNN", torch.backends.cudnn.version(), "NCCL", torch.cuda.nccl.version())' > "$2/environment-final.txt"
export PRE_COMMIT_HOME="$env_dir/pre-commit-cache"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy SKIP
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0=''
ca_bundle=$(python -c 'import certifi; print(certifi.where())')
export SSL_CERT_FILE="$ca_bundle" REQUESTS_CA_BUNDLE="$ca_bundle" CURL_CA_BUNDLE="$ca_bundle" PIP_CERT="$ca_bundle" GIT_SSL_CAINFO="$ca_bundle"
export UV_DEFAULT_INDEX=https://pypi.org/simple PIP_INDEX_URL=https://pypi.org/simple
export PATH="$env_dir/bin:/usr/local/bin:/opt/.venv/bin:$PATH"
# Preserve uv's live resolver log; pre-commit itself buffers hook output.
export MUON_REAL_UV MUON_UV_LOG
MUON_REAL_UV=$(command -v uv)
MUON_UV_LOG="$2/uv-lock.log"
mkdir -p "$2/tool-bin"
cat > "$2/tool-bin/uv" <<'UV_WRAPPER'
#!/bin/bash
set -o pipefail
"$MUON_REAL_UV" "$@" 2>&1 | tee -a "$MUON_UV_LOG"
UV_WRAPPER
chmod +x "$2/tool-bin/uv"
export PATH="$2/tool-bin:$PATH"
git init -q
# Retain the input repository's base so existing large tracked files are not
# misidentified as newly added files by check-added-large-files.
source_repo=$(python -c 'import json; print(json.load(open("research/muon/local_environment.json"))["source_repository"])')
base_commit=$(python -c 'import json; print(json.load(open("research/muon/local_environment.json"))["base_commit"])')
git fetch --quiet --no-tags "$source_repo" "$base_commit"
git reset --mixed "$base_commit"
git add .
set +e
python -m pre_commit run --all-files 2>&1 | tee "$2/pre-commit-first.log"
first_status=${PIPESTATUS[0]}
set -e
if [ "$first_status" != 0 ]; then
  # Hooks can rewrite formatting, generated CLI docs and lockfiles.
  python -m pre_commit run --all-files 2>&1 | tee "$2/pre-commit-final.log"
fi
git diff --name-only > "$2/hook-modified-files.txt"
python -m pytest -q tests/test_megatron_muon_config.py tests/test_megatron_optimizer_config.py tests/test_megatron_async_save.py 2>&1 | tee "$2/pytest-final.log"
INNER
