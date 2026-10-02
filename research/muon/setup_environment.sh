#!/bin/bash
set -euo pipefail
workspace=$1
output=$2
mkdir -p "$output"
squeue -j 972320 > "$output/allocation.txt"
srun --mpi=none --jobid=972320 --overlap -N1 -n1 \
  apptainer exec --nv --bind /storage:/storage \
  /storage/openpsi/images/areal-dev-sglang-0821.sif \
  bash -s -- "$workspace" "$output" <<'INNER'
set -euo pipefail
workspace=$1
output=$2
cd "$workspace"
echo "environment_workspace=$workspace"
env_dir=$(python3 -c 'import json; print(json.load(open("research/muon/local_environment.json"))["venv"])')
python3 -m venv --system-site-packages "$env_dir"
site_dir=$("$env_dir/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
echo /opt/.venv/lib/python3.12/site-packages > "$site_dir/container-packages.pth"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=http.proxy GIT_CONFIG_VALUE_0=''
ca_bundle=$(python3 -c 'import certifi; print(certifi.where())')
export SSL_CERT_FILE="$ca_bundle" REQUESTS_CA_BUNDLE="$ca_bundle" CURL_CA_BUNDLE="$ca_bundle" PIP_CERT="$ca_bundle" GIT_SSL_CAINFO="$ca_bundle"
"$env_dir/bin/python" -m pip install --index-url https://pypi.org/simple --no-deps megatron-core==0.19.0 emerging-optimizers==0.3.0 uv==0.11.8 2>&1 | tee "$output/install.log"
"$env_dir/bin/python" -m pip install --no-deps 'https://github.com/sitabulaixizawaluduo/Megatron-Bridge/releases/download/areal-v0.6.0-exaone45-fix.2/megatron_bridge-0.6.0-py3-none-any.whl' 2>&1 | tee "$output/bridge-install.log"
# Keep the locked Transformers version supplied by the image. Remove only an
# isolated trial install; pip does not uninstall packages outside this venv.
if [ "$("$env_dir/bin/python" -c 'import importlib.metadata as m; print(m.version("transformers"))')" != 5.3.0 ]; then
  "$env_dir/bin/python" -m pip uninstall -y transformers
fi
source research/muon/container_environment.sh
export PYTHONPATH="$workspace${PYTHONPATH:+:$PYTHONPATH}"
"$env_dir/bin/python" -c 'import sys,importlib.metadata as m; print(sys.executable,sys.version); print([(n,m.version(n)) for n in ["torch","megatron-core","emerging-optimizers","megatron-bridge","transformers","sglang"]]); import areal.engine.megatron_engine' 2>&1 | tee "$output/import.log"
"$env_dir/bin/python" -m pytest -q tests/test_megatron_muon_config.py tests/test_megatron_optimizer_config.py tests/test_megatron_async_save.py 2>&1 | tee "$output/pytest.log"
INNER
