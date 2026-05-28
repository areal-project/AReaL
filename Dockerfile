# Supports sglang and vllm variants via the VARIANT build argument.
# VARIANT is declared early so each variant gets the correct torch version
# and C++ extensions compiled against it.
#
# Usage:
#   docker build -t areal-runtime:dev-sglang .                          # default (sglang)
#   docker build --build-arg VARIANT=vllm -t areal-runtime:dev-vllm .    # vllm variant

# ============================================================
# BUILDER STAGE: compile C++ extensions and install all deps
# ============================================================
FROM lmsysorg/sglang:v0.5.10.post1-runtime AS builder

# Inference backend selector: sglang (default) or vllm
ARG VARIANT=sglang

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive

# Build-time system dependencies (includes compilers + dev headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    ccache \
    kmod \
    libibverbs-dev \
    librdmacm-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip and uv
RUN pip install -U pip uv

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Build-time environment variables
ENV NVTE_WITH_USERBUFFERS=1
ENV NVTE_FRAMEWORK=pytorch
ENV MPI_HOME=/usr/local/mpi
ENV TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0 9.0a"
ENV MAX_JOBS=32

# Set VIRTUAL_ENV so uv pip install targets the venv
ENV VIRTUAL_ENV=/opt/.venv
ENV CUDA_HOME=/usr/local/cuda

##############################################################
# Install base torch (version is variant-specific)
##############################################################

# Create venv and install torch with CUDA support
# Version is variant-specific: sglang pins 2.9.1, vllm pins 2.10.0
RUN uv venv $VIRTUAL_ENV \
    && if [ "$VARIANT" = "vllm" ]; then TORCH_VER="2.10.0"; else TORCH_VER="2.9.1"; fi \
    && uv pip install --index-url https://download.pytorch.org/whl/cu129 \
    "torch==${TORCH_VER}+cu129" "torchaudio" "torchvision"

RUN uv pip install "setuptools>=77.0.3,<80" pybind11 nvidia-mathdx wheel

##############################################################
# Install heavy C++ dependencies
# These require only torch and rarely change.
# Moving these BEFORE uv sync prevents recompilation when
# pyproject.toml/uv.lock changes (C++ packages stay cached).
##############################################################

# Install torch memory saver
RUN TMS_CUDA_MAJOR=12 uv pip install --no-build-isolation --no-cache-dir --force-reinstall \
    git+https://github.com/fzyzcjy/torch_memory_saver.git

# Install grouped_gemm (for MoE models)
RUN uv pip install --no-build-isolation --no-cache-dir \
    git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

# Install apex (NVIDIA apex for mixed precision training)
RUN NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 \
    uv pip -v install --disable-pip-version-check --no-cache-dir --no-build-isolation \
    git+https://github.com/NVIDIA/apex.git

# Install transformer engine (for FP8 training)
RUN uv pip -v install --no-build-isolation --no-cache-dir \
    git+https://github.com/NVIDIA/TransformerEngine.git@stable

# FlashMLA (Multi-head Latent Attention for DeepSeek-V3)
RUN git clone https://github.com/deepseek-ai/FlashMLA.git /flash-mla \
    && cd /flash-mla && git checkout 71c7379 \
    && git submodule update --init --recursive \
    && uv pip install -v . --no-build-isolation --no-cache-dir \
    && rm -rf /flash-mla

# DeepGEMM (FP8 GEMM library for DeepSeek-V3)
RUN git clone https://github.com/deepseek-ai/DeepGEMM /DeepGEMM \
    && cd /DeepGEMM && git checkout d30fc36 \
    && git submodule update --init --recursive \
    && uv pip install -v . --no-build-isolation --no-cache-dir \
    && rm -rf /DeepGEMM

# DeepEP (Expert Parallelism communication library for MoE)
# Note: TORCH_CUDA_ARCH_LIST="9.0" enables SM90 features and aggressive PTX instructions
# The NVSHMEM path is auto-detected from nvidia.nvshmem module installed above
RUN git clone https://github.com/deepseek-ai/DeepEP /DeepEP \
    && cd /DeepEP && git checkout 567632d \
    && TORCH_CUDA_ARCH_LIST="9.0 9.0a" uv pip install -v . --no-build-isolation --no-cache-dir \
    && rm -rf /DeepEP

# conv1d, required by Qwen-3.5
RUN git clone https://github.com/Dao-AILab/causal-conv1d -b v1.6.0 /causal-conv1d \
    && cd /causal-conv1d \
    && uv pip install -v . --no-build-isolation --no-cache-dir \
    && rm -rf /causal-conv1d

# flash-attn 2: download pre-built wheel, strip local version, repack & install
RUN set -ex \
    && FA_VER="2.8.3" \
    && FA_RELEASE="v0.7.16" \
    && if [ "$VARIANT" = "vllm" ]; then TORCH_TAG="torch2.10"; else TORCH_TAG="torch2.9"; fi \
    && PY_TAG=$(python3 -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')") \
    && LOCAL="+cu128${TORCH_TAG}" \
    && WHL="flash_attn-${FA_VER}${LOCAL}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl" \
    && URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/${FA_RELEASE}/${WHL}" \
    && WORK="/tmp/flash-attn-repack" \
    && mkdir -p "$WORK" \
    && curl -fSL --retry 3 -o "$WORK/$WHL" "$URL" \
    && $VIRTUAL_ENV/bin/wheel unpack "$WORK/$WHL" -d "$WORK/unpacked" \
    && SRC="$WORK/unpacked/flash_attn-${FA_VER}${LOCAL}" \
    && sed -i "s/^Version: .*/Version: ${FA_VER}/" "$SRC/flash_attn-${FA_VER}${LOCAL}.dist-info/METADATA" \
    && mv "$SRC/flash_attn-${FA_VER}${LOCAL}.dist-info" "$SRC/flash_attn-${FA_VER}.dist-info" \
    && mv "$SRC" "$WORK/unpacked/flash_attn-${FA_VER}" \
    && $VIRTUAL_ENV/bin/wheel pack "$WORK/unpacked/flash_attn-${FA_VER}" -d "$WORK" \
    && uv pip install "$WORK/flash_attn-${FA_VER}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl" --no-build-isolation \
    && rm -rf "$WORK"

# flash-attn-3: install pre-built wheel (C extension only) + Python interface from source
RUN set -ex \
    && FA3_VER="3.0.0" \
    && FA3_RELEASE="v0.8.2" \
    && FA3_SRC_TAG="v2.8.3" \
    && if [ "$VARIANT" = "vllm" ]; then TORCH_TAG="torch2.10"; else TORCH_TAG="torch2.9"; fi \
    && LOCAL="+cu128${TORCH_TAG}gite2743ab" \
    && WHL="flash_attn_3-${FA3_VER}${LOCAL}-cp39-abi3-linux_x86_64.whl" \
    && URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/${FA3_RELEASE}/${WHL}" \
    && curl -fSL --retry 3 -o "/tmp/${WHL}" "$URL" \
    && uv pip install "/tmp/${WHL}" --no-build-isolation \
    && rm -f "/tmp/${WHL}" \
    && PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") \
    && SITE_PKG="$VIRTUAL_ENV/lib/python${PY_VER}/site-packages/flash_attn_3" \
    && mkdir -p "$SITE_PKG" \
    && curl -fSL --retry 3 -o "$SITE_PKG/flash_attn_interface.py" \
       "https://raw.githubusercontent.com/Dao-AILab/flash-attention/${FA3_SRC_TAG}/hopper/flash_attn_interface.py" \
    && touch "$SITE_PKG/__init__.py"

##############################################################
# Install project dependencies from pyproject.toml
# --active: target the existing $VIRTUAL_ENV instead of creating a new .venv
# --inexact: keep C++ packages (apex, flash-attn, etc.) not tracked in the lockfile
# --no-install-project: install dependencies only, not the areal package itself
##############################################################

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=/tmp/sglang/pyproject.toml \
    --mount=type=bind,source=pyproject.vllm.toml,target=/tmp/vllm/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=/tmp/sglang/uv.lock \
    --mount=type=bind,source=uv.vllm.lock,target=/tmp/vllm/uv.lock \
    case "$VARIANT" in \
      sglang) PROJECT_DIR=/tmp/sglang ;; \
      vllm) PROJECT_DIR=/tmp/vllm ;; \
      *) echo "Invalid VARIANT=$VARIANT (expected: sglang|vllm)" >&2; exit 1 ;; \
    esac \
    && uv sync --active --inexact --no-install-project --no-build-isolation \
       --extra cuda --extra sandbox --group dev --project "$PROJECT_DIR"

# Misc fixes (apply in builder so the venv is clean when copied)
RUN uv pip uninstall pynvml
# Update setuptools to fix a wandb bug; install nvidia-ml-py to replace pynvml
RUN uv pip install --no-cache-dir -U setuptools nvidia-ml-py

# ============================================================
# RUNTIME STAGE: lean final image (no build tools, no source
# checkouts, no intermediate compilation artifacts)
# ============================================================
FROM lmsysorg/sglang:v0.5.10.post1-runtime

ARG VARIANT=sglang

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive

# Runtime-only system dependencies (no cmake, ccache — build tools stay in builder)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    net-tools \
    unzip \
    kmod \
    libibverbs-dev \
    librdmacm-dev \
    ibverbs-utils \
    rdmacm-utils \
    python3-pyverbs \
    opensm \
    ibutils \
    perftest \
    python3-venv \
    tmux \
    lsof \
    nvtop \
    rsync \
    dnsutils \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Remove libcudnn9 to avoid conflicts with torch
RUN apt-get --purge remove -y --allow-change-held-packages libcudnn9* \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Install pip and uv (needed for editable install and nanobot-ai)
RUN pip install --no-cache-dir -U pip uv

WORKDIR /AReaL

# Runtime environment variables
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV NVTE_WITH_USERBUFFERS=1
ENV NVTE_FRAMEWORK=pytorch
ENV MPI_HOME=/usr/local/mpi
ENV VIRTUAL_ENV=/opt/.venv
ENV CUDA_HOME=/usr/local/cuda

# Copy the fully-built venv from builder (all C++ extensions included)
COPY --from=builder /opt/.venv /opt/.venv

##############################################################
# Install Node.js and npm-based tools
##############################################################

# Install Node.js via fnm and Claude Code
ENV FNM_DIR=/root/.fnm
ENV NODE_VERSION=24.13.0
ENV PATH="$FNM_DIR/aliases/default/bin:/root/.local/bin:$PATH"
RUN set -ex \
    && curl -fsSL https://fnm.vercel.app/install | bash -s -- --install-dir "$FNM_DIR" --skip-shell \
    && eval "$($FNM_DIR/fnm env --shell bash)" \
    && $FNM_DIR/fnm install $NODE_VERSION \
    && $FNM_DIR/fnm default $NODE_VERSION \
    && npm install -g npm@latest \
    && npm install -g @openai/codex @google/gemini-cli openclaw@latest \
    && curl -fsSL https://claude.ai/install.sh | bash \
    && curl -fsSL https://opencode.ai/install | bash \
    && npm cache clean --force \
    && rm -rf /root/.cache/uv /root/.cache/pip /tmp/*

ENV PATH="/root/.cargo/bin:$PATH"
RUN curl -fsSL \
    https://github.com/zeroclaw-labs/zeroclaw/releases/latest/download/install.sh \
    | bash -s -- --skip-onboard \
    && rm -rf /tmp/*

##############################################################
# Install AReaL from local source (last for fast iteration)
##############################################################

# Copy AReaL source code from build context (checked out by CI)
COPY . /AReaL

# Install areal package in editable mode without dependencies
# Using pip install instead of uv sync to avoid overwriting C++ packages
RUN uv pip install --no-cache-dir --no-deps -e /AReaL

# Place executables in the environment at the front of the path
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Reset entrypoint (some base images set custom entrypoints; this ensures /bin/bash)
ENTRYPOINT []
CMD ["/bin/bash"]
