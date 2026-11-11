# Installation (AMD ROCm)

**Note**: AMD GPU support lives on `main`. There is no published ROCm image yet, so you
build one from
[`Dockerfile.rocm`](https://github.com/areal-project/AReaL/blob/main/Dockerfile.rocm) as
described below.

## Prerequisites

### Hardware Requirements

The following hardware configurations have been tested:

- **GPU**: 8x AMD Instinct MI308X (gfx942) or 8x AMD Instinct MI355X (gfx950) per node
- **CPU**: 224 logical cores per node (MI308X node) or 256 logical cores per node
  (MI355X node)
- **Memory**: about 880 GB per node (MI308X node) or about 3 TB per node (MI355X node)

**Note**: MI300X and MI325X use the gfx942 architecture, and MI350X and MI355X use the
gfx950 architecture.

### Software Requirements

`Dockerfile.rocm` builds on `rocm/primus:v26.5` and keeps the base image's ROCm build of
PyTorch.

| Component          | Version                                                        |
| ------------------ | -------------------------------------------------------------- |
| Operating System   | Ubuntu 24.04 or any system meeting the requirements below      |
| AMD GPU driver     | amdgpu 6.16.13                                                 |
| Docker             | 29.1.3                                                         |
| Base image         | `rocm/primus:v26.5`                                            |
| ROCm               | 7.15                                                           |
| PyTorch            | 2.12.0                                                         |
| Transformer Engine | 2.15 ([ROCm build](https://github.com/ROCm/TransformerEngine)) |
| SGLang             | 0.5.10.post1, built from source (`sglang` variant)             |
| vLLM               | 0.19.1, built from source (`vllm` variant)                     |
| Megatron-Core      | core_v0.19.0                                                   |
| Transformers       | 5.3.0                                                          |
| Ray                | 2.59.0                                                         |

**Note**: This tutorial does not cover the installation of the AMD GPU driver or of
shared storage. See the
[ROCm installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
for the driver.

## Runtime Environment

The full list of pinned Python packages is in
[`requirements-rocm.txt`](https://github.com/areal-project/AReaL/blob/main/requirements-rocm.txt).

### Build the Image

Run the build from the root of the AReaL repository. The image has two variants that
differ only in the inference backend:

```bash
git clone https://github.com/areal-project/AReaL
cd AReaL

# Note: set to gfx950 for MI350X/MI355X!
GPU_ARCH=gfx942

# SGLang inference backend (default)
docker build -f Dockerfile.rocm --build-arg GPU_ARCH=${GPU_ARCH} \
    -t areal-runtime:dev-rocm-sglang-${GPU_ARCH} .

# vLLM inference backend
docker build -f Dockerfile.rocm --build-arg VARIANT=vllm --build-arg GPU_ARCH=${GPU_ARCH} \
    -t areal-runtime:dev-rocm-vllm-${GPU_ARCH} .
```

The build compiles SGLang or vLLM from source, so it takes a while. At the end, it
checks that the ROCm PyTorch was not replaced and that the inference backend imports.

**Notes**:

- Each image targets a single GPU architecture, `gfx942` by default. `GPU_ARCH` sets the
  target architecture for every kernel in the image: the SGLang kernels, the vLLM
  kernels, and the kernels aiter compiles at runtime. An image built for one
  architecture does not run on another. For MI350X/MI355X machines, set it to `gfx950`.
- If `docker build` fails because the `buildx` plugin is missing, prefix the command
  with `DOCKER_BUILDKIT=0` to use Docker's legacy builder. `Dockerfile.rocm` builds with
  it as well.
- During the build, aiter prints
  `ROCm/HIP JIT runtime not available: No module named 'jax'. CK and HIP ops are disabled.`
  This is because no GPU is visible at build time and does not affect the image; aiter
  loads normally when the container runs on a GPU.

### Create Container

```bash
docker run -it --name areal-rocm \
   --device=/dev/kfd --device=/dev/dri --group-add video \
   --ipc=host --network host \
   --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
   -v /path/to/mount:/path/to/mount \
   areal-runtime:dev-rocm-sglang-gfx942 \
   /bin/bash
```

- `/dev/kfd` and `/dev/dri` expose the GPUs to the container. `--ipc=host` gives the
  container the host's shared memory, so `--shm-size` is not needed. Keep
  `--cap-add=SYS_PTRACE`: sharing GPU memory between processes uses the `pidfd_getfd`
  system call, which fails with `Operation not permitted` without this capability.
- For the vLLM variant, replace the image tag with the vLLM tag.
- With `--network host`, AReaL's services listen on the node IP instead of the loopback
  address. Read [Required Settings](#rocm-required-settings) before launching a job.

**For multi-node training**: mount a shared storage path on every node and into the
container. It holds checkpoints and logs.

### Use Your Own AReaL Checkout

The image installs the AReaL source it was built from under `/AReaL`. To use a different
checkout, reinstall from it to override the bundled copy:

```bash
cd /path/to/mount/AReaL
pip install --no-deps --no-build-isolation -e .
```

Keep `--no-deps`. `pyproject.toml` pins a CUDA PyTorch, and resolving dependencies would
replace the image's ROCm build.

### Custom Environment Installation

We recommend the image. To install AReaL into your own ROCm environment instead:

- Do not run `uv sync --extra cuda`. The PyTorch, SGLang and vLLM packages on PyPI are
  CUDA builds and overwrite the ROCm installation.

- Follow `Dockerfile.rocm` step by step. It installs every package with `--no-deps` and
  compiles SGLang and vLLM against the installed ROCm PyTorch.

- Set the environment variables the image sets with `ENV`:

  - Set `GPU_ARCHS`, `PYTORCH_ROCM_ARCH` and `AMDGPU_TARGET` to your GPU architecture,
    for example `gfx942`. aiter cannot resolve `GPU_ARCHS=native` and compiles kernels
    for the wrong architecture.
  - Set `LD_LIBRARY_PATH` in the Dockerfile's order so the ROCm SDK libraries come
    first. If two copies of LLVM are loaded, `torch_memory_saver` and Transformer Engine
    exit with `Option spirv-expand-step registered more than once`.
  - With Megatron, set `NVTE_FLASH_ATTN=1` and `NVTE_FUSED_ATTN=1`. If either is `0`,
    Megatron's `attention_backend=auto` fails an assertion at startup.

- If you install SGLang yourself and plan to set `SGLANG_USE_AITER=1`, run the command
  below once after installing it and before training. It patches SGLang 0.5.10.post1,
  the version AReaL currently pairs with, so MoE weights keep the layout aiter expects
  after online weight updates; it refuses to run on any other SGLang version. Without
  aiter the patch is not needed:

  ```bash
  python -m areal.engine.sglang_aiter_moe_layout
  ```

## Required Settings

<a id="rocm-required-settings"></a>

1. **Admin API key.** AReaL's HTTP services expose admin endpoints that require an admin
   API key. The default key is in the public source, so AReaL refuses to start a service
   on a non-loopback address with the default key and reports
   `Refusing to start server on non-loopback host ... default admin API key`. Inside a
   container, services listen on the container IP or node IP whether or not you use
   `--network host`, so set one of the following:

   - A unique key, for example `rollout.agent.admin_api_key=<your-key>`. The error
     message names the field to set; v2 controllers read `rollout.admin_api_key` and
     `actor.admin_api_key`.
   - On a trusted network only, `export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1` to keep the
     default key.

1. **SGLang attention backend.** AReaL's default SGLang `attention_backend` is `fa3`,
   but the ROCm `sgl-kernel` does not include FA3. On ROCm, AReaL automatically replaces
   `fa3` with `aiter`, or with `triton` when `sglang.enable_deterministic_inference` is
   on, because SGLang's deterministic mode accepts only `flashinfer`, `fa3` and
   `triton`. Any other backend you set explicitly is passed to SGLang unchanged, for
   example `+sglang.attention_backend=triton`.

## Environment Variables

| Variable                          | Purpose                                                                                                                                                                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SGLANG_USE_AITER=1`              | Optional, off by default. Uses aiter kernels for SGLang's MoE, LayerNorm and other operators; attention is not affected. You may enable it for MoE models, which then need the MoE layout patch above.  |
| `VLLM_ROCM_USE_AITER=1`           | Optional, off by default. Uses aiter kernels in vLLM. The first start compiles aiter kernels and can exceed the default 300 s setup timeout; add `+rollout.setup_timeout=1200`.                         |
| `AREAL_ALLOW_DEFAULT_ADMIN_KEY=1` | Required when a service starts on a non-loopback address without a custom `admin_api_key`; otherwise AReaL refuses to start. Use it only on a trusted network.                                          |
| `VLLM_SERVER_DEV_MODE=1`          | Optional, needed only for vLLM colocation; export it before training. vLLM 0.19.1 serves the `/sleep` and `/wake_up` endpoints colocation needs only in dev mode. Separated deployments do not need it. |

**Notes**:

- `VLLM_SERVER_DEV_MODE=1` dev mode exposes unauthenticated debug endpoints on the vLLM
  server. Do not expose vLLM ports to an untrusted network.
- AReaL selects GPUs with `CUDA_VISIBLE_DEVICES` on ROCm as well. If
  `HIP_VISIBLE_DEVICES` or `ROCR_VISIBLE_DEVICES` is already set in your shell or
  container (check with `env | grep VISIBLE_DEVICES`; it may come from a launch script,
  the image's `ENV`, or the cluster scheduler), `unset` it before launching so that only
  `CUDA_VISIBLE_DEVICES` controls device visibility.

## Troubleshooting

- **Leftover processes after a failure.** When a rollout service fails to start, AReaL
  prints `Deleting ... workers for role rollout`, but the `areal.infra.rpc.rpc_server`
  processes it started may stay alive, and the shell or pipe waiting on their output
  does not return. Check with `ps aux | grep areal.infra.rpc.rpc_server` and kill them
  before retrying.
- **aiter reports `cannot open shared object file` for a kernel under
  `~/.aiter/build/`.** An earlier build of that kernel failed, for example because
  `GPU_ARCHS` was wrong or because the home directory is shared between gfx942 and
  gfx950 machines, and aiter keeps loading the broken directory. Delete that kernel's
  directory under `~/.aiter/build/` and run again.

## Run an Example

Once the environment is ready, start with the GSM8K GRPO example from the
[Quickstart](quickstart.md): it trains Qwen2.5-1.5B-Instruct with GRPO on GSM8K on a
single node with 8 GPUs, 4 for inference and 4 for training. On ROCm, add the
[required settings](#rocm-required-settings).

In the SGLang container:

```bash
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1  # trusted networks only; or set admin_api_key
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local
```

In the vLLM container, switch the inference backend:

```bash
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1  # trusted networks only; or set admin_api_key
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    rollout.backend=vllm:d4p1t1
```

After each training step, the log prints `Train step N/M done.`. Seeing that line means
inference, training and weight updates all work. The example trains for 10 epochs by
default; for a quick environment check, add `+total_train_steps=3`. Keep the leading
`+`, because the example YAML files have no such key.
