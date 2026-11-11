# 安装指南（AMD ROCm）

**注意**：AMD GPU 支持位于 `main` 分支。目前没有发布 ROCm 镜像，需要按下文用
[`Dockerfile.rocm`](https://github.com/areal-project/AReaL/blob/main/Dockerfile.rocm)
自行构建。

## 前置要求

### 硬件要求

以下硬件配置已经过测试：

- **GPU**：每节点 8 × AMD Instinct MI308X（gfx942），或每节点 8 × AMD Instinct MI355X（gfx950）
- **CPU**：每节点 224 个逻辑核（MI308X 节点）或 256 个逻辑核（MI355X 节点）
- **内存**：每节点 1 TB（MI308X 节点）或 3 TB（MI355X 节点）
- **网络**：节点内 8 张 GPU 通过 AMD Infinity Fabric 两两直连（MI308X 每条链路 64 GB/s，MI355X 每条链路 144
  GB/s）；节点间建议每节点配置 8 × 400 Gb/s 的 InfiniBand 或 RoCE 网卡，共 3.2 Tbps

**注意**：MI300X 和 MI325X 使用 gfx942 架构，MI350X 和 MI355X 使用 gfx950 架构。

### 软件要求

`Dockerfile.rocm` 基于 `rocm/primus:v26.5` 构建，并保留基础镜像中 ROCm 版的 PyTorch。

| 组件               | 版本                                                         |
| ------------------ | ------------------------------------------------------------ |
| 操作系统           | Ubuntu 24.04 或满足以下要求的任何系统                        |
| AMD GPU 驱动       | amdgpu 6.16.13                                               |
| Docker             | 29.1.3                                                       |
| 基础镜像           | `rocm/primus:v26.5`                                          |
| ROCm               | 7.15                                                         |
| PyTorch            | 2.12.0                                                       |
| Transformer Engine | 2.15（[ROCm 版](https://github.com/ROCm/TransformerEngine)） |
| SGLang             | 0.5.10.post1，从源码构建（`sglang` 版本）                    |
| vLLM               | 0.19.1，从源码构建（`vllm` 版本）                            |
| Megatron-Core      | core_v0.19.0                                                 |
| Transformers       | 5.3.0                                                        |
| Ray                | 2.59.0                                                       |

**注意**：本教程不涵盖 AMD GPU 驱动和共享存储的安装。驱动安装请参考
[ROCm 安装指南](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)。

## 运行环境

完整的 Python 依赖版本列表见
[`requirements-rocm.txt`](https://github.com/areal-project/AReaL/blob/main/requirements-rocm.txt)。

### 构建镜像

在 AReaL 仓库根目录下执行构建。镜像有两个版本，区别只在推理后端：

```bash
git clone https://github.com/areal-project/AReaL
cd AReaL

# 注：MI350X/MI355X 请设为 gfx950！
GPU_ARCH=gfx942

# SGLang 推理后端（默认）
docker build -f Dockerfile.rocm --build-arg GPU_ARCH=${GPU_ARCH} \
    -t areal-runtime:dev-rocm-sglang-${GPU_ARCH} .

# vLLM 推理后端
docker build -f Dockerfile.rocm --build-arg VARIANT=vllm --build-arg GPU_ARCH=${GPU_ARCH} \
    -t areal-runtime:dev-rocm-vllm-${GPU_ARCH} .
```

构建过程会从源码编译 SGLang 或 vLLM，需要一些时间。构建最后会检查 ROCm 版 PyTorch 没有被替换，并且推理后端可以正常导入。

**注意**：

- 每个镜像只针对一种 GPU 架构，默认是 `gfx942`。`GPU_ARCH` 决定镜像里所有 kernel 的目标架构，包括 SGLang 的 kernel、vLLM
  的 kernel，以及 aiter 在运行时编译的 kernel。为一种架构构建的镜像不能在另一种架构上运行。如果要使用 MI350X/MI355X 机器，请设为
  `gfx950`。
- 如果 `docker build` 报错提示缺少 `buildx` 插件，请在命令前加上 `DOCKER_BUILDKIT=0`，改用 Docker
  自带的旧版构建器。`Dockerfile.rocm` 用旧版构建器也能正常构建。
- 构建过程中 aiter 会打印
  `ROCm/HIP JIT runtime not available: No module named 'jax'. CK and HIP ops are disabled.`
  这是因为构建时看不到 GPU，不影响镜像本身；容器在 GPU 上运行时 aiter 会正常加载。

### 创建容器

```bash
docker run -it --name areal-rocm \
   --device=/dev/kfd --device=/dev/dri --group-add video \
   --ipc=host --network host \
   --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
   -v /path/to/mount:/path/to/mount \
   areal-runtime:dev-rocm-sglang-gfx942 \
   /bin/bash
```

- `/dev/kfd` 和 `/dev/dri` 用于把 GPU 暴露给容器。`--ipc=host` 让容器直接使用主机的共享内存，因此不需要
  `--shm-size`。请保留 `--cap-add=SYS_PTRACE`：进程间共享显存要用到 `pidfd_getfd` 系统调用，缺少这项权限会报
  `Operation not permitted`。
- 使用 vLLM 版本时，把镜像标签换成 vLLM 对应的标签。
- 使用 `--network host` 时，AReaL 的服务会监听节点
  IP，而不是回环地址。启动任务前请先看[必需的设置](#rocm-required-settings)。

**对于多节点训练**：请确保在每个节点上挂载共享存储路径，并挂载到容器中。此路径将用于保存检查点和日志。

### 使用自己的 AReaL 代码

镜像在 `/AReaL` 下安装了构建时所用的 AReaL 源码。如果要使用其他代码目录，在其中重新安装即可覆盖镜像自带的版本：

```bash
cd /path/to/mount/AReaL
pip install --no-deps --no-build-isolation -e .
```

请保留 `--no-deps`。`pyproject.toml` 中固定的是 CUDA 版 PyTorch，解析依赖会替换掉镜像中的 ROCm 版。

### 自定义环境安装

我们推荐使用镜像。如果要把 AReaL 安装到自己的 ROCm 环境中：

- 不要运行 `uv sync --extra cuda`。PyPI 上的 PyTorch、SGLang 和 vLLM 都是 CUDA 版，会覆盖 ROCm 安装。

- 按照 `Dockerfile.rocm` 逐步安装。它用 `--no-deps` 安装每个包，并基于已安装的 ROCm 版 PyTorch 编译 SGLang 和
  vLLM。

- 设置镜像通过 `ENV` 设置的环境变量：

  - 把 `GPU_ARCHS`、`PYTORCH_ROCM_ARCH` 和 `AMDGPU_TARGET` 设为您的 GPU 架构，例如 `gfx942`。aiter
    无法解析 `GPU_ARCHS=native`，会把 kernel 编译成错误的架构。
  - 按 Dockerfile 的顺序设置 `LD_LIBRARY_PATH`，让 ROCm SDK 的库排在最前面。如果加载了两份 LLVM，
    `torch_memory_saver` 和 Transformer Engine 会报
    `Option spirv-expand-step registered more than once` 并退出。
  - 使用 Megatron 时设置 `NVTE_FLASH_ATTN=1` 和 `NVTE_FUSED_ATTN=1`。其中任何一个为 `0` 时，Megatron 的
    `attention_backend=auto` 会在启动时断言失败。

- 如果自己安装 SGLang 并打算开启 `SGLANG_USE_AITER=1`，请在安装之后、开始训练之前执行一次下面的命令。它会给 AReaL 当前对应的 SGLang
  0.5.10.post1 版本打补丁，使在线权重更新后 MoE 权重仍保持 aiter 要求的布局；遇到其他 SGLang 版本时会拒绝执行。不开启 aiter
  时不需要这个补丁：

  ```bash
  python -m areal.engine.sglang_aiter_moe_layout
  ```

## 必需的设置

<a id="rocm-required-settings"></a>

1. **管理 API 密钥。** AReaL 的 HTTP 服务带有管理接口，调用时需要管理 API 密钥。默认密钥写在公开的源码里，
   所以服务监听在非回环地址上时，AReaL 会拒绝用默认密钥启动，并报
   `Refusing to start server on non-loopback host ... default admin API key`。在容器里运行时，无论是否使用
   `--network host`，服务监听的都是容器 IP 或节点 IP，因此启动时需要以下任选一种设置：

   - 设置一个唯一的密钥，例如 `rollout.agent.admin_api_key=<你的密钥>`。报错信息会指出需要设置的字段；v2 控制器读
     `rollout.admin_api_key` 和 `actor.admin_api_key`。
   - 仅在可信网络中，设置 `export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1`，继续使用默认密钥。

1. **SGLang 的 attention 后端。** AReaL 给 SGLang 的默认 `attention_backend` 是 `fa3`，而 ROCm 版的
   `sgl-kernel` 不包含 FA3。在 ROCm 上，AReaL 会自动把 `fa3` 换成 `aiter`；开启
   `sglang.enable_deterministic_inference` 时则换成 `triton`，因为 SGLang 在确定性模式下只接受
   `flashinfer`、`fa3` 和 `triton`。您显式设置的其他后端会原样传给 SGLang，例如
   `+sglang.attention_backend=triton`。

## 环境变量

| 变量                              | 作用                                                                                                                                                |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SGLANG_USE_AITER=1`              | 可选，默认关闭。让 SGLang 的 MoE、LayerNorm 等算子使用 aiter kernel，不影响 attention。MoE 模型可以开启，开启后需要上面的 MoE 布局补丁。            |
| `VLLM_ROCM_USE_AITER=1`           | 可选，默认关闭。让 vLLM 使用 aiter kernel。首次启动要编译 aiter kernel，可能超过默认 300 秒的启动超时，需要加 `+rollout.setup_timeout=1200`。       |
| `AREAL_ALLOW_DEFAULT_ADMIN_KEY=1` | 在非回环地址上启动服务、且没有设置自定义 `admin_api_key` 时必须设置，否则 AReaL 会拒绝启动。只应在可信网络中使用。                                  |
| `VLLM_SERVER_DEV_MODE=1`          | 可选，只有 vLLM 共卡模式需要，请在启动训练前 export。vLLM 0.19.1 只在开发模式下提供共卡模式所需的 `/sleep` 和 `/wake_up` 接口。分离部署不需要设置。 |

**注意**：

- `VLLM_SERVER_DEV_MODE=1` 开发模式会在 vLLM 服务上暴露未鉴权的调试接口，请不要把 vLLM 端口暴露到不可信的网络。
- AReaL 在 ROCm 上同样通过 `CUDA_VISIBLE_DEVICES` 选择 GPU。如果当前 shell 或容器里已经设置了
  `HIP_VISIBLE_DEVICES` 或 `ROCR_VISIBLE_DEVICES`（可用 `env | grep VISIBLE_DEVICES`
  检查，它们可能来自启动脚本、镜像的 `ENV` 或集群调度器），请在启动前用 `unset` 清除，确保只由 `CUDA_VISIBLE_DEVICES` 控制设备可见性。

## 问题排查

- **失败后残留进程。** rollout 服务启动失败时，AReaL 会打印 `Deleting ... workers for role rollout`，但它启动的
  `areal.infra.rpc.rpc_server` 进程可能仍然存活，等待其输出的 shell 或管道也不会返回。请用
  `ps aux | grep areal.infra.rpc.rpc_server` 检查，并在重试前结束这些进程。
- **aiter 报 `~/.aiter/build/` 下的某个 kernel `cannot open shared object file`。** 原因是该
  kernel 之前的编译失败了，例如 `GPU_ARCHS` 设置错误，或者 home 目录在 gfx942 和 gfx950 机器之间共享，而 aiter
  会一直加载这个坏掉的目录。请删除 `~/.aiter/build/` 下该 kernel 的目录后重新运行。

## 运行示例

装好环境后，可以先运行[快速入门](quickstart.md)中的 GSM8K GRPO 示例：用 GRPO 在 GSM8K 数据集上训练
Qwen2.5-1.5B-Instruct，单节点 8 张 GPU，4 张用于推理、4 张用于训练。在 ROCm
上，请加上[必需的设置](#rocm-required-settings)。

在 SGLang 容器中：

```bash
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1  # 仅限可信网络；或者设置 admin_api_key
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local
```

在 vLLM 容器中，切换推理后端：

```bash
export AREAL_ALLOW_DEFAULT_ADMIN_KEY=1  # 仅限可信网络；或者设置 admin_api_key
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    rollout.backend=vllm:d4p1t1
```

每完成一步训练，日志会打印一行 `Train step N/M done.`，看到这一行就说明推理、训练和权重更新都已跑通。示例默认训练 10 个
epoch；只想快速检查环境时，可以加上 `+total_train_steps=3`。请保留开头的 `+`，因为示例 YAML 文件里没有这个键。
