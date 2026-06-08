# 安装指南

## 前置要求

### 硬件要求

以下硬件配置经过充分测试：

- **GPU**：每节点 8x H800
- **CPU**：每节点 64 核
- **内存**：每节点 1TB
- **网络**：NVSwitch + RoCE 3.2 Tbps
- **存储**：
  - 单节点实验需要 1TB 本地存储
  - 分布式实验需要 10TB 共享存储（NAS）

### 软件要求

| 组件                     |                                                                                  版本                                                                                  |
| ------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 操作系统                 |                                                            CentOS 7 / Ubuntu 22.04 或满足以下要求的任何系统                                                            |
| NVIDIA 驱动              |                                                                               550.127.08                                                                               |
| CUDA                     |                                                                                  12.8                                                                                  |
| Git LFS                  | 用于下载模型、数据集和 AReaL 代码。请参阅[安装指南](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) |
| Docker                   |                                                                                 27.5.1                                                                                 |
| NVIDIA Container Toolkit |                             请参阅[安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)                              |
| AReaL 镜像               |                                `ghcr.io/areal-project/areal-runtime:v1.0.4-sglang`（默认）或 `v1.0.4-vllm`。包含运行时依赖和 Ray 组件。                                |

**注意**：本教程不涵盖 NVIDIA 驱动、CUDA 或共享存储挂载的安装，因为这些取决于您具体的节点配置和系统版本。请独立完成这些安装。

## 运行时环境

**对于多节点训练**：确保在每个节点上挂载共享存储路径（如果使用 Docker，也要挂载到容器中）。此路径将用于保存检查点和日志。

### 方式 1：Docker（推荐）

我们推荐使用 Docker 和提供的镜像。Dockerfile 位于 AReaL 仓库的顶级目录。

```bash
docker pull ghcr.io/areal-project/areal-runtime:v1.0.4-sglang
docker run -it --name areal-node1 \
   --privileged --gpus all --network host \
   --shm-size 700g -v /path/to/mount:/path/to/mount \
   ghcr.io/areal-project/areal-runtime:v1.0.4-sglang \
   /bin/bash
git clone https://github.com/areal-project/AReaL /path/to/mount/AReaL
cd /path/to/mount/AReaL
uv pip install -e . --no-deps
```

vLLM 变体的 Docker 镜像也可使用： `ghcr.io/areal-project/areal-runtime:v1.0.4-vllm`。如果您偏好使用 vLLM
作为推理后端，请将上述命令中的镜像标签替换为该变体。

### 方式 2：自定义环境安装

1. 安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。

1. 克隆仓库：

```bash
git clone https://github.com/areal-project/AReaL
cd AReaL
```

3. （可选）指定自定义索引 URL。

在 `pyproject.toml` 中添加以下内容：

```
[[tool.uv.index]]
url = "https://mirrors.aliyun.com/pypi/simple/"  # 更改为您首选的镜像站
default = true
```

如果网络连接稳定，可以跳过此步骤。

4. 使用 uv sync 安装依赖：

```bash
# 在 Linux 上使用 CUDA 时添加 `--extra cuda` 以获得完整功能
uv sync --extra cuda
# 或者不带 CUDA 支持
# uv sync
# 或者包含开发和测试的额外包
# uv sync --group dev
```

这将安装 CUDA 依赖的训练包（Megatron、Torch Memory Saver）以及 **SGLang** 作为默认推理后端。这些包需要 Linux x86_64 和
CUDA 12.x 及兼容的 NVIDIA 驱动。

#### 使用 vLLM 替代 SGLang

SGLang 和 vLLM 锁定了互不兼容的 `torch` / `torchao` 版本，因此它们位于独立的 pyproject 文件中。默认的
`pyproject.toml` 使用 SGLang；若需使用 vLLM，请使用 `pyproject.vllm.toml`：

```bash
cp pyproject.vllm.toml pyproject.toml
cp uv.vllm.lock uv.lock
uv sync --extra cuda
```

也可以不替换 `pyproject.toml`：

```bash
uv pip install -r pyproject.vllm.toml --extra cuda
```

#### Flash Attention 预编译 Wheel

Flash Attention v2 包含在 `--extra cuda` 中，但 PyPI 仅提供源码分发包，从源码编译耗时约 30 分钟。为跳过编译，请在运行
`uv sync` **之前**安装**预编译 wheel**。

Flash Attention wheel 在编译时与特定 PyTorch 版本绑定。SGLang（默认 `pyproject.toml`）使用 **torch
2.9**，vLLM（`pyproject.vllm.toml`）使用 **torch 2.10**，请选择对应的 wheel。将 `cpXYZ` 替换为您的 Python
版本（3.11 对应 `cp311`，3.12 对应 `cp312`）。

**SGLang**（默认，torch 2.9）：

```bash
# Python 3.12
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
# Python 3.11
# uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl"

uv sync --extra cuda
```

**vLLM**（torch 2.10，需要 `pyproject.vllm.toml`）：

```bash
cp pyproject.vllm.toml pyproject.toml
cp uv.vllm.lock uv.lock
# Python 3.12
uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
# Python 3.11
# uv pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl"

uv sync --extra cuda
```

浏览所有可用 wheel： <https://github.com/mjun0812/flash-attention-prebuild-wheels/releases>。

同样的 `uv sync` 命令也适用于 macOS 和不带 CUDA 支持的 Linux。CUDA 包（包括 flash-attn） 会通过平台标记自动跳过。但是，需要
CUDA 的训练和推理功能将不可用。此配置仅适用于开发、测试和非 GPU 工作流。

您也可以单独安装各个 extra，而不是完整的 `cuda` 捆绑包：

- `sglang`：SGLang 推理引擎（在 `pyproject.toml` 中）
- `vllm`：vLLM 推理引擎（在 `pyproject.vllm.toml` 中）
- `megatron`：Megatron 训练后端
- `tms`：Torch Memory Saver
- `flash-attn`：Flash Attention v2
- `kernels`：Hugging Face Kernels 运行时
- `cuda-train`：仅训练包（megatron + tms + kernels，不含推理后端）
- `cuda`：cuda-train + 推理后端 + flash-attn

### 在训练中使用 Hugging Face Kernels

安装 `kernels` extra 只会让 [Hugging Face Kernels](https://github.com/huggingface/kernels)
运行时可用；训练仍然保持现有默认值，除非您在配置中显式启用。

对训练引擎配置（例如 `actor`、`critic` 或 `teacher`）使用以下字段：

- `attn_impl`：选择注意力后端。除了 `sdpa`、`flash_attention_2` 等内置后端外，还可以填写 Hugging Face kernels 仓库
  ID，例如 `kernels-community/flash-attn` 或
  `kernels-community/flash-attn@main:flash_attn_varlen_func`。
- `use_kernels`：设为 `true` 后，会在模型创建完成后对模型执行 kernelize。

示例：

```yaml
actor:
  attn_impl: kernels-community/flash-attn
  use_kernels: true
```

为了获得可预测的行为，建议显式指定 kernels 仓库 ID，而不是依赖 `flash_attention_*` 的自动回退。

### 额外的 CUDA 包（可选，手动安装）

Docker 镜像包含 `pyproject.toml` 中没有的额外编译包。这些包需要 CUDA，必须从源码编译。如果使用自定义环境（非
Docker）且需要这些包的优化（例如 FP8 训练、融合 Adam 内核），请在运行 `uv sync --extra cuda` 后手动安装：

| 包                | 用途                              | 安装命令                                                                                                                                                          |
| ----------------- | --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| grouped_gemm      | Megatron 中的 MoE 模型支持        | `uv pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@v1.1.4`                                                                       |
| NVIDIA apex       | Megatron 中的融合 Adam 等         | `NVCC_APPEND_FLAGS="--threads 4" APEX_PARALLEL_BUILD=8 APEX_CPP_EXT=1 APEX_CUDA_EXT=1 uv pip install --no-build-isolation git+https://github.com/NVIDIA/apex.git` |
| TransformerEngine | Megatron 中的 FP8 训练、优化 GEMM | `uv pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable`                                                                  |
| flash-attn-3      | Flash Attention v3（Hopper）      | 从源码构建，请参阅 [Dockerfile](https://github.com/areal-project/AReaL/blob/main/Dockerfile)                                                                      |

**重要**：这些包需要 `--no-build-isolation`，因为它们需要访问已安装的 PyTorch 进行 CUDA 编译。先通过
`uv sync --extra cuda` 安装 PyTorch，然后再尝试安装这些包。

### DeepSeek-V3 优化包（可选）

为了以最佳性能运行 DeepSeek-V3 风格的 MoE 模型，Docker 镜像还包含以下包。这些包有复杂的构建要求和 GPU 架构约束。**请参阅
[Dockerfile](https://github.com/areal-project/AReaL/blob/main/Dockerfile)
获取确切的安装命令和环境变量。**

| 包                     | 用途                             | GPU 要求       |
| ---------------------- | -------------------------------- | -------------- |
| FlashMLA               | Multi-head Latent Attention 内核 | SM90+ (Hopper) |
| DeepGEMM               | MoE 的 FP8 GEMM 库               | SM90+ (Hopper) |
| DeepEP                 | Expert Parallelism 通信库        | SM80+ (Ampere) |
| flash-linear-attention | 使用 Triton 内核的线性注意力     | 任意 GPU       |
| NVSHMEM                | DeepEP 节点间通信必需            | 任意 GPU       |

**注意**：

- FlashMLA 和 DeepGEMM 需要 Hopper（H100/H800）或更新的 GPU。
- DeepEP 需要 NVSHMEM 用于节点间和低延迟功能。Docker 镜像通过 pip 安装 `nvidia-nvshmem-cu12`，DeepEP 会自动检测。
- flash-linear-attention 使用纯 Triton 内核，可在任意 GPU 上运行。
- 这些包在 Docker 构建期间从 GitHub 克隆和编译。手动安装时，确保为 FlashMLA 和 DeepGEMM 初始化 git
  子模块（`git submodule update --init --recursive`），因为它们依赖 CUTLASS。

5. 验证 AReaL 安装：

我们提供了一个脚本来验证 AReaL 安装。只需运行：

```bash
uv run python3 areal/tools/validate_installation.py
```

验证通过后，您就可以开始使用了！

(install-skypilot)=

## （可选）安装 SkyPilot

SkyPilot 帮助您在 17 种以上不同云平台或您自己的 Kubernetes 基础设施上轻松运行 AReaL。有关 SkyPilot 的更多详细信息，请参阅
[SkyPilot 文档](https://docs.skypilot.co/en/latest/overview.html)。以下是 在 GCP 或 Kubernetes
上设置 SkyPilot 的最小步骤。

### 使用 pip 安装 SkyPilot

```bash
# 在您的 conda 环境中
# 注意：SkyPilot 需要 3.7 <= python <= 3.13
pip install -U "skypilot[gcp,kubernetes]"
```

### GCP 设置

```bash
# 安装 Google Cloud SDK
conda install -y -c conda-forge google-cloud-sdk

# 初始化 gcloud 并选择您的账户/项目
gcloud init

#（可选）显式选择项目
gcloud config set project <PROJECT_ID>

# 创建应用程序默认凭据
gcloud auth application-default login
```

### Kubernetes 设置

请参阅
[SkyPilot Kubernetes 设置指南](https://docs.skypilot.co/en/latest/reference/kubernetes/kubernetes-setup.html)，了解如何为
SkyPilot 设置 Kubernetes 集群的详细指南。

### 验证

```bash
sky check
```

如果显示 `GCP: enabled` 或 `Kubernetes: enabled`，您就可以将 SkyPilot 与 AReaL 一起使用了。请参阅
[SkyPilot 示例](https://github.com/areal-project/AReaL/blob/main/examples/skypilot/README.md)
获取运行 AReaL 的详细指南。更多选项和详细信息，请参阅官方
[SkyPilot 安装指南](https://docs.skypilot.co/en/latest/getting-started/installation.html)。

## （可选）为分布式训练启动 Ray 集群

在第一个节点上，启动 Ray Head：

```bash
ray start --head
```

在所有其他节点上，启动 Ray Workers：

```bash
# 替换为第一个节点的实际 IP 地址
RAY_HEAD_IP=xxx.xxx.xxx.xxx
ray start --address $RAY_HEAD_IP
```

运行 `ray status` 时应该可以看到 Ray 资源状态。

在 AReaL 的训练命令中适当设置 `n_nodes` 参数，AReaL 将自动检测资源并将 worker 分配到集群。

## 下一步

查看[快速入门部分](quickstart.md)启动您的第一个 AReaL 任务。
