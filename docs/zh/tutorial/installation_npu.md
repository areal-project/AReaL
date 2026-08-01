# 安装指南（昇腾 NPU）

**注意**：昇腾 NPU 支持在
[`ascend-v1.0.4`](https://github.com/areal-project/AReaL/tree/ascend-v1.0.4) 分支维护而非
`main` 分支。本文档中引用的所有代码、配置和示例均指该分支——请按照下文说明检出该分支。

## 前置要求

### 硬件要求

以下硬件配置已经过充分测试：

- **NPU**：每节点 16 个 NPU
- **CPU**：每节点 64 核
- **内存**：每节点 1TB
- **网络**：RoCE 3.2 Tbps
- **存储**：
  - 1TB 本地存储用于单节点实验
  - 10TB 共享存储（NAS）用于分布式实验

### 软件要求

| 组件            |                                                                                  版本                                                                                  |
| --------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| 操作系统        |                                                                Ubuntu、EulerOS 或满足以下要求的任何系统                                                                |
| 昇腾 HDK        |                                                                                 25.2.1                                                                                 |
| CANN            |                                                                                 9.0.0                                                                                  |
| Git LFS         | 下载模型、数据集和 AReaL 代码所需。请参阅[安装指南](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) |
| Docker          |                                                                                 27.2.0                                                                                 |
| AReaL 镜像 (A2) |                                                           `ghcr.io/hwvanici/areal_npu:v1.0.4-a2`（详见下文）                                                           |
| AReaL 镜像 (A3) |                                                           `ghcr.io/hwvanici/areal_npu:v1.0.4-a3`（详见下文）                                                           |

**注意**：本教程不涵盖 CANN 的安装或共享存储挂载，因为这些取决于您特定的节点配置和系统版本。请独立完成这些安装。您可以从 vLLM-Ascend
社区查看更多详情[此页面](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)。

## 运行环境

我们建议使用 Docker 和我们提供的 NPU 镜像。A2 镜像面向 Atlas A2，A3 镜像面向 Atlas A3；两者基于相同的方法构建（参见
`ascend-v1.0.4` 分支上的
[`Dockerfile.a2`](https://github.com/areal-project/AReaL/blob/ascend-v1.0.4/Dockerfile.a2)
/
[`Dockerfile.a3`](https://github.com/areal-project/AReaL/blob/ascend-v1.0.4/Dockerfile.a3)），并预装以下依赖栈：

| 组件            |                           版本                           |
| --------------- | :------------------------------------------------------: |
| 基础镜像        | `quay.io/ascend/cann:9.0.0`（Ubuntu 22.04、Python 3.11） |
| torch_npu       |                       2.9.0.post2                        |
| vLLM-Ascend     |                         v0.18.0                          |
| triton-ascend   |                          3.2.1                           |
| Megatron-Core   |                         v0.16.1                          |
| MindSpeed       |               core_r0.16.0（固定 commit）                |
| Megatron-Bridge |        固定 commit，与 Megatron-Core 0.16.x 兼容         |

其余 AReaL Python 依赖均已根据 `pyproject.npu.toml` 预装。Megatron-LM、MindSpeed 和 Megatron-Bridge
的源码位于 `/areal-workspace` 目录下，并通过 `PYTHONPATH` 使其可导入。

### 创建容器

```bash
WORK_DIR=<your_workspace>
CONTAINER_WORK_DIR=<your_container_workspace>

# 根据您的硬件类型使用 A2/A3 镜像
# IMAGE=ghcr.io/hwvanici/areal_npu:v1.0.4-a2
IMAGE=ghcr.io/hwvanici/areal_npu:v1.0.4-a3
CONTAINER_NAME=areal_npu

cd ${WORK_DIR}

docker pull ${IMAGE}

docker run -itd --cap-add=SYS_PTRACE --net=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci8 \
--device=/dev/davinci9 \
--device=/dev/davinci10 \
--device=/dev/davinci11 \
--device=/dev/davinci12 \
--device=/dev/davinci13 \
--device=/dev/davinci14 \
--device=/dev/davinci15 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
--shm-size=1200g \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /var/log/npu/:/usr/slog \
-v ${WORK_DIR}:${CONTAINER_WORK_DIR} \
--privileged=true \
--name ${CONTAINER_NAME} \
${IMAGE}  \
/bin/bash
```

**对于多节点训练**：请确保在每个节点上挂载共享存储路径（如果使用 Docker，也要挂载到容器中）。此路径将用于保存检查点和日志。

### 自定义环境安装

该镜像在 `/AReaL` 目录下包含一份内置的 AReaL 源代码副本，但该副本可能已过时。建议删除该目录，并使用最新源码重新安装 AReaL。此操作只会替换 AReaL
包本身——所有依赖（包括位于 `/areal-workspace` 下的 MindSpeed 和 Megatron-Bridge）都不会受到影响。

```bash
rm -rf /AReaL

git clone https://github.com/areal-project/AReaL
cd AReaL

# 切换到 ascend 分支
git checkout ascend-v1.0.4

# 安装 AReaL。所有 Python 依赖（来自 pyproject.npu.toml）已预装在镜像中，
# 因此只需安装 AReaL 包本身。
uv pip install --no-deps -e . --system
```

## （可选）启动 Ray 集群用于分布式训练

在第一个节点上，启动 Ray Head：

```bash
ray start --head
```

在所有其他节点上，启动 Ray Worker：

```bash
# 替换为第一个节点的实际 IP 地址
RAY_HEAD_IP=xxx.xxx.xxx.xxx
ray start --address $RAY_HEAD_IP
```

运行 `ray status` 时应该可以看到 Ray 资源状态显示。

正确设置 AReaL 训练命令中的 `n_nodes` 参数，然后 AReaL 的训练脚本将自动检测资源并为集群分配 worker。

## 下一步

查看[快速入门部分](quickstart.md)来熟悉 AReaL 任务的启动方式。在 NPU 上，我们建议从 `ascend-v1.0.4`
分支上的视觉语言模型（VLM）GRPO 示例
[`examples/vlm_npu/`](https://github.com/areal-project/AReaL/tree/ascend-v1.0.4/examples/vlm_npu)
开始，这些示例在 Geometry3K 数据集上进行训练。例如，在单节点上（16 个 NPU，vLLM 推理 + Megatron 训练）训练 Qwen2.5-VL-3B：

```bash
# 部分模型会被 vllm-ascend 优化，但优化后的变体可能不适用于 RLHF 训练。
# 启动前请先禁用。
export USE_OPTIMIZED_MODEL=0

python examples/vlm/geometry3k_grpo.py \
    --config examples/vlm_npu/qwen2_5_vl_3b_geometry3k_grpo.yaml
```

完整的配置列表（Qwen2.5-VL、Qwen3-VL、Qwen3.5、Qwen3.6 稠密与 MoE 模型）、数据集准备以及基于 Ray 的多节点训练，请参阅
`ascend-v1.0.4` 分支上的
[`examples/vlm_npu/README.md`](https://github.com/areal-project/AReaL/blob/ascend-v1.0.4/examples/vlm_npu/README.md)。如果要运行多节点训练，请在启动任务之前确保您的
Ray 集群已按上述说明启动。

**注意**：在昇腾 NPU 上，推理（rollout）通过 `vllm` 引擎（vLLM-Ascend 插件）支持；SGLang 不可用。`fsdp` 和
`megatron`（通过 [MindSpeed](https://gitcode.com/Ascend/MindSpeed)）训练引擎均已支持。
