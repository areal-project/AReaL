# 安装指南（昇腾 NPU）

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
| CANN            |                                                                                 8.5.0                                                                                  |
| Git LFS         | 下载模型、数据集和 AReaL 代码所需。请参阅[安装指南](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) |
| Docker          |                                                                                 27.2.0                                                                                 |
| AReaL 镜像 (A2) |                                                `swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:v1.0.1-a2`（详见下文）                                                |
| AReaL 镜像 (A3) |                                                `swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:v1.0.1-a3`（详见下文）                                                |

**注意**：本教程不涵盖 CANN 的安装或共享存储挂载，因为这些取决于您特定的节点配置和系统版本。请独立完成这些安装。您可以从 vLLM-Ascend
社区查看更多详情[此页面](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)。

## 运行环境

我们建议使用 Docker 和我们提供的 NPU 镜像，其中包含 CANN 以及预构建的 vLLM 和 vLLM-Ascend。

### 创建容器

```bash
WORK_DIR=<your_workspace>
CONTAINER_WORK_DIR=<your_container_workspace>

# 根据您的硬件类型使用 A2/A3 镜像
# IMAGE=swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:v1.0.1-a2
IMAGE=swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:v1.0.1-a3
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

该镜像在 `/AReaL` 目录下包含一份内置的 AReaL 源代码副本，但该副本可能已过时。建议删除该目录，并从源码重新安装 AReaL。

```bash
rm -rf /AReaL

git clone https://github.com/areal-project/AReaL
cd AReaL

# 切换到 ascend 分支
git checkout ascend-v1.0.1

# 安装 AReaL
pip install -e . --no-deps
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

查看[快速入门部分](quickstart.md)来启动您的第一个 AReaL 任务。要在 NPU 上运行，请进行以下更改：

- **训练脚本：** 使用 `examples/math/gsm8k_rl.py`
- **配置文件：** 使用提供的 NPU 配置文件 `examples/math/gsm8k_grpo_npu.yaml`

按照那里的说明进行操作。如果要使用 Ray 运行多节点训练，请在启动任务之前确保您的 Ray 集群已按上述说明启动。

**注意**：目前，昇腾 NPU 上仅支持 `fsdp` 训练引擎和 `vllm` 推理引擎（通过 vLLM-Ascend 插件）。`megatron` 引擎（通过
[MindSpeed](https://gitcode.com/Ascend/MindSpeed)）目前仍处于实验阶段。
