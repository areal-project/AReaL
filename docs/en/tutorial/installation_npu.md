# Installation (Ascend NPU)

## Prerequisites

### Hardware Requirements

The following hardware configuration has been extensively tested:

- **NPU**: 16x NPU per node
- **CPU**: 64 cores per node
- **Memory**: 1TB per node
- **Network**: RoCE 3.2 Tbps
- **Storage**:
  - 1TB local storage for single-node experiments
  - 10TB shared storage (NAS) for distributed experiments

### Software Requirements

| Component        |                                                                                                Version                                                                                                 |
| ---------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Operating System |                                                                      Ubuntu, EulerOS or any system meeting the requirements below                                                                      |
| Ascend HDK       |                                                                                                 25.2.1                                                                                                 |
| CANN             |                                                                                                 8.5.0                                                                                                  |
| Git LFS          | Required for downloading models, datasets, and AReaL code. See [installation guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) |
| Docker           |                                                                                                 27.2.0                                                                                                 |
| AReaL Image (A2) |                                                            `swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:v1.0.1-a2` (see details below)                                                            |
| AReaL Image (A3) |                                                            `swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:v1.0.1-a3` (see details below)                                                            |

**Note**: This tutorial does not cover the installation of CANN, or shared storage
mounting, as these depend on your specific node configuration and system version. Please
complete these installations independently. You can check out more details from the
vLLM-Ascend community at
[this page](https://docs.vllm.ai/projects/ascend/en/latest/installation.html).

## Runtime Environment

We recommend using Docker with our provided image for NPU, which contains CANN and
pre-built vLLM and vLLM-Ascend.

### Create Container

```bash
WORK_DIR=<your_workspace>
CONTAINER_WORK_DIR=<your_container_workspace>

# Use A2/A3 image depending on your hardware type
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

**For multi-node training**: Ensure a shared storage path is mounted on every node (and
mounted to the container if you are using Docker). This path will be used to save
checkpoints and logs.

### Custom Environment Installation

The image includes a built-in copy of the AReaL source code under `/AReaL`, but it may
be out of date. We recommend removing it and installing AReaL from source.

```bash
rm -rf /AReaL

git clone https://github.com/areal-project/AReaL
cd AReaL

# Checkout to ascend branch
git checkout ascend-v1.0.1

# Install AReaL
pip install -e . --no-deps
```

## (Optional) Launch Ray Cluster for Distributed Training

On the first node, start the Ray Head:

```bash
ray start --head
```

On all other nodes, start the Ray Worker:

```bash
# Replace with the actual IP address of the first node
RAY_HEAD_IP=xxx.xxx.xxx.xxx
ray start --address $RAY_HEAD_IP
```

You should see the Ray resource status displayed when running `ray status`.

Properly set the `n_nodes` argument in AReaL's training command, then AReaL's training
script will automatically detect the resources and allocate workers to the cluster.

## Next Steps

Check the [quickstart section](quickstart.md) to launch your first AReaL job. To run on
NPU, make the following changes:

- **Training script:** use `examples/math/gsm8k_rl.py`
- **Configuration file:** use the provided NPU configuration file
  `examples/math/gsm8k_grpo_npu.yaml`

Follow the instructions there. If you want to run multi-node training with Ray, make
sure your Ray cluster is started as described above before launching the job.

**Note**: Currently, only the `fsdp` training engine and the `vllm` rollout engine
(through the vLLM-Ascend plugin) are supported on Ascend NPU. The `megatron` engine
(through [MindSpeed](https://gitcode.com/Ascend/MindSpeed)) is currently experimental.
