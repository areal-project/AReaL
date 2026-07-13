# Installation (Ascend NPU)

**Note**: Ascend NPU support is maintained on the
[`ascend-v1.0.4`](https://github.com/areal-project/AReaL/tree/ascend-v1.0.4) branch
rather than `main`. All code, configurations, and examples referenced in this document
refer to that branch — make sure to check it out as described below.

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
| CANN             |                                                                                                 9.0.0                                                                                                  |
| Git LFS          | Required for downloading models, datasets, and AReaL code. See [installation guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) |
| Docker           |                                                                                                 27.2.0                                                                                                 |
| AReaL Image (A2) |                                                                       `ghcr.io/hwvanici/areal_npu:v1.0.4-a2` (see details below)                                                                       |
| AReaL Image (A3) |                                                                       `ghcr.io/hwvanici/areal_npu:v1.0.4-a3` (see details below)                                                                       |

**Note**: This tutorial does not cover the installation of CANN, or shared storage
mounting, as these depend on your specific node configuration and system version. Please
complete these installations independently. You can check out more details from the
vLLM-Ascend community at
[this page](https://docs.vllm.ai/projects/ascend/en/latest/installation.html).

## Runtime Environment

We recommend using Docker with our provided image for NPU. The A2 image targets Atlas A2
and the A3 image targets Atlas A3; both are built from the same recipe (see
[`Dockerfile.a2`](https://github.com/areal-project/AReaL/blob/ascend-v1.0.4/Dockerfile.a2)
/
[`Dockerfile.a3`](https://github.com/areal-project/AReaL/blob/ascend-v1.0.4/Dockerfile.a3)
on the `ascend-v1.0.4` branch) and ship the following pre-built stack:

| Component       |                         Version                         |
| --------------- | :-----------------------------------------------------: |
| Base image      | `quay.io/ascend/cann:9.0.0` (Ubuntu 22.04, Python 3.11) |
| torch_npu       |                       2.9.0.post2                       |
| vLLM-Ascend     |                         v0.18.0                         |
| triton-ascend   |                          3.2.1                          |
| Megatron-Core   |                         v0.16.1                         |
| MindSpeed       |              core_r0.16.0 (pinned commit)               |
| Megatron-Bridge |   pinned commit compatible with Megatron-Core 0.16.x    |

All other AReaL Python dependencies are pre-installed from `pyproject.npu.toml`.
Megatron-LM, MindSpeed, and Megatron-Bridge sources live under `/areal-workspace` and
are made importable via `PYTHONPATH`.

### Create Container

```bash
WORK_DIR=<your_workspace>
CONTAINER_WORK_DIR=<your_container_workspace>

# Use A2/A3 image depending on your hardware type
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

**For multi-node training**: Ensure a shared storage path is mounted on every node (and
mounted to the container if you are using Docker). This path will be used to save
checkpoints and logs.

### Custom Environment Installation

The image includes a built-in copy of the AReaL source code under `/AReaL`, but it may
be out of date. We recommend removing it and installing AReaL from the latest source.
This only replaces the AReaL package itself — all dependencies (including MindSpeed and
Megatron-Bridge under `/areal-workspace`) stay untouched.

```bash
rm -rf /AReaL

git clone https://github.com/areal-project/AReaL
cd AReaL

# Checkout to ascend branch
git checkout ascend-v1.0.4

# Install AReaL. All Python dependencies (from pyproject.npu.toml) are already
# installed in the image, so only the AReaL package itself needs to be installed.
uv pip install --no-deps -e . --system
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

Check the [quickstart section](quickstart.md) to get familiar with launching AReaL jobs.
On NPU, we recommend starting from the vision-language model (VLM) GRPO examples in
[`examples/vlm_npu/`](https://github.com/areal-project/AReaL/tree/ascend-v1.0.4/examples/vlm_npu)
on the `ascend-v1.0.4` branch, which train on the Geometry3K dataset. For example, to
train Qwen2.5-VL-3B on a single node (16 NPUs, vLLM rollout + Megatron training):

```bash
# Some models are optimized by vllm-ascend, but the optimized variants
# may be unsuitable for RLHF training. Disable them before launching.
export USE_OPTIMIZED_MODEL=0

python examples/vlm/geometry3k_grpo.py \
    --config examples/vlm_npu/qwen2_5_vl_3b_geometry3k_grpo.yaml
```

See the
[`examples/vlm_npu/README.md`](https://github.com/areal-project/AReaL/blob/ascend-v1.0.4/examples/vlm_npu/README.md)
on the `ascend-v1.0.4` branch for the full list of configurations (Qwen2.5-VL, Qwen3-VL,
Qwen3.5, Qwen3.6 dense and MoE), dataset preparation, and multi-node training with Ray.
If you want to run multi-node training, make sure your Ray cluster is started as
described above before launching the job.

**Note**: On Ascend NPU, rollout is supported through the `vllm` engine (via the
vLLM-Ascend plugin); SGLang is not available. Both the `fsdp` and `megatron` (through
[MindSpeed](https://gitcode.com/Ascend/MindSpeed)) training engines are supported.
