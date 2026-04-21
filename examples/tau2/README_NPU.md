# Running tau2-airline on Ascend NPU

This guide explains how to set up and run **tau2-airline** training on an **Ascend NPU**
(Atlas A3) environment. For a general explanation about the training pipeline, see
[this](README.md).

______________________________________________________________________

# 1. Environment Setup

## 1.1 Create the Container

Create a container using the following image:

`swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:v1.0.1-a3`

You can use the following commands to create the container:

```bash
WORK_DIR=<your_workspace>
CONTAINER_WORK_DIR=<your_container_workspace>

IMAGE=swr.cn-north-9.myhuaweicloud.com/areal/areal_npu:v1.0.1-a3
CONTAINER_NAME=areal_npu

cd "${WORK_DIR}"

docker pull "${IMAGE}"

docker run -itd \
  --cap-add=SYS_PTRACE \
  --net=host \
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
  -v "${WORK_DIR}:${CONTAINER_WORK_DIR}" \
  --privileged=true \
  --name "${CONTAINER_NAME}" \
  "${IMAGE}" \
  /bin/bash
```

______________________________________________________________________

# 2. Install tau2-bench

Clone the following repository and install the benchmark:

```bash
git clone https://github.com/dhh1995/tau2-bench.git
cd tau2-bench
git checkout dhh/async-and-custom-completion
pip install -e .
cd ..
```

______________________________________________________________________

# 3. Install AReaL

Clone the **AReaL** repository and check out the `ascend-v1.0.4` branch:

```bash
git clone https://github.com/inclusionAI/AReaL
cd AReal
git checkout ascend-v1.0.4
pip install -e . --no-deps
```

______________________________________________________________________

# 4. Prepare tau2 Training

Once the setup is complete, follow the steps below.

______________________________________________________________________

# 4.1 Launch the User Simulator

Start the **vLLM OpenAI-compatible API server** on one of your nodes.

Make sure the file `qwen3_nonthinking.jinja` is available. Adjust the port accordingly.

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /path/to/Qwen3-30B-A3B \
  --served-model-name Qwen3-30B-A3B \
  --host 0.0.0.0 \
  --port 30000 \
  --enforce-eager \
  --data-parallel-size 4 \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --chat-template ./qwen3_nonthinking.jinja
```

______________________________________________________________________

# 4.2 Initialize Ray

Ensure **Ray is initialized on all nodes** and that the same codebase is available on
each node.

### Start the Ray Head (first node)

```bash
cd AReal
ray start --head
```

### Start Ray Workers (other nodes)

```bash
cd AReal

# Replace with the actual IP address of the head node
RAY_HEAD_IP=xxx.xxx.xxx.xxx

ray start --address="${RAY_HEAD_IP}:6379"
```

You can verify the cluster by running:

```bash
ray status
```

______________________________________________________________________

# 4.3 Run tau2-airline on 4 nodes

Use the provided YAML file for NPU: `config_30b_moe_airline_npu.yaml`.

For a **4-node setup**:

- Allocate **1 node for the user simulator**.
- Set the following in your config:

```yaml
rollout:
  backend: "vllm:d4t4"

actor:
  backend: "megatron:(attn:d2p4t4|ffn:d1p4e8)"
```

______________________________________________________________________

# 5. Configuration Notes

Before running the benchmark:

1. Update the YAML configuration file `config_30b_moe_airline_npu.yaml` with the correct
   paths.
1. Set `user_llm_base_url` to the **IP address and port of the node running the user
   simulator** and `user_llm` to the model name.
1. Ensure the **vLLM server is started with `--enforce-eager`**.
1. In the YAML configuration, make sure you have:

```yaml
actor:
  enable_tree_training: false

vllm:
  enforce_eager: true

rollout:
  pause_grace_period: 30
```

______________________________________________________________________

# 6. Launch Training

Run the following command:

```bash
python3 examples/tau2/train.py \
  --config examples/tau2/config_30b_moe_airline_npu.yaml
```

Additionally, you can increase the `timeout` of `workflow_kwargs` in `train.py` if you
have observed too many timeouts.
