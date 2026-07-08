# Training VLMs with GRPO on Ascend NPU

This directory contains examples for training vision-language models (VLMs) with GRPO on
Ascend NPU (Atlas A3):

| Config                                 | Model           | Dataset    | Rollout + Train Backend                          | Scale            |
| -------------------------------------- | --------------- | ---------- | ------------------------------------------------ | ---------------- |
| `qwen2_5_vl_3b_geometry3k_grpo.yaml`   | Qwen2.5-VL-3B   | Geometry3K | `vllm:d8` + `megatron:d4t2`                      | 1 node, 16 NPUs  |
| `qwen3_vl_2b_geometry3k_grpo.yaml`     | Qwen3-VL-2B     | Geometry3K | `vllm:d8` + `megatron:d4t2`                      | 1 node, 16 NPUs  |
| `qwen3_5_2b_geometry3k_grpo.yaml`      | Qwen3.5-2B      | Geometry3K | `vllm:d8` + `megatron:d2p4`                      | 1 node, 16 NPUs  |
| `qwen3_6_27b_geometry3k_grpo.yaml`     | Qwen3.6-27B     | Geometry3K | `vllm:d4t2` + `megatron:p2t4`                    | 1 node, 16 NPUs  |
| `qwen3_6_35b_a3b_geometry3k_grpo.yaml` | Qwen3.6-35B-A3B | Geometry3K | `vllm:d8t2` + `megatron:(attn:d2p4t2\|ffn:p4e4)` | 2 nodes, 32 NPUs |
| `qwen2_5_vl_3b_fsdp_virl39k_grpo.yaml` | Qwen2.5-VL-3B   | ViRL39K    | `vllm:d32` + `fsdp:d16`                          | 3 nodes, 48 NPUs |

The single-node Geometry3K examples run with the `local` scheduler; the multi-node
examples are scheduled via Ray.

______________________________________________________________________

# 1. Environment Setup

## 1.1 Create the Container

Create a container using the latest AReaL NPU image (A3) and the container creation
commands from the
[NPU installation guide](https://areal-project.github.io/AReaL/en/tutorial/installation_npu.html).

For multi-node training, create the container on every node.

## 1.2 Install AReaL

Inside the container, clone the AReaL repository and check out the `ascend-v1.0.4`
branch:

```bash
git clone https://github.com/inclusionAI/AReaL
cd AReaL
git checkout ascend-v1.0.4
pip install -e . --no-deps
```

______________________________________________________________________

# 2. Prepare Datasets

> If your cluster cannot reach `huggingface.co` directly, set a mirror before
> downloading, e.g. `export HF_ENDPOINT=https://hf-mirror.com`.

## 2.1 Geometry3K

The Geometry3K configs point `train_dataset.path` / `valid_dataset.path` to
[`hiyouga/geometry3k`](https://huggingface.co/datasets/hiyouga/geometry3k), which is
downloaded automatically from the HuggingFace Hub on first run.

To pre-download it instead (recommended for offline clusters):

```bash
huggingface-cli download hiyouga/geometry3k --repo-type dataset --local-dir data/geometry3k
```

Then override the dataset paths when launching training:

```bash
... train_dataset.path=data/geometry3k valid_dataset.path=data/geometry3k
```

## 2.2 ViRL39K

Download [`TIGER-Lab/ViRL39K`](https://huggingface.co/datasets/TIGER-Lab/ViRL39K) and
extract the image archive next to the parquet file:

```bash
huggingface-cli download TIGER-Lab/ViRL39K --repo-type dataset --local-dir data/ViRL39K
cd data/ViRL39K && unzip images.zip && cd -
```

The resulting layout should be:

```
data/ViRL39K/
├── 39Krelease.parquet
└── images/
    ├── Processed-xxx-0.jpg
    └── ...
```

The ViRL39K config already sets `train_dataset.path: data/ViRL39K/39Krelease.parquet`
(relative to the AReaL repository root). The dataset loader expects the `images/` folder
to sit in the same directory as the parquet file. For multi-node training, the dataset
must be available at the same path on every node — place it on shared storage (NAS) or
copy it to each node.

______________________________________________________________________

# 3. Single-Node Training (Geometry3K)

Before launching, disable vllm-ascend optimized models. Some models are optimized by
vllm-ascend, but the optimized variants may be unsuitable for RLHF training:

```bash
export USE_OPTIMIZED_MODEL=0
```

All single-node examples use 16 NPUs, split between vLLM rollout and Megatron
(MindSpeed) training as listed in the table above. Launch the one you need:

### Qwen2.5-VL-3B

```bash
python examples/vlm/geometry3k_grpo.py \
    --config examples/vlm_npu/qwen2_5_vl_3b_geometry3k_grpo.yaml
```

### Qwen3-VL-2B

```bash
python examples/vlm/geometry3k_grpo.py \
    --config examples/vlm_npu/qwen3_vl_2b_geometry3k_grpo.yaml
```

### Qwen3.5-2B

```bash
python examples/vlm/geometry3k_grpo.py \
    --config examples/vlm_npu/qwen3_5_2b_geometry3k_grpo.yaml
```

### Qwen3.6-27B

```bash
python examples/vlm/geometry3k_grpo.py \
    --config examples/vlm_npu/qwen3_6_27b_geometry3k_grpo.yaml
```

The Geometry3K examples share the same dataset and GRPO training configuration but use
different model architectures and parallelism strategies. The smaller 2B/3B models are
more suitable for environments with limited resources.

______________________________________________________________________

# 4. Multi-Node Training

Two examples run on multiple nodes via the Ray scheduler:

- **Qwen3.6-35B-A3B (MoE) on Geometry3K** — 2 nodes × 16 NPUs, `vllm:d8t2` rollout +
  Megatron training with expert parallelism.
- **Qwen2.5-VL-3B on ViRL39K** — 3 nodes × 16 NPUs, split into 32 NPUs for vLLM rollout
  and 16 NPUs for FSDP training (`vllm:d32` + `fsdp:d16`), with long-context generation
  (16384 max new tokens).

## 4.1 Preparation

- Make sure the same AReaL codebase and dataset paths are available on **all nodes**
  (see [Section 2](#2-prepare-datasets)).
- Point `cluster.fileroot` and `cluster.name_resolve.nfs_record_root` in the config to
  shared storage (NAS) accessible from all nodes.
- Export `USE_OPTIMIZED_MODEL=0` before starting Ray on each node so that workers
  inherit it. (The ViRL39K config also sets it explicitly via
  `actor.scheduling_spec.env_vars`.)

## 4.2 Initialize the Ray Cluster

### Start the Ray Head (first node)

```bash
cd AReaL
ray start --head
```

### Start Ray Workers (other nodes)

```bash
cd AReaL

# Replace with the actual IP address of the head node
RAY_HEAD_IP=xxx.xxx.xxx.xxx

ray start --address="${RAY_HEAD_IP}:6379"
```

You can verify the cluster by running:

```bash
ray status
```

All nodes declared in the config (`cluster.n_nodes` × `cluster.n_gpus_per_node`) must be
visible in the Ray cluster before launching.

## 4.3 Launch Training

Run the training command on the head node.

### Qwen3.6-35B-A3B on Geometry3K (2 nodes)

```bash
python examples/vlm/geometry3k_grpo.py \
    --config examples/vlm_npu/qwen3_6_35b_a3b_geometry3k_grpo.yaml
```

### Qwen2.5-VL-3B on ViRL39K (3 nodes)

```bash
python examples/vlm_npu/virl39k_grpo.py \
    --config examples/vlm_npu/qwen2_5_vl_3b_fsdp_virl39k_grpo.yaml
```

______________________________________________________________________

# 5. Benchmark Results

## 5.1 Testing Qwen2.5-VL-3B

### Hardware

The following hardware configuration has been extensively tested:

- **NPU**: 16x NPU per node
- **CPU**: 64 cores per node
- **Memory**: 1TB per node
- **Network**: RoCE 3.2 Tbps
- **Storage**:
  - 1TB local storage for single-node experiments
  - 10TB shared storage (NAS) for distributed experiments

### Key Contributions

- Trained Qwen2.5VL-3B-instruct model upto 70 epochs with (4 cards+ 4 cards) train-infer
  configuration. Took around 19hr to finish full training.
- Trained model is tested with more than one benchmark using VLMEvalKit.

### Results

We trained Qwen2.5-VL-3B for 70 epochs on Geometry3K and evaluated the checkpoints using
VLMEvalKit on out of distribution tasks such as MathVision, MathVista, and LogicVista.
The training was performed on both NPU and GPU and results are as follows:

| Method     | LogicVista | MathVision_mini | MathVista_mini | Avg.     |
| ---------- | ---------- | --------------- | -------------- | -------- |
| Base Model | 31.0       | 18.3            | 52.3           | 33.8     |
| GRPO-GPU   | 35.4       | 20.9            | 55.9           | **37.4** |
| GRPO-NPU   | 35.3       | 20.5            | 54.7           | **36.8** |

## 5.2 AReaL vs. verl: Multi-node Training Performance

We test performance of AReaL for large-scale multi-node training with long context
generation and compare this performance with verl synchronous training with the same
training settings. All training and evaluation is done on Ascend NPU.

AReaL asynchronous settings

| Framework | Nodes | N_GPUS Per Node | Train Dataset       | Max Generated Tokens | Max Head Offpolicyness | Batch Size | Allocation Mode |
| --------- | ----- | --------------- | ------------------- | -------------------- | ---------------------- | ---------- | --------------- |
| AReaL     | 3     | 16              | `TIGER-Lab/ViRL39K` | 16384                | 4                      | 528        | vllm:d32+d16    |
| verl      | 3     | 16              | `TIGER-Lab/ViRL39K` | 16384                | -                      | 528        | -               |

### Training setup

We trained Qwen2.5-VL-3B following the settings in the above table:

- **AReaL config**: `examples/vlm_npu/qwen2_5_vl_3b_fsdp_virl39k_grpo.yaml`
- **Dataset**: [`TIGER-Lab/ViRL39K`](https://huggingface.co/datasets/TIGER-Lab/ViRL39K)

### Performance Comparison

We compare the training time and out-of-distribution (OOD) performance of both
frameworks. For OOD evaluation, we use VLMEvalKit and report `Avg@8` accuracy to report
the performance.

| Framework | Method   | Checkpoint | Training Time | LogicVista | MathVision_mini | WeMath   | DynaMath | MathVerse | MMMU_Pro_v | Avg. |
| --------- | -------- | ---------- | ------------- | ---------- | --------------- | -------- | -------- | --------- | ---------- | ---- |
| verl      | GRPO-NPU | Epoch 1    | 6.8 hours     | 33.0       | 18.8            | **19.6** | 32.9     | **31.3**  | **23.5**   | 26.5 |
| AReaL     | GRPO-NPU | Epoch 2    | **4.3 hours** | 33.8       | 20.1            | 17.4     | 34.9     | 29.6      | 22.3       | 26.3 |
| AReaL     | GRPO-NPU | Epoch 3    | **6.6 hours** | **34.1**   | **20.3**        | 18.9     | **35.7** | 30.2      | 22.5       | 27.0 |

Under identical hardware and training configurations, AReaL reaches two training epochs
in 4.3 hours, whereas verl requires 6.8 hours to complete a single epoch. Despite the
shorter wall-clock time, AReaL achieves comparable OOD performance at that stage (Avg@8:
26.3 vs. 26.5). With additional training (Epoch 3), AReaL surpasses verl in overall
accuracy (27.0) while still requiring less total training time (6.6 hours). These
results suggest that AReaL’s asynchronous training strategy improves time-to-performance
efficiency for large-scale, long-context GRPO training without sacrificing downstream
generalization.
