# 运行环境与固定输入

本说明使用 Linux amd64、8×H20、Python 3.12。训练 GPU、CPU sandbox 与控制 LLM 分开配置。**现有 H20
已通过环境预检；本文整理后的全新机器部署尚未验收。** 不需要为了整理文档重跑训练。

## 训练容器

现有容器使用的公开镜像 digest 已记录在 [inputs.lock.json](reproduction/inputs.lock.json)：

```bash
# Linux GPU 主机；DEMO_HOME 使用仓库外的绝对路径。
export DEMO_HOME=/path/to/elastic-demo-data
mkdir -p "$DEMO_HOME"
export TRAINING_IMAGE=ghcr.io/areal-project/areal-runtime@sha256:751c93268976c2157d27e125e6b5599f29ed14a148f90d71b18c27892d0bb226

docker pull "$TRAINING_IMAGE"
docker run --rm -it --name elastic-demo --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/AReaL -v "$DEMO_HOME":/demo \
  -w /workspace/AReaL --entrypoint /bin/bash "$TRAINING_IMAGE"

# 容器内：使用镜像已有 Python，并加载本次 demo 源码。
export PATH=/opt/.venv/bin:$PATH
export PYTHONPATH=/workspace/AReaL
nvidia-smi
python -c 'import torch, sglang, awex, megatron.core, transformer_engine; print(torch.__version__, torch.cuda.device_count())'
```

请确保八卡可用，输出目录位于支持 SQLite WAL／flock 的本地文件系统。SDK 源码、venv、kubectl、kubeconfig
也必须在容器内可见；`runtime.json` 中填写容器内路径。使用现有 Kubernetes 时，把其 kubeconfig 引用的证书一并挂载，并检查 API
地址在容器内可达。

已运行环境观测：Torch 2.9.1+cu129、SGLang 0.5.10.post1、AWEX 0.8.1、Megatron Core 0.17.0、MBridge
0.15.1、TransformerEngine 2.14.1+27de67e1、Transformers 5.3.0。完整观测快照保存在项目外层
`artifacts/elastic-inline-20261030/reference-results/environment-observed.json`。这是**现有容器实测版本**；尚未核验其与新拉取镜像内所有包完全相同。仓库
Dockerfile 部分上游依赖未固定，当前不声称重新 build 能得到相同镜像。

## 模型与 SWE-Gym 数据

在训练环境内执行；Hugging Face 下载使用已有 `huggingface_hub`：

```bash
mkdir -p /demo/model /demo/data /demo/src
python - <<'PY'
import json
from huggingface_hub import snapshot_download, hf_hub_download
lock = json.load(open('examples/elastic/reproduction/inputs.lock.json'))
snapshot_download(lock['model']['repo'], revision=lock['model']['revision'], local_dir='/demo/model')
hf_hub_download(lock['dataset']['dataset'], repo_type='dataset',
    revision=lock['dataset']['revision'], filename='data/train-00000-of-00001.parquet', local_dir='/demo/data')
PY

git clone https://github.com/SWE-Gym/SWE-Bench-Fork.git /demo/src/SWE-Bench-Fork
git -C /demo/src/SWE-Bench-Fork checkout 242429c188fcfd06aad13fce9a54d450470bf0ac

python -m examples.elastic.swe.prepare_data \
  --gym-parquet /demo/data/data/train-00000-of-00001.parquet \
  --dataset-revision bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb \
  --harness /demo/src/SWE-Bench-Fork \
  --harness-revision 242429c188fcfd06aad13fce9a54d450470bf0ac \
  --images examples/elastic/reproduction/task-images.json \
  --count 8 --seed 1 --repeats 2 --output /demo/data/tasks.jsonl
```

[Qwen 模型](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554)
的三个权重分片及 tokenizer／配置哈希已与现用模型匹配。数据和镜像清单在运行前还会进行哈希检查。Gold patch 仅用于 `.fixtures/`
环境验证，不进入训练 JSONL。

锁定的首遍顺序为
DVC-1700、pandas-57232、Dask-10159、pandas-56016、mypy-15006、MONAI-5486、pandas-53975、Dask-7305；第二遍精确顺序在
lock 的 `row_order` 中。原生错误补采可能改变实际训练消费的题目组合，所以还要保留 TRAINING 事件。

## 独立 SWE SDK

源码匹配核验：现部署 AEnvironment 的全部 67 个 Python 文件、SWEAgent 的全部 22 个 Python 文件分别与以下公开 revision
一致。不需要另外施加 Python SDK 源码补丁。

```bash
git clone https://github.com/inclusionAI/AEnvironment.git /demo/src/AEnvironment
git -C /demo/src/AEnvironment checkout 667f743428d358f4fb84857835d447cd1d9b31ba
git clone https://github.com/areal-project/AReaL-SWEAgent.git /demo/src/AReaL-SWEAgent
git -C /demo/src/AReaL-SWEAgent checkout f144900d5e0a019bf5381b624c442b49999ffa4d

uv venv --python python3.12 /demo/sdk-venv
uv --no-config pip install --python /demo/sdk-venv/bin/python \
  -e /demo/src/AEnvironment/aenv \
  -r examples/elastic/reproduction/sdk-requirements.txt

PYTHONPATH=/demo/src/AEnvironment/aenv/src:/demo/src/AReaL-SWEAgent \
  /demo/sdk-venv/bin/python -c 'from aweagent.envs.swe import AenvSWE; from aenv.core.environment import Environment'
```

训练侧 Pydantic 与 SDK 的 `<2.12` 约束不同，不能把 SDK 装进训练
Python。[sdk-requirements.txt](reproduction/sdk-requirements.txt) 固定现有环境全部 86 个非 editable
包的版本；AEnvironment 单独从固定源码安装。现有 87 包依赖一致性检查通过，干净重装尚未验收。此清单未包含 wheel 文件哈希。

适配已包含在本 demo：`swe/sandbox_worker.py` 设置 HTTP transport timeout、适配 SDK 的 `old_srt`
参数拼写、保存 sandbox ownership；`swe/sandbox.py` 管理独立进程及确认删除；`swe/gym.py` 打包固定 SWE-Gym
评分脚本；`swe/workflow.py` 对评分异常返回 0 并记录诊断。无需再修改上游 SDK。

## AEnvironment 与任务镜像

独立单机 CPU sandbox 部署命令见 [swe/deploy/README.md](swe/deploy/README.md)。它包含服务／shell
镜像构建、kind／Helm、注册与端口转发；部署文件来自已运行环境的准备配置，本轮尚未重新部署验收。

任务镜像使用 [task-images.json](reproduction/task-images.json) 中八个公开 digest 拉取。AEnvironment 的
`imagePrefix:datasource` 接口要求本地任务别名；别名只是部署映射，不能代替原始 digest 记录。部署后按模板建立私有 runtime 文件：

```bash
mkdir -p /demo/private
cp examples/elastic/swe/runtime.example.json /demo/private/runtime.json
# 编辑其中 SDK Python/PYTHONPATH、kubectl、kubeconfig 为容器内实际路径。
# environment: autopilot-swe-bash@0.1.0
# system_url: http://127.0.0.1（SDK 在该主机上使用 control 8080 / data 8081）
```

只检查 `http://127.0.0.1:80` 会产生错误结论。轻量预检：

```bash
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8083/health
curl --fail http://127.0.0.1:8083/env/autopilot-swe-bash/0.1.0
```

首次部署应逐题执行真实空／gold 验证，包含 sandbox 创建、工具、评分和删除；不加载模型，不训练：

```bash
for fixture in /demo/data/tasks.fixtures/*.json; do
  task=$(basename "$fixture" .json)
  python -m examples.elastic.swe.validate \
    --fixture "$fixture" --runtime /demo/private/runtime.json \
    --output "/demo/validation/$task" || exit 1
done
```

此验证严格要求空 patch=0、gold patch=1，不使用训练中的“异常记零”容错。现有部署此前八题验证通过；新机器应自己完成验证。随后执行主 README
的输入预检与两轮运行。
